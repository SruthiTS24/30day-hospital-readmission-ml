import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import gdown

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="30-Day Readmission Risk Predictor", layout="centered")

# ---------------------------------------------------------
# Model file is too large for GitHub (175MB), so it is hosted
# on Google Drive and downloaded once on first app run.
# Replace YOUR_FILE_ID_HERE with the actual Google Drive file ID.
# ---------------------------------------------------------
# ---------------------------------------------------------
# Resolve paths relative to THIS script's location, since
# Streamlit Cloud's working directory is the repo root,
# not the folder containing app.py
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE_ID = "1LMIdPhmHcd3ILx0AAWzriFx2BEj-5KWa"
MODEL_PATH = os.path.join(SCRIPT_DIR, "rf_tuned_model.pkl")

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only, ~175MB)..."):
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

# ---------------------------------------------------------
# Load model and artifacts (cached so they only load once)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    ensure_model_downloaded()
    rf_tuned = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(os.path.join(SCRIPT_DIR, "feature_columns.pkl"))
    final_threshold = joblib.load(os.path.join(SCRIPT_DIR, "final_threshold.pkl"))
    default_row = joblib.load(os.path.join(SCRIPT_DIR, "default_row.pkl"))
    rf_model_only = rf_tuned.named_steps["rf"]
    explainer = shap.TreeExplainer(rf_model_only)
    return rf_tuned, feature_columns, final_threshold, default_row, explainer

rf_tuned, feature_columns, final_threshold, default_row, explainer = load_artifacts()

# ---------------------------------------------------------
# Feature builder — converts simple user input into the full
# feature row the model expects (same logic developed and
# tested in the modeling notebook)
# ---------------------------------------------------------
def build_patient_features(age_bracket, gender, time_in_hospital,
                            num_medications, number_inpatient, number_emergency):
    row = default_row.copy()

    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    row['age_ordinal'] = age_order.index(age_bracket)

    row['gender_Male'] = 1 if gender == 'Male' else 0
    if 'gender_Unknown/Invalid' in row.index:
        row['gender_Unknown/Invalid'] = 0

    row['time_in_hospital'] = time_in_hospital
    row['num_medications'] = num_medications
    row['number_inpatient'] = number_inpatient
    row['number_emergency'] = number_emergency

    row = row[feature_columns]
    return row

# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------
st.title("30-Day Hospital Readmission Risk Predictor")
st.markdown(
    "This tool estimates a patient's risk of hospital readmission within "
    "30 days, based on a Random Forest model trained on the UCI Diabetes "
    "130-US Hospitals dataset (1999-2008). **This is a prototype for "
    "portfolio/demonstration purposes and is not a validated clinical tool.**"
)

st.divider()
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age_bracket = st.selectbox(
        "Age Range",
        ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
        index=6
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)

with col2:
    num_medications = st.slider("Number of Medications", 1, 80, 15)
    number_inpatient = st.slider("Prior Inpatient Visits (past year)", 0, 20, 0)
    number_emergency = st.slider("Prior Emergency Visits (past year)", 0, 20, 0)

st.divider()

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict Readmission Risk", type="primary"):
    patient_row = build_patient_features(
        age_bracket, gender, time_in_hospital,
        num_medications, number_inpatient, number_emergency
    )
    patient_df = pd.DataFrame([patient_row])

    proba = rf_tuned.predict_proba(patient_df)[:, 1][0]
    is_high_risk = proba >= final_threshold

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Readmission Risk Score", f"{proba:.1%}")
    with col2:
        if is_high_risk:
            st.error("**High Risk** of 30-day readmission")
        else:
            st.success("**Lower Risk** of 30-day readmission")

    st.caption(f"Classification threshold: {final_threshold:.2f} "
               f"(selected to balance catching true readmissions against "
               f"over-flagging low-risk patients)")

    # -----------------------------------------------------
    # Feature importance context
    # -----------------------------------------------------
    st.divider()
    st.subheader("What typically drives this model's predictions?")
    st.markdown(
        "Based on SHAP analysis performed during model development, this "
        "model's predictions are influenced by factors including prior "
        "hospital visit history, medication patterns, diagnosis category, "
        "and admission details. Live per-patient SHAP explanations are not "
        "shown here, as this deep Random Forest model produced numerically "
        "unstable SHAP outputs for individual predictions during testing "
        "(a known limitation of TreeExplainer on unconstrained tree "
        "ensembles). Full global and individual SHAP analysis, computed "
        "and verified offline, is documented in the project notebook."
    )

st.divider()
st.caption(
    "Data source: UCI Machine Learning Repository — Diabetes 130-US "
    "Hospitals for Years 1999-2008. Model: Random Forest (tuned), "
    "Test ROC-AUC = 0.591. Built as a data science capstone project."
)
