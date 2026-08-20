# 30-Day Hospital Readmission Risk Prediction: A SQL-Driven and Explainable Machine Learning Approach

A full end-to-end data science project predicting 30-day hospital readmission risk for diabetic patients, using SQL-based relational data extraction, machine learning, explainable AI (SHAP), and a deployed interactive web application.

**Author:** Sruthi T S

**Affiliation:** Entri Elevate

**Project Date:** August 21, 2026

**Live Application:** Deployed via Streamlit Cloud : https://30-day-hospital-readmission-predictor.streamlit.app/

---

## Problem Statement

Hospital readmissions within 30 days of discharge represent a significant challenge for healthcare systems, both clinically and financially. Under the U.S. Centers for Medicare & Medicaid Services' Hospital Readmissions Reduction Program, hospitals face financial penalties for excess readmissions, making early identification of at-risk patients a priority for care management and discharge planning. Despite established clinical guidelines for post-discharge care, many patients — particularly those with chronic conditions like diabetes — are readmitted due to inconsistent follow-up, inadequate risk stratification, or gaps in identifying high-risk patients before discharge. This project addresses the need for a data-driven approach to flag patients at elevated readmission risk, using structured hospital encounter data.

## Objective

The objective of this project is to build an end-to-end machine learning pipeline that predicts whether a patient will be readmitted to the hospital within 30 days of discharge, using structured clinical and administrative data. Specifically, this project aims to:

1. Extract and structure hospital encounter data using a relational (SQL-based) approach, reflecting how such data is typically organized in real hospital information systems.
2. Perform thorough data cleaning, exploratory analysis, and feature engineering to prepare the data for modeling.
3. Train and evaluate machine learning models to predict 30-day readmission risk, addressing the significant class imbalance inherent in this problem.
4. Apply explainable AI techniques (SHAP) to identify the key clinical and administrative factors driving readmission risk, making model predictions interpretable for potential clinical or administrative use.
5. Deploy the final model as an interactive web application, allowing users to input patient information and receive a readmission risk estimate along with an explanation of the prediction.

## Data Description

This project uses the **Diabetes 130-US Hospitals dataset (1999–2008)**, sourced from the UCI Machine Learning Repository. The dataset represents ten years of clinical care data from 130 US hospitals and integrated delivery networks, comprising **101,766 hospital encounters** for patients diagnosed with diabetes.

Each row represents a single inpatient encounter, restricted to admissions that involved a diabetes-related diagnosis, a length of stay between 1 and 14 days, at least one laboratory test, and at least one medication administered. The dataset includes 47 features covering patient demographics (age, race, gender), admission details (admission type, source, and discharge disposition), clinical information (diagnosis codes, number of lab procedures, number of diagnoses), and medication data (23 diabetes-related medications, insulin usage, and whether medication was changed during the encounter).

The target variable, `readmitted`, originally recorded whether a patient was not readmitted, readmitted after 30 days, or readmitted within 30 days. For this project, the target was converted into a binary variable (`readmitted_30d`) indicating whether a patient was readmitted within 30 days of discharge — the outcome of primary clinical and financial interest.

**Data source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

**Known limitation:** The dataset spans 1999–2008 and uses ICD-9 diagnosis coding, which was officially replaced by ICD-10 in the US in 2015. As such, this project should be understood as a methodology demonstration rather than a production-ready clinical tool.

---

## Project Pipeline

### 1. Data Extraction (SQL)
Raw data was split into four relational tables (`patients`, `encounters`, `diagnoses`, `procedures_meds`) — reflecting how patient data is typically structured across linked systems in real hospital IT environments — loaded into a SQLite database, and reconstructed into a single modeling dataset using SQL JOIN queries.

### 2. Preprocessing
- Handled missing values with column-specific, evidence-based decisions (e.g., dropped `weight` at 96.85% missing; retained `A1Cresult`/`max_glu_serum` with an explicit "Not Tested" category, since missingness itself is clinically meaningful)
- Mapped numeric admission/discharge/source IDs to readable clinical labels
- Applied **Clinical Cohort Filtering**, removing 2,423 encounters, where the patient expired or was discharged to hospice — outcomes that do not reflect genuine "non-readmission" and would otherwise mislead the model

### 3. Exploratory Data Analysis
Univariate and bivariate analysis (with sample-size verification to avoid small-sample noise) identified prior inpatient visits and prior emergency visits as the strongest marginal predictors of 30-day readmission, alongside a clear age-related trend and elevated risk among patients discharged to rehabilitation or skilled nursing facilities.

### 4. Feature Engineering
- Grouped 700+ ICD-9 diagnosis codes into clinically meaningful categories (Circulatory, Respiratory, Diabetes, etc.), following standard ICD-9 chapter classifications
- Removed 15 near-zero-variance medication columns (>99% single-category dominance)
- Applied ordinal encoding (age, lab results, medication dosage changes) and one-hot encoding (demographics, admission details, diagnosis categories)
- Final dataset: 99,343 rows × 120 fully numeric features

### 5. Modeling
- Addressed class imbalance (88.6% / 11.4%) using SMOTE, applied to training data only
- Trained and compared Logistic Regression, Random Forest, and XGBoost; selected Random Forest based on strongest minority-class performance
- Tuned Random Forest via `RandomizedSearchCV`, identifying and correcting a SMOTE/cross-validation data leakage issue (inflated CV ROC-AUC of 0.95 vs. genuine test performance of 0.59)
- Selected a classification threshold (0.32) using cross-validated out-of-fold predictions on training data only, avoiding test-set leakage in the threshold selection process

**Final Model Performance (Test Set, Threshold = 0.32):**

| Metric | Value |
|---|---|
| Accuracy | 0.80 |
| Precision (Readmitted) | 0.17 |
| Recall (Readmitted) | 0.19 |
| F1-score (Readmitted) | 0.18 |
| ROC-AUC | 0.591 |

### 6. Generalization Check and Overfitting Investigation

To assess whether the tuned Random Forest was genuinely learning generalizable patterns, its performance was compared against a Dummy Classifier baseline and checked for overfitting via train-test gap.

| Check | Result |
|---|---|
| Dummy Classifier ROC-AUC | 0.496 |
| Tuned Random Forest ROC-AUC | 0.591 |
| Train ROC-AUC | 1.00 |
| Test ROC-AUC | 0.591 |
| Train-Test Gap | 0.409 |

The model performs meaningfully better than random guessing, confirming genuine predictive signal. However, the large train-test gap indicated the unconstrained model fits training data very closely. To investigate whether this could be improved, two depth-constrained variants were tested:

| Model | Train ROC-AUC | Test ROC-AUC | Gap |
|---|---|---|---|
| Original Tuned RF (final) | 1.00 | 0.591 | 0.409 |
| max_depth = 20 | 0.88 | 0.568 | 0.312 |
| max_depth = 10 | 0.59 | 0.542 | 0.048 |

Constraining tree depth reduced the train-test gap but also reduced test performance in both cases — indicating the model's added complexity was capturing genuine, if modest, predictive signal rather than pure noise. The unconstrained tuned Random Forest was retained as the final model based on its superior real-world (test-set) performance, despite the overfitting gap.

### 7. Explainability (SHAP)
Applied SHAP (TreeExplainer) to identify global and individual feature contributions to readmission risk. Analysis revealed the model relies more heavily on sparse categorical patterns (e.g., specific diagnosis categories, admission sources) than the broader clinical indicators identified in EDA (`number_inpatient` ranked 21st in SHAP importance despite being the strongest EDA predictor) — a finding linked to signs of overfitting (Train ROC-AUC = 1.00 vs. Test ROC-AUC = 0.591) investigated via a depth-constraint analysis, which confirmed the unconstrained model retained superior real-world performance despite the gap.

### 8. Deployment
The final model was deployed as an interactive Streamlit web application. Users input simplified patient information (age, gender, time in hospital, medications, prior inpatient/emergency visits) and receive a readmission risk score, risk classification, and a SHAP-based explanation of the prediction.

---

## Key Findings

- **Strongest predictors (EDA):** Prior inpatient visits and prior emergency visits showed the clearest, sample-size-verified relationship with 30-day readmission.
- **Model performance:** The final model performs meaningfully better than random guessing (ROC-AUC 0.591 vs. 0.496 for a baseline dummy classifier), though with modest recall — reflecting the genuine difficulty of this prediction task from administrative/clinical data alone.
- **Overfitting:** The unconstrained Random Forest showed signs of overfitting (Train AUC = 1.00). Depth-constrained alternatives (max_depth = 10, 20) were tested but reduced real-world test performance, so the original model was retained as the better-performing option.

---

## Tech Stack

- **Data extraction:** Python, SQLite, SQL
- **Data processing & analysis:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn, imbalanced-learn (SMOTE), XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit, Google Drive (model hosting), gdown

---

## Project Structure

``````````````````````````````````````````
30day-hospital-readmission-ml/
├── app/
│ ├── app.py
│ ├── requirements.txt
│ ├── feature_columns.pkl
│ ├── final_threshold.pkl
│ ├── scaler.pkl
│ ├── default_row.pkl
│ └── shap_background_sample.pkl
├── notebooks/
│ ├── 01_Hospital_Readmission_Data_Prep.ipynb
│ ├── 02_Hospital_Readmission_Modeling_Final.ipynb
| ├── 03_Hospital_Readmission_Risk_pred.ipynb
└── README.md
`````````````````````````````````````````````````

*(Note: the trained model file, `rf_tuned_model.pkl`, is ~175MB and hosted on Google Drive; the app downloads it automatically on first run.)*

---

## How to Run Locally

```bash
git clone https://github.com/SruthiTS24/30day-hospital-readmission-ml.git
cd 30day-hospital-readmission-ml/app
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations

- Dataset spans 1999–2008 and uses pre-ICD-10 diagnosis coding; a production system would require retraining on current, ICD-10-coded data.
- Model shows signs of overfitting and relies partly on sparse categorical patterns rather than the cleanest clinical signals identified in EDA.
- Live SHAP explanations required a batching workaround to remain numerically stable for individual predictions, a known limitation of TreeExplainer on deep, unconstrained tree ensembles.
- This is a portfolio/demonstration project, not a validated clinical decision-support tool.

## Future Work

- Apply regularization or feature selection to reduce reliance on sparse categorical splits and better align model behavior with the strongest clinical predictors.
- Explore alternative models (e.g., LightGBM, CatBoost) for potentially better generalization on this imbalanced, high-cardinality dataset.
- Extend diagnosis-grouping logic to support ICD-10 codes for applicability to current hospital data.
- Incorporate richer clinical data sources (e.g., vitals, lab trends) where available.

---
