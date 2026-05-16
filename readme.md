# 🛡️ StrokeGuard — Stroke Risk Prediction System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-strokeguard--tn019.streamlit.app-ff4b4b?style=for-the-badge&logo=streamlit)](https://strokeguard-tn019.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189fdd?style=for-the-badge)](https://xgboost.readthedocs.io)

An end-to-end clinical stroke risk prediction system built on 5,110 patient records. Compares 5 ML models, engineers 8+ clinical features, applies SHAP explainability, performs SQL cohort analysis, and deploys a 4-page interactive Streamlit dashboard.

**[→ Live Demo](https://strokeguard-tn019.streamlit.app)**

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Risk Scorer** | Input patient vitals → real-time stroke probability with risk tier + SHAP waterfall |
| **Model Comparison** | Interactive ROC curves and metrics table for all 5 models |
| **Cohort Explorer** | Bar charts from SQL cohort analysis across age, glucose, BMI, hypertension |
| **Feature Importance** | Global SHAP summary + plain-English clinical interpretation of top 10 drivers |

---

## Results

| Model | ROC-AUC | Recall (Stroke) | F1 |
|---|---|---|---|
| Logistic Regression | 0.7905 | 0.54 | 0.29 |
| Decision Tree | 0.7501 | 0.64 | 0.19 |
| Random Forest | 0.7754 | 0.44 | 0.20 |
| XGBoost | 0.7767 | 0.82 | 0.18 |
| **LightGBM** | **0.7855** | **0.82** | **0.18** |

> Threshold tuned via precision-recall curve to maximise recall on the minority stroke class (5% prevalence). An 82% recall means 82 out of 100 true stroke patients are correctly flagged — clinically more important than raw AUC on an imbalanced dataset.

---

## Engineered Features

| Feature | Description |
|---|---|
| `bmi_risk_tier` | Ordinal BMI category: Underweight / Normal / Overweight / Obese |
| `glucose_category` | Normal (<114) / Prediabetic (114–140) / Diabetic (>140) |
| `age_hypertension_interaction` | age × hypertension — amplifies risk in elderly hypertensives |
| `age_heart_disease_interaction` | age × heart_disease |
| `age_group` | Binned age: <18 / 18–40 / 40–60 / 60+ |
| `vascular_risk_score` | Sum of hypertension + heart_disease + diabetic flag (0–3) |
| `bmi_glucose_interaction` | BMI × avg_glucose_level — metabolic syndrome proxy |
| `senior_hypertensive` | Binary: age > 60 AND hypertension == 1 |

---

## Project Structure

```
clinicalrisk/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── notebooks/
│   └── eda.ipynb                  # 10 EDA charts
├── src/
│   ├── preprocess.py              # Cleaning, imputation, encoding
│   ├── features.py                # Feature engineering pipeline
│   ├── train.py                   # 5-model training with SMOTE + CV
│   ├── evaluate.py                # Metrics, ROC/PR curves, confusion matrices
│   ├── explain.py                 # SHAP summary, beeswarm, waterfall plots
│   └── database.py                # SQLite loader + 5 cohort queries
├── models/
│   ├── best_model.pkl
│   ├── best_model_meta.pkl
│   ├── model_comparison.csv
│   ├── shap_data.pkl
│   ├── shap_plots/
│   └── cohort_results/
├── app/
│   └── streamlit_app.py           # 4-page Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/clinicalrisk.git
cd clinicalrisk
pip install -r requirements.txt

# 2. Add dataset
# Download from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
# Place at data/healthcare-dataset-stroke-data.csv

# 3. Run pipeline
python src/preprocess.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/database.py
python src/explain.py

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Technical Approach

**Class Imbalance (95:5 ratio)**
SMOTE applied exclusively inside each CV fold via `imblearn.Pipeline` — never on the test set. Tree-based models use `class_weight='balanced_subsample'`; XGBoost/LightGBM use `scale_pos_weight=18`.

**No Data Leakage**
`StandardScaler` lives inside the model pipeline after SMOTE. BMI imputation uses median grouped by age bracket computed on train only. Feature engineering operates on raw values with no fit state.

**Threshold Tuning**
After selecting the best model by ROC-AUC, the classification threshold is tuned on the precision-recall curve to maximise F1 for the stroke class — pushing recall to 0.82.

**SHAP Explainability**
`TreeExplainer` for tree-based models, `LinearExplainer` for logistic regression. Per-patient waterfall plots rendered live in the dashboard via Plotly.

**SQL Cohort Analysis**
Full dataset loaded into SQLite. Five queries surface stroke prevalence across age groups, glucose categories, BMI tiers, hypertension status, and composite vascular risk scores.

---

## Dataset

[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — Kaggle  
5,110 records · 12 features · Binary classification (stroke: 0/1)

---

## Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `imbalanced-learn` `SHAP` `Streamlit` `Plotly` `SQLite` `Pandas` `NumPy`