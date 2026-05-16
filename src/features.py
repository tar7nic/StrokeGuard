import pandas as pd
import numpy as np
import joblib
import os

MODELS_DIR = "models"

ENGINEERED_FEATURES = [
    "bmi_risk_tier",
    "glucose_category",
    "age_hypertension_interaction",
    "age_heart_disease_interaction",
    "age_group",
    "vascular_risk_score",
    "bmi_glucose_interaction",
    "senior_hypertensive",
]


def add_bmi_risk_tier(df):
    df = df.copy()
    bmi = df["bmi"]
    df["bmi_risk_tier"] = 1
    df.loc[bmi < 18.5, "bmi_risk_tier"] = 0
    df.loc[(bmi >= 25) & (bmi < 30), "bmi_risk_tier"] = 2
    df.loc[bmi >= 30, "bmi_risk_tier"] = 3
    return df


def add_glucose_category(df):
    df = df.copy()
    g = df["avg_glucose_level"]
    df["glucose_category"] = 0
    df.loc[(g >= 114) & (g < 140), "glucose_category"] = 1
    df.loc[g >= 140, "glucose_category"] = 2
    return df


def add_age_hypertension_interaction(df):
    df = df.copy()
    df["age_hypertension_interaction"] = df["age"] * df["hypertension"]
    return df


def add_age_heart_disease_interaction(df):
    df = df.copy()
    df["age_heart_disease_interaction"] = df["age"] * df["heart_disease"]
    return df


def add_age_group(df):
    df = df.copy()
    age = df["age"]
    df["age_group"] = 0
    df.loc[(age >= 18) & (age < 40), "age_group"] = 1
    df.loc[(age >= 40) & (age < 60), "age_group"] = 2
    df.loc[age >= 60, "age_group"] = 3
    return df


def add_vascular_risk_score(df):
    df = df.copy()
    diabetic = (df["glucose_category"] == 2).astype(int)
    df["vascular_risk_score"] = df["hypertension"] + df["heart_disease"] + diabetic
    return df


def add_bmi_glucose_interaction(df):
    df = df.copy()
    df["bmi_glucose_interaction"] = df["bmi"] * df["avg_glucose_level"]
    return df


def add_senior_hypertensive(df):
    df = df.copy()
    df["senior_hypertensive"] = ((df["age"] > 60) & (df["hypertension"] == 1)).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_bmi_risk_tier(df)
    df = add_glucose_category(df)
    df = add_age_hypertension_interaction(df)
    df = add_age_heart_disease_interaction(df)
    df = add_age_group(df)
    df = add_vascular_risk_score(df)
    df = add_bmi_glucose_interaction(df)
    df = add_senior_hypertensive(df)
    return df


def run_feature_pipeline(models_dir: str = MODELS_DIR):
    X_train = pd.read_csv(f"{models_dir}/X_train.csv")
    X_test = pd.read_csv(f"{models_dir}/X_test.csv")
    y_train = pd.read_csv(f"{models_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{models_dir}/y_test.csv").squeeze()

    X_train_fe = build_features(X_train)
    X_test_fe = build_features(X_test)

    X_train_fe.to_csv(f"{models_dir}/X_train_fe.csv", index=False)
    X_test_fe.to_csv(f"{models_dir}/X_test_fe.csv", index=False)

    print(f"Train shape after feature engineering: {X_train_fe.shape}")
    print(f"Test shape after feature engineering:  {X_test_fe.shape}")
    print(f"All features ({X_train_fe.shape[1]}): {list(X_train_fe.columns)}")

    return X_train_fe, X_test_fe, y_train, y_test


if __name__ == "__main__":
    run_feature_pipeline()