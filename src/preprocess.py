import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

RAW_PATH = "data/healthcare-dataset-stroke-data.csv"
MODELS_DIR = "models"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=["id"], inplace=True)
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    return df


def impute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_bracket"] = pd.cut(df["age"], bins=[0, 18, 40, 60, 120], labels=["child", "young", "middle", "senior"])
    bmi_medians = df.groupby("age_bracket", observed=True)["bmi"].median()
    def fill_bmi(row):
        if pd.isna(row["bmi"]):
            return bmi_medians[row["age_bracket"]]
        return row["bmi"]
    df["bmi"] = df.apply(fill_bmi, axis=1)
    df.drop(columns=["age_bracket"], inplace=True)
    return df


def impute_smoking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mode_by_gender = df.groupby("gender")["smoking_status"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "never smoked"
    )
    mask = df["smoking_status"].isna() | (df["smoking_status"] == "")
    df.loc[mask, "smoking_status"] = df.loc[mask, "gender"].map(mode_by_gender)
    return df


ORDINAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if fit:
        encoders = {}
        for col in ORDINAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in ORDINAL_COLS:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df, encoders


def preprocess_pipeline(path: str = RAW_PATH, test_size: float = 0.2, random_state: int = 42):
    df = load_raw(path)
    df = impute_bmi(df)
    df = impute_smoking(df)
    df, encoders = encode_categoricals(df, fit=True)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, f"{MODELS_DIR}/encoders.pkl")

    X_train.to_csv(f"{MODELS_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{MODELS_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{MODELS_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{MODELS_DIR}/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test, encoders


def preprocess_single_patient(patient_dict: dict, encoders: dict, feature_cols: list) -> pd.DataFrame:
    df = pd.DataFrame([patient_dict])
    if df["bmi"].isna().any():
        df = impute_bmi(df)
    if df["smoking_status"].isna().any():
        df = impute_smoking(df)
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders = preprocess_pipeline()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Stroke rate train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")
    print(f"Features: {list(X_train.columns)}")