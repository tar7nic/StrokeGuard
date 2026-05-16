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


# ── BMI imputation ────────────────────────────────────────────────────────────
# FIX: medians are computed ONLY on training data and saved to disk.
# Both test-set imputation and single-patient inference use the saved medians,
# preventing any leakage from test rows into the imputation values.

def compute_bmi_medians(df_train: pd.DataFrame) -> pd.Series:
    """Compute BMI medians by age bracket from training data only."""
    tmp = df_train.copy()
    tmp["age_bracket"] = pd.cut(
        tmp["age"], bins=[0, 18, 40, 60, 120],
        labels=["child", "young", "middle", "senior"]
    )
    return tmp.groupby("age_bracket", observed=True)["bmi"].median()


def impute_bmi(df: pd.DataFrame, bmi_medians: pd.Series) -> pd.DataFrame:
    """Impute missing BMI using pre-computed training medians."""
    df = df.copy()
    # Cast to float64 first — avoids pandas AssertionError when the column is
    # integer-typed (no NaNs present yet) and we try to assign float medians.
    df["bmi"] = df["bmi"].astype("float64")
    df["age_bracket"] = pd.cut(
        df["age"], bins=[0, 18, 40, 60, 120],
        labels=["child", "young", "middle", "senior"]
    )
    mask = df["bmi"].isna()
    if mask.any():
        df.loc[mask, "bmi"] = df.loc[mask, "age_bracket"].map(bmi_medians).astype("float64")
    # Fallback: any remaining NaNs (unmatched bracket) get overall median
    if df["bmi"].isna().any():
        df["bmi"].fillna(float(bmi_medians.mean()), inplace=True)
    df.drop(columns=["age_bracket"], inplace=True)
    return df


# ── Smoking imputation ────────────────────────────────────────────────────────
# FIX: modes computed from training data and saved, applied to test + inference.

def compute_smoking_modes(df_train: pd.DataFrame) -> pd.Series:
    """Compute smoking status mode by gender from training data only."""
    return df_train.groupby("gender")["smoking_status"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "never smoked"
    )


def impute_smoking(df: pd.DataFrame, smoking_modes: pd.Series) -> pd.DataFrame:
    """Impute missing smoking status using pre-computed training modes."""
    df = df.copy()
    mask = df["smoking_status"].isna() | (df["smoking_status"] == "")
    df.loc[mask, "smoking_status"] = df.loc[mask, "gender"].map(smoking_modes)
    # Fallback if gender not in modes
    if df["smoking_status"].isna().any():
        df["smoking_status"].fillna("never smoked", inplace=True)
    return df


# ── Encoding ──────────────────────────────────────────────────────────────────

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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess_pipeline(path: str = RAW_PATH, test_size: float = 0.2, random_state: int = 42):
    df = load_raw(path)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # Split FIRST — then compute imputation stats from training data only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Compute stats from training data only
    bmi_medians    = compute_bmi_medians(X_train)
    smoking_modes  = compute_smoking_modes(X_train)

    # Apply imputation to both splits using training-derived stats
    X_train = impute_bmi(X_train, bmi_medians)
    X_test  = impute_bmi(X_test,  bmi_medians)
    X_train = impute_smoking(X_train, smoking_modes)
    X_test  = impute_smoking(X_test,  smoking_modes)

    # Fit encoders on training data only, apply to both
    X_train, encoders = encode_categoricals(X_train, fit=True)
    X_test,  _        = encode_categoricals(X_test,  encoders=encoders, fit=False)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save all inference artifacts
    joblib.dump(encoders,      f"{MODELS_DIR}/encoders.pkl")
    joblib.dump(bmi_medians,   f"{MODELS_DIR}/bmi_medians.pkl")
    joblib.dump(smoking_modes, f"{MODELS_DIR}/smoking_modes.pkl")

    X_train.to_csv(f"{MODELS_DIR}/X_train.csv", index=False)
    X_test.to_csv( f"{MODELS_DIR}/X_test.csv",  index=False)
    y_train.to_csv(f"{MODELS_DIR}/y_train.csv", index=False)
    y_test.to_csv( f"{MODELS_DIR}/y_test.csv",  index=False)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Stroke rate — train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")
    print(f"BMI missing — train: {X_train['bmi'].isna().sum()}, test: {X_test['bmi'].isna().sum()}")
    print(f"Features: {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test, encoders


# ── Single-patient inference ──────────────────────────────────────────────────

def preprocess_single_patient(patient_dict: dict, encoders: dict,
                               bmi_medians: pd.Series, smoking_modes: pd.Series,
                               feature_cols: list) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline to a single patient dict.
    Uses the SAME saved medians/modes as training — no recomputation.
    """
    df = pd.DataFrame([patient_dict])

    if df["bmi"].isna().any():
        df = impute_bmi(df, bmi_medians)

    if df["smoking_status"].isna().any() or (df["smoking_status"] == "").any():
        df = impute_smoking(df, smoking_modes)

    df, _ = encode_categoricals(df, encoders=encoders, fit=False)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


if __name__ == "__main__":
    preprocess_pipeline()