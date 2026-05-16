import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

MODELS_DIR = "models"
SHAP_DIR = "models/shap_plots"


def load_best_model_and_data(models_dir: str = MODELS_DIR):
    model  = joblib.load(f"{models_dir}/best_model.pkl")
    meta   = joblib.load(f"{models_dir}/best_model_meta.pkl")
    X_test  = pd.read_csv(f"{models_dir}/X_test_fe.csv")
    X_train = pd.read_csv(f"{models_dir}/X_train_fe.csv")   # needed for SHAP background
    y_test  = pd.read_csv(f"{models_dir}/y_test.csv").squeeze()
    return model, meta, X_train, X_test, y_test


def get_classifier_and_transformed(pipeline, X: pd.DataFrame, is_train: bool = False):
    """
    Extract the fitted classifier and apply all pre-classifier steps except SMOTE.
    Returns (clf, X_transformed).

    For tree models, X_transformed is the pre-scaler data (original feature space),
    which is what TreeExplainer needs. For linear models, StandardScaler is applied.
    """
    steps = dict(pipeline.steps)
    clf   = pipeline.named_steps["clf"]
    model_type = type(clf).__name__

    X_transformed = X.copy()

    for name, step in pipeline.steps[:-1]:
        if name == "smote":
            continue  # SMOTE is fit-only, skip at inference
        X_transformed = pd.DataFrame(
            step.transform(X_transformed),
            columns=X_transformed.columns
        )

    return clf, X_transformed


def compute_shap_values(clf, X_test_transformed: pd.DataFrame,
                        X_train_transformed: pd.DataFrame):
    """
    Compute SHAP values using the correct explainer for each model type.

    FIX: LinearExplainer background must be the TRAINING distribution, not test data.
    Using test data as background shifts the expected value and produces incorrect
    SHAP attributions. The background is now X_train_transformed (scaled training set
    for LR, raw features for tree models — but tree models don't use a background).

    Tree models (XGB/LGBM/RF): TreeExplainer — no background dataset needed.
    Linear models (LR): LinearExplainer — background = full scaled training set.
    """
    model_type = type(clf).__name__

    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                      "DecisionTreeClassifier"):
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_transformed)
        if isinstance(shap_values, list):   # RF/DT return per-class list
            shap_values = shap_values[1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[1])
        else:
            expected_value = float(expected_value)

    else:
        # FIX: background = training data (not test data)
        explainer   = shap.LinearExplainer(
            clf, X_train_transformed, feature_perturbation="interventional"
        )
        shap_values    = explainer.shap_values(X_test_transformed)
        expected_value = float(explainer.expected_value)

    return explainer, shap_values, expected_value


def plot_summary_bar(shap_values, X_transformed: pd.DataFrame, save_dir: str = SHAP_DIR):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, plot_type="bar",
                      max_display=10, show=False)
    plt.title("Global Feature Importance — Top 10 SHAP Features",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/shap_summary_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_beeswarm(shap_values, X_transformed: pd.DataFrame, save_dir: str = SHAP_DIR):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_transformed, plot_type="dot",
                      max_display=10, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Distribution",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_waterfall_high_risk(shap_values, expected_value,
                              X_transformed: pd.DataFrame,
                              y_test: pd.Series, save_dir: str = SHAP_DIR):
    stroke_indices   = np.where(y_test.values == 1)[0]
    y_prob_shap      = shap_values[stroke_indices].sum(axis=1)
    highest_risk_local = np.argmax(y_prob_shap)
    patient_idx      = stroke_indices[highest_risk_local]

    patient_shap   = shap_values[patient_idx]
    feature_names  = X_transformed.columns.tolist()
    patient_values = X_transformed.iloc[patient_idx].values

    sorted_idx   = np.argsort(np.abs(patient_shap))[::-1][:10]
    top_features = [feature_names[i] for i in sorted_idx]
    top_shap     = patient_shap[sorted_idx]
    top_values   = patient_values[sorted_idx]

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_shap]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(top_features))
    ax.barh(list(y_pos), top_shap[::-1], color=colors[::-1])
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{top_features[::-1][i]} = {top_values[::-1][i]:.2f}"
         for i in range(len(top_features))],
        fontsize=10
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on stroke probability)", fontsize=11)
    ax.set_title(
        f"SHAP Waterfall — Highest Risk Patient (idx={patient_idx})\n"
        f"Base value: {expected_value:.4f}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{save_dir}/shap_waterfall_high_risk.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return patient_idx


def save_top10_features(shap_values, X_transformed: pd.DataFrame,
                        save_dir: str = SHAP_DIR):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature":       X_transformed.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).head(10).reset_index(drop=True)
    feature_importance.to_csv(f"{save_dir}/top10_shap_features.csv", index=False)
    print(f"\n  Top 10 SHAP features:")
    print(feature_importance.to_string(index=False))
    return feature_importance


def run_explain_pipeline(models_dir: str = MODELS_DIR, shap_dir: str = SHAP_DIR):
    os.makedirs(shap_dir, exist_ok=True)

    print("=== Loading best model and data ===")
    pipeline, meta, X_train, X_test, y_test = load_best_model_and_data(models_dir)
    print(f"  Model: {meta['model_name']}  |  Threshold: {meta['threshold']:.4f}")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("\n=== Transforming features (skip SMOTE, apply other steps) ===")
    clf, X_test_transformed  = get_classifier_and_transformed(pipeline, X_test)
    clf, X_train_transformed = get_classifier_and_transformed(pipeline, X_train)
    print(f"  Test transformed shape:  {X_test_transformed.shape}")
    print(f"  Train transformed shape: {X_train_transformed.shape}")

    print("\n=== Computing SHAP values ===")
    explainer, shap_values, expected_value = compute_shap_values(
        clf, X_test_transformed, X_train_transformed
    )
    print(f"  SHAP values shape:  {shap_values.shape}")
    print(f"  Expected value:     {expected_value:.4f}")

    print("\n=== Generating plots ===")
    plot_summary_bar(shap_values, X_test_transformed, shap_dir)
    plot_beeswarm(shap_values, X_test_transformed, shap_dir)
    patient_idx = plot_waterfall_high_risk(
        shap_values, expected_value, X_test_transformed, y_test, shap_dir
    )
    save_top10_features(shap_values, X_test_transformed, shap_dir)

    joblib.dump({
        "shap_values":           shap_values,
        "expected_value":        expected_value,
        "feature_names":         X_test_transformed.columns.tolist(),
        "high_risk_patient_idx": patient_idx
    }, f"{models_dir}/shap_data.pkl")

    print(f"\n  SHAP data saved to {models_dir}/shap_data.pkl")
    print(f"  All plots saved to {shap_dir}/")


if __name__ == "__main__":
    run_explain_pipeline()