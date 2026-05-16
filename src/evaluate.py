import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)

MODELS_DIR = "models"
MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]


def load_test_data(models_dir: str = MODELS_DIR):
    X_test = pd.read_csv(f"{models_dir}/X_test_fe.csv")
    y_test = pd.read_csv(f"{models_dir}/y_test.csv").squeeze()
    return X_test, y_test


def load_all_models(models_dir: str = MODELS_DIR):
    models = {}
    for name in MODEL_NAMES:
        path = f"{models_dir}/{name}.pkl"
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def compute_metrics(model, X_test, y_test, threshold: float = 0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def print_full_report(name: str, metrics: dict, y_test, threshold: float):
    print(f"\n{'='*60}")
    print(f"  MODEL: {name.upper().replace('_', ' ')}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"{'='*60}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision    : {metrics['avg_precision']:.4f}")
    print(f"  F1 (stroke)      : {metrics['f1']:.4f}")
    print(f"  Recall (stroke)  : {metrics['recall']:.4f}")
    print(f"  Precision(stroke): {metrics['precision']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={metrics['tn']:>5}  FP={metrics['fp']:>5}")
    print(f"    FN={metrics['fn']:>5}  TP={metrics['tp']:>5}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, metrics["y_pred"], target_names=["No Stroke", "Stroke"], zero_division=0))


def plot_roc_curves(models: dict, X_test, y_test, save_dir: str = MODELS_DIR):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name.replace('_', ' ').title()} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_recall_curves(models: dict, X_test, y_test, save_dir: str = MODELS_DIR):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name.replace('_', ' ').title()} (AP={ap:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/pr_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(models: dict, X_test, y_test, threshold_map: dict, save_dir: str = MODELS_DIR):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        threshold = threshold_map.get(name, 0.5)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Stroke", "Stroke"])
        ax.set_yticklabels(["No Stroke", "Stroke"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
    plt.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_evaluation_report(models_dir: str = MODELS_DIR):
    X_test, y_test = load_test_data(models_dir)
    models = load_all_models(models_dir)

    meta = joblib.load(f"{models_dir}/best_model_meta.pkl")
    best_name = meta["model_name"]
    optimal_threshold = meta["threshold"]

    threshold_map = {name: 0.5 for name in models}
    threshold_map[best_name] = optimal_threshold

    print("\n" + "="*60)
    print("  CLINICALRISK — FULL EVALUATION REPORT")
    print("="*60)

    summary_rows = []
    for name, model in models.items():
        threshold = threshold_map[name]
        metrics = compute_metrics(model, X_test, y_test, threshold=threshold)
        print_full_report(name, metrics, y_test, threshold)
        summary_rows.append({
            "model": name,
            "threshold": threshold,
            "roc_auc": metrics["roc_auc"],
            "avg_precision": metrics["avg_precision"],
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
        })

    summary_df = pd.DataFrame(summary_rows).round(4)
    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(summary_df[["model", "roc_auc", "recall", "f1", "precision", "tp", "fn"]].to_string(index=False))

    target_auc = summary_df["roc_auc"].max() >= 0.80
    target_recall = summary_df["recall"].max() >= 0.70
    print(f"\n  Target AUC ≥ 0.80  : {'✓ MET' if target_auc else '✗ NOT MET'} (best={summary_df['roc_auc'].max():.4f})")
    print(f"  Target Recall ≥ 0.70 : {'✓ MET' if target_recall else '✗ NOT MET'} (best={summary_df['recall'].max():.4f})")

    print("\n  Generating plots...")
    plot_roc_curves(models, X_test, y_test, models_dir)
    plot_precision_recall_curves(models, X_test, y_test, models_dir)
    plot_confusion_matrices(models, X_test, y_test, threshold_map, models_dir)

    return summary_df


if __name__ == "__main__":
    generate_evaluation_report()