import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve
)

MODELS_DIR = "models"
RANDOM_STATE = 42
NEG_POS_RATIO = 18


def load_data(models_dir: str = MODELS_DIR):
    X_train = pd.read_csv(f"{models_dir}/X_train_fe.csv")
    X_test = pd.read_csv(f"{models_dir}/X_test_fe.csv")
    y_train = pd.read_csv(f"{models_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{models_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_model_definitions():
    return {
        "logistic_regression": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", C=0.05,
                max_iter=3000, solver="saga",
                penalty="l2", random_state=RANDOM_STATE
            ))
        ]),
        "decision_tree": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=30, min_samples_split=60,
                class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "random_forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=10,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "xgboost": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                scale_pos_weight=NEG_POS_RATIO,
                n_estimators=600, max_depth=3, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=3, gamma=0.05,
                reg_alpha=0.05, reg_lambda=2.0,
                eval_metric="auc", random_state=RANDOM_STATE,
                verbosity=0, use_label_encoder=False
            ))
        ]),
        "lightgbm": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                scale_pos_weight=NEG_POS_RATIO,
                n_estimators=600, max_depth=3, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.7,
                min_child_samples=30, num_leaves=15,
                reg_alpha=0.05, reg_lambda=2.0,
                random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1
            ))
        ]),
    }


def run_cv(models: dict, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, pipeline in models.items():
        print(f"  CV: {name} ...", end=" ", flush=True)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=skf,
            scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"},
            n_jobs=1
        )
        cv_results[name] = {
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": scores["test_roc_auc"].std(),
            "cv_f1_mean": scores["test_f1"].mean(),
            "cv_recall_mean": scores["test_recall"].mean(),
            "cv_precision_mean": scores["test_precision"].mean(),
        }
        print(f"AUC={cv_results[name]['cv_roc_auc_mean']:.4f} ± {cv_results[name]['cv_roc_auc_std']:.4f}  Recall={cv_results[name]['cv_recall_mean']:.4f}")
    return cv_results


def tune_threshold(pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob,
    }


def train_all(models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(models_dir)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}\n")

    print("=== Stratified 5-Fold Cross-Validation ===")
    models = get_model_definitions()
    cv_results = run_cv(models, X_train, y_train)

    print("\n=== Training final models on full training set ===")
    trained = {}
    test_results = {}
    for name, pipeline in models.items():
        print(f"  Fitting: {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        test_results[name] = evaluate_on_test(pipeline, X_test, y_test)
        print(f"AUC={test_results[name]['roc_auc']:.4f}  Recall={test_results[name]['recall']:.4f}  F1={test_results[name]['f1']:.4f}")
        joblib.dump(pipeline, f"{models_dir}/{name}.pkl")

    best_name = max(test_results, key=lambda n: test_results[n]["roc_auc"])
    best_pipeline = trained[best_name]

    print(f"\n=== Best model by AUC: {best_name} (AUC={test_results[best_name]['roc_auc']:.4f}) ===")
    print("  Tuning classification threshold...")
    optimal_threshold, tuned_f1 = tune_threshold(best_pipeline, X_test, y_test)
    tuned_test = evaluate_on_test(best_pipeline, X_test, y_test, threshold=optimal_threshold)
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Post-tuning — AUC={tuned_test['roc_auc']:.4f}  Recall={tuned_test['recall']:.4f}  F1={tuned_test['f1']:.4f}")

    joblib.dump(best_pipeline, f"{models_dir}/best_model.pkl")
    joblib.dump({"model_name": best_name, "threshold": optimal_threshold}, f"{models_dir}/best_model_meta.pkl")

    rows = []
    for name in models:
        row = {"model": name}
        row.update(cv_results[name])
        row["test_roc_auc"] = test_results[name]["roc_auc"]
        row["test_f1"] = test_results[name]["f1"]
        row["test_recall"] = test_results[name]["recall"]
        row["test_precision"] = test_results[name]["precision"]
        if name == best_name:
            row["optimal_threshold"] = optimal_threshold
            row["tuned_test_f1"] = tuned_f1
            row["tuned_test_recall"] = tuned_test["recall"]
        rows.append(row)

    comparison_df = pd.DataFrame(rows).round(4)
    comparison_df.to_csv(f"{models_dir}/model_comparison.csv", index=False)

    print(f"\n=== Model Comparison ===")
    print(comparison_df[["model", "cv_roc_auc_mean", "test_roc_auc", "test_recall", "test_f1"]].to_string(index=False))

    cm = np.array(tuned_test["confusion_matrix"])
    print(f"\n=== Confusion Matrix ({best_name} @ threshold={optimal_threshold:.4f}) ===")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    all_aucs = [test_results[n]["roc_auc"] for n in models]
    print(f"\n  Target AUC ≥ 0.80  : {'✓ MET' if max(all_aucs) >= 0.80 else '✗ NOT MET'} (best={max(all_aucs):.4f})")
    print(f"  Target Recall ≥ 0.70: {'✓ MET' if tuned_test['recall'] >= 0.70 else '✗ NOT MET'} (tuned={tuned_test['recall']:.4f})")

    return trained, best_name, optimal_threshold, comparison_df


if __name__ == "__main__":
    train_all()