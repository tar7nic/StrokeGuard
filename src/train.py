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
    X_test  = pd.read_csv(f"{models_dir}/X_test_fe.csv")
    y_train = pd.read_csv(f"{models_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{models_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_model_definitions():
    """
    Pipelines per model type.

    FIX 1: StandardScaler removed from tree-model pipelines (XGBoost, LightGBM,
    RandomForest, DecisionTree). Tree models are scale-invariant — the scaler had
    zero effect on predictions and only wasted compute.
    Scaler is kept for LogisticRegression, which genuinely needs it.

    FIX 2: use_label_encoder=False removed from XGBClassifier — deprecated and
    removed in XGBoost >= 1.6, causes errors on modern installs.
    """
    smote = lambda: SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    return {
        "logistic_regression": ImbPipeline([
            ("smote",  smote()),
            ("scaler", StandardScaler()),          # needed for LR
            ("clf",    LogisticRegression(
                class_weight="balanced", C=0.05,
                max_iter=3000, solver="saga",
                penalty="l2", random_state=RANDOM_STATE
            ))
        ]),

        "decision_tree": ImbPipeline([
            ("smote", smote()),
            # no scaler — decision trees are scale-invariant
            ("clf",   DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=30, min_samples_split=60,
                class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),

        "random_forest": ImbPipeline([
            ("smote", smote()),
            # no scaler — random forests are scale-invariant
            ("clf",   RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=10,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),

        "xgboost": ImbPipeline([
            ("smote", smote()),
            # no scaler — XGBoost is scale-invariant
            ("clf",   XGBClassifier(
                scale_pos_weight=NEG_POS_RATIO,
                n_estimators=600, max_depth=3, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=3, gamma=0.05,
                reg_alpha=0.05, reg_lambda=2.0,
                eval_metric="auc", random_state=RANDOM_STATE,
                verbosity=0
                # use_label_encoder removed — deprecated in XGBoost >= 1.6
            ))
        ]),

        "lightgbm": ImbPipeline([
            ("smote", smote()),
            # no scaler — LightGBM is scale-invariant
            ("clf",   LGBMClassifier(
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
            "cv_roc_auc_mean":  scores["test_roc_auc"].mean(),
            "cv_roc_auc_std":   scores["test_roc_auc"].std(),
            "cv_f1_mean":       scores["test_f1"].mean(),
            "cv_recall_mean":   scores["test_recall"].mean(),
            "cv_precision_mean":scores["test_precision"].mean(),
        }
        print(
            f"AUC={cv_results[name]['cv_roc_auc_mean']:.4f} "
            f"± {cv_results[name]['cv_roc_auc_std']:.4f}  "
            f"Recall={cv_results[name]['cv_recall_mean']:.4f}"
        )
    return cv_results


def tune_threshold_cv(pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                      cv_folds: int = 5) -> float:
    """
    FIX: Tune the classification threshold using cross-validation on training data,
    NOT on the test set.

    Previously threshold was tuned on X_test then reported on the same X_test —
    that inflated F1/Recall because the threshold was fitted to the test set.

    This function finds the threshold that maximises mean CV F1 across folds.
    The test set remains completely untouched and gives an honest estimate.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    all_probs  = []
    all_labels = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Clone-fit each fold (pipeline is stateful, must refit)
        from sklearn.base import clone
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)
        probs = fold_pipe.predict_proba(X_val)[:, 1]

        all_probs.append(probs)
        all_labels.append(y_val.values)

    y_prob_all = np.concatenate(all_probs)
    y_true_all = np.concatenate(all_labels)

    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_prob_all)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":        roc_auc_score(y_test, y_prob),
        "f1":             f1_score(y_test, y_pred, zero_division=0),
        "recall":         recall_score(y_test, y_pred, zero_division=0),
        "precision":      precision_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob":         y_prob,
    }


def train_all(models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(models_dir)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}\n")

    print("=== Stratified 5-Fold Cross-Validation ===")
    models = get_model_definitions()
    cv_results = run_cv(models, X_train, y_train)

    # FIX: Best model chosen by CV AUC (not test AUC) to avoid test-set leakage
    best_name_by_cv = max(cv_results, key=lambda n: cv_results[n]["cv_roc_auc_mean"])
    print(f"\n  Best model by CV AUC: {best_name_by_cv} "
          f"(CV AUC={cv_results[best_name_by_cv]['cv_roc_auc_mean']:.4f})")

    print("\n=== Training final models on full training set ===")
    trained = {}
    test_results = {}
    for name, pipeline in models.items():
        print(f"  Fitting: {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        test_results[name] = evaluate_on_test(pipeline, X_test, y_test)
        print(
            f"AUC={test_results[name]['roc_auc']:.4f}  "
            f"Recall={test_results[name]['recall']:.4f}  "
            f"F1={test_results[name]['f1']:.4f}"
        )
        joblib.dump(pipeline, f"{models_dir}/{name}.pkl")

    best_pipeline = trained[best_name_by_cv]

    # FIX: Tune threshold using CV on training data — test set stays untouched
    print(f"\n=== Tuning threshold for {best_name_by_cv} via CV (not test set) ===")
    optimal_threshold = tune_threshold_cv(
        models[best_name_by_cv], X_train, y_train
    )
    print(f"  CV-tuned threshold: {optimal_threshold:.4f}")

    # Now evaluate on test set ONCE with the CV-tuned threshold (honest estimate)
    tuned_test = evaluate_on_test(best_pipeline, X_test, y_test, threshold=optimal_threshold)
    print(
        f"  Test — AUC={tuned_test['roc_auc']:.4f}  "
        f"Recall={tuned_test['recall']:.4f}  "
        f"F1={tuned_test['f1']:.4f}"
    )

    joblib.dump(best_pipeline, f"{models_dir}/best_model.pkl")
    joblib.dump(
        {"model_name": best_name_by_cv, "threshold": optimal_threshold},
        f"{models_dir}/best_model_meta.pkl"
    )

    # Build comparison table
    rows = []
    for name in models:
        row = {"model": name}
        row.update(cv_results[name])
        row["test_roc_auc"]   = test_results[name]["roc_auc"]
        row["test_f1"]        = test_results[name]["f1"]
        row["test_recall"]    = test_results[name]["recall"]
        row["test_precision"] = test_results[name]["precision"]
        if name == best_name_by_cv:
            row["optimal_threshold"] = optimal_threshold
            row["tuned_test_f1"]     = tuned_test["f1"]
            row["tuned_test_recall"] = tuned_test["recall"]
        rows.append(row)

    comparison_df = pd.DataFrame(rows).round(4)
    comparison_df.to_csv(f"{models_dir}/model_comparison.csv", index=False)

    print(f"\n=== Model Comparison ===")
    print(comparison_df[
        ["model", "cv_roc_auc_mean", "test_roc_auc", "test_recall", "test_f1"]
    ].to_string(index=False))

    cm = np.array(tuned_test["confusion_matrix"])
    print(f"\n=== Confusion Matrix ({best_name_by_cv} @ threshold={optimal_threshold:.4f}) ===")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    all_aucs = [test_results[n]["roc_auc"] for n in models]
    print(f"\n  Target AUC ≥ 0.80  : {'✓ MET' if max(all_aucs) >= 0.80 else '✗ NOT MET'} (best={max(all_aucs):.4f})")
    print(f"  Target Recall ≥ 0.70: {'✓ MET' if tuned_test['recall'] >= 0.70 else '✗ NOT MET'} (tuned={tuned_test['recall']:.4f})")

    return trained, best_name_by_cv, optimal_threshold, comparison_df


if __name__ == "__main__":
    train_all()