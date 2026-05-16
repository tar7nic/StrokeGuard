import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.features import build_features
from src.preprocess import encode_categoricals

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODELS_DIR = "models"
MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]

ORDINAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

GENDER_OPTIONS       = ["Male", "Female"]
MARRIED_OPTIONS      = ["Yes", "No"]
WORK_OPTIONS         = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
RESIDENCE_OPTIONS    = ["Urban", "Rural"]
SMOKING_OPTIONS      = ["never smoked", "formerly smoked", "smokes", "Unknown"]

# ─────────────────────────────────────────────
# Cached loaders — loaded ONCE per server start
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all_models():
    models = {}
    for name in MODEL_NAMES:
        path = f"{MODELS_DIR}/{name}.pkl"
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource(show_spinner="Loading encoders…")
def load_encoders():
    return joblib.load(f"{MODELS_DIR}/encoders.pkl")

@st.cache_resource(show_spinner="Loading best model metadata…")
def load_best_meta():
    meta = joblib.load(f"{MODELS_DIR}/best_model_meta.pkl")
    best_pipeline = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    return best_pipeline, meta  # meta = {"model_name": ..., "threshold": ...}

@st.cache_resource(show_spinner="Loading test data…")
def load_test_data():
    X_test  = pd.read_csv(f"{MODELS_DIR}/X_test_fe.csv")
    y_test  = pd.read_csv(f"{MODELS_DIR}/y_test.csv").squeeze()
    X_train = pd.read_csv(f"{MODELS_DIR}/X_train_fe.csv")
    return X_test, y_test, X_train

@st.cache_data(show_spinner="Loading model comparison…")
def load_comparison():
    path = f"{MODELS_DIR}/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ─────────────────────────────────────────────
# Patient input → feature-engineered DataFrame
# ─────────────────────────────────────────────
def build_patient_df(inputs: dict, encoders: dict, feature_cols: list) -> pd.DataFrame:
    """
    Takes raw UI inputs, applies the SAME preprocessing & feature engineering
    pipeline that was used during training. Returns a 1-row DataFrame with
    exactly the columns the model was trained on, in the correct order.
    """
    df = pd.DataFrame([inputs])

    # Encode categoricals using the SAVED encoders (no re-fitting)
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)

    # Feature engineering (pure deterministic transforms — no randomness)
    df = build_features(df)

    # Enforce exact column order from training
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# ─────────────────────────────────────────────
# SHAP helpers — correct per model type
# ─────────────────────────────────────────────
def get_shap_values_for_patient(pipeline, patient_df: pd.DataFrame, X_train_bg: pd.DataFrame = None):
    """
    Returns (shap_vals_1d, feature_display_df, expected_value).

    - Tree models: TreeExplainer on pre-scaler data (original feature space).
    - Linear models: LinearExplainer MUST have a proper background dataset
      (the full training set scaled), otherwise SHAP values are near-zero garbage.
    """
    steps = dict(pipeline.steps)
    clf   = steps["clf"]
    model_type = type(clf).__name__

    X_for_shap = patient_df.copy()
    if "scaler" in steps:
        scaler = steps["scaler"]
        X_scaled = pd.DataFrame(
            scaler.transform(X_for_shap),
            columns=X_for_shap.columns
        )
    else:
        X_scaled = X_for_shap.copy()

    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                      "DecisionTreeClassifier"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_for_shap)
        if isinstance(sv, list):
            sv = sv[1]
        shap_1d = sv[0]
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[1])
        display_df = X_for_shap   # original values shown to user

    else:
        # LinearExplainer REQUIRES a meaningful background distribution.
        # Using a single row gives near-zero SHAP values. Use scaled training data.
        if X_train_bg is not None and "scaler" in steps:
            scaler = steps["scaler"]
            bg_scaled = pd.DataFrame(
                scaler.transform(X_train_bg),
                columns=X_train_bg.columns
            )
        else:
            bg_scaled = X_scaled  # fallback (poor but won't crash)

        explainer = shap.LinearExplainer(clf, bg_scaled, feature_perturbation="interventional")
        sv = explainer.shap_values(X_scaled)
        shap_1d = sv[0] if sv.ndim > 1 else sv
        ev = float(explainer.expected_value)
        display_df = X_for_shap   # show original (unscaled) values to user

    return shap_1d, display_df, ev

# ─────────────────────────────────────────────
# Gauge chart — value and range both in [0, 1]
# ─────────────────────────────────────────────
def make_gauge(probability: float, threshold: float) -> go.Figure:
    """
    probability : float in [0, 1]
    threshold   : float in [0, 1]  ← the trained optimal threshold
    
    The gauge axis is [0, 1]. We display percentages only in tick labels.
    This is the fix for the needle rendering at the wrong position.
    """
    pct = probability * 100

    if probability < threshold * 0.6:
        color = "#2ecc71"
        label = "Low Risk"
    elif probability < threshold:
        color = "#f39c12"
        label = "Moderate Risk"
    elif probability < threshold * 1.5:
        color = "#e74c3c"
        label = "High Risk"
    else:
        color = "#8e1c1c"
        label = "Very High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,           # ← in [0,1], same unit as axis range
        number={
            "valueformat": ".1%",    # displayed as percentage text
            "font": {"size": 36, "color": color}
        },
        delta={
            "reference": threshold,
            "valueformat": ".1%",
            "increasing": {"color": "#e74c3c"},
            "decreasing": {"color": "#2ecc71"},
        },
        gauge={
            "axis": {
                "range": [0, 1],     # ← same unit as value
                "tickformat": ".0%",
                "tickfont": {"size": 11},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#cccccc",
            "steps": [
                {"range": [0,                threshold * 0.6], "color": "#d5f5e3"},
                {"range": [threshold * 0.6,  threshold],       "color": "#fdebd0"},
                {"range": [threshold,        threshold * 1.5], "color": "#fadbd8"},
                {"range": [threshold * 1.5,  1.0],             "color": "rgba(192,57,43,0.13)"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 3},
                "thickness": 0.75,
                "value": threshold,   # ← in [0,1]
            },
        },
        title={"text": f"<b>Stroke Probability</b><br><span style='font-size:13px;color:{color}'>{label}</span>",
               "font": {"size": 16}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Georgia, serif"},
    )
    return fig

# ─────────────────────────────────────────────
# SHAP bar chart via matplotlib (no Plotly bugs)
# ─────────────────────────────────────────────
def make_shap_bar(shap_vals: np.ndarray, feature_names: list, feature_values: np.ndarray,
                  top_n: int = 10):
    idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_names  = [feature_names[i] for i in idx]
    top_shap   = shap_vals[idx]
    top_vals   = feature_values[idx]

    colors = ["#c0392b" if v > 0 else "#2980b9" for v in top_shap]
    y_pos  = range(len(top_names))
    labels = [f"{n}  [{v:.2f}]" for n, v in zip(top_names[::-1], top_vals[::-1])]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(y_pos), top_shap[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#2c3e50", linewidth=1.2)
    ax.set_xlabel("SHAP value  (↑ increases risk,  ↓ decreases risk)", fontsize=10)
    ax.set_title("Feature Contributions to This Prediction", fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# ROC / PR curves (reuse evaluate.py logic inline)
# ─────────────────────────────────────────────
def make_roc_figure(models: dict, X_test, y_test):
    from sklearn.metrics import roc_curve, roc_auc_score
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig = go.Figure()
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})",
            line=dict(color=color, width=2)
        ))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(color="grey", dash="dash"), name="Random", showlegend=True))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05),
        height=430,
        template="plotly_white",
    )
    return fig

def make_pr_figure(models: dict, X_test, y_test):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig = go.Figure()
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name.replace('_', ' ').title()} (AP={ap:.3f})",
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=430,
        template="plotly_white",
    )
    return fig

# ─────────────────────────────────────────────
# Metrics summary table
# ─────────────────────────────────────────────
def compute_all_metrics(models: dict, X_test, y_test, threshold_map: dict) -> pd.DataFrame:
    from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                                 precision_score, average_precision_score)
    rows = []
    for name, model in models.items():
        thr   = threshold_map.get(name, 0.5)
        yp    = model.predict_proba(X_test)[:, 1]
        ypred = (yp >= thr).astype(int)
        rows.append({
            "Model":      name.replace("_", " ").title(),
            "Threshold":  f"{thr:.3f}",
            "ROC-AUC":    round(roc_auc_score(y_test, yp), 4),
            "Avg Prec":   round(average_precision_score(y_test, yp), 4),
            "F1":         round(f1_score(y_test, ypred, zero_division=0), 4),
            "Recall":     round(recall_score(y_test, ypred, zero_division=0), 4),
            "Precision":  round(precision_score(y_test, ypred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# PAGE: Risk Evaluation
# ─────────────────────────────────────────────
def page_risk_evaluation(best_pipeline, meta, encoders, feature_cols, all_models: dict = None, X_train: pd.DataFrame = None):
    st.title("🩺 Patient Risk Evaluation")
    st.markdown(
        "Enter patient details below. The prediction uses the **best trained model** "
        f"(`{meta['model_name'].replace('_', ' ').title()}`) with its **optimal classification "
        f"threshold of {meta['threshold']:.3f}** derived from the precision-recall curve."
    )
    threshold = float(meta["threshold"])

    # ── Input form ──────────────────────────────────────────────────────────
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age    = st.slider("Age", 1, 100, 55)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            ever_married   = st.selectbox("Ever Married", MARRIED_OPTIONS)
            residence_type = st.selectbox("Residence Type", RESIDENCE_OPTIONS)

        with col2:
            st.subheader("Medical History")
            hypertension  = st.checkbox("Hypertension", value=False)
            heart_disease = st.checkbox("Heart Disease", value=False)
            avg_glucose   = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=1.0)
            bmi           = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)

        with col3:
            st.subheader("Lifestyle")
            work_type      = st.selectbox("Work Type", WORK_OPTIONS)
            smoking_status = st.selectbox("Smoking Status", SMOKING_OPTIONS)

        submitted = st.form_submit_button("🔍 Predict Stroke Risk", use_container_width=True)

    if not submitted:
        st.info("Fill in the patient details and click **Predict** to see results.")
        return

    # ── Build input ──────────────────────────────────────────────────────────
    raw_inputs = {
        "gender":          gender,
        "age":             float(age),
        "hypertension":    int(hypertension),
        "heart_disease":   int(heart_disease),
        "ever_married":    ever_married,
        "work_type":       work_type,
        "Residence_type":  residence_type,
        "avg_glucose_level": float(avg_glucose),
        "bmi":             float(bmi),
        "smoking_status":  smoking_status,
    }

    patient_df = build_patient_df(raw_inputs, encoders, feature_cols)

    # ── Predict — call pipeline directly (deterministic) ────────────────────
    # predict_proba on the full pipeline handles scaler internally.
    # SMOTE step is skipped at inference automatically by ImbPipeline.
    probability = float(best_pipeline.predict_proba(patient_df)[0, 1])
    prediction  = int(probability >= threshold)   # ← same threshold used everywhere

    # ── Risk label derived from threshold, not hardcoded numbers ────────────
    if probability < threshold * 0.6:
        risk_label = "🟢 Low Risk"
        risk_color = "#2ecc71"
    elif probability < threshold:
        risk_label = "🟡 Moderate Risk"
        risk_color = "#f39c12"
    elif probability < threshold * 1.5:
        risk_label = "🔴 High Risk"
        risk_color = "#e74c3c"
    else:
        risk_label = "🚨 Very High Risk"
        risk_color = "#8e1c1c"

    # ── Display ──────────────────────────────────────────────────────────────
    st.markdown("---")
    g_col, m_col = st.columns([1.4, 1])

    with g_col:
        st.plotly_chart(make_gauge(probability, threshold), use_container_width=True)

    with m_col:
        st.markdown(f"### {risk_label}")
        st.markdown(f"**Stroke Probability:** `{probability:.1%}`")
        st.markdown(f"**Classification:** `{'Stroke' if prediction == 1 else 'No Stroke'}`")
        st.markdown(f"**Decision Threshold:** `{threshold:.3f}`")
        st.markdown(f"**Model:** `{meta['model_name'].replace('_', ' ').title()}`")
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ Elevated stroke risk detected. Clinical review recommended.")
        else:
            st.success("✅ Risk is below the clinical threshold.")


    # ── All-model probability comparison ─────────────────────────────────────
    if all_models:
        st.markdown("### 📊 All Model Probabilities")
        st.caption(
            "Logistic Regression with SMOTE/class-weight can produce compressed probabilities — "
            "tree-based models (XGBoost, LightGBM, Random Forest) are generally better calibrated "
            "and more reliable for high-risk patients."
        )
        mcols = st.columns(len(all_models))
        model_display_names = {
            "logistic_regression": "Logistic Reg",
            "decision_tree": "Decision Tree",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
        }
        for col_ui, (mname, mpipeline) in zip(mcols, all_models.items()):
            mp = float(mpipeline.predict_proba(patient_df)[0, 1])
            is_best = (mname == meta["model_name"])
            mp_thr  = float(meta["threshold"]) if is_best else 0.5
            mp_pred = "Stroke" if mp >= mp_thr else "No Stroke"
            clr     = "#e74c3c" if mp_pred == "Stroke" else "#2ecc71"
            label   = model_display_names.get(mname, mname)
            star    = "⭐ Best" if is_best else label
            col_ui.markdown(
                f"<div style='text-align:center;padding:10px 4px;border:1px solid #444;"
                f"border-radius:8px;background:#1a1a2e'>"
                f"<div style='font-size:11px;color:#aaa;margin-bottom:4px'>{star}</div>"
                f"<div style='font-size:22px;font-weight:700;color:{clr}'>{mp:.1%}</div>"
                f"<div style='font-size:11px;color:{clr}'>{mp_pred}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("---")

    # ── SHAP Explanation ─────────────────────────────────────────────────────
    st.markdown("### 🔬 SHAP Feature Explanation")
    st.caption(
        "Each bar shows how much a feature pushed the prediction **toward** (red, +) "
        "or **away from** (blue, −) a stroke classification. Feature values in brackets."
    )
    try:
        shap_vals, display_df, base_val = get_shap_values_for_patient(best_pipeline, patient_df, X_train_bg=X_train)
        feat_names = list(patient_df.columns)
        feat_vals  = display_df.iloc[0].values

        fig_shap = make_shap_bar(shap_vals, feat_names, feat_vals, top_n=10)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        st.caption(f"SHAP base value (model prior): `{base_val:.4f}` | "
                   f"Sum of SHAP values: `{shap_vals.sum():.4f}` | "
                   f"Approx model output (log-odds/score): `{base_val + shap_vals.sum():.4f}`")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    # ── Feature summary table ────────────────────────────────────────────────
    with st.expander("📋 Processed Feature Values Sent to Model"):
        st.dataframe(patient_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: Performance Metrics
# ─────────────────────────────────────────────
def page_performance(models: dict, best_meta: dict, X_test, y_test, X_train=None):
    st.title("📊 Model Performance")

    best_name = best_meta["model_name"]
    opt_thr   = float(best_meta["threshold"])

    # Threshold map: optimal threshold for best model, 0.5 for others
    threshold_map = {n: 0.5 for n in models}
    threshold_map[best_name] = opt_thr

    metrics_df = compute_all_metrics(models, X_test, y_test, threshold_map)

    st.subheader("Summary Table")
    st.dataframe(
        metrics_df.style.highlight_max(subset=["ROC-AUC", "Recall", "F1"], color="#d5f5e3")
                        .highlight_min(subset=["ROC-AUC", "Recall", "F1"], color="#fadbd8"),
        use_container_width=True, hide_index=True
    )

    # Target checkboxes
    best_auc    = metrics_df["ROC-AUC"].max()
    best_recall = metrics_df["Recall"].max()
    col1, col2 = st.columns(2)
    col1.metric("Best ROC-AUC", f"{best_auc:.4f}",
                delta="✓ Target Met" if best_auc >= 0.80 else "✗ Target Not Met")
    col2.metric("Best Recall", f"{best_recall:.4f}",
                delta="✓ Target Met" if best_recall >= 0.70 else "✗ Target Not Met")

    st.markdown("---")
    tab1, tab2 = st.tabs(["ROC Curves", "Precision-Recall Curves"])
    with tab1:
        st.plotly_chart(make_roc_figure(models, X_test, y_test), use_container_width=True)
    with tab2:
        st.plotly_chart(make_pr_figure(models, X_test, y_test), use_container_width=True)

    # Comparison CSV if it exists
    comp = load_comparison()
    if comp is not None:
        with st.expander("📄 Full Training CV + Test Comparison"):
            st.dataframe(comp, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# Main app entry point
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="StrokeGuard — Stroke Prediction",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global style ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
        html, body, [class*="css"]  { font-family: 'Source Sans 3', sans-serif; }
        h1, h2, h3 { font-family: 'Lora', serif; }
        .stButton>button {
            background: #1a3a5c; color: white; border-radius: 6px;
            font-weight: 600; border: none; padding: 0.6rem 1.2rem;
        }
        .stButton>button:hover { background: #254e7a; }
        .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Load resources ────────────────────────────────────────────────────────
    try:
        models        = load_all_models()
        encoders      = load_encoders()
        best_pipeline, meta = load_best_meta()
        X_test, y_test, X_train = load_test_data()
    except FileNotFoundError as e:
        st.error(f"Required model files not found: {e}\n\nRun `preprocess.py → features.py → train.py` first.")
        st.stop()

    # Feature columns come from the saved test CSV (ground truth column order)
    feature_cols = list(X_test.columns)

    # ── Sidebar navigation ────────────────────────────────────────────────────
    st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.sidebar.title("StrokeGuard")
    st.sidebar.caption("Stroke Risk Prediction System")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🩺 Risk Evaluation", "📊 Performance Metrics"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Active Model:** {meta['model_name'].replace('_', ' ').title()}\n\n"
        f"**Threshold:** `{float(meta['threshold']):.4f}`"
    )

    # ── Route ─────────────────────────────────────────────────────────────────
    if page == "🩺 Risk Evaluation":
        page_risk_evaluation(best_pipeline, meta, encoders, feature_cols, models, X_train)
    else:
        page_performance(models, meta, X_test, y_test, X_train)


if __name__ == "__main__":
    main()