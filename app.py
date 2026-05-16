import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StrokeGuard — Stroke Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #0f1117;
    color: #e8eaf0;
}
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1f2937;
}
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2540 100%);
    border: 1px solid #2a3350;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7fa3;
    margin-top: 4px;
}
.risk-low { color: #34d399; }
.risk-medium { color: #fbbf24; }
.risk-high { color: #f87171; }
.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}
.badge-low { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
.badge-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
.badge-high { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }
.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4b5fa3;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2937;
}
h1, h2, h3 { color: #e8eaf0; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #9aa3b8 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"
COHORT_DIR = "models/cohort_results"
SHAP_DIR = "models/shap_plots"


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    meta = joblib.load(f"{MODELS_DIR}/best_model_meta.pkl")
    encoders = joblib.load(f"{MODELS_DIR}/encoders.pkl")
    shap_data = joblib.load(f"{MODELS_DIR}/shap_data.pkl")
    comparison_df = pd.read_csv(f"{MODELS_DIR}/model_comparison.csv")
    return pipeline, meta, encoders, shap_data, comparison_df


@st.cache_data
def load_cohort_data():
    q1 = pd.read_csv(f"{COHORT_DIR}/q1_stroke_by_age_group.csv")
    q2 = pd.read_csv(f"{COHORT_DIR}/q2_stroke_by_glucose.csv")
    q3 = pd.read_csv(f"{COHORT_DIR}/q3_stroke_by_bmi_tier.csv")
    q4 = pd.read_csv(f"{COHORT_DIR}/q4_stroke_by_hypertension.csv")
    q5 = pd.read_csv(f"{COHORT_DIR}/q5_top_risk_cohorts.csv")
    return q1, q2, q3, q4, q5


def get_classifier_and_transformed(pipeline, X: pd.DataFrame):
    X_t = X.copy()
    for name, step in pipeline.steps[:-1]:
        if name == "smote":
            continue
        X_t = pd.DataFrame(step.transform(X_t), columns=X_t.columns)
    return pipeline.named_steps["clf"], X_t


def encode_patient(patient_dict: dict, encoders: dict, feature_cols: list) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df = pd.DataFrame([patient_dict])
    ordinal_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in ordinal_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


def build_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bmi = df["bmi"]
    df["bmi_risk_tier"] = 1
    df.loc[bmi < 18.5, "bmi_risk_tier"] = 0
    df.loc[(bmi >= 25) & (bmi < 30), "bmi_risk_tier"] = 2
    df.loc[bmi >= 30, "bmi_risk_tier"] = 3

    g = df["avg_glucose_level"]
    df["glucose_category"] = 0
    df.loc[(g >= 114) & (g < 140), "glucose_category"] = 1
    df.loc[g >= 140, "glucose_category"] = 2

    df["age_hypertension_interaction"] = df["age"] * df["hypertension"]
    df["age_heart_disease_interaction"] = df["age"] * df["heart_disease"]

    age = df["age"]
    df["age_group"] = 0
    df.loc[(age >= 18) & (age < 40), "age_group"] = 1
    df.loc[(age >= 40) & (age < 60), "age_group"] = 2
    df.loc[age >= 60, "age_group"] = 3

    diabetic = (df["glucose_category"] == 2).astype(int)
    df["vascular_risk_score"] = df["hypertension"] + df["heart_disease"] + diabetic
    df["bmi_glucose_interaction"] = df["bmi"] * df["avg_glucose_level"]
    df["senior_hypertensive"] = ((df["age"] > 60) & (df["hypertension"] == 1)).astype(int)
    return df


def plotly_dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa3b8", family="DM Sans"),
        xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    )


pipeline, meta, encoders, shap_data, comparison_df = load_artifacts()
q1, q2, q3, q4, q5 = load_cohort_data()

BASE_FEATURE_COLS = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
                     "gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FE_COLS = list(pd.read_csv(f"{MODELS_DIR}/X_train_fe.csv").columns)

st.sidebar.markdown("## 🛡️ StrokeGuard")
st.sidebar.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Risk Scorer", "Model Comparison", "Cohort Explorer", "Feature Importance"],
                         label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='section-header'>Model Active</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"`{meta['model_name'].replace('_', ' ').title()}`")
st.sidebar.markdown(f"Threshold: `{meta['threshold']:.4f}`")


if page == "Risk Scorer":
    st.markdown("# Risk Scorer")
    st.markdown("<div class='section-header'>Enter patient vitals to compute real-time stroke risk</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 1, 100, 55)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with col2:
        avg_glucose_level = st.number_input("Avg Glucose Level", 50.0, 300.0, 120.0, step=0.5)
        bmi = st.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    if st.button("Compute Stroke Risk", type="primary", use_container_width=True):
        patient_raw = {
            "age": age, "hypertension": hypertension, "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level, "bmi": bmi, "gender": gender,
            "ever_married": ever_married, "work_type": work_type,
            "Residence_type": residence, "smoking_status": smoking
        }

        df_enc = pd.DataFrame([patient_raw])
        for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
            df_enc[col] = encoders[col].transform(df_enc[col].astype(str))

        df_fe = build_patient_features(df_enc)
        df_fe = df_fe.reindex(columns=FE_COLS, fill_value=0)

        prob = pipeline.predict_proba(df_fe)[0][1]
        threshold = meta["threshold"]
        prediction = int(prob >= threshold)

        all_probs = pd.Series(pipeline.predict_proba(pd.read_csv(f"{MODELS_DIR}/X_test_fe.csv"))[:, 1])
        low_cut = all_probs.quantile(0.55)
        high_cut = all_probs.quantile(0.85)

        if prob < low_cut:
            risk_label, badge_class, value_class = "Low Risk", "badge-low", "risk-low"
        elif prob < high_cut:
            risk_label, badge_class, value_class = "Medium Risk", "badge-medium", "risk-medium"
        else:
            risk_label, badge_class, value_class = "High Risk", "badge-high", "risk-high"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value {value_class}'>{prob*100:.1f}%</div>
                <div class='metric-label'>Stroke Probability</div>
                <div style='color:#6b7fa3; font-size:0.72rem; margin-top:6px;'>Relative to population distribution</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='margin-top:8px'><span class='risk-badge {badge_class}'>{risk_label}</span></div>
                <div class='metric-label' style='margin-top:12px'>Risk Classification</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#6b7fa3'>{meta['threshold']:.2f}</div>
                <div class='metric-label'>Decision Threshold</div>
            </div>""", unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#e8eaf0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7fa3"},
                "bar": {"color": "#f87171" if prob >= 0.6 else "#fbbf24" if prob >= 0.3 else "#34d399"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(52,211,153,0.1)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.1)"},
                    {"range": [60, 100], "color": "rgba(248,113,113,0.1)"},
                ],
                "threshold": {"line": {"color": "#e8eaf0", "width": 2}, "value": threshold * 100},
                "bgcolor": "rgba(0,0,0,0)",
            }
        ))
        gauge.update_layout(height=250, margin=dict(t=20, b=10), **plotly_dark_layout())
        st.plotly_chart(gauge, use_container_width=True)

        try:
            clf, X_t = get_classifier_and_transformed(pipeline, df_fe)
            model_type = type(clf).__name__
            if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_t)
                if isinstance(sv, list):
                    sv = sv[1]
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = ev[1]
            else:
                explainer = shap.LinearExplainer(clf, X_t, feature_perturbation="interventional")
                sv = explainer.shap_values(X_t)
                ev = explainer.expected_value

            patient_sv = sv[0]
            feature_names = X_t.columns.tolist()
            patient_vals = X_t.iloc[0].values
            sorted_idx = np.argsort(np.abs(patient_sv))[::-1][:10]

            top_f = [feature_names[i] for i in sorted_idx][::-1]
            top_sv = patient_sv[sorted_idx][::-1]
            top_v = patient_vals[sorted_idx][::-1]
            colors = ["#f87171" if v > 0 else "#60a5fa" for v in top_sv]

            wf = go.Figure(go.Bar(
                x=top_sv, y=[f"{top_f[i]} = {top_v[i]:.2f}" for i in range(len(top_f))],
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in top_sv], textposition="outside"
            ))
            wf.update_layout(
                title="SHAP Waterfall — This Patient",
                height=380, margin=dict(l=10, r=60, t=40, b=10),
                **plotly_dark_layout()
            )
            st.plotly_chart(wf, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP waterfall unavailable: {e}")


elif page == "Model Comparison":
    st.markdown("# Model Comparison")
    st.markdown("<div class='section-header'>Stratified 5-fold CV + held-out test set evaluation</div>", unsafe_allow_html=True)

    display_cols = ["model", "cv_roc_auc_mean", "cv_roc_auc_std", "test_roc_auc", "test_recall", "test_f1", "test_precision"]
    available = [c for c in display_cols if c in comparison_df.columns]
    st.dataframe(
        comparison_df[available].style.highlight_max(
            subset=[c for c in ["cv_roc_auc_mean", "test_roc_auc", "test_recall", "test_f1"] if c in available],
            color="#1e3a2f"
        ).format({c: "{:.4f}" for c in available if c != "model"}),
        use_container_width=True, height=220
    )

    st.markdown("---")

    from sklearn.metrics import roc_curve, roc_auc_score
    X_test_fe = pd.read_csv(f"{MODELS_DIR}/X_test_fe.csv")
    y_test = pd.read_csv(f"{MODELS_DIR}/y_test.csv").squeeze()

    model_names = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]
    colors_roc = ["#60a5fa", "#a78bfa", "#34d399", "#f87171", "#fbbf24"]

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="#374151"), name="Random"))
    for mname, color in zip(model_names, colors_roc):
        path = f"{MODELS_DIR}/{mname}.pkl"
        if not os.path.exists(path):
            continue
        m = joblib.load(path)
        y_prob = m.predict_proba(X_test_fe)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{mname.replace('_',' ').title()} (AUC={auc:.4f})",
            line=dict(color=color, width=2)
        ))

    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=460, legend=dict(x=0.55, y=0.1),
        **plotly_dark_layout()
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_auc = go.Figure(go.Bar(
            x=comparison_df["model"].str.replace("_", " ").str.title(),
            y=comparison_df["test_roc_auc"],
            marker_color=colors_roc[:len(comparison_df)],
            text=comparison_df["test_roc_auc"].round(4), textposition="outside"
        ))
        fig_auc.update_layout(title="Test ROC-AUC", height=320, **plotly_dark_layout())
        st.plotly_chart(fig_auc, use_container_width=True)
    with c2:
        fig_rec = go.Figure(go.Bar(
            x=comparison_df["model"].str.replace("_", " ").str.title(),
            y=comparison_df["test_recall"],
            marker_color=colors_roc[:len(comparison_df)],
            text=comparison_df["test_recall"].round(4), textposition="outside"
        ))
        fig_rec.update_layout(title="Test Recall (Stroke Class)", height=320, **plotly_dark_layout())
        st.plotly_chart(fig_rec, use_container_width=True)


elif page == "Cohort Explorer":
    st.markdown("# Cohort Explorer")
    st.markdown("<div class='section-header'>SQL cohort analysis — stroke prevalence across clinical segments</div>", unsafe_allow_html=True)

    def cohort_bar(df, x_col, y_col, title, color="#60a5fa"):
        fig = go.Figure(go.Bar(
            x=df[x_col], y=df[y_col],
            marker_color=color, text=df[y_col].round(2).astype(str) + "%",
            textposition="outside"
        ))
        fig.update_layout(title=title, yaxis_title="Stroke Rate (%)", height=320, **plotly_dark_layout())
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(cohort_bar(q1, "age_group", "stroke_rate_pct",
                                    "Stroke Rate by Age Group", "#60a5fa"), use_container_width=True)
    with c2:
        st.plotly_chart(cohort_bar(q2, "glucose_category", "stroke_rate_pct",
                                    "Stroke Rate by Glucose Category", "#f87171"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(cohort_bar(q3, "bmi_tier", "stroke_rate_pct",
                                    "Stroke Rate by BMI Tier", "#34d399"), use_container_width=True)
    with c4:
        st.plotly_chart(cohort_bar(q4, "hypertension_status", "stroke_rate_pct",
                                    "Stroke Rate by Hypertension Status", "#fbbf24"), use_container_width=True)

    st.markdown("### Top 5 Highest-Risk Cohorts")
    st.dataframe(q5.style.background_gradient(subset=["stroke_rate_pct"], cmap="Reds"),
                 use_container_width=True)


elif page == "Feature Importance":
    st.markdown("# Feature Importance")
    st.markdown("<div class='section-header'>Global SHAP analysis — top 10 clinical risk drivers</div>", unsafe_allow_html=True)

    sv = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top_features = [feature_names[i] for i in top_idx]
    top_shap = mean_abs[top_idx]

    fig_imp = go.Figure(go.Bar(
        x=top_shap[::-1],
        y=top_features[::-1],
        orientation="h",
        marker=dict(
            color=top_shap[::-1],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#f87171"]],
            showscale=False
        ),
        text=[f"{v:.4f}" for v in top_shap[::-1]],
        textposition="outside"
    ))
    fig_imp.update_layout(
        title="Mean |SHAP Value| — Top 10 Features",
        xaxis_title="Mean Absolute SHAP Value",
        height=420, margin=dict(l=10, r=80, t=40, b=10),
        **plotly_dark_layout()
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### Clinical Interpretation")
    interpretations = {
        "age": "Older patients carry exponentially higher stroke risk — the single strongest predictor.",
        "age_squared": "Non-linear age effect: risk accelerates after ~60 years.",
        "avg_glucose_level": "Elevated glucose indicates metabolic stress and vascular damage.",
        "glucose_squared": "Quadratic glucose term captures extreme hyperglycemia risk.",
        "age_hypertension_interaction": "Hypertension compounds with age — elderly hypertensive patients are highest risk.",
        "age_heart_disease_interaction": "Heart disease becomes markedly more dangerous with advancing age.",
        "bmi_glucose_interaction": "Metabolic syndrome proxy: high BMI + high glucose together indicate vascular risk.",
        "vascular_risk_score": "Composite of hypertension, heart disease, and diabetes — summarises overall vascular burden.",
        "senior_hypertensive": "Binary flag for the highest-risk demographic: over-60 with hypertension.",
        "hypertension": "Standalone hypertension directly elevates stroke probability.",
        "heart_disease": "Pre-existing cardiac conditions significantly increase stroke incidence.",
        "bmi_risk_tier": "Obesity tier correlates with stroke, though weaker than metabolic markers.",
        "glucose_category": "Diabetic-range glucose carries 2-3× higher stroke rate vs normal.",
        "age_group": "Ordinal age bracket — confirms stroke is concentrated in 60+ cohort.",
        "bmi": "BMI as continuous variable; effect partially captured by bmi_risk_tier.",
        "senior_hypertensive": "Flags elderly hypertensives — a high-priority clinical screening target.",
    }

    for i, feat in enumerate(top_features):
        interp = interpretations.get(feat, "Contributes to stroke risk prediction — see SHAP beeswarm for directionality.")
        st.markdown(f"""
        <div class='metric-card' style='padding:14px 20px; margin-bottom:8px;'>
            <span style='font-family:DM Mono,monospace; color:#60a5fa; font-size:0.85rem;'>#{i+1} {feat}</span>
            <span style='color:#6b7fa3; font-size:0.75rem; margin-left:12px;'>SHAP={top_shap[i]:.4f}</span>
            <div style='color:#c4cde0; font-size:0.88rem; margin-top:6px;'>{interp}</div>
        </div>""", unsafe_allow_html=True)