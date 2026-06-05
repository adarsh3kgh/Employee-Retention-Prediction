import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Main background */
.main { background-color: #0f172a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2rem; font-weight: 700; color: #f8fafc; }
.metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

/* Section header */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #334155;
}

/* Risk badge */
.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fecaca;
}
.risk-low {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: #a7f3d0;
}

/* Input card */
.input-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Prediction probability bar */
.prob-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}

/* Stbutton */
div.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 12px 32px;
    transition: all 0.2s;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
}

/* Tab styling */
[data-testid="stTabs"] button {
    background: transparent;
    color: #94a3b8;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f8fafc;
    border-bottom: 2px solid #6366f1;
}

/* Dataframe */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Form labels */
label { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

/* Number input / selectbox */
input, select { background-color: #0f172a !important; color: #f8fafc !important; border: 1px solid #334155 !important; border-radius: 6px !important; }

/* Slider */
[data-testid="stSlider"] > div { color: #6366f1; }

/* Tips box */
.tip-box {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.tip-title { font-weight: 600; color: #a5b4fc; margin-bottom: 4px; }

/* Feature importance bar */
.fi-bar-bg {
    background: #0f172a;
    border-radius: 4px;
    height: 10px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("employee_retention_model.pkl")

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("hr_employee_churn_data.csv")
    except FileNotFoundError:
        return None

model = load_model()
df_dataset = load_dataset()

FEATURE_NAMES = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company", "Work_accident",
    "promotion_last_5years", "salary_low", "salary_medium"
]
FEATURE_IMPORTANCES = [0.279, 0.095, 0.164, 0.046, 0.321, 0.040, 0.011, 0.021, 0.022]
FEATURE_LABELS = [
    "Satisfaction Level", "Last Evaluation", "# Projects",
    "Monthly Hours", "Years at Company", "Work Accident",
    "Promotion (5y)", "Salary: Low", "Salary: Medium"
]

LOG_PATH = "predictions_log.csv"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👥 HR Analytics")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool uses an **XGBoost** model trained on 15,000 employee records "
        "to predict churn risk with high accuracy."
    )
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("- **Model:** XGBoost Classifier")
    st.markdown("- **Training size:** 14,999 records")
    st.markdown("- **Churn rate (base):** 23.8%")
    st.markdown("---")
    st.markdown("### Top Churn Drivers")

    sorted_fi = sorted(zip(FEATURE_LABELS, FEATURE_IMPORTANCES), key=lambda x: -x[1])
    for label, imp in sorted_fi[:5]:
        pct = int(imp * 100)
        st.markdown(f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#94a3b8">
                <span>{label}</span><span style="color:#a5b4fc">{pct}%</span>
            </div>
            <div style="background:#0f172a;border-radius:4px;height:6px;margin-top:4px">
                <div style="background:linear-gradient(90deg,#6366f1,#3b82f6);width:{pct}%;height:6px;border-radius:4px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.75rem;text-align:center'>Built with Streamlit + XGBoost</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 8px">
    <div style="font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#3b82f6,#06b6d4);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2">
        Employee Retention Predictor
    </div>
    <div style="color:#64748b;font-size:1rem;margin-top:8px;font-weight:400">
        AI-powered churn risk assessment · Predict · Analyze · Retain
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI STRIP (if dataset loaded)
# ─────────────────────────────────────────────
if df_dataset is not None:
    k1, k2, k3, k4 = st.columns(4)
    total = len(df_dataset)
    churned = df_dataset['left'].sum()
    retained = total - churned
    avg_sat = df_dataset['satisfaction_level'].mean()

    for col, val, label, color in [
        (k1, f"{total:,}", "Total Employees", "#6366f1"),
        (k2, f"{retained:,}", "Retained", "#10b981"),
        (k3, f"{churned:,}", "Churned", "#ef4444"),
        (k4, f"{avg_sat:.2f}", "Avg Satisfaction", "#f59e0b"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_predict, tab_history, tab_explore = st.tabs(["🔮 Predict", "📋 History", "📊 Explore Data"])


# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Employee Details</div>', unsafe_allow_html=True)

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                emp_name = st.text_input("Employee Name", placeholder="e.g. Riya Sharma")
            with c2:
                emp_id = st.text_input("Employee ID", placeholder="e.g. EMP-1042")

        st.markdown('<div class="section-header" style="margin-top:16px">Performance & Workload</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            satisfaction_level = st.slider(
                "😊 Satisfaction Level", 0.0, 1.0, 0.5, 0.01,
                help="Employee self-reported satisfaction (0 = very dissatisfied, 1 = very satisfied)"
            )
            number_project = st.number_input("📁 Number of Projects", 1, 10, 3,
                help="Total projects currently assigned")
            time_spend_company = st.number_input("🏢 Years at Company", 1, 10, 3,
                help="Total years the employee has spent in the company")
        with c2:
            last_evaluation = st.slider(
                "⭐ Last Evaluation Score", 0.0, 1.0, 0.6, 0.01,
                help="Most recent performance evaluation score"
            )
            average_montly_hours = st.number_input("⏱ Monthly Hours", 50, 350, 160,
                help="Average working hours per month")

        st.markdown('<div class="section-header" style="margin-top:16px">HR Attributes</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            Work_accident = st.selectbox("🚑 Work Accident", ["No", "Yes"])
        with c2:
            promotion_last_5years = st.selectbox("🏆 Promotion (5 Yrs)", ["No", "Yes"])
        with c3:
            salary = st.selectbox("💰 Salary Band", ["Low", "Medium", "High"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Run Prediction", use_container_width=True)

    # ─── RESULT COLUMN ───
    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if not predict_btn:
            st.markdown("""
            <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;
                padding:48px 24px;text-align:center;color:#475569">
                <div style="font-size:3rem;margin-bottom:12px">🔮</div>
                <div style="font-size:1rem;font-weight:500;color:#64748b">
                    Fill in the employee details<br>and click <b>Run Prediction</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── VALIDATION ──
            error = None
            if number_project > 6 and average_montly_hours < 100:
                error = "⚠️ Unrealistic input: Too many projects with very low working hours."

            if error:
                st.error(error)
            else:
                wa = 1 if Work_accident == "Yes" else 0
                pr = 1 if promotion_last_5years == "Yes" else 0
                low = 1 if salary == "Low" else 0
                med = 1 if salary == "Medium" else 0

                features = np.array([[satisfaction_level, last_evaluation,
                                      number_project, average_montly_hours,
                                      time_spend_company, wa, pr, low, med]])

                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0]
                churn_prob = prob[1]
                retain_prob = prob[0]

                # ── Risk Badge ──
                if pred == 1:
                    badge_class = "risk-high"
                    icon = "🚨"
                    label_text = "High Risk of Leaving"
                    prob_color = "#ef4444"
                else:
                    badge_class = "risk-low"
                    icon = "✅"
                    label_text = "Likely to Stay"
                    prob_color = "#10b981"

                st.markdown(f'<div class="{badge_class}">{icon} {label_text}</div>', unsafe_allow_html=True)

                # ── Probability Gauge ──
                st.markdown("<br>", unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(churn_prob * 100, 1),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Churn Probability", "font": {"color": "#94a3b8", "size": 14}},
                    number={"suffix": "%", "font": {"color": "#f8fafc", "size": 32}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
                        "bar": {"color": prob_color, "thickness": 0.3},
                        "bgcolor": "#0f172a",
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 30], "color": "#052e16"},
                            {"range": [30, 60], "color": "#451a03"},
                            {"range": [60, 100], "color": "#450a0a"},
                        ],
                        "threshold": {"line": {"color": prob_color, "width": 3}, "value": churn_prob * 100},
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=220,
                    margin=dict(l=30, r=30, t=40, b=10),
                    font={"color": "#94a3b8"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

                # ── Confidence bars ──
                c_stay, c_leave = st.columns(2)
                c_stay.metric("Stay Probability", f"{retain_prob*100:.1f}%")
                c_leave.metric("Leave Probability", f"{churn_prob*100:.1f}%")

                # ── Feature Impact (SHAP-style visual) ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Key Risk Factors</div>', unsafe_allow_html=True)

                input_vals = [satisfaction_level, last_evaluation, number_project,
                              average_montly_hours, time_spend_company, wa, pr, low, med]
                contributions = [v * i for v, i in zip(input_vals, FEATURE_IMPORTANCES)]
                top_idx = sorted(range(len(contributions)), key=lambda i: -abs(contributions[i]))[:5]

                fig_bar = go.Figure()
                colors = ["#ef4444" if contributions[i] > 0.12 else "#6366f1" for i in top_idx]
                fig_bar.add_trace(go.Bar(
                    x=[FEATURE_LABELS[i] for i in top_idx],
                    y=[contributions[i] for i in top_idx],
                    marker_color=colors,
                    text=[f"{contributions[i]:.3f}" for i in top_idx],
                    textposition="outside",
                    textfont={"color": "#94a3b8", "size": 11}
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=60),
                    xaxis={"showgrid": False, "tickfont": {"color": "#64748b", "size": 10}},
                    yaxis={"showgrid": True, "gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

                # ── Recommendations ──
                st.markdown('<div class="section-header">💡 HR Recommendations</div>', unsafe_allow_html=True)
                tips = []
                if satisfaction_level < 0.45:
                    tips.append(("Low Satisfaction", "Consider a one-on-one to address concerns, explore role fit."))
                if number_project > 5:
                    tips.append(("Overloaded", "Redistribute projects — employee may be burning out."))
                if average_montly_hours > 230:
                    tips.append(("Overworked", "High working hours detected. Review workload & deadlines."))
                if promotion_last_5years == "No" and time_spend_company >= 4:
                    tips.append(("No Recent Promotion", "Consider a growth conversation or recognition program."))
                if salary == "Low":
                    tips.append(("Low Salary Band", "Benchmark compensation against market to stay competitive."))
                if last_evaluation > 0.8 and satisfaction_level < 0.5:
                    tips.append(("High Performer, Low Morale", "Top performer at risk — prioritize retention package."))

                if tips:
                    for t, desc in tips:
                        st.markdown(f"""
                        <div class="tip-box">
                            <div class="tip-title">⚡ {t}</div>
                            {desc}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tip-box">
                        <div class="tip-title">✅ No immediate concerns</div>
                        Employee appears to have a healthy profile. Keep up regular check-ins.
                    </div>
                    """, unsafe_allow_html=True)

                # ── Save Log ──
                new_row = pd.DataFrame([{
                    "Name": emp_name or "N/A",
                    "Employee_ID": emp_id or "N/A",
                    "Satisfaction": satisfaction_level,
                    "Projects": number_project,
                    "Monthly_Hrs": average_montly_hours,
                    "Years": time_spend_company,
                    "Salary": salary,
                    "Prediction": label_text,
                    "Churn_Prob_%": round(churn_prob * 100, 1),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }])

                if os.path.exists(LOG_PATH):
                    updated = pd.concat([pd.read_csv(LOG_PATH), new_row], ignore_index=True)
                else:
                    updated = new_row
                updated.to_csv(LOG_PATH, index=False)

                st.success("✅ Prediction saved to history.")


# ══════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)

    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH).sort_values("Timestamp", ascending=False)

        # Summary strip
        total_pred = len(log_df)
        high_risk = (log_df["Prediction"] == "High Risk of Leaving").sum()
        low_risk = total_pred - high_risk

        c1, c2, c3 = st.columns(3)
        for col, val, label, col_color in [
            (c1, total_pred, "Total Predictions", "#6366f1"),
            (c2, high_risk, "High Risk", "#ef4444"),
            (c3, low_risk, "Likely to Stay", "#10b981"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{col_color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter
        filt = st.selectbox("Filter by prediction:", ["All", "High Risk of Leaving", "Likely to Stay"])
        if filt != "All":
            log_df = log_df[log_df["Prediction"] == filt]

        # Color rows
        def highlight_pred(val):
            if val == "High Risk of Leaving":
                return "background-color: #450a0a; color: #fecaca"
            return "background-color: #052e16; color: #a7f3d0"

        styled = log_df.style.applymap(highlight_pred, subset=["Prediction"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download
        csv_dl = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download CSV", csv_dl, "predictions_log.csv", "text/csv")

        if st.button("🗑️  Clear History"):
            os.remove(LOG_PATH)
            st.success("History cleared. Refresh the page.")
    else:
        st.markdown("""
        <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;
            padding:48px;text-align:center;color:#475569">
            <div style="font-size:2.5rem;margin-bottom:8px">📋</div>
            No prediction history yet. Run a prediction to get started.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — EXPLORE DATA
# ══════════════════════════════════════════════
with tab_explore:
    if df_dataset is None:
        st.warning("Place `hr_employee_churn_data.csv` in the same directory to enable this tab.")
    else:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

        row1c1, row1c2 = st.columns(2)

        # Chart 1: Churn distribution donut
        with row1c1:
            churn_counts = df_dataset["left"].value_counts()
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Retained", "Churned"],
                values=[churn_counts[0], churn_counts[1]],
                hole=0.6,
                marker_colors=["#10b981", "#ef4444"],
                textinfo="percent+label",
                textfont={"color": "#f8fafc"},
            )])
            fig_donut.update_layout(
                title={"text": "Overall Churn Distribution", "font": {"color": "#94a3b8", "size": 14}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend={"font": {"color": "#94a3b8"}},
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                annotations=[{"text": f"<b>{len(df_dataset):,}</b><br>Employees",
                               "x": 0.5, "y": 0.5, "font": {"size": 14, "color": "#f8fafc"},
                               "showarrow": False}]
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        # Chart 2: Salary vs Churn
        with row1c2:
            sal_churn = df_dataset.groupby(["salary", "left"]).size().reset_index(name="count")
            sal_churn["status"] = sal_churn["left"].map({0: "Retained", 1: "Churned"})
            fig_sal = px.bar(
                sal_churn, x="salary", y="count", color="status",
                color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
                barmode="group",
                category_orders={"salary": ["low", "medium", "high"]},
                title="Salary Band vs Churn",
            )
            fig_sal.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                title_font={"color": "#94a3b8", "size": 14},
                legend={"font": {"color": "#94a3b8"}, "title": {"text": ""}},
                xaxis={"showgrid": False, "tickfont": {"color": "#64748b"}},
                yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
            )
            st.plotly_chart(fig_sal, use_container_width=True, config={"displayModeBar": False})

        row2c1, row2c2 = st.columns(2)

        # Chart 3: Satisfaction vs Churn box
        with row2c1:
            fig_sat = go.Figure()
            for val, name, color in [(0, "Retained", "#10b981"), (1, "Churned", "#ef4444")]:
                fig_sat.add_trace(go.Box(
                    y=df_dataset[df_dataset["left"] == val]["satisfaction_level"],
                    name=name,
                    marker_color=color,
                    line={"color": color},
                    fillcolor=color + "33",
                ))
            fig_sat.update_layout(
                title={"text": "Satisfaction Level by Status", "font": {"color": "#94a3b8", "size": 14}},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                legend={"font": {"color": "#94a3b8"}},
                xaxis={"showgrid": False, "tickfont": {"color": "#64748b"}},
                yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
            )
            st.plotly_chart(fig_sat, use_container_width=True, config={"displayModeBar": False})

        # Chart 4: Monthly hours vs projects scatter
        with row2c2:
            sample = df_dataset.sample(min(500, len(df_dataset)), random_state=42)
            fig_sc = px.scatter(
                sample,
                x="average_montly_hours", y="satisfaction_level",
                color=sample["left"].map({0: "Retained", 1: "Churned"}),
                color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
                title="Monthly Hours vs Satisfaction",
                opacity=0.6,
            )
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                title_font={"color": "#94a3b8", "size": 14},
                legend={"font": {"color": "#94a3b8"}, "title": {"text": ""}},
                xaxis={"showgrid": False, "tickfont": {"color": "#64748b"}},
                yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
            )
            st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

        # Chart 5: Feature importance
        st.markdown('<div class="section-header" style="margin-top:8px">Model Feature Importance</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({"Feature": FEATURE_LABELS, "Importance": FEATURE_IMPORTANCES})
        fi_df = fi_df.sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#1e293b"], [0.5, "#6366f1"], [1, "#3b82f6"]],
                showscale=False,
            ),
            text=[f"{v*100:.1f}%" for v in fi_df["Importance"]],
            textposition="outside",
            textfont={"color": "#94a3b8"},
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=60, t=10, b=10),
            xaxis={"showgrid": False, "showticklabels": False},
            yaxis={"tickfont": {"color": "#94a3b8", "size": 12}},
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

        # Raw data toggle
        with st.expander("🔍 View Raw Dataset (first 100 rows)"):
            st.dataframe(df_dataset.head(100), use_container_width=True, hide_index=True)
