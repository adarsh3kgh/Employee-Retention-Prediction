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

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.main { background-color: #0f172a; }

.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fecaca;
    margin-bottom: 12px;
}
.risk-low {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    color: #a7f3d0;
    margin-bottom: 12px;
}

.tip-box {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #cbd5e1;
}
.tip-title { font-weight: 600; color: #a5b4fc; margin-bottom: 3px; }

div.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 28px;
    width: 100%;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    transform: translateY(-1px);
}

[data-testid="stTabs"] button { color: #64748b; font-weight: 500; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f8fafc;
    border-bottom: 2px solid #6366f1;
}

.stat-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 2px;
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

FEATURE_NAMES   = ["satisfaction_level","last_evaluation","number_project",
                   "average_montly_hours","time_spend_company","Work_accident",
                   "promotion_last_5years","salary_low","salary_medium"]
FEATURE_LABELS  = ["Satisfaction Level","Last Evaluation","# Projects",
                   "Monthly Hours","Years at Company","Work Accident",
                   "Promotion (5y)","Salary: Low","Salary: Medium"]
FEATURE_IMPORTANCES = [0.279,0.095,0.164,0.046,0.321,0.040,0.011,0.021,0.022]
LOG_PATH = "predictions_log.csv"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👥 HR Analytics")
    st.markdown("---")
    st.markdown("**XGBoost** model trained on 15 K employee records to predict churn risk.")
    st.markdown("---")
    st.markdown("### Top Churn Drivers")
    sorted_fi = sorted(zip(FEATURE_LABELS, FEATURE_IMPORTANCES), key=lambda x: -x[1])
    for label, imp in sorted_fi[:5]:
        pct = int(imp * 100)
        st.markdown(f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8">
                <span>{label}</span><span style="color:#a5b4fc">{pct}%</span>
            </div>
            <div style="background:#0f172a;border-radius:4px;height:5px;margin-top:4px">
                <div style="background:linear-gradient(90deg,#6366f1,#3b82f6);width:{pct}%;height:5px;border-radius:4px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#475569;font-size:0.72rem;text-align:center'>Streamlit · XGBoost · Plotly</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 20px">
    <div style="font-size:2.2rem;font-weight:800;
        background:linear-gradient(135deg,#6366f1,#3b82f6,#06b6d4);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;line-height:1.2">
        Employee Retention Predictor
    </div>
    <div style="color:#475569;font-size:0.9rem;margin-top:6px">
        AI-powered churn risk · results update live as you adjust inputs
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_predict, tab_history, tab_explore = st.tabs(["🔮 Predict", "📋 History", "📊 Explore Data"])


# ══════════════════════════════════════════════
# TAB 1 — PREDICT  (live, no button needed)
# ══════════════════════════════════════════════
with tab_predict:

    col_form, col_result = st.columns([1.05, 0.95], gap="large")

    # ── LEFT: INPUTS ──────────────────────────
    with col_form:

        st.markdown('<div class="section-header">Employee Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            emp_name = st.text_input("Employee Name", placeholder="e.g. Riya Sharma")
        with c2:
            emp_id = st.text_input("Employee ID", placeholder="e.g. EMP-1042")

        st.markdown('<div class="section-header" style="margin-top:14px">Performance & Workload</div>', unsafe_allow_html=True)

        satisfaction_level = st.slider(
            "😊 Satisfaction Level", 0.0, 1.0, 0.5, 0.01,
            help="0 = very dissatisfied · 1 = very satisfied"
        )
        last_evaluation = st.slider(
            "⭐ Last Evaluation Score", 0.0, 1.0, 0.6, 0.01,
            help="Most recent performance score"
        )

        c1, c2 = st.columns(2)
        with c1:
            number_project = st.number_input("📁 Projects", 1, 10, 3)
            time_spend_company = st.number_input("🏢 Years at Company", 1, 10, 3)
        with c2:
            average_montly_hours = st.number_input("⏱ Monthly Hours", 50, 350, 160)

        st.markdown('<div class="section-header" style="margin-top:14px">HR Attributes</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            Work_accident = st.selectbox("🚑 Work Accident", ["No", "Yes"])
        with c2:
            promotion_last_5years = st.selectbox("🏆 Promotion (5y)", ["No", "Yes"])
        with c3:
            salary = st.selectbox("💰 Salary", ["Low", "Medium", "High"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Save button (only saves, prediction already live)
        save_btn = st.button("💾  Save to History", use_container_width=True)

    # ── RIGHT: LIVE RESULT ────────────────────
    with col_result:
        st.markdown('<div class="section-header">Live Prediction</div>', unsafe_allow_html=True)

        # Encode
        wa  = 1 if Work_accident == "Yes" else 0
        pr  = 1 if promotion_last_5years == "Yes" else 0
        low = 1 if salary == "Low" else 0
        med = 1 if salary == "Medium" else 0

        # Validate
        if number_project > 6 and average_montly_hours < 100:
            st.error("⚠️ Unrealistic: too many projects with very low hours.")
        else:
            features = np.array([[satisfaction_level, last_evaluation,
                                  number_project, average_montly_hours,
                                  time_spend_company, wa, pr, low, med]])
            pred       = model.predict(features)[0]
            prob       = model.predict_proba(features)[0]
            churn_prob = prob[1]
            retain_prob = prob[0]

            if pred == 1:
                badge_class = "risk-high"
                icon        = "🚨"
                label_text  = "High Risk of Leaving"
                prob_color  = "#ef4444"
            else:
                badge_class = "risk-low"
                icon        = "✅"
                label_text  = "Likely to Stay"
                prob_color  = "#10b981"

            # Badge
            st.markdown(f'<div class="{badge_class}">{icon} {label_text}</div>', unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(churn_prob * 100, 1),
                domain={"x": [0,1], "y": [0,1]},
                title={"text": "Churn Probability", "font": {"color": "#94a3b8", "size": 13}},
                number={"suffix": "%", "font": {"color": "#f8fafc", "size": 30}},
                gauge={
                    "axis": {"range": [0,100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
                    "bar": {"color": prob_color, "thickness": 0.28},
                    "bgcolor": "#0f172a",
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, 30],  "color": "#052e16"},
                        {"range": [30, 60], "color": "#451a03"},
                        {"range": [60,100], "color": "#450a0a"},
                    ],
                    "threshold": {"line": {"color": prob_color, "width": 3}, "value": churn_prob*100},
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=210,
                margin=dict(l=28, r=28, t=36, b=8),
                font={"color": "#94a3b8"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # Stay / Leave pills
            c_s, c_l = st.columns(2)
            c_s.metric("Stay Probability",  f"{retain_prob*100:.1f}%")
            c_l.metric("Leave Probability", f"{churn_prob*100:.1f}%")

            # Key Risk Factors bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Key Risk Factors</div>', unsafe_allow_html=True)
            input_vals    = [satisfaction_level, last_evaluation, number_project,
                             average_montly_hours, time_spend_company, wa, pr, low, med]
            contributions = [v * i for v, i in zip(input_vals, FEATURE_IMPORTANCES)]
            top_idx = sorted(range(len(contributions)), key=lambda i: -abs(contributions[i]))[:5]
            bar_colors = ["#ef4444" if contributions[i] > 0.12 else "#6366f1" for i in top_idx]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[FEATURE_LABELS[i] for i in top_idx],
                y=[contributions[i] for i in top_idx],
                marker_color=bar_colors,
                text=[f"{contributions[i]:.3f}" for i in top_idx],
                textposition="outside",
                textfont={"color": "#94a3b8", "size": 10}
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(l=8, r=8, t=8, b=55),
                xaxis={"showgrid": False, "tickfont": {"color": "#64748b", "size": 9}},
                yaxis={"showgrid": True, "gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # Recommendations
            st.markdown('<div class="section-header">💡 HR Recommendations</div>', unsafe_allow_html=True)
            tips = []
            if satisfaction_level < 0.45:
                tips.append(("Low Satisfaction", "Schedule a one-on-one; explore role fit and grievances."))
            if number_project > 5:
                tips.append(("Overloaded", "Redistribute projects — burnout risk is high."))
            if average_montly_hours > 230:
                tips.append(("Overworked", "High hours detected. Review workload and deadlines."))
            if promotion_last_5years == "No" and time_spend_company >= 4:
                tips.append(("No Recent Promotion", "Consider a growth conversation or recognition award."))
            if salary == "Low":
                tips.append(("Low Salary", "Benchmark compensation against market rates."))
            if last_evaluation > 0.8 and satisfaction_level < 0.5:
                tips.append(("High Performer, Low Morale", "Top talent at risk — prioritise retention package."))

            if tips:
                for t, desc in tips:
                    st.markdown(f'<div class="tip-box"><div class="tip-title">⚡ {t}</div>{desc}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="tip-box"><div class="tip-title">✅ No immediate concerns</div>'
                            'Profile looks healthy. Keep up regular check-ins.</div>', unsafe_allow_html=True)

            # ── Save on button click
            if save_btn:
                if not emp_name and not emp_id:
                    st.warning("Please enter at least a name or ID before saving.")
                else:
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
                    st.success("✅ Saved to history!")


# ══════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)

    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH).sort_values("Timestamp", ascending=False)

        total_pred = len(log_df)
        high_risk  = (log_df["Prediction"] == "High Risk of Leaving").sum()
        low_risk   = total_pred - high_risk

        c1, c2, c3 = st.columns(3)
        for col, val, label, color in [
            (c1, total_pred, "Total Predictions", "#6366f1"),
            (c2, high_risk,  "High Risk",         "#ef4444"),
            (c3, low_risk,   "Likely to Stay",    "#10b981"),
        ]:
            col.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                padding:16px;text-align:center">
                <div style="font-size:1.8rem;font-weight:700;color:{color}">{val}</div>
                <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.08em;margin-top:4px">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        filt = st.selectbox("Filter", ["All", "High Risk of Leaving", "Likely to Stay"])
        if filt != "All":
            log_df = log_df[log_df["Prediction"] == filt]

        def highlight_pred(val):
            if val == "High Risk of Leaving":
                return "background-color:#450a0a;color:#fecaca"
            return "background-color:#052e16;color:#a7f3d0"

        styled = log_df.style.applymap(highlight_pred, subset=["Prediction"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        col_dl, col_cl = st.columns([1, 1])
        with col_dl:
            st.download_button("⬇️ Download CSV",
                               log_df.to_csv(index=False).encode("utf-8"),
                               "predictions_log.csv", "text/csv", use_container_width=True)
        with col_cl:
            if st.button("🗑️ Clear History", use_container_width=True):
                os.remove(LOG_PATH)
                st.success("History cleared. Refresh.")

    else:
        st.markdown("""
        <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;
            padding:48px;text-align:center;color:#475569">
            <div style="font-size:2.5rem;margin-bottom:8px">📋</div>
            No records yet. Run a prediction and click <b>Save to History</b>.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — EXPLORE DATA  (interactive filters)
# ══════════════════════════════════════════════
with tab_explore:
    if df_dataset is None:
        st.warning("Place `hr_employee_churn_data.csv` in the same folder to enable this tab.")
    else:
        # ── Interactive filters ──────────────────
        st.markdown('<div class="section-header">Interactive Filters</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            sal_filter = st.multiselect(
                "💰 Salary Band",
                options=["low", "medium", "high"],
                default=["low", "medium", "high"]
            )
        with fc2:
            proj_range = st.slider("📁 Projects Range", 1, 10, (1, 10))
        with fc3:
            sat_range = st.slider("😊 Satisfaction Range", 0.0, 1.0, (0.0, 1.0), 0.05)

        # Apply filters
        df_f = df_dataset[
            df_dataset["salary"].isin(sal_filter) &
            df_dataset["number_project"].between(*proj_range) &
            df_dataset["satisfaction_level"].between(*sat_range)
        ]

        n_total   = len(df_f)
        n_churned = int(df_f["left"].sum())
        n_kept    = n_total - n_churned
        churn_pct = round(n_churned / n_total * 100, 1) if n_total else 0

        # Filtered stats row
        s1, s2, s3, s4 = st.columns(4)
        for col, val, label, color in [
            (s1, f"{n_total:,}",    "Filtered Employees", "#6366f1"),
            (s2, f"{n_kept:,}",     "Retained",           "#10b981"),
            (s3, f"{n_churned:,}",  "Churned",            "#ef4444"),
            (s4, f"{churn_pct}%",   "Churn Rate",         "#f59e0b"),
        ]:
            col.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                padding:14px;text-align:center;margin-bottom:14px">
                <div style="font-size:1.6rem;font-weight:700;color:{color}">{val}</div>
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.08em;margin-top:3px">{label}</div>
            </div>""", unsafe_allow_html=True)

        if n_total == 0:
            st.warning("No data matches the selected filters.")
        else:
            row1c1, row1c2 = st.columns(2)

            # Chart 1: Churn donut
            with row1c1:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=["Retained", "Churned"],
                    values=[n_kept, n_churned],
                    hole=0.62,
                    marker_colors=["#10b981", "#ef4444"],
                    textinfo="percent+label",
                    textfont={"color": "#f8fafc", "size": 12},
                )])
                fig_donut.update_layout(
                    title={"text": "Churn Distribution (filtered)", "font": {"color": "#94a3b8", "size": 13}},
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend={"font": {"color": "#94a3b8"}},
                    height=290,
                    margin=dict(l=10, r=10, t=40, b=10),
                    annotations=[{"text": f"<b>{n_total:,}</b><br>employees",
                                  "x": 0.5, "y": 0.5,
                                  "font": {"size": 13, "color": "#f8fafc"}, "showarrow": False}]
                )
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

            # Chart 2: Salary vs Churn (filtered)
            with row1c2:
                sal_churn = df_f.groupby(["salary", "left"]).size().reset_index(name="count")
                sal_churn["status"] = sal_churn["left"].map({0: "Retained", 1: "Churned"})
                fig_sal = px.bar(
                    sal_churn, x="salary", y="count", color="status",
                    color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
                    barmode="group",
                    category_orders={"salary": ["low", "medium", "high"]},
                    title="Salary Band vs Churn (filtered)",
                )
                fig_sal.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=290, margin=dict(l=10, r=10, t=40, b=10),
                    title_font={"color": "#94a3b8", "size": 13},
                    legend={"font": {"color": "#94a3b8"}, "title": {"text": ""}},
                    xaxis={"showgrid": False, "tickfont": {"color": "#64748b"}},
                    yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
                )
                st.plotly_chart(fig_sal, use_container_width=True, config={"displayModeBar": False})

            row2c1, row2c2 = st.columns(2)

            # Chart 3: Satisfaction boxplot
            with row2c1:
                fig_sat = go.Figure()
                for val, name, color, fill in [
                    (0, "Retained", "#10b981", "rgba(16,185,129,0.2)"),
                    (1, "Churned",  "#ef4444", "rgba(239,68,68,0.2)"),
                ]:
                    fig_sat.add_trace(go.Box(
                        y=df_f[df_f["left"] == val]["satisfaction_level"],
                        name=name,
                        marker_color=color,
                        line={"color": color},
                        fillcolor=fill,
                    ))
                fig_sat.update_layout(
                    title={"text": "Satisfaction by Status (filtered)", "font": {"color": "#94a3b8", "size": 13}},
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=290, margin=dict(l=10, r=10, t=40, b=10),
                    legend={"font": {"color": "#94a3b8"}},
                    xaxis={"showgrid": False, "tickfont": {"color": "#64748b"}},
                    yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
                )
                st.plotly_chart(fig_sat, use_container_width=True, config={"displayModeBar": False})

            # Chart 4: Hours vs Satisfaction scatter
            with row2c2:
                sample = df_f.sample(min(600, len(df_f)), random_state=42)
                fig_sc = px.scatter(
                    sample,
                    x="average_montly_hours", y="satisfaction_level",
                    color=sample["left"].map({0: "Retained", 1: "Churned"}),
                    color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
                    title="Monthly Hours vs Satisfaction (filtered)",
                    opacity=0.55,
                )
                fig_sc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=290, margin=dict(l=10, r=10, t=40, b=10),
                    title_font={"color": "#94a3b8", "size": 13},
                    legend={"font": {"color": "#94a3b8"}, "title": {"text": ""}},
                    xaxis={"showgrid": False, "tickfont": {"color": "#64748b"}},
                    yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#64748b"}},
                )
                st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

            # Chart 5: Projects vs Churn rate bar
            st.markdown('<div class="section-header" style="margin-top:4px">Churn Rate by Number of Projects</div>', unsafe_allow_html=True)
            proj_churn = (
                df_f.groupby("number_project")["left"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "churn_rate", "count": "employees"})
            )
            proj_churn["churn_pct"] = (proj_churn["churn_rate"] * 100).round(1)
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Bar(
                x=proj_churn["number_project"].astype(str),
                y=proj_churn["churn_pct"],
                marker_color=[
                    "#ef4444" if v > 40 else "#f59e0b" if v > 20 else "#10b981"
                    for v in proj_churn["churn_pct"]
                ],
                text=[f"{v}%" for v in proj_churn["churn_pct"]],
                textposition="outside",
                textfont={"color": "#94a3b8", "size": 11},
                customdata=proj_churn["employees"],
                hovertemplate="Projects: %{x}<br>Churn Rate: %{y}%<br>Employees: %{customdata}<extra></extra>",
            ))
            fig_proj.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=10, r=10, t=10, b=40),
                xaxis={"title": "Number of Projects", "showgrid": False,
                       "tickfont": {"color": "#64748b"}, "title_font": {"color": "#64748b"}},
                yaxis={"title": "Churn Rate (%)", "gridcolor": "#1e293b",
                       "tickfont": {"color": "#64748b"}, "title_font": {"color": "#64748b"}},
                showlegend=False,
            )
            st.plotly_chart(fig_proj, use_container_width=True, config={"displayModeBar": False})

            # Chart 6: Feature importance
            st.markdown('<div class="section-header">Model Feature Importance</div>', unsafe_allow_html=True)
            fi_df = pd.DataFrame({"Feature": FEATURE_LABELS, "Importance": FEATURE_IMPORTANCES}).sort_values("Importance")
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0,"#1e293b"],[0.5,"#6366f1"],[1,"#3b82f6"]],
                    showscale=False,
                ),
                text=[f"{v*100:.1f}%" for v in fi_df["Importance"]],
                textposition="outside",
                textfont={"color": "#94a3b8"},
            ))
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=10, r=55, t=10, b=10),
                xaxis={"showgrid": False, "showticklabels": False},
                yaxis={"tickfont": {"color": "#94a3b8", "size": 11}},
            )
            st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

            # Raw data expander
            with st.expander("🔍 View Filtered Dataset"):
                st.markdown(f"**{len(df_f):,} rows** match current filters")
                st.dataframe(df_f.head(200), use_container_width=True, hide_index=True)
