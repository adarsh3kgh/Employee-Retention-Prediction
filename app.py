import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Retention Predictor", layout="centered")

# ---------- LOAD MODEL ----------
model = joblib.load("employee_retention_model.pkl")

# ---------- CSS ----------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 12px;
}
div.stButton > button {
    display: block;
    margin: 0 auto;
    width: 200px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">Employee Retention Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter employee details</div>', unsafe_allow_html=True)

# ---------- FORM ----------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("prediction_form"):

        emp_name = st.text_input("Employee Name")
        emp_id = st.text_input("Employee ID")

        col1, col2 = st.columns(2)

        with col1:
            satisfaction_level = st.number_input("Satisfaction Level (0–1)", 0.0, 1.0, 0.5)
            number_project = st.number_input("Number of Projects", 1, 10, 3)
            time_spend_company = st.number_input("Years at Company", 1, 10, 3)
            Work_accident = st.selectbox("Work Accident", ["No", "Yes"])

        with col2:
            last_evaluation = st.number_input("Last Evaluation (0–1)", 0.0, 1.0, 0.5)
            average_montly_hours = st.number_input("Monthly Hours", 50, 350, 160)
            promotion_last_5years = st.selectbox("Promotion (Last 5 Years)", ["No", "Yes"])
            salary = st.selectbox("Salary", ["Low", "Medium", "High"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ENCODING ----------
Work_accident = 1 if Work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0
low = 1 if salary == "Low" else 0
medium = 1 if salary == "Medium" else 0

# ---------- PREDICTION ----------
if submitted:

    # -------- VALIDATION --------
    if number_project > 6 and average_montly_hours < 100:
        st.error("Unrealistic input: Too many projects with very low working hours")
        st.stop()

    features = np.array([[satisfaction_level,
                          last_evaluation,
                          number_project,
                          average_montly_hours,
                          time_spend_company,
                          Work_accident,
                          promotion_last_5years,
                          low,
                          medium]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "High Risk of Leaving"
        color = "#7f1d1d"
    else:
        result = "Likely to Stay"
        color = "#064e3b"

    st.markdown("---")

    st.markdown(f"""
    <div style="
        background:{color};
        padding:20px;
        border-radius:12px;
        text-align:center;
        font-size:20px;
        font-weight:600;">
        {result}
    </div>
    """, unsafe_allow_html=True)

    st.caption("Prediction is based on historical data.")

    # ---------- SAVE RESULT ----------
    new_data = pd.DataFrame([{
        "Name": emp_name,
        "Employee_ID": emp_id,
        "Prediction": result,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    file_path = "predictions_log.csv"

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        updated = pd.concat([df_old, new_data], ignore_index=True)
    else:
        updated = new_data

    updated.to_csv(file_path, index=False)

    st.success("Saved successfully")

# ---------- HISTORY ----------
st.markdown("---")
st.subheader("Previous Predictions")

file_path = "predictions_log.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    df = df.sort_values(by="Timestamp", ascending=False)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- CLEAR BUTTON ----------
    if st.button("Clear History"):
        os.remove(file_path)
        st.success("History cleared. Refresh page.")

else:
    st.write("No records yet")