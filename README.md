# 👥 Employee Retention Predictor

> An AI-powered HR analytics web app that predicts employee churn risk in real time using a trained XGBoost model.

---

## 🌐 Live Demo

**[https://employee-retention-prediction-dntja9qdeibgc42d73vsdz.streamlit.app](https://employee-retention-prediction-dntja9qdeibgc42d73vsdz.streamlit.app)**

---

## 📌 Overview

Employee churn is one of the most costly problems HR teams face. This app helps organizations identify employees at high risk of leaving — before they resign — using a machine learning model trained on 14,999 employee records with **98.6% accuracy**.

Simply adjust the employee's profile using the interactive sliders and dropdowns, and the prediction updates **live** — no submit button needed.

---

## ✨ Features

### 🔮 Predict Tab
- **Live predictions** — results update instantly as inputs change
- **Churn probability gauge** — visual speedometer from 0–100%
- **Risk badge** — clear High Risk / Likely to Stay verdict
- **Key Risk Factors chart** — top 5 contributing features for that specific employee
- **Smart HR Recommendations** — contextual tips based on the employee's profile (e.g. overloaded, stuck without promotion, low satisfaction)
- **Save to History** — log predictions with a timestamp

### 📋 History Tab
- View all saved predictions in a colour-coded table
- Filter by prediction outcome
- Download history as CSV
- Clear history with one click

### 📊 Explore Data Tab
- **Interactive filters** — Salary Band, Projects Range, Satisfaction Range
- All charts respond live to filter changes
- Churn distribution donut chart
- Salary Band vs Churn grouped bar chart
- Satisfaction Level boxplot by churn status
- Monthly Hours vs Satisfaction scatter plot
- Churn Rate by Number of Projects bar chart
- Model Feature Importance horizontal bar chart
- Raw filtered dataset viewer

---

## 🧠 Model

| Property | Detail |
|---|---|
| Algorithm | XGBoost Classifier |
| Training samples | 14,999 employees |
| Accuracy | **98.6%** |
| Precision (churn) | 99% |
| Recall (churn) | 95% |
| F1-Score (churn) | 97% |

### Features Used

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee self-reported satisfaction (0–1) |
| `last_evaluation` | Most recent performance evaluation score (0–1) |
| `number_project` | Number of projects currently assigned |
| `average_montly_hours` | Average working hours per month |
| `time_spend_company` | Years at the company |
| `Work_accident` | Whether the employee had a work accident (Yes/No) |
| `promotion_last_5years` | Whether promoted in the last 5 years (Yes/No) |
| `salary` | Salary band (Low / Medium / High) |

### Top Churn Drivers

```
Years at Company      32.1%  ████████████████████████
Satisfaction Level    27.9%  █████████████████████
Number of Projects    16.4%  █████████████
Last Evaluation        9.5%  ███████
Monthly Hours          4.6%  ████
```

### Churn Risk Profiles

**🚨 High Risk employee typically has:**
- Satisfaction level below 0.45
- 6–7 projects (overloaded) or just 2 (disengaged)
- Monthly hours above 200
- 4–6 years at the company with no promotion
- High evaluation score but low satisfaction (unrecognised top performer)

**✅ Low Risk employee typically has:**
- Satisfaction level above 0.65
- 3–4 projects
- Monthly hours between 150–180
- Received a promotion recently

---

## 📂 Project Structure

```
employee-retention-prediction/
│
├── app.py                          # Main Streamlit application
├── employee_retention_model.pkl    # Trained XGBoost model
├── hr_employee_churn_data.csv      # Training dataset (14,999 records)
├── requirements.txt                # Python dependencies
├── Major_Project_Employee_Retention_Prediction.ipynb  # Training notebook
└── README.md                       # This file
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/employee-retention-prediction.git
cd employee-retention-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
joblib
numpy
pandas
plotly
xgboost
scikit-learn
```

---

## 📊 Dataset

The dataset contains **14,999 employee records** with a churn rate of **23.8%**.

| Column | Type | Description |
|---|---|---|
| `empid` | int | Employee ID |
| `satisfaction_level` | float | Satisfaction score (0–1) |
| `last_evaluation` | float | Last performance score (0–1) |
| `number_project` | int | Projects assigned |
| `average_montly_hours` | int | Avg monthly working hours |
| `time_spend_company` | int | Years at company |
| `Work_accident` | int | Had a work accident (0/1) |
| `promotion_last_5years` | int | Promoted in 5 years (0/1) |
| `salary` | str | Salary band (low/medium/high) |
| `left` | int | Target — left the company (1) or stayed (0) |

---


## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [XGBoost](https://xgboost.readthedocs.io) | Churn prediction model |
| [Plotly](https://plotly.com) | Interactive charts |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [scikit-learn](https://scikit-learn.org) | Model evaluation utilities |
| [Joblib](https://joblib.readthedocs.io) | Model serialisation |

---

## 👤 Author

Built as a major project on Employee Retention Prediction using machine learning and interactive data visualization.

---

## 📄 License

This project is for educational purposes. Feel free to fork, modify, and build on it.
