import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- Theme Mode Toggle in Sidebar ---
mode = st.sidebar.radio("Choose Theme Mode", options=["Light", "Dark"])

# CSS for Light Mode
light_css = """
<style>
.stApp {
    background-color: #f0f4f8;
    color: #333333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    text-align: center;
    padding: 1rem 0;
    color: #0d3b66;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.result-card, .med-card {
    background-color: white !important;
    padding: 1.2rem 1.5rem !important;
    border-radius: 15px !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.12) !important;
    margin-bottom: 1.5rem !important;
    color: #333333 !important;
}
.section-title {
    color: #0d3b66;
    font-weight: 600;
    margin-bottom: 1rem;
}
.med-card p {
    margin: 0.25rem 0;
}
.sidebar .sidebar-content {
    background: #dde6f0 !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    color: #000000 !important;
}
</style>
"""

# CSS for Dark Mode
dark_css = """
<style>
.stApp {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    text-align: center;
    padding: 1rem 0;
    color: #70A9A1;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.result-card, .med-card {
    background-color: #1E1E1E !important;
    padding: 1.2rem 1.5rem !important;
    border-radius: 15px !important;
    box-shadow: 0 3px 8px rgba(255,255,255,0.1) !important;
    margin-bottom: 1.5rem !important;
    color: #E0E0E0 !important;
}
.section-title {
    color: #70A9A1;
    font-weight: 600;
    margin-bottom: 1rem;
}
.med-card p {
    margin: 0.25rem 0;
}
.sidebar .sidebar-content {
    background: #22333B !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}
</style>
"""

# Inject relevant CSS based on mode
if mode == "Dark":
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Medicine database (unchanged)
MEDICINES = {
    "Diabetes": [
        {"name": "Metformin 500mg", "class": "Biguanide", "safety": "Low risk", "mechanism": "Decreases glucose production"},
        {"name": "Sitagliptin 100mg", "class": "DPP4 Inhibitor", "safety": "Low risk", "mechanism": "Inhibits DPP4 enzyme"},
        {"name": "Glipizide 5mg", "class": "Sulfonylurea", "safety": "Medium risk", "mechanism": "Stimulates insulin release"},
    ],
    "Hypertension": [
        {"name": "Lisinopril 10mg", "class": "ACE Inhibitor", "safety": "Low risk", "mechanism": "Blocks angiotensin conversion"},
        {"name": "Amlodipine 5mg", "class": "Calcium Channel Blocker", "safety": "Low risk", "mechanism": "Blocks calcium channels"},
        {"name": "Metoprolol 50mg", "class": "Beta Blocker", "safety": "Medium risk", "mechanism": "Blocks beta receptors"},
    ],
    "Hypertension_Diabetes": [
        {"name": "Lisinopril/HCTZ Combo", "class": "ACE Inhibitor + Diuretic", "safety": "Medium risk", "mechanism": "Blood pressure control + diuretic effect"},
        {"name": "Amlodipine/Metformin Combo", "class": "Calcium blocker + Biguanide", "safety": "Low risk", "mechanism": "Combined effect on BP and glucose"},
    ],
    "Tachycardia": [
        {"name": "Propranolol 40mg", "class": "Beta Blocker", "safety": "Medium risk", "mechanism": "Slows heart rate"},
        {"name": "Diltiazem 120mg", "class": "Calcium Channel Blocker", "safety": "Medium risk", "mechanism": "Reduces heart rate and BP"},
    ],
    "Age_Related_Hypertension": [
        {"name": "Aspirin 81mg", "class": "Antiplatelet", "safety": "Low risk", "mechanism": "Prevents blood clots"},
        {"name": "Atorvastatin 20mg", "class": "Statin", "safety": "Low risk", "mechanism": "Lowers cholesterol"},
    ],
    "Healthy": [
        {"name": "Multivitamin Daily", "class": "Multivitamin", "safety": "Low risk", "mechanism": "Provides essential nutrients"},
        {"name": "Vitamin D3 1000IU", "class": "Vitamin", "safety": "Low risk", "mechanism": "Supports bone health"},
        {"name": "Omega-3 Fish Oil", "class": "Supplement", "safety": "Low risk", "mechanism": "Supports heart health"},
    ],
}

def risk_score(bp, glucose, hr, age):
    score = 100
    if bp > 140:
        score -= 30
    elif 120 < bp <= 140:
        score -= 15
    if glucose > 126:
        score -= 30
    elif 100 < glucose <= 126:
        score -= 15
    if hr > 100:
        score -= 20
    if age > 65:
        score -= 10
    return max(score, 0)

def plot_gauge(param, value, max_val, thresholds, title):
    colors = []
    for t in thresholds:
        colors.append(t[1])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range': [None, max_val]},
               'steps': [{'range': [0, thresholds[0][0]], 'color': colors[0]},
                         {'range': [thresholds[0][0], thresholds[1][0]], 'color': colors[1]},
                         {'range': [thresholds[1][0], max_val], 'color': colors[2]}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': thresholds[1][0]}}
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# Page Header
st.markdown('<div class="header">Personalized Healthcare Recommendation System</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("Enter Your Health Parameters")
age = st.sidebar.slider("Age (years)", 18, 90, 45)
bp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 90, 200, 120)
glucose = st.sidebar.slider("Blood Glucose Level (mg/dL)", 70, 300, 100)
hr = st.sidebar.slider("Heart Rate (bpm)", 50, 120, 72)

if st.sidebar.button("Predict Disease"):
    X = pd.DataFrame([[age, bp, glucose, hr]], columns=["age", "blood_pressure", "glucose_level", "heart_rate"])
    prediction = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0])

    st.subheader("Prediction Result")

    color_map = {
        "Healthy": "#008000",
        "Diabetes": "#FFA500",
        "Hypertension": "#FFA500",
        "Hypertension_Diabetes": "#FF0000",
        "Tachycardia": "#FF0000",
        "Age_Related_Hypertension": "#FFA500",
    }

    color = color_map.get(prediction, "#0000FF")

    st.markdown(
        f"<h3 style='color:{color}; font-weight: bold;'>Predicted Condition: {prediction} (Confidence: {proba:.1%})</h3>",
        unsafe_allow_html=True,
    )

    if color == "#008000":
        st.success("Your health parameters are within normal ranges.")
    elif color == "#FFA500":
        st.warning("Moderate risk detected. Consider consulting a healthcare professional.")
    else:
        st.error("High risk detected. Please seek medical advice promptly.")

    st.subheader("Health Parameters")
    col1, col2 = st.columns(2)
    with col1:
        plot_gauge("Blood Pressure", bp, 200, [(120, "#008000"), (140, "#FFD700"), (200, "#FF6347")], "Systolic Blood Pressure")
        plot_gauge("Heart Rate", hr, 120, [(60, "#008000"), (100, "#FFD700"), (120, "#FF6347")], "Heart Rate")
    with col2:
        plot_gauge("Blood Glucose", glucose, 300, [(100, "#008000"), (126, "#FFD700"), (300, "#FF6347")], "Blood Glucose")
        st.metric("Age", age)

    score = risk_score(bp, glucose, hr, age)
    st.subheader("Overall Health Risk Score")
    if score >= 80:
        st.success(f"Excellent! Your health score is {score}/100.")
    elif 60 <= score < 80:
        st.warning(f"Moderate risk: Your health score is {score}/100. Consider lifestyle changes.")
    else:
        st.error(f"High risk: Your health score is {score}/100. Consult a healthcare provider.")

    st.subheader("Medicine Recommendations")
    meds = MEDICINES.get(prediction, MEDICINES["Healthy"])
    for med in meds:
        st.markdown(f"**{med['name']}** ({med['class']}) - {med['safety']}")
        st.write(f"Mechanism: {med['mechanism']}")
        st.markdown("---")

    st.subheader("Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Distribution")
        try:
            df = pd.read_csv("dataset.csv")
            diagnosis_counts = df['diagnosis'].value_counts()

            fig_pie = px.pie(
                values=diagnosis_counts.values,
                names=diagnosis_counts.index,
                title="Distribution of Diagnoses in Dataset"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        except:
            st.info("Dataset not available for analysis")

    with col2:
        st.write("Your Risk Factors")
        risk_factors = []
        risk_levels = []

        if bp > 140:
            risk_factors.append("High Blood Pressure")
            risk_levels.append("High")
        elif bp > 120:
            risk_factors.append("Elevated Blood Pressure")
            risk_levels.append("Medium")

        if glucose > 126:
            risk_factors.append("High Blood Glucose")
            risk_levels.append("High")
        elif glucose > 100:
            risk_factors.append("Elevated Blood Glucose")
            risk_levels.append("Medium")

        if hr > 100:
            risk_factors.append("High Heart Rate")
            risk_levels.append("Medium")

        if age > 65:
            risk_factors.append("Age Factor")
            risk_levels.append("Medium")

        if risk_factors:
            risk_df = pd.DataFrame({
                'Risk Factor': risk_factors,
                'Level': risk_levels
            })

            fig_bar = px.bar(
                risk_df,
                x='Risk Factor',
                y=[1]*len(risk_factors),
                color='Level',
                title="Identified Risk Factors",
                color_discrete_map={'High': '#FF6B6B', 'Medium': '#FFD93D', 'Low': '#6BCF7F'}
            )
            fig_bar.update_layout(showlegend=True, yaxis_title="Risk Present", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("No significant risk factors identified!")

    st.subheader("System Information")
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Model Type", "Logistic Regression")

    with col4:
        try:
            df = pd.read_csv("dataset.csv")
            st.metric("Training Samples", len(df))
        except:
            st.metric("Training Samples", "N/A")

    with col5:
        st.metric("Prediction Confidence", f"{proba:.1%}")

else:
    st.info("Use the sidebar to input your health parameters and click 'Predict Disease'.")
