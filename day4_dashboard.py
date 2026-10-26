"""
Day 4 - Streamlit Dashboard
Combines everything: loads the trained DNN, runs a prediction, gets a
Gen AI explanation + severity, and lets the user chat about the alert
or generate a weekly report.

RUN WITH:
    streamlit run day4_dashboard.py
"""

import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

from day2_genai_explain import explain_attack
from chatbot_report import chat_about_alert, generate_weekly_report

st.set_page_config(page_title="IoT Security Dashboard", layout="wide")

# ---------- LOAD SAVED MODEL + PREPROCESSING (from Day 1) ----------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("models/iot_dnn_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, scaler, label_encoder, feature_columns

model, scaler, label_encoder, feature_columns = load_artifacts()

# ---------- SESSION STATE FOR ALERT HISTORY (used by weekly report) ----------
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

st.title("🛡️ IoT Security Dashboard")
st.caption("DNN detection + Generative AI explanation, severity, chat, and reporting")

# ---------- SIDEBAR: SIMULATE / UPLOAD TRAFFIC ----------
st.sidebar.header("Test an alert")
device_name = st.sidebar.text_input("Device name", "Smart Camera")

st.sidebar.write("Enter feature values (or use random test values):")
if st.sidebar.button("Use random sample values"):
    sample_values = np.random.rand(len(feature_columns)) * 100
else:
    sample_values = np.zeros(len(feature_columns))

feature_inputs = []
for i, col in enumerate(feature_columns):
    val = st.sidebar.number_input(col, value=float(sample_values[i]), key=col)
    feature_inputs.append(val)

run_button = st.sidebar.button("Run detection", type="primary")

# ---------- MAIN: RUN DNN + GEN AI ----------
if run_button:
    X = np.array(feature_inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled, verbose=0)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    attack_type = label_encoder.inverse_transform([class_index])[0]

    st.subheader("Stage 2 — DNN Detection Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Device", device_name)
    col2.metric("Attack type", attack_type)
    col3.metric("Confidence", f"{confidence:.1f}%")

    with st.spinner("Asking Generative AI for an explanation..."):
        result = explain_attack(device_name, attack_type, confidence)

    severity = result.get("severity", "Medium")
    severity_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

    st.subheader("Stage 3 — Generative AI Explanation")
    st.markdown(f"**Severity:** {severity_colors.get(severity, '⚪')} {severity}")
    st.write(result.get("explanation", ""))

    st.markdown("**Recommended actions:**")
    for rec in result.get("recommendations", []):
        st.write(f"- {rec}")

    # log this alert for the weekly report
    st.session_state.alert_log.append({
        "device": device_name,
        "attack_type": attack_type,
        "confidence": confidence,
        "severity": severity,
    })

    # ---------- CHATBOT ----------
    st.subheader("Ask about this alert")
    question = st.text_input("Type a question, e.g. 'How dangerous is this?'")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = chat_about_alert(device_name, attack_type, confidence, question)
        st.info(answer)

# ---------- WEEKLY REPORT ----------
st.divider()
st.subheader("Stage 4 — Weekly Report")
st.write(f"Alerts logged this session: {len(st.session_state.alert_log)}")

if st.button("Generate weekly report"):
    if len(st.session_state.alert_log) == 0:
        st.warning("Run at least one detection first.")
    else:
        with st.spinner("Generating report..."):
            path = generate_weekly_report(st.session_state.alert_log)
        with open(path, "rb") as f:
            st.download_button("Download PDF report", f, file_name="weekly_report.pdf")
