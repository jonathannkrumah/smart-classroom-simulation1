# Streamlit dashboard

import streamlit as st
import pandas as pd
# import plotly.graph_objects as go
from classroom_sim import run_simulation
from ml_integration import predict_environment

st.set_page_config(page_title="Smart Classroom Simulator", layout="wide")

st.title("🏫 Smart Classroom Simulation Dashboard")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
num_students = st.sidebar.slider("Number of Students", 10, 50, 30)
room_size = st.sidebar.slider("Room Size (m²)", 50, 200, 100)
simulation_hours = st.sidebar.slider("Simulation Duration (hours)", 1, 8, 2)

# Manual environment controls
st.sidebar.header("Manual Environmental Inputs")
temp = st.sidebar.slider("Temperature (°C)", 18, 32, 22)
co2 = st.sidebar.slider("CO₂ (ppm)", 400, 2000, 500)
light = st.sidebar.slider("Light (lux)", 100, 1000, 400)

# Run simulation button
if st.sidebar.button("▶️ Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # This would run your full simulation
        # For demo, we'll show a mockup
        st.success(f"Simulation completed for {simulation_hours} hours!")
        
        # Show results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Focus Score", "87%", "+5%")
        with col2:
            st.metric("Avg CO₂ Level", "680 ppm", "-120 ppm")
        with col3:
            st.metric("Interventions Triggered", "3", "-2 from baseline")

# Real-time prediction display
st.header("📊 Real-Time Environment Assessment")
features = {
    'temperature': temp,
    'co2': co2,
    'humidity': 50,
    'light': light,
    'noise': 50
}

prediction, confidence = predict_environment(features)
status_color = "🟢" if prediction == "conducive" else "🔴"

st.subheader(f"Current Status: {status_color} {prediction.upper()}")
st.progress(confidence, text=f"Confidence: {confidence:.1%}")

# Visualization
st.header("📈 Environmental Trends")
# Add Plotly charts here for simulated data

# Download simulation results
st.download_button(
    label="📥 Download Simulation Report",
    data=pd.DataFrame([features]).to_csv(),
    file_name="simulation_report.csv"
)

st.info("This dashboard demonstrates the predictive simulation framework. Adjust parameters to see how classroom conditions affect predicted learning focus.")