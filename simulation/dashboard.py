"""
Interactive Streamlit dashboard for smart classroom simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.classroom_sim import run_simulation, ClassroomSimulation
from simulation.ml_integration import predict_environment, load_model

# Page configuration
st.set_page_config(
    page_title="Smart Classroom Simulation",
    page_icon="🏫",
    layout="wide"
)

# Title
st.title("🏫 Smart Classroom Simulation Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Basic parameters
    num_students = st.slider("Number of Students", 10, 50, 30)
    room_size = st.slider("Room Size (m²)", 50, 200, 100)
    simulation_hours = st.slider("Simulation Duration (hours)", 1, 8, 2)
    
    st.markdown("---")
    st.header("Manual Controls")
    
    # Manual environment controls (for testing)
    use_manual = st.checkbox("Use Manual Controls", False)
    
    if use_manual:
        temp = st.slider("Temperature (°C)", 18.0, 32.0, 22.0, 0.1)
        co2 = st.slider("CO₂ (ppm)", 400, 2000, 500)
        humidity = st.slider("Humidity (%)", 20, 80, 50)
        light = st.slider("Light (lux)", 100, 1000, 400)
        noise = st.slider("Noise (dB)", 30, 80, 45)
    
    st.markdown("---")
    
    # Run button
    run_button = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard demonstrates a **predictive simulation framework** 
    for optimizing classroom environments using IoT data and machine learning.
    
    **Features:**
    - Real-time ML predictions
    - Environmental monitoring
    - Automated interventions
    - Performance analytics
    """)

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Students", num_students, delta=None)
with col2:
    st.metric("Room Size", f"{room_size} m²", delta=None)
with col3:
    st.metric("Sim Duration", f"{simulation_hours} hours", delta=None)

# Current environment assessment (if manual mode)
if use_manual:
    st.markdown("---")
    st.header("📊 Current Environment Assessment")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Temperature", f"{temp}°C")
    with col2:
        st.metric("CO₂", f"{co2} ppm")
    with col3:
        st.metric("Humidity", f"{humidity}%")
    with col4:
        st.metric("Light", f"{light} lux")
    with col5:
        st.metric("Noise", f"{noise} dB")
    
    # Get ML prediction
    features = {
        'temperature': temp,
        'co2': co2,
        'humidity': humidity,
        'light': light,
        'noise': noise
    }
    
    prediction, confidence = predict_environment(features)
    
    # Display prediction
    st.subheader("ML Model Prediction")
    if prediction == "conducive":
        st.success(f"✅ **{prediction.upper()}** (Confidence: {confidence:.1%})")
    else:
        st.error(f"⚠️ **{prediction.upper()}** (Confidence: {confidence:.1%})")
    
    # Recommendation
    st.subheader("Recommendations")
    if prediction == "non-conducive":
        if co2 > 800:
            st.warning("• Increase ventilation - CO₂ levels too high")
        if temp > 27:
            st.warning("• Activate cooling - Temperature too high")
        if light < 250:
            st.warning("• Increase lighting - Too dim")
        if noise > 65:
            st.warning("• Address noise sources - Excessive noise")
    else:
        st.info("✓ Environment is optimal for learning")

# Run simulation
if run_button:
    st.markdown("---")
    st.header("📈 Simulation Results")
    
    with st.spinner("Running simulation..."):
        # Run the simulation
        log_data = run_simulation(hours=simulation_hours, num_students=num_students)
    
    # Convert logs to DataFrame
    df = pd.DataFrame(log_data)
    
    if not df.empty:
        # Summary metrics
        st.subheader("Summary Statistics")
        
        conducive_pct = (df['prediction'] == 'conducive').mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
        with col2:
            st.metric("Avg CO₂", f"{df['co2'].mean():.0f} ppm")
        with col3:
            st.metric("Time Conducive", f"{conducive_pct:.1f}%")
        with col4:
            st.metric("Interventions", f"{df[df['prediction']=='non-conducive'].shape[0]}")
        
        # Environmental trends
        st.subheader("Environmental Trends")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Temperature & CO₂", "Light & Noise", "Prediction History"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'],
                                    mode='lines', name='Temperature (°C)',
                                    line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df['time'], y=df['co2']/40,  # Scale for visualization
                                    mode='lines', name='CO₂ (ppm/40)',
                                    line=dict(color='blue', dash='dash')))
            fig.update_layout(title='Temperature and CO₂ Trends',
                            xaxis_title='Time (minutes)',
                            yaxis_title='Value',
                            hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['light'],
                                    mode='lines', name='Light (lux)',
                                    line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=df['time'], y=df['noise'],
                                    mode='lines', name='Noise (dB)',
                                    line=dict(color='orange')))
            fig.update_layout(title='Light and Noise Trends',
                            xaxis_title='Time (minutes)',
                            yaxis_title='Value',
                            hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Convert predictions to numeric for visualization
            df['pred_numeric'] = (df['prediction'] == 'conducive').astype(int)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['pred_numeric'],
                                    mode='markers+lines',
                                    name='Conducive (1) / Non-conducive (0)',
                                    line=dict(shape='hv', color='green')))
            fig.add_trace(go.Scatter(x=df['time'], y=df['confidence'],
                                    mode='lines', name='Confidence',
                                    line=dict(color='gray', dash='dot')))
            fig.update_layout(title='Prediction History',
                            xaxis_title='Time (minutes)',
                            yaxis_title='State',
                            yaxis=dict(tickvals=[0, 1],
                                      ticktext=['Non-Conducive', 'Conducive']))
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Simulation Data (CSV)",
            data=csv,
            file_name=f"classroom_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("**Jonathan Nkrumah** | A Simulation-Based IoT Framework for Optimizing Classroom Environments")