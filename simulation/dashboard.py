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

from simulation.classroom_sim import (
    run_simulation,
    ATTENTION_THRESHOLDS,
    COMFORT_THRESHOLDS,
    evaluate_features_zone,
    fuse_model_zone_status,
)
from simulation.ml_integration import predict_environment
from validation.hardware_test import (
    parse_sample_line,
    baseline_label as hil_baseline_label,
    intervention_recommendations as hil_intervention_recommendations,
    normalize_serial_port,
    open_serial as hil_open_serial,
)

# Page configuration
st.set_page_config(
    page_title="Smart Classroom Simulation",
    page_icon="🏫",
    layout="wide"
)

# Title
st.title("🏫 Smart Classroom Simulation Dashboard")
st.markdown("---")

if "hil_records" not in st.session_state:
    st.session_state.hil_records = []
if "simulation_df" not in st.session_state:
    st.session_state.simulation_df = None


def manual_recommendations(features):
    """Return actionable guidance from attention-first three-zone thresholds."""
    tips = []

    temperature = float(features['temperature'])
    co2 = float(features['co2'])
    humidity = float(features['humidity'])
    light = float(features['light'])

    # CO2
    if co2 > COMFORT_THRESHOLDS['co2']['high']:
        tips.append(("error", f"Urgent: Increase ventilation immediately (CO₂ {co2:.0f}ppm > {COMFORT_THRESHOLDS['co2']['high']:.0f}ppm comfort limit)."))
    elif co2 > ATTENTION_THRESHOLDS['co2']['high']:
        tips.append(("warning", f"Increase ventilation to return to attention target (CO₂ {co2:.0f}ppm > {ATTENTION_THRESHOLDS['co2']['high']:.0f}ppm)."))

    # Temperature
    t_att_low = ATTENTION_THRESHOLDS['temperature']['low']
    t_att_high = ATTENTION_THRESHOLDS['temperature']['high']
    t_comf_low = COMFORT_THRESHOLDS['temperature']['low']
    t_comf_high = COMFORT_THRESHOLDS['temperature']['high']
    if temperature > t_comf_high:
        tips.append(("error", f"Activate strong cooling (Temp {temperature:.1f}°C > {t_comf_high:.1f}°C comfort limit)."))
    elif temperature < t_comf_low:
        tips.append(("error", f"Activate strong heating (Temp {temperature:.1f}°C < {t_comf_low:.1f}°C comfort limit)."))
    elif temperature > t_att_high:
        tips.append(("warning", f"Slight cooling recommended (Temp {temperature:.1f}°C above attention range {t_att_low:.1f}–{t_att_high:.1f}°C)."))
    elif temperature < t_att_low:
        tips.append(("warning", f"Slight heating recommended (Temp {temperature:.1f}°C below attention range {t_att_low:.1f}–{t_att_high:.1f}°C)."))

    # Humidity
    h_att_low = ATTENTION_THRESHOLDS['humidity']['low']
    h_att_high = ATTENTION_THRESHOLDS['humidity']['high']
    h_comf_low = COMFORT_THRESHOLDS['humidity']['low']
    h_comf_high = COMFORT_THRESHOLDS['humidity']['high']
    if humidity > h_comf_high:
        tips.append(("error", f"Enable dehumidification (Humidity {humidity:.1f}% > {h_comf_high:.1f}% comfort limit)."))
    elif humidity < h_comf_low:
        tips.append(("error", f"Enable humidification (Humidity {humidity:.1f}% < {h_comf_low:.1f}% comfort limit)."))
    elif humidity > h_att_high:
        tips.append(("warning", f"Minor dehumidification recommended (Humidity {humidity:.1f}% above attention range {h_att_low:.1f}–{h_att_high:.1f}%)."))
    elif humidity < h_att_low:
        tips.append(("warning", f"Minor humidification recommended (Humidity {humidity:.1f}% below attention range {h_att_low:.1f}–{h_att_high:.1f}%)."))

    # Light
    l_att_low = ATTENTION_THRESHOLDS['light']['low']
    l_att_high = ATTENTION_THRESHOLDS['light']['high']
    l_comf_low = COMFORT_THRESHOLDS['light']['low']
    l_comf_high = COMFORT_THRESHOLDS['light']['high']
    if light < l_comf_low:
        tips.append(("error", f"Increase lighting strongly (Light {light:.0f} lux < {l_comf_low:.0f} lux comfort limit)."))
    elif light > l_comf_high:
        tips.append(("warning", f"Reduce glare/brightness (Light {light:.0f} lux > {l_comf_high:.0f} lux comfort high bound)."))
    elif light < l_att_low:
        tips.append(("warning", f"Increase lighting to attention target (Light {light:.0f} lux < {l_att_low:.0f} lux)."))
    elif light > l_att_high:
        tips.append(("warning", f"Reduce lighting to attention target (Light {light:.0f} lux > {l_att_high:.0f} lux)."))

    return tips


def read_hil_batch(port, baud, timeout, max_samples, room_size, start_hour):
    """Read a batch of live testbed samples and evaluate them with the shared model."""
    port = normalize_serial_port(port)
    serial_conn = hil_open_serial(port, baud, timeout=timeout)
    records = []

    try:
        while len(records) < max_samples:
            raw = serial_conn.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            sample = parse_sample_line(raw)
            if sample is None:
                records.append({
                    "raw": raw,
                    "status": "skipped",
                    "reason": "Malformed input",
                })
                continue

            features = sample.as_features()
            prediction, confidence = predict_environment(
                features,
                context={
                    "room_size": room_size,
                    "start_hour": start_hour,
                    "current_minute": len(records),
                    "datetime": sample.timestamp,
                },
            )
            zone_state = evaluate_features_zone(features)
            fused = fuse_model_zone_status(prediction, zone_state, confidence)
            baseline = hil_baseline_label(features)
            recs = hil_intervention_recommendations(features)

            records.append(
                {
                    "timestamp": sample.timestamp,
                    "temperature": sample.temperature,
                    "humidity": sample.humidity,
                    "co2": sample.co2,
                    "light": sample.light,
                    "model_prediction": prediction,
                    "confidence": float(confidence),
                    "overall_zone": zone_state["overall_zone"],
                    "final_status": fused["final_status"],
                    "disagreement": bool(fused["disagreement"]),
                    "rationale": fused["rationale"],
                    "baseline_prediction": baseline,
                    "recommendations": recs,
                    "raw": raw,
                    "status": "ok",
                }
            )
    finally:
        serial_conn.close()

    return records

# Sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Basic parameters
    num_students = st.slider("Number of Students", 10, 50, 30)
    room_size = st.slider("Room Size (m²)", 50, 200, 100)
    simulation_hours = st.slider("Simulation Duration (hours)", 1, 8, 2)
    start_hour = st.slider("Start Hour", 0, 23, 9)
    random_seed = st.number_input("Random Seed (optional)", min_value=0, max_value=999999, value=42)
    
    st.markdown("---")
    st.header("Manual Controls")
    
    # Manual environment controls (for testing single prediction)
    use_manual = st.checkbox("Use Manual Controls (single prediction)", False)

    if use_manual:
        st.markdown("---")
        st.header("Manual Prediction Inputs")
        temp = st.slider("Temperature (°C)", 18.0, 32.0, 22.0, 0.1)
        co2 = st.slider("CO₂ (ppm)", 400, 2000, 500)
        humidity = st.slider("Humidity (%)", 20, 80, 50)
        light = st.slider("Light (lux)", 100, 1000, 400)

    st.markdown("---")
    st.header("Simulation Initial Conditions")
    use_custom_initials = st.checkbox("Set custom initial conditions", True)

    initial_conditions = None
    if use_custom_initials:
        init_temp = st.slider("Initial Temperature (°C)", 16.0, 35.0, 22.0, 0.1)
        init_co2 = st.slider("Initial CO₂ (ppm)", 380, 2200, 450)
        init_humidity = st.slider("Initial Humidity (%)", 15, 90, 50)
        init_light = st.slider("Initial Light (lux)", 50, 1200, 450)

        initial_conditions = {
            'temperature': init_temp,
            'co2': init_co2,
            'humidity': init_humidity,
            'light': init_light,
        }

    with st.expander("Advanced Dynamics (thesis mode)", expanded=False):
        co2_prod = st.slider("CO₂ production per student/min", 0.001, 0.03, 0.008, 0.001)
        co2_decay = st.slider("CO₂ decay rate", 0.001, 0.08, 0.02, 0.001)
        monitor_interval = st.slider("Model check interval (minutes)", 1, 30, 10)

    sim_config = {
        'co2_production_per_student': co2_prod,
        'co2_decay_rate': co2_decay,
        'monitor_interval_minutes': monitor_interval,
    }
    
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

main_tab1, main_tab2, main_tab3 = st.tabs(["Simulation", "HIL / Live Testbed", "Comparison"])

with main_tab1:
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Temperature", f"{temp}°C")
        with col2:
            st.metric("CO₂", f"{co2} ppm")
        with col3:
            st.metric("Humidity", f"{humidity}%")
        with col4:
            st.metric("Light", f"{light} lux")
        
        # Get ML prediction
        features = {
            'temperature': temp,
            'co2': co2,
            'humidity': humidity,
            'light': light,
            'occupancy': num_students,
            'occupancy_count': num_students,
        }
        
        prediction, confidence = predict_environment(
            features,
            context={
                'room_size': room_size,
                'start_hour': datetime.now().hour,
                'current_minute': 0,
                'datetime': datetime.now(),
            }
        )
        
        # Unified status (model + evidence-based zone)
        zone_state = evaluate_features_zone(features)
        fused = fuse_model_zone_status(prediction, zone_state, confidence)
        final_status = fused['final_status']

        st.subheader("Decision Status")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Model", prediction)
        with c2:
            st.metric("Zone", zone_state['overall_zone'])
        with c3:
            st.metric("Final", final_status)

        if final_status == "conducive":
            st.success(f"✅ Final: **{final_status.upper()}** (confidence: {confidence:.1%})")
        elif final_status == "acceptable":
            st.warning(f"⚠️ Final: **{final_status.upper()}** (confidence: {confidence:.1%})")
        else:
            st.error(f"⛔ Final: **{final_status.upper()}** (confidence: {confidence:.1%})")

        st.caption(f"Decision rationale: {fused['rationale']}")
        if fused.get('disagreement'):
            st.info("Model and threshold-zone policy disagree for this input.")
        
        # Recommendation
        st.subheader("Recommendations")
        recs = manual_recommendations(features)
        if recs:
            for level, message in recs:
                if level == "error":
                    st.error(f"• {message}")
                elif level == "warning":
                    st.warning(f"• {message}")
                else:
                    st.info(f"• {message}")
        else:
            st.success("✓ All factors are within evidence-based attention ranges.")

    # Run simulation
    if run_button:
        st.markdown("---")
        st.header("📈 Simulation Results")
        
        with st.spinner("Running simulation..."):
            # Run the simulation
            log_data = run_simulation(
                hours=simulation_hours,
                num_students=num_students,
                room_size=room_size,
                start_hour=start_hour,
                initial_conditions=initial_conditions,
                sim_config=sim_config,
                random_seed=int(random_seed),
            )
        
        # Convert logs to DataFrame
        df = pd.DataFrame(log_data)
        st.session_state.simulation_df = df
        
        if not df.empty:
            # Summary metrics
            st.subheader("Summary Statistics")
            
            final_col = 'final_status' if 'final_status' in df.columns else 'prediction'
            conducive_pct = (df[final_col] == 'conducive').mean() * 100
            total_interventions = int(df['intervention_count'].sum()) if 'intervention_count' in df.columns else 0
            disagreement_count = int(df['model_zone_disagreement'].sum()) if 'model_zone_disagreement' in df.columns else 0
            optimal_pct = (df['overall_zone'] == 'optimal').mean() * 100 if 'overall_zone' in df.columns else 0
            acceptable_pct = (df['overall_zone'] == 'acceptable').mean() * 100 if 'overall_zone' in df.columns else 0
            non_conducive_zone_pct = (df['overall_zone'] == 'non-conducive').mean() * 100 if 'overall_zone' in df.columns else 0
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
            with col2:
                st.metric("Avg CO₂", f"{df['co2'].mean():.0f} ppm")
            with col3:
                st.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
            with col4:
                st.metric("Avg Light", f"{df['light'].mean():.0f} lux")
            with col5:
                st.metric("Time Conducive", f"{conducive_pct:.1f}%")
            with col6:
                st.metric("Interventions", f"{total_interventions}")

            z1, z2, z3, z4 = st.columns(4)
            with z1:
                st.metric("Optimal Zone", f"{optimal_pct:.1f}%")
            with z2:
                st.metric("Acceptable Zone", f"{acceptable_pct:.1f}%")
            with z3:
                st.metric("Non-Conducive Zone", f"{non_conducive_zone_pct:.1f}%")
            with z4:
                st.metric("Model-Zone Disagreements", disagreement_count)
            
            # Environmental trends
            st.subheader("Environmental Trends")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Temperature, CO₂ & Humidity", "Light", "Prediction History"])
        
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'],
                                        mode='lines', name='Temperature (°C)',
                                        line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['co2']/40,  # Scale for visualization
                                        mode='lines', name='CO₂ (ppm/40)',
                                        line=dict(color='blue', dash='dash')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['humidity'],
                                        mode='lines', name='Humidity (%)',
                                        line=dict(color='cyan', dash='dot')))
                fig.update_layout(title='Temperature, CO₂, and Humidity Trends',
                                xaxis_title='Time (minutes)',
                                yaxis_title='Value',
                                hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['time'], y=df['light'],
                                        mode='lines', name='Light (lux)',
                                        line=dict(color='yellow')))
                fig.update_layout(title='Light Trend',
                                xaxis_title='Time (minutes)',
                                yaxis_title='Value',
                                hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Convert final status to numeric for visualization
                status_for_plot = final_col
                df['pred_numeric'] = (df[status_for_plot] == 'conducive').astype(int)
                point_colors = np.where(df[status_for_plot] == 'conducive', '#00cc66', '#ff4d4f')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['time'], y=df['pred_numeric'],
                                        mode='markers+lines',
                                        name='Final status: Conducive (1) / Other (0)',
                                        line=dict(shape='hv', color='green'),
                                        marker=dict(color=point_colors, size=9),
                                        text=df[status_for_plot].astype(str),
                                        customdata=df['confidence'].astype(float),
                                        hovertemplate=(
                                            'Time: %{x} min'
                                            '<br>Status: %{text}'
                                            '<br>Confidence: %{customdata:.1%}'
                                            '<extra></extra>'
                                        )))
                fig.add_trace(go.Scatter(x=df['time'], y=df['confidence'],
                                        mode='lines', name='Confidence',
                                        line=dict(color='gray', dash='dot')))
                fig.update_layout(title='Prediction History',
                                xaxis_title='Time (minutes)',
                                yaxis_title='State',
                                yaxis=dict(tickvals=[0, 1],
                                        ticktext=['Non-Conducive', 'Conducive']))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Zone Causes & Interventions")
            if 'non_conducive_factors' in df.columns and 'interventions_triggered' in df.columns and 'overall_zone' in df.columns:
                non_optimal_df = df[df['overall_zone'] != 'optimal'].copy()
                if not non_optimal_df.empty:
                    non_optimal_df['causing_factors'] = (
                        non_optimal_df['non_conducive_factors'].replace('', pd.NA)
                        .fillna(non_optimal_df.get('acceptable_factors', '').replace('', pd.NA))
                        .fillna('Model-specific pattern')
                    )
                    non_optimal_df['triggered_interventions'] = non_optimal_df['interventions_triggered'].replace('', 'No direct actuator action')
                    display_cols = [
                        'time',
                        'model_prediction',
                        'final_status',
                        'model_zone_disagreement',
                        'overall_zone',
                        'temperature',
                        'co2',
                        'humidity',
                        'light',
                        'confidence',
                        'causing_factors',
                        'zone_trigger_reason',
                        'triggered_interventions',
                    ]
                    st.dataframe(non_optimal_df[display_cols], use_container_width=True)
                else:
                    st.info("No non-optimal zone points in this run.")
            else:
                st.info("Cause/intervention tracking columns not available for this run.")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Data (CSV)",
                data=csv,
                file_name=f"classroom_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

with main_tab2:
    st.header("🔌 Live HIL / Testbed")
    st.caption("Connect your serial testbed here to stream live sensor values and evaluate them with the same model used in simulation.")
    st.info("Windows Arduino default supported: COM7 at 9700 baud. If you typed COMP7 by mistake, it will be normalized to COM7.")

    c1, c2, c3 = st.columns(3)
    with c1:
        hil_port = st.text_input("Serial Port", value="COM7")
    with c2:
        hil_baud = st.number_input("Baud Rate", min_value=9600, max_value=921600, value=9700, step=100)
    with c3:
        hil_timeout = st.number_input("Timeout (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    c4, c5, c6 = st.columns(3)
    with c4:
        hil_max_samples = st.number_input("Samples to Read", min_value=1, max_value=500, value=20, step=1)
    with c5:
        hil_room_size = st.number_input("Room Size Context", min_value=50, max_value=300, value=room_size, step=1)
    with c6:
        hil_start_hour = st.number_input("Start Hour Context", min_value=0, max_value=23, value=start_hour, step=1)

    if st.button("▶️ Read Live Testbed Batch", type="primary"):
        with st.spinner("Reading live testbed data..."):
            try:
                hil_records = read_hil_batch(
                    port=hil_port,
                    baud=int(hil_baud),
                    timeout=float(hil_timeout),
                    max_samples=int(hil_max_samples),
                    room_size=int(hil_room_size),
                    start_hour=int(hil_start_hour),
                )
                st.session_state.hil_records = hil_records
                st.success(f"Read {len(hil_records)} live records from the testbed.")
            except Exception as exc:
                st.error(f"Could not read from the testbed: {exc}")

    hil_records = st.session_state.hil_records
    if hil_records:
        hil_df = pd.DataFrame(hil_records)
        hil_df_ok = hil_df[hil_df['status'] == 'ok'].copy() if 'status' in hil_df.columns else hil_df.copy()

        st.subheader("HIL Summary")
        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            st.metric("Records", len(hil_df_ok))
        with hc2:
            st.metric("Model Conducive", f"{(hil_df_ok['model_prediction'] == 'conducive').mean() * 100:.1f}%")
        with hc3:
            st.metric("Final Conducive", f"{(hil_df_ok['final_status'] == 'conducive').mean() * 100:.1f}%")
        with hc4:
            st.metric("Disagreements", int(hil_df_ok['disagreement'].sum()))

        st.subheader("Live HIL Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hil_df_ok.index, y=(hil_df_ok['final_status'] == 'conducive').astype(int),
                                 mode='markers+lines', name='Final Status',
                                 line=dict(color='green', shape='hv'),
                                 marker=dict(color=np.where(hil_df_ok['final_status'] == 'conducive', '#00cc66', '#ff4d4f')),
                                 customdata=hil_df_ok['confidence'],
                                 text=hil_df_ok['final_status'],
                                 hovertemplate='Sample: %{x}<br>Status: %{text}<br>Confidence: %{customdata:.1%}<extra></extra>'))
        fig.add_trace(go.Scatter(x=hil_df_ok.index, y=hil_df_ok['confidence'], mode='lines', name='Confidence', line=dict(color='gray', dash='dot')))
        fig.update_layout(title='HIL Final Status and Confidence', xaxis_title='Sample', yaxis_title='State', yaxis=dict(tickvals=[0, 1], ticktext=['Non-Conducive', 'Conducive']))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Live HIL Records")
        st.dataframe(
            hil_df_ok[[c for c in ['timestamp', 'temperature', 'humidity', 'co2', 'light', 'model_prediction', 'confidence', 'overall_zone', 'final_status', 'disagreement', 'rationale', 'baseline_prediction'] if c in hil_df_ok.columns]],
            use_container_width=True,
        )

        st.download_button(
            label="📥 Download HIL Data (CSV)",
            data=hil_df_ok.to_csv(index=False),
            file_name=f"hil_testbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

with main_tab3:
    st.header("📊 Simulation vs HIL Comparison")
    sim_df = st.session_state.simulation_df
    hil_df = pd.DataFrame(st.session_state.hil_records) if st.session_state.hil_records else None

    if sim_df is None or sim_df.empty:
        st.info("Run a simulation first to populate the comparison view.")
    elif hil_df is None or hil_df.empty:
        st.info("Capture a live HIL batch first to populate the comparison view.")
    else:
        sim_final_col = 'final_status' if 'final_status' in sim_df.columns else 'prediction'
        sim_summary = {
            'source': 'Simulation',
            'avg_temp': sim_df['temperature'].mean(),
            'avg_humidity': sim_df['humidity'].mean(),
            'avg_co2': sim_df['co2'].mean(),
            'avg_light': sim_df['light'].mean(),
            'conducive_pct': (sim_df[sim_final_col] == 'conducive').mean() * 100,
        }

        hil_ok = hil_df[hil_df['status'] == 'ok'].copy() if 'status' in hil_df.columns else hil_df.copy()
        hil_summary = {
            'source': 'HIL',
            'avg_temp': hil_ok['temperature'].mean(),
            'avg_humidity': hil_ok['humidity'].mean(),
            'avg_co2': hil_ok['co2'].mean(),
            'avg_light': hil_ok['light'].mean(),
            'conducive_pct': (hil_ok['final_status'] == 'conducive').mean() * 100,
        }

        comp_df = pd.DataFrame([sim_summary, hil_summary])
        st.dataframe(comp_df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avg Temp', x=comp_df['source'], y=comp_df['avg_temp']))
        fig.add_trace(go.Bar(name='Avg Humidity', x=comp_df['source'], y=comp_df['avg_humidity']))
        fig.add_trace(go.Bar(name='Avg CO₂', x=comp_df['source'], y=comp_df['avg_co2']))
        fig.add_trace(go.Bar(name='Avg Light', x=comp_df['source'], y=comp_df['avg_light']))
        fig.update_layout(barmode='group', title='Simulation vs HIL Environmental Averages', xaxis_title='Source', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Conducive %', x=comp_df['source'], y=comp_df['conducive_pct']))
        fig2.update_layout(title='Conducive Percentage Comparison', xaxis_title='Source', yaxis_title='Conducive %')
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Jonathan Nkrumah** | A Simulation-Based IoT Framework for Optimizing Classroom Environments")