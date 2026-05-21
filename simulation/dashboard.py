""" Interactive Streamlit dashboard for smart classroom simulation """
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime
from pathlib import Path
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_DIR = Path(__file__).resolve().parents[1]

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

ACADEMIC_COLORS = {
    "background": "#f7f4ee",
    "paper": "#ffffff",
    "text": "#1f2937",
    "grid": "#d6d0c4",
    "spine": "#9ca3af",
    "title": "#111827",
    "temp": "#8b1e3f",
    "co2": "#1f5aa6",
    "humidity": "#2a6f97",
    "light": "#b7791f",
    "confidence": "#4b5563",
    "green": "#2e7d32",
    "red": "#b23a48",
    "amber": "#b45309",
}

pio.templates["academic_classroom"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=ACADEMIC_COLORS["paper"],
        plot_bgcolor=ACADEMIC_COLORS["background"],
        font=dict(family="Georgia, Times New Roman, serif", color=ACADEMIC_COLORS["text"], size=20),
        title=dict(font=dict(family="Georgia, Times New Roman, serif", color=ACADEMIC_COLORS["title"], size=34)),
        colorway=[ACADEMIC_COLORS["temp"], ACADEMIC_COLORS["co2"], ACADEMIC_COLORS["humidity"], ACADEMIC_COLORS["light"], ACADEMIC_COLORS["green"], ACADEMIC_COLORS["confidence"]],
        xaxis=dict(
            showgrid=True,
            gridcolor=ACADEMIC_COLORS["grid"],
            zeroline=False,
            linecolor=ACADEMIC_COLORS["spine"],
            mirror=True,
            ticks="outside",
            tickfont=dict(size=18),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=ACADEMIC_COLORS["grid"],
            zeroline=False,
            linecolor=ACADEMIC_COLORS["spine"],
            mirror=True,
            ticks="outside",
            tickfont=dict(size=18),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=ACADEMIC_COLORS["grid"],
            borderwidth=1,
            font=dict(size=18),
        ),
        margin=dict(l=80, r=40, t=100, b=80),
    )
)
pio.templates.default = "academic_classroom"


PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "smart_classroom_chart",
        "height": 900,
        "width": 1600,
        "scale": 4,
    },
}


def styled_plotly_chart(fig):
    fig.update_layout(
        template="academic_classroom",
        width=None,
        height=580,
        font=dict(family="Georgia, Times New Roman, serif", color=ACADEMIC_COLORS["text"], size=20),
        title=dict(x=0.02, xanchor="left", font=dict(size=34)),
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=18)),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=ACADEMIC_COLORS["grid"],
            font=dict(color=ACADEMIC_COLORS["text"], size=18),
        ),
    )
    fig.update_traces(
        selector=dict(type="scatter", mode="lines"),
        line=dict(width=4),
    )
    fig.update_traces(
        selector=dict(type="bar"),
        marker=dict(line=dict(color=ACADEMIC_COLORS["paper"], width=1.5)),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

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


def normalize_serial_port(port):
    port = str(port).strip()
    return port.upper() if port else "COM6"


def apply_refined_calibration(features):
    """Apply threshold-aligned calibration to HIL sensor features."""
    calibrated = dict(features)
    calibrated['temperature'] = float(features['temperature']) - 1.0
    calibrated['humidity'] = float(features['humidity']) * 0.6
    calibrated['co2'] = float(features['co2']) - 120.0
    calibrated_light = float(features['light']) * 0.8
    calibrated['light'] = float(np.clip(calibrated_light, 300.0, 650.0))
    return calibrated


def calibrate_hil_dataframe(feed_df: pd.DataFrame) -> pd.DataFrame:
    """Apply refined calibration to bridge feed rows when core columns are present."""
    if feed_df is None or feed_df.empty:
        return feed_df

    calibrated_df = feed_df.copy()

    if 'temperature' in calibrated_df.columns:
        calibrated_df['temperature_raw'] = pd.to_numeric(calibrated_df['temperature'], errors='coerce')
        calibrated_df['temperature'] = calibrated_df['temperature_raw'] - 1.0

    if 'humidity' in calibrated_df.columns:
        calibrated_df['humidity_raw'] = pd.to_numeric(calibrated_df['humidity'], errors='coerce')
        calibrated_df['humidity'] = calibrated_df['humidity_raw'] * 0.6

    if 'co2' in calibrated_df.columns:
        calibrated_df['co2_raw_ppm'] = pd.to_numeric(calibrated_df['co2'], errors='coerce')
        calibrated_df['co2'] = calibrated_df['co2_raw_ppm'] - 120.0

    if 'light' in calibrated_df.columns:
        calibrated_df['light_raw'] = pd.to_numeric(calibrated_df['light'], errors='coerce')
        light_cal = calibrated_df['light_raw'] * 0.8
        calibrated_df['light'] = light_cal.clip(lower=300.0, upper=650.0)

    return calibrated_df


def hil_baseline_label(features):
    zone_state = evaluate_features_zone(features)
    overall = zone_state.get("overall_zone", "optimal")
    if overall == "optimal":
        return "conducive"
    if overall == "acceptable":
        return "acceptable"
    return "non-conducive"


def hil_intervention_recommendations(features):
    tips = manual_recommendations(features)
    return "; ".join(msg for _, msg in tips)

def read_hil_batch(port, baud, timeout, max_samples):
    """Read a batch of live testbed samples and apply refined calibration."""
    port = normalize_serial_port(port)
    serial_conn = hil_open_serial(port, baud, timeout=timeout)
    records = []
    empty_reads = 0
    max_empty_reads = max(10, int(max_samples) * 5)
    
    try:
        while len(records) < max_samples:
            raw = serial_conn.readline().decode("utf-8", errors="ignore").strip()

            # For request/response firmware, request a frame only if passive read is empty.
            if not raw:
                serial_conn.write((datetime.now().isoformat() + "\n").encode("utf-8"))
                raw = serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    empty_reads += 1
                    if empty_reads >= max_empty_reads:
                        records.append({
                            "raw": "",
                            "status": "skipped",
                            "reason": "No response from device",
                        })
                        break
                    continue
            empty_reads = 0
            sample = parse_sample_line(raw)
            if sample is None:
                records.append({
                    "raw": raw,
                    "status": "skipped",
                    "reason": "Malformed input",
                })
                continue
            raw_features = sample.as_features()
            features = apply_refined_calibration(raw_features)
            prediction, confidence = predict_environment(features)
            zone_state = evaluate_features_zone(features)
            fused = fuse_model_zone_status(prediction, zone_state, confidence)
            baseline = hil_baseline_label(features)
            recs = hil_intervention_recommendations(features)
            records.append(
                {
                    "timestamp": sample.timestamp,
                    "temperature_raw": sample.temperature,
                    "temperature": features['temperature'],
                    "humidity_raw": sample.humidity,
                    "humidity": features['humidity'],
                    "co2": features['co2'],
                    "co2_raw_ppm": sample.co2,
                    "co2_used_ppm": features['co2'],
                    "light_raw": sample.light,
                    "light": features['light'],
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

def _safe_series_mean(df, column_name):
    if column_name not in df.columns:
        return None
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())

def _safe_series_max(df, column_name):
    if column_name not in df.columns:
        return None
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.max())


def normalize_simulation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    normalized = df.copy()

    rename_map = {}
    if "temp" in normalized.columns and "temperature" not in normalized.columns:
        rename_map["temp"] = "temperature"
    if "zone" in normalized.columns and "overall_zone" not in normalized.columns:
        rename_map["zone"] = "overall_zone"
    if "agreement" in normalized.columns and "agreement_score" not in normalized.columns:
        rename_map["agreement"] = "agreement_score"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "model_prediction" not in normalized.columns and "prediction" in normalized.columns:
        normalized["model_prediction"] = normalized["prediction"]

    if "model_zone_disagreement" not in normalized.columns and "disagreement" in normalized.columns:
        normalized["model_zone_disagreement"] = normalized["disagreement"].astype(int)

    if "intervention_count" not in normalized.columns and "interventions" in normalized.columns:
        def _count(v):
            if isinstance(v, (list, tuple, set)):
                return len(v)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0
            text = str(v).strip()
            if not text or text in {"[]", "None", "nan"}:
                return 0
            if ";" in text:
                return len([part for part in text.split(";") if part.strip()])
            return 1

        normalized["intervention_count"] = normalized["interventions"].apply(_count)

    if "interventions_triggered" not in normalized.columns and "interventions" in normalized.columns:
        def _join(v):
            if isinstance(v, (list, tuple, set)):
                return "; ".join(str(item) for item in v)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            return str(v)

        normalized["interventions_triggered"] = normalized["interventions"].apply(_join)

    if "confidence" not in normalized.columns:
        normalized["confidence"] = np.nan

    return normalized

def load_bridge_feed_records(feed_path: str, max_rows: int = 300, apply_calibration: bool = True):
    if not feed_path or not os.path.exists(feed_path):
        return []
    try:
        feed_df = pd.read_csv(feed_path)
    except Exception:
        return []
    if feed_df.empty:
        return []
    if max_rows > 0:
        feed_df = feed_df.tail(int(max_rows)).copy()
    if apply_calibration:
        feed_df = calibrate_hil_dataframe(feed_df)
    if "status" not in feed_df.columns:
        feed_df["status"] = "ok"
    records = feed_df.to_dict("records")
    return records

def _hil_column_map(use_calibrated: bool):
    return {
        'temperature': 'temperature' if use_calibrated else 'temperature_raw',
        'humidity': 'humidity' if use_calibrated else 'humidity_raw',
        'co2': 'co2' if use_calibrated else 'co2_raw_ppm',
        'light': 'light' if use_calibrated else 'light_raw',
    }


def _hil_series(df: pd.DataFrame, preferred: str, fallback: str):
    if preferred in df.columns:
        return pd.to_numeric(df[preferred], errors='coerce')
    if fallback in df.columns:
        return pd.to_numeric(df[fallback], errors='coerce')
    return pd.Series([np.nan] * len(df), index=df.index)


def render_hil_results(hil_records, use_calibrated: bool = True):
    if not hil_records:
        st.info("No HIL records available yet.")
        return
    hil_df = pd.DataFrame(hil_records)
    hil_df_ok = hil_df[hil_df['status'] == 'ok'].copy() if 'status' in hil_df.columns else hil_df.copy()
    if hil_df_ok.empty:
        st.warning("No valid HIL samples found in the selected source.")
        return
    hil_plot_df = hil_df_ok.copy()
    if 'timestamp' in hil_plot_df.columns:
        hil_plot_df['timestamp'] = pd.to_datetime(hil_plot_df['timestamp'], errors='coerce')
        hil_plot_df = hil_plot_df.sort_values('timestamp')
    if hil_plot_df.empty:
        hil_plot_df = hil_df_ok.copy()
    hil_plot_df = hil_plot_df.reset_index(drop=True)
    hil_x = hil_plot_df['timestamp'] if 'timestamp' in hil_plot_df.columns and hil_plot_df['timestamp'].notna().any() else hil_plot_df.index
    hil_cols = _hil_column_map(use_calibrated)
    temp_series = _hil_series(hil_plot_df, hil_cols['temperature'], 'temperature')
    humidity_series = _hil_series(hil_plot_df, hil_cols['humidity'], 'humidity')
    co2_series = _hil_series(hil_plot_df, hil_cols['co2'], 'co2')
    light_series = _hil_series(hil_plot_df, hil_cols['light'], 'light')
    value_label = 'Calibrated' if use_calibrated else 'Raw'
    
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
    
    hil_agreement_rate = (hil_df_ok['model_prediction'] == hil_df_ok['baseline_prediction']).mean() * 100 if 'baseline_prediction' in hil_df_ok.columns else None
    hil_recommendations = int(hil_df_ok['recommendations'].fillna('').astype(str).ne('').sum()) if 'recommendations' in hil_df_ok.columns else 0
    hc5, hc6, hc7, hc8 = st.columns(4)
    with hc5:
        st.metric("Baseline Agreement", f"{hil_agreement_rate:.1f}%" if hil_agreement_rate is not None else "N/A")
    with hc6:
        st.metric("Recommended Actuations", hil_recommendations)
    with hc7:
        st.metric("Avg Confidence", f"{hil_df_ok['confidence'].mean():.1%}" if 'confidence' in hil_df_ok.columns else "N/A")
    with hc8:
        st.metric("Final Non-Conducive", f"{(hil_df_ok['final_status'] != 'conducive').mean() * 100:.1f}%")
    
    st.subheader("HIL Trends")
    hil_tab1, hil_tab2, hil_tab3, hil_tab4 = st.tabs(["Temperature & Humidity", "CO₂", "Light", "Prediction History"])
    
    with hil_tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hil_x, y=temp_series, mode='lines', name='Temperature (°C)', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=hil_x, y=humidity_series, mode='lines', name='Humidity (%)', line=dict(color='cyan', dash='dot')))
        fig.update_layout(title=f'HIL Temperature and Humidity Trends ({value_label})', xaxis_title='Time' if 'timestamp' in hil_plot_df.columns else 'Sample', yaxis_title='Value', hovermode='x unified')
        styled_plotly_chart(fig)

    with hil_tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hil_x, y=co2_series, mode='lines', name='CO₂ (ppm)', line=dict(color='blue')))
        fig.update_layout(title=f'HIL CO₂ Trend ({value_label})', xaxis_title='Time' if 'timestamp' in hil_plot_df.columns else 'Sample', yaxis_title='CO₂ (ppm)', hovermode='x unified')
        styled_plotly_chart(fig)
    
    with hil_tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hil_x, y=light_series, mode='lines', name='Light (lux)', line=dict(color='orange')))
        fig.update_layout(title=f'HIL Light Trend ({value_label})', xaxis_title='Time' if 'timestamp' in hil_plot_df.columns else 'Sample', yaxis_title='Value', hovermode='x unified')
        styled_plotly_chart(fig)
    
    with hil_tab4:
        hil_plot_df = hil_plot_df.copy()
        hil_plot_df['pred_numeric'] = (hil_plot_df['final_status'] == 'conducive').astype(int)
        point_colors = np.where(hil_plot_df['final_status'] == 'conducive', '#00cc66', '#ff4d4f')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hil_x, y=hil_plot_df['pred_numeric'],
            mode='markers+lines',
            name='Final status: Conducive (1) / Other (0)',
            line=dict(shape='hv', color='green'),
            marker=dict(color=point_colors, size=9),
            text=hil_plot_df['final_status'].astype(str),
            customdata=hil_plot_df['confidence'].astype(float),
            hovertemplate=(
                'Sample: %{x}' '<br>Status: %{text}' '<br>Confidence: %{customdata:.1%}' '<extra></extra>'
            )
        ))
        fig.add_trace(go.Scatter(x=hil_x, y=hil_plot_df['confidence'], mode='lines', name='Confidence', line=dict(color='gray', dash='dot')))
        fig.update_layout(title='HIL Prediction History', xaxis_title='Time' if 'timestamp' in hil_plot_df.columns else 'Sample', yaxis_title='State', yaxis=dict(tickvals=[0, 1], ticktext=['Non-Conducive', 'Conducive']))
        styled_plotly_chart(fig)
    
    st.subheader("Live HIL Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hil_x, y=(hil_plot_df['final_status'] == 'conducive').astype(int),
        mode='markers+lines',
        name='Final Status',
        line=dict(color='green', shape='hv'),
        marker=dict(color=np.where(hil_plot_df['final_status'] == 'conducive', '#00cc66', '#ff4d4f')),
        customdata=hil_plot_df['confidence'],
        text=hil_plot_df['final_status'],
        hovertemplate='Sample: %{x}<br>Status: %{text}<br>Confidence: %{customdata:.1%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(x=hil_x, y=hil_plot_df['confidence'], mode='lines', name='Confidence', line=dict(color='gray', dash='dot')))
    fig.update_layout(title='HIL Final Status and Confidence', xaxis_title='Time' if 'timestamp' in hil_plot_df.columns else 'Sample', yaxis_title='State', yaxis=dict(tickvals=[0, 1], ticktext=['Non-Conducive', 'Conducive']))
    styled_plotly_chart(fig)
    
    st.subheader("Live HIL Records")
    display_cols = ['timestamp', hil_cols['temperature'], hil_cols['humidity'], hil_cols['co2'], hil_cols['light'], 'model_prediction', 'confidence', 'overall_zone', 'final_status', 'disagreement', 'rationale', 'baseline_prediction', 'recommendations']
    label_map = {
        hil_cols['temperature']: f"temperature_{'cal' if use_calibrated else 'raw'}",
        hil_cols['humidity']: f"humidity_{'cal' if use_calibrated else 'raw'}",
        hil_cols['co2']: f"co2_{'cal' if use_calibrated else 'raw'}",
        hil_cols['light']: f"light_{'cal' if use_calibrated else 'raw'}",
    }
    table_df = hil_df_ok[[c for c in display_cols if c in hil_df_ok.columns]].rename(columns=label_map)
    st.dataframe(
        table_df,
        use_container_width=True,
    )
    st.download_button(
        label="📥 Download HIL Data (CSV)",
        data=hil_df_ok.to_csv(index=False),
        file_name=f"hil_testbed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

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
    This dashboard demonstrates a **predictive simulation framework** for optimizing classroom environments using IoT data and machine learning.
    
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
            )
            
            # Convert logs to DataFrame
            df = normalize_simulation_dataframe(pd.DataFrame(log_data))
            st.session_state.simulation_df = df
            
            if not df.empty:
                # Summary metrics
                st.subheader("Summary Statistics")
                final_col = 'final_status' if 'final_status' in df.columns else 'prediction'
                conducive_pct = (df[final_col] == 'conducive').mean() * 100
                total_interventions = int(df['intervention_count'].sum()) if 'intervention_count' in df.columns else 0
                total_actuations = int(_safe_series_max(df, 'total_actuations') or 0)
                disagreement_count = int(df['model_zone_disagreement'].sum()) if 'model_zone_disagreement' in df.columns else 0
                optimal_pct = (df['overall_zone'] == 'optimal').mean() * 100 if 'overall_zone' in df.columns else 0
                acceptable_pct = (df['overall_zone'] == 'acceptable').mean() * 100 if 'overall_zone' in df.columns else 0
                non_conducive_zone_pct = (df['overall_zone'] == 'non-conducive').mean() * 100 if 'overall_zone' in df.columns else 0
                avg_agreement_score = _safe_series_mean(df, 'agreement_score')
                max_attention_drift = _safe_series_max(df, 'attention_drift_streak')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
                with col2:
                    st.metric("Avg CO₂", f"{df['co2'].mean():.0f} ppm")
                with col3:
                    st.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
                with col4:
                    st.metric("Avg Light", f"{df['light'].mean():.0f} lux")
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Time Conducive", f"{conducive_pct:.1f}%")
                with m2:
                    st.metric("Interventions", f"{total_interventions}")
                with m3:
                    st.metric("Total Actuations", f"{total_actuations}")
                with m4:
                    st.metric("Model-Zone Disagreements", disagreement_count)
                
                z1, z2, z3, z4 = st.columns(4)
                with z1:
                    st.metric("Optimal Zone", f"{optimal_pct:.1f}%")
                with z2:
                    st.metric("Acceptable Zone", f"{acceptable_pct:.1f}%")
                with z3:
                    st.metric("Non-Conducive Zone", f"{non_conducive_zone_pct:.1f}%")
                with z4:
                    st.metric("Avg Agreement Score", f"{avg_agreement_score:.2f}" if avg_agreement_score is not None else "N/A")
                
                z5, z6, z7, z8 = st.columns(4)
                with z5:
                    st.metric("Max Drift Streak", f"{int(max_attention_drift)}" if max_attention_drift is not None else "N/A")
                with z6:
                    st.metric("Model Agreement", f"{avg_agreement_score:.2f}" if avg_agreement_score is not None else "N/A")
                with z7:
                    st.metric("Safety Overrides", int(df['overall_zone'].eq('non-conducive').sum()) if 'overall_zone' in df.columns else 0)
                with z8:
                    st.metric("Triggered Rows", int((df['intervention_count'] > 0).sum()) if 'intervention_count' in df.columns else 0)
                
                # Environmental trends
                st.subheader("Environmental Trends")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Temperature & Humidity", "CO₂", "Light", "Prediction History"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'], mode='lines', name='Temperature (°C)', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=df['time'], y=df['humidity'], mode='lines', name='Humidity (%)', line=dict(color='cyan', dash='dot')))
                    fig.update_layout(title='Temperature and Humidity Trends', xaxis_title='Time (minutes)', yaxis_title='Value', hovermode='x unified')
                    styled_plotly_chart(fig)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['time'], y=df['co2'], mode='lines', name='CO₂ (ppm)', line=dict(color='blue')))
                    fig.update_layout(title='CO₂ Trend', xaxis_title='Time (minutes)', yaxis_title='CO₂ (ppm)', hovermode='x unified')
                    styled_plotly_chart(fig)
                
                with tab3:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['time'], y=df['light'], mode='lines', name='Light (lux)', line=dict(color='yellow')))
                    fig.update_layout(title='Light Trend', xaxis_title='Time (minutes)', yaxis_title='Value', hovermode='x unified')
                    styled_plotly_chart(fig)
                
                with tab4:
                    # Convert final status to numeric for visualization
                    status_for_plot = final_col
                    df['pred_numeric'] = (df[status_for_plot] == 'conducive').astype(int)
                    point_colors = np.where(df[status_for_plot] == 'conducive', '#00cc66', '#ff4d4f')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['time'], y=df['pred_numeric'],
                        mode='markers+lines',
                        name='Final status: Conducive (1) / Other (0)',
                        line=dict(shape='hv', color='green'),
                        marker=dict(color=point_colors, size=9),
                        text=df[status_for_plot].astype(str),
                        customdata=df['confidence'].astype(float),
                        hovertemplate=(
                            'Time: %{x} min' '<br>Status: %{text}' '<br>Confidence: %{customdata:.1%}' '<extra></extra>'
                        )
                    ))
                    fig.add_trace(go.Scatter(x=df['time'], y=df['confidence'], mode='lines', name='Confidence', line=dict(color='gray', dash='dot')))
                    fig.update_layout(title='Prediction History', xaxis_title='Time (minutes)', yaxis_title='State', yaxis=dict(tickvals=[0, 1], ticktext=['Non-Conducive', 'Conducive']))
                    styled_plotly_chart(fig)
                
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
                            'time', 'model_prediction', 'final_status', 'model_zone_disagreement',
                            'agreement_score', 'attention_drift_streak', 'overall_zone',
                            'temperature', 'co2', 'humidity', 'light', 'confidence',
                            'total_actuations', 'causing_factors', 'zone_trigger_reason', 'triggered_interventions',
                        ]
                        available_display_cols = [col for col in display_cols if col in non_optimal_df.columns]
                        st.dataframe(non_optimal_df[available_display_cols], use_container_width=True)
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
    
    if (not run_button) and st.session_state.simulation_df is not None and not st.session_state.simulation_df.empty:
        st.markdown("---")
        st.header("📈 Simulation Results")
        st.caption("Showing latest simulation run stored in session. Click Run Simulation to refresh.")
        df = normalize_simulation_dataframe(st.session_state.simulation_df.copy())
        
        # Summary metrics
        st.subheader("Summary Statistics")
        final_col = 'final_status' if 'final_status' in df.columns else 'prediction'
        conducive_pct = (df[final_col] == 'conducive').mean() * 100
        total_interventions = int(df['intervention_count'].sum()) if 'intervention_count' in df.columns else 0
        total_actuations = int(_safe_series_max(df, 'total_actuations') or 0)
        disagreement_count = int(df['model_zone_disagreement'].sum()) if 'model_zone_disagreement' in df.columns else 0
        optimal_pct = (df['overall_zone'] == 'optimal').mean() * 100 if 'overall_zone' in df.columns else 0
        acceptable_pct = (df['overall_zone'] == 'acceptable').mean() * 100 if 'overall_zone' in df.columns else 0
        non_conducive_zone_pct = (df['overall_zone'] == 'non-conducive').mean() * 100 if 'overall_zone' in df.columns else 0
        avg_agreement_score = _safe_series_mean(df, 'agreement_score')
        max_attention_drift = _safe_series_max(df, 'attention_drift_streak')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
        with col2:
            st.metric("Avg CO₂", f"{df['co2'].mean():.0f} ppm")
        with col3:
            st.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
        with col4:
            st.metric("Avg Light", f"{df['light'].mean():.0f} lux")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Time Conducive", f"{conducive_pct:.1f}%")
        with m2:
            st.metric("Interventions", f"{total_interventions}")
        with m3:
            st.metric("Total Actuations", f"{total_actuations}")
        with m4:
            st.metric("Model-Zone Disagreements", disagreement_count)
        
        z1, z2, z3, z4 = st.columns(4)
        with z1:
            st.metric("Optimal Zone", f"{optimal_pct:.1f}%")
        with z2:
            st.metric("Acceptable Zone", f"{acceptable_pct:.1f}%")
        with z3:
            st.metric("Non-Conducive Zone", f"{non_conducive_zone_pct:.1f}%")
        with z4:
            st.metric("Avg Agreement Score", f"{avg_agreement_score:.2f}" if avg_agreement_score is not None else "N/A")
        
        z5, z6, z7, z8 = st.columns(4)
        with z5:
            st.metric("Max Drift Streak", f"{int(max_attention_drift)}" if max_attention_drift is not None else "N/A")
        with z6:
            st.metric("Model Agreement", f"{avg_agreement_score:.2f}" if avg_agreement_score is not None else "N/A")
        with z7:
            st.metric("Safety Overrides", int(df['overall_zone'].eq('non-conducive').sum()) if 'overall_zone' in df.columns else 0)
        with z8:
            st.metric("Triggered Rows", int((df['intervention_count'] > 0).sum()) if 'intervention_count' in df.columns else 0)
        
        st.subheader("Environmental Trends")
        tab1, tab2, tab3, tab4 = st.tabs(["Temperature & Humidity", "CO₂", "Light", "Prediction History"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['temperature'], mode='lines', name='Temperature (°C)', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df['time'], y=df['humidity'], mode='lines', name='Humidity (%)', line=dict(color='cyan', dash='dot')))
            fig.update_layout(title='Temperature and Humidity Trends', xaxis_title='Time (minutes)', yaxis_title='Value', hovermode='x unified')
            styled_plotly_chart(fig)

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['co2'], mode='lines', name='CO₂ (ppm)', line=dict(color='blue')))
            fig.update_layout(title='CO₂ Trend', xaxis_title='Time (minutes)', yaxis_title='CO₂ (ppm)', hovermode='x unified')
            styled_plotly_chart(fig)
        
        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['time'], y=df['light'], mode='lines', name='Light (lux)', line=dict(color='yellow')))
            fig.update_layout(title='Light Trend', xaxis_title='Time (minutes)', yaxis_title='Value', hovermode='x unified')
            styled_plotly_chart(fig)
        
        with tab4:
            status_for_plot = final_col
            plot_df = df.copy()
            plot_df['pred_numeric'] = (plot_df[status_for_plot] == 'conducive').astype(int)
            point_colors = np.where(plot_df[status_for_plot] == 'conducive', '#00cc66', '#ff4d4f')
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df['time'], y=plot_df['pred_numeric'],
                mode='markers+lines',
                name='Final status: Conducive (1) / Other (0)',
                line=dict(shape='hv', color='green'),
                marker=dict(color=point_colors, size=9),
                text=plot_df[status_for_plot].astype(str),
                customdata=plot_df['confidence'].astype(float),
                hovertemplate=(
                    'Time: %{x} min' '<br>Status: %{text}' '<br>Confidence: %{customdata:.1%}' '<extra></extra>'
                )
            ))
            fig.add_trace(go.Scatter(x=plot_df['time'], y=plot_df['confidence'], mode='lines', name='Confidence', line=dict(color='gray', dash='dot')))
            fig.update_layout(title='Prediction History', xaxis_title='Time (minutes)', yaxis_title='State', yaxis=dict(tickvals=[0, 1], ticktext=['Non-Conducive', 'Conducive']))
            styled_plotly_chart(fig)
        
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
                    'time', 'model_prediction', 'final_status', 'model_zone_disagreement',
                    'agreement_score', 'attention_drift_streak', 'overall_zone',
                    'temperature', 'co2', 'humidity', 'light', 'confidence',
                    'total_actuations', 'causing_factors', 'zone_trigger_reason', 'triggered_interventions',
                ]
                available_display_cols = [col for col in display_cols if col in non_optimal_df.columns]
                st.dataframe(non_optimal_df[available_display_cols], use_container_width=True)
            else:
                st.info("No non-optimal zone points in this run.")
        else:
            st.info("Cause/intervention tracking columns not available for this run.")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Simulation Data (CSV)",
            data=csv,
            file_name=f"classroom_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with main_tab2:
    st.header("🔌 Live HIL / Testbed")
    st.caption("Connect your serial testbed here to stream live sensor values. HIL applies refined calibration before prediction and control logic.")
    st.info("Windows Arduino default supported: COM6 at 9600 baud. If you typed COMP6 by mistake, it will be normalized to COM6.")
    st.success("Calibration active: Temperature -1.0°C, Humidity ×0.6, CO₂ -120 ppm, Light ×0.8 clipped to 300-650 lux.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        hil_port = st.text_input("Serial Port", value="COM6")
    with c2:
        hil_baud = st.number_input("Baud Rate", min_value=9600, max_value=921600, value=9600, step=100)
    with c3:
        hil_timeout = st.number_input("Timeout (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
    c4 = st.columns(1)[0]
    with c4:
        hil_max_samples = st.number_input("Samples to Read", min_value=1, max_value=500, value=20, step=1)
    hil_use_calibrated_view = st.checkbox("Display calibrated sensor values", value=True, help="Switch HIL trends/table between calibrated and raw sensor values.")
    
    if st.button("▶️ Read Live Testbed Batch", type="primary"):
        with st.spinner("Reading live testbed data..."):
            try:
                hil_records = read_hil_batch(
                    port=hil_port,
                    baud=int(hil_baud),
                    timeout=float(hil_timeout),
                    max_samples=int(hil_max_samples),
                )
                st.session_state.hil_records = hil_records
                st.session_state.hil_source = "direct"
                st.success(f"Read {len(hil_records)} live records from the testbed.")
            except Exception as exc:
                message = str(exc)
                if "Access is denied" in message or "PermissionError" in message:
                    st.error(
                        "Could not read from the testbed: serial port is busy. "
                        "If bridge_v2 is running, COM6 is already open there. "
                        "Use 'Bridge Forward Feed' mode in this dashboard, or stop the bridge before direct serial read."
                    )
                else:
                    st.error(f"Could not read from the testbed: {exc}")
    
    if st.session_state.get("hil_source") == "direct" and st.session_state.hil_records:
        st.markdown("---")
        st.subheader("Direct Read Results")
        render_hil_results(st.session_state.hil_records, use_calibrated=hil_use_calibrated_view)
    
    st.markdown("---")
    st.subheader("Bridge Forward Feed (Optional)")
    st.caption("Use this when running validation/testbed_simulation_bridge_v2.py with --forward-live.")
    b1, b2 = st.columns(2)
    with b1:
        bridge_feed_path = st.text_input("Bridge Feed CSV", value=str(ROOT_DIR / "validation" / "live_bridge_feed.csv"))
    with b2:
        bridge_max_rows = st.number_input("Max Rows to Display", min_value=50, max_value=5000, value=300, step=50)
    b3, b4, b5 = st.columns(3)
    with b3:
        use_bridge_feed = st.checkbox("Use Bridge Feed as Live Source", value=False)
    with b4:
        bridge_auto_refresh = st.checkbox("Auto Refresh Feed", value=False)
    with b5:
        bridge_refresh_seconds = st.number_input("Refresh Every (sec)", min_value=1, max_value=30, value=2, step=1)

    if use_bridge_feed:
        st.warning("Bridge feed is not guaranteed to be raw-only. For calibrated raw MQ135 values, use Direct Read mode above.")
    
    if st.button("🔄 Load Bridge Feed") or use_bridge_feed:
        bridge_records = load_bridge_feed_records(bridge_feed_path, int(bridge_max_rows), apply_calibration=True)
        if bridge_records:
            st.session_state.hil_records = bridge_records
            st.session_state.hil_source = "feed"
            st.success(f"Loaded {len(bridge_records)} records from bridge feed.")
        else:
            st.warning("No bridge feed records found yet. Make sure the bridge script is running and forwarding CSV rows.")
    
    if use_bridge_feed and bridge_auto_refresh:
        st.caption(f"Auto-refresh active: every {bridge_refresh_seconds}s")
        time.sleep(int(bridge_refresh_seconds))
        st.experimental_rerun()
    
    if st.session_state.get("hil_source") == "feed" and st.session_state.hil_records:
        st.markdown("---")
        st.subheader("Bridge Feed Results")
        render_hil_results(st.session_state.hil_records, use_calibrated=hil_use_calibrated_view)

with main_tab3:
    st.header("📊 Simulation vs HIL Comparison")
    sim_df = st.session_state.simulation_df
    hil_df = pd.DataFrame(st.session_state.hil_records) if st.session_state.hil_records else None
    st.caption("Note: only one process can use COM6 at a time. If bridge_v2 is running, use Bridge Forward Feed below (do not click direct serial read).")
    comparison_mode = st.radio("HIL comparison basis", ["Calibrated", "Raw"], horizontal=True, index=0)
    use_calibrated_compare = comparison_mode == "Calibrated"
    compare_cols = _hil_column_map(use_calibrated_compare)
    
    if sim_df is None or sim_df.empty:
        st.info("Run a simulation first to populate the comparison view.")
    elif hil_df is None or hil_df.empty:
        st.info("Capture a live HIL batch first to populate the comparison view.")
    else:
        sim_final_col = 'final_status' if 'final_status' in sim_df.columns else 'prediction'
        sim_agreement_score = _safe_series_mean(sim_df, 'agreement_score')
        sim_total_actuations = int(_safe_series_max(sim_df, 'total_actuations') or 0)
        sim_disagreements = int(sim_df['model_zone_disagreement'].sum()) if 'model_zone_disagreement' in sim_df.columns else 0
        
        sim_summary = {
            'source': 'Simulation',
            'avg_temp': sim_df['temperature'].mean(),
            'avg_humidity': sim_df['humidity'].mean(),
            'avg_co2': sim_df['co2'].mean(),
            'avg_light': sim_df['light'].mean(),
            'conducive_pct': (sim_df[sim_final_col] == 'conducive').mean() * 100,
            'agreement_score': sim_agreement_score,
            'total_actuations': sim_total_actuations,
            'disagreements': sim_disagreements,
        }
        
        hil_ok = hil_df[hil_df['status'] == 'ok'].copy() if 'status' in hil_df.columns else hil_df.copy()
        hil_agreement_rate = (hil_ok['model_prediction'] == hil_ok['baseline_prediction']).mean() * 100 if 'baseline_prediction' in hil_ok.columns else None
        hil_recommendations = int(hil_ok['recommendations'].fillna('').astype(str).ne('').sum()) if 'recommendations' in hil_ok.columns else 0
        hil_disagreements = int(hil_ok['disagreement'].sum()) if 'disagreement' in hil_ok.columns else 0
        
        hil_temp = _hil_series(hil_ok, compare_cols['temperature'], 'temperature')
        hil_humidity = _hil_series(hil_ok, compare_cols['humidity'], 'humidity')
        hil_co2 = _hil_series(hil_ok, compare_cols['co2'], 'co2')
        hil_light = _hil_series(hil_ok, compare_cols['light'], 'light')

        hil_summary = {
            'source': 'HIL',
            'avg_temp': float(hil_temp.mean()) if not hil_temp.dropna().empty else np.nan,
            'avg_humidity': float(hil_humidity.mean()) if not hil_humidity.dropna().empty else np.nan,
            'avg_co2': float(hil_co2.mean()) if not hil_co2.dropna().empty else np.nan,
            'avg_light': float(hil_light.mean()) if not hil_light.dropna().empty else np.nan,
            'conducive_pct': (hil_ok['final_status'] == 'conducive').mean() * 100,
            'agreement_score': hil_agreement_rate,
            'total_actuations': hil_recommendations,
            'disagreements': hil_disagreements,
        }
        
        comp_df = pd.DataFrame([sim_summary, hil_summary])
        st.dataframe(comp_df, use_container_width=True)
        
        st.subheader(f"Validation Summary ({comparison_mode} HIL values)")
        temp_gap = abs(float(sim_summary['avg_temp']) - float(hil_summary['avg_temp']))
        humidity_gap = abs(float(sim_summary['avg_humidity']) - float(hil_summary['avg_humidity']))
        co2_gap = abs(float(sim_summary['avg_co2']) - float(hil_summary['avg_co2']))
        light_gap = abs(float(sim_summary['avg_light']) - float(hil_summary['avg_light']))
        conducive_gap = abs(float(sim_summary['conducive_pct']) - float(hil_summary['conducive_pct']))
        
        checks = [
            ("Temperature gap <= 2.0°C", temp_gap <= 2.0),
            ("Humidity gap <= 10%", humidity_gap <= 10.0),
            ("CO₂ gap <= 400 ppm", co2_gap <= 400.0),
            ("Light gap <= 150 lux", light_gap <= 150.0),
            ("Conducive rate gap <= 20%", conducive_gap <= 20.0),
        ]
        pass_count = sum(1 for _, passed in checks if passed)
        
        v1, v2, v3, v4, v5 = st.columns(5)
        with v1:
            st.metric("Temp Gap", f"{temp_gap:.2f}°C")
        with v2:
            st.metric("Humidity Gap", f"{humidity_gap:.2f}%")
        with v3:
            st.metric("CO₂ Gap", f"{co2_gap:.1f} ppm")
        with v4:
            st.metric("Light Gap", f"{light_gap:.1f} lux")
        with v5:
            st.metric("Conducive Gap", f"{conducive_gap:.1f}%")
        
        st.caption(f"Validation checks passed: {pass_count}/{len(checks)}")
        for label, passed in checks:
            if passed:
                st.success(f"PASS: {label}")
            else:
                st.warning(f"REVIEW: {label}")
        
        if pass_count == len(checks):
            st.success("Overall validation status: aligned within configured tolerance bands.")
        elif pass_count >= 3:
            st.info("Overall validation status: partially aligned. Calibrate sensors and review thresholds for closer matching.")
        else:
            st.error("Overall validation status: weak alignment. Recheck calibration, feature scaling, and test conditions.")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Simulation Agreement", f"{sim_agreement_score:.2f}" if sim_agreement_score is not None else "N/A")
        with m2:
            st.metric("Simulation Actuations", sim_total_actuations)
        with m3:
            st.metric("HIL Baseline Agreement", f"{hil_agreement_rate:.1f}%" if hil_agreement_rate is not None else "N/A")
        with m4:
            st.metric("HIL Recommended Actuations", hil_recommendations)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avg Temp', x=comp_df['source'], y=comp_df['avg_temp']))
        fig.add_trace(go.Bar(name='Avg Humidity', x=comp_df['source'], y=comp_df['avg_humidity']))
        fig.add_trace(go.Bar(name='Avg CO₂', x=comp_df['source'], y=comp_df['avg_co2']))
        fig.add_trace(go.Bar(name='Avg Light', x=comp_df['source'], y=comp_df['avg_light']))
        fig.update_layout(barmode='group', title='Simulation vs HIL Environmental Averages', xaxis_title='Source', yaxis_title='Value')
        styled_plotly_chart(fig)
        
        fig_agreement = go.Figure()
        fig_agreement.add_trace(go.Bar(name='Agreement / Baseline Match', x=comp_df['source'], y=comp_df['agreement_score']))
        fig_agreement.add_trace(go.Bar(name='Actuation Count', x=comp_df['source'], y=comp_df['total_actuations']))
        fig_agreement.add_trace(go.Bar(name='Disagreements', x=comp_df['source'], y=comp_df['disagreements']))
        fig_agreement.update_layout(
            barmode='group',
            title='Simulation vs HIL Decision Metrics',
            xaxis_title='Source',
            yaxis_title='Value',
        )
        styled_plotly_chart(fig_agreement)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Conducive %', x=comp_df['source'], y=comp_df['conducive_pct'], marker=dict(color=ACADEMIC_COLORS['green'])))
        fig2.update_layout(title='Conducive Percentage Comparison', xaxis_title='Source', yaxis_title='Conducive %')
        styled_plotly_chart(fig2)

# Footer
st.markdown("---")
st.markdown("**Jonathan Nkrumah** | A Simulation-Based IoT Framework for Optimizing Classroom Environments")