"""
Streamlit Dashboard for Predictive Maintenance Monitoring (Standalone)
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (No Changes) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-critical { color: #d62728; font-weight: bold; }
    .risk-high { color: #ff7f0e; font-weight: bold; }
    .risk-medium { color: #ffbb00; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Model & Data Loading ---

@st.cache_data
def load_sensor_data():
    """Load sensor data from CSV"""
    try:
        df = pd.read_csv('data/sensor_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the `main.py` script first to generate data.")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature columns."""
    model_dir = Path("models")
    try:
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(model_dir / "feature_cols.json", "r") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error("Model files not found in the 'models/' directory. Please run `main.py` first.")
        return None, None, None

# --- Helper Functions for Prediction ---

def create_features_from_reading(reading_dict, feature_cols):
    """
    Create a feature vector from a dictionary of sensor readings.
    NOTE: This uses the same simplified logic as the original API. For a real-world
    application, this should be replaced with a proper feature engineering pipeline
    that uses historical data.
    """
    base_features = [
        reading_dict['temperature'], reading_dict['vibration'], reading_dict['voltage'],
        reading_dict['pressure'], reading_dict['rpm']
    ]
    feature_vector = []
    for feat in base_features:
        feature_vector.extend([feat, feat, feat * 0.1, feat, feat])
    
    feature_vector.extend([0, 0, 0]) # Rate of change approximated as 0
    feature_vector.append(reading_dict['temperature'] * reading_dict['vibration'])
    feature_vector.append(reading_dict['voltage'] * reading_dict['rpm'])
    
    while len(feature_vector) < len(feature_cols):
        feature_vector.append(0)
        
    return feature_vector[:len(feature_cols)]

def predict_failure(input_data, model, scaler, feature_cols):
    """
    Predict failure probability from a dictionary of sensor data.
    """
    # Create and scale the feature vector
    features = create_features_from_reading(input_data, feature_cols)
    features_scaled = scaler.transform([features])
    
    # Make prediction
    probability = model.predict_proba(features_scaled)[0][1]
    
    # Determine risk level and recommendation
    if probability >= 0.7:
        risk_level, recommendation = "CRITICAL", "Immediate maintenance required. Stop machine operation."
    elif probability >= 0.4:
        risk_level, recommendation = "HIGH", "Schedule maintenance within 24 hours."
    elif probability >= 0.2:
        risk_level, recommendation = "MEDIUM", "Monitor closely. Schedule preventive maintenance."
    else:
        risk_level, recommendation = "LOW", "Normal operation. Continue routine monitoring."
        
    return {
        'failure_probability': probability,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#ffbb00', 'LOW': '#2ca02c'}
    return colors.get(risk_level, '#gray')

# --- Main Dashboard ---
def main():
    # Header
    st.markdown('<p class="main-header">⚙️ Predictive Maintenance Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Page", ["Overview", "Real-Time Monitoring", "Machine Analysis"])
        st.markdown("---")
        st.subheader("System Status")
        st.info("This is a standalone dashboard. Model is loaded locally.")

    # Load data and model
    df = load_sensor_data()
    model, scaler, feature_cols = load_model_artifacts()
    
    if df is None or model is None:
        st.warning("Please run the `main.py` training script to generate data and model files.")
        return
    
    # Page Routing
    if page == "Overview":
        show_overview(df)
    elif page == "Real-Time Monitoring":
        show_realtime_monitoring(df, model, scaler, feature_cols)
    elif page == "Machine Analysis":
        show_machine_analysis(df) # This function would need to be created if desired

def show_overview(df):
    """Overview page with key metrics (No Changes from original)"""
    st.header("System Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Machines", df['machine_id'].nunique())
    with col2: st.metric("Total Records", f"{len(df):,}")
    with col3: st.metric("Failure Rate", f"{df['failure'].mean() * 100:.2f}%")
    with col4: st.metric("Total Failures", int(df['failure'].sum()))
    with col5: st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Failure Events Over Time")
        daily_failures = df.groupby(df['timestamp'].dt.date)['failure'].sum().reset_index()
        daily_failures.columns = ['Date', 'Failures']
        fig = px.line(daily_failures, x='Date', y='Failures', title="Daily Failure Count")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Machine Failure Distribution")
        machine_failures = df.groupby('machine_id')['failure'].sum().reset_index()
        fig = px.bar(machine_failures, x='machine_id', y='failure', title="Failures by Machine")
        st.plotly_chart(fig, use_container_width=True)

def show_realtime_monitoring(df, model, scaler, feature_cols):
    """Real-time monitoring page with local prediction"""
    st.header("Real-Time Machine Monitoring")
    
    machines = sorted(df['machine_id'].unique())
    selected_machine = st.selectbox("Select Machine", machines)
    
    latest = df[df['machine_id'] == selected_machine].sort_values('timestamp', ascending=False).iloc[0]
    
    st.subheader(f"Current Status: {selected_machine}")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Temperature", f"{latest['temperature']:.1f}°C")
    with col2: st.metric("Vibration", f"{latest['vibration']:.2f} m/s²")
    with col3: st.metric("Voltage", f"{latest['voltage']:.1f}V")
    with col4: st.metric("Pressure", f"{latest['pressure']:.1f} PSI")
    with col5: st.metric("RPM", f"{latest['rpm']:.0f}")
    
    if st.button("🔮 Predict Failure Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing sensor data..."):
            prediction_data = {
                "temperature": float(latest['temperature']), "vibration": float(latest['vibration']),
                "voltage": float(latest['voltage']), "pressure": float(latest['pressure']),
                "rpm": float(latest['rpm'])
            }
            
            result = predict_failure(prediction_data, model, scaler, feature_cols)
            
            risk_color = get_risk_color(result['risk_level'])
            st.markdown(f"""
                <div style='padding: 2rem; background-color: {risk_color}20; border-radius: 10px; border: 2px solid {risk_color}'>
                    <h2 style='color: {risk_color}; margin: 0;'>Risk Level: {result['risk_level']}</h2>
                    <h3>Failure Probability: {result['failure_probability']*100:.1f}%</h3>
                    <p style='font-size: 1.1rem;'><strong>Recommendation:</strong> {result['recommendation']}</p>
                </div>
            """, unsafe_allow_html=True)

import streamlit as st
import plotly.express as px

def show_machine_analysis(df):
    st.header("Detailed Machine Analysis")

    st.sidebar.header("Analysis Filters")
    machines = sorted(df['machine_id'].unique())
    selected_machines = st.sidebar.multiselect(
        "Select Machines to Analyze",
        options=machines,
        default=[machines[0]]
    )

    sensor_cols = ['temperature', 'vibration', 'voltage', 'pressure', 'rpm']
    selected_sensors = st.sidebar.multiselect(
        "Select Sensors to Display",
        options=sensor_cols,
        default=['temperature', 'vibration']
    )

    if not selected_machines or not selected_sensors:
        st.warning("Please select at least one machine and one sensor from the sidebar to view the analysis.")
        return

    for machine in selected_machines:
        st.subheader(f"Sensor Readings for {machine}")

        machine_df = df[df['machine_id'] == machine].copy()
        if not pd.api.types.is_datetime64_any_dtype(machine_df['timestamp']):
            machine_df['timestamp'] = pd.to_datetime(machine_df['timestamp'])

        # Create line plot
        fig = px.line(
            machine_df,
            x='timestamp',
            y=selected_sensors,
            title=f"Time Series Data for {machine}"
        )

        # Add failure events as vertical lines using shapes
        failure_events = machine_df[machine_df['failure'] == 1]['timestamp']
        for event_ts in failure_events:
            fig.add_shape(
                type="line",
                x0=event_ts,
                x1=event_ts,
                y0=machine_df[selected_sensors].min().min(),
                y1=machine_df[selected_sensors].max().max(),
                line=dict(color="red", width=2, dash="dash")
            )
            # Optionally, add annotation
            fig.add_annotation(
                x=event_ts,
                y=machine_df[selected_sensors].max().max(),
                text="Failure",
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-40
            )

        st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()