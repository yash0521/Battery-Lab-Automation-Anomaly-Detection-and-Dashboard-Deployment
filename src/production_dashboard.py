# src/production_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from anomaly_detector import BatteryAnomalyDetector

st.set_page_config(
    page_title="🔋 Battery Lab Automation",
    page_icon="⚡",
    layout="wide"
)

# Initialize
detector = BatteryAnomalyDetector()

# Sidebar
st.sidebar.header("⚙️ Controls")
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Individual Cycle", "Batch Analysis", "ML Comparison"]
)

# Main content
st.title("🔋 Battery Lab Automation Dashboard")
st.markdown("**Professional Analytics for Electric Vehicle Battery Performance**")

uploaded_file = st.file_uploader("Upload Battery Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cycles", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Quality", f"{100-missing_pct:.1f}%")
    
    if analysis_type == "Individual Cycle":
        # Your existing individual analysis
        selected_cycle = st.selectbox("Select Cycle:", df['cycle_id'].unique())
        selected_row = df[df['cycle_id'] == selected_cycle].iloc[0]
        
        score, is_anomaly = detector.detect_anomaly(
            selected_row['temp_max'],
            selected_row['current_mean'], 
            selected_row['voltage_mean'],
            df
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomaly Score", f"{score:.2f}")
        with col2:
            status = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
            st.metric("Status", status)
        
        st.dataframe(selected_row.to_frame())
    
    elif analysis_type == "Batch Analysis":
        # ML analysis for all cycles
        with st.spinner("Analyzing all cycles..."):
            results = detector.detect_ml_anomalies(df)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Z-Score Anomalies", results['zscore_anomaly'].sum())
        with col2:
            st.metric("ML Anomalies", results['ml_anomaly'].sum())
        with col3:
            agreement = (results['zscore_anomaly'] == results['ml_anomaly']).mean()
            st.metric("Method Agreement", f"{agreement:.1%}")
        
        # Visualization
        fig = px.scatter(
            results.merge(df, on='cycle_id'),
            x='temp_max',
            y='current_mean',
            color='zscore_anomaly',
            title="Anomaly Detection Results",
            labels={'zscore_anomaly': 'Anomaly'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        csv = results.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "results.csv")
    
    else:  # ML Comparison
        # Side-by-side comparison
        st.subheader("🤖 Algorithm Comparison")
        
        results = detector.detect_ml_anomalies(df)
        
        # Confusion matrix style comparison
        comparison_data = []
        for _, row in results.iterrows():
            comparison_data.append({
                'Cycle': row['cycle_id'],
                'Z-Score': '✅ Normal' if row['zscore_anomaly'] == 0 else '⚠️ Anomaly',
                'ML (Isolation Forest)': '✅ Normal' if row['ml_anomaly'] == 0 else '⚠️ Anomaly',
                'Score': row['zscore_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)

else:
    st.info("👆 Upload your battery data CSV to begin analysis")
    
    # Demo mode
    if st.button("🎮 Try Demo Mode"):
        st.balloons()
        st.success("Demo feature coming soon! Upload your data to get started.")