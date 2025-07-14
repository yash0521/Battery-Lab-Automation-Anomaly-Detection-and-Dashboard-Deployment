import streamlit as st
import pandas as pd
import numpy as np
from anomaly_detector import BatteryAnomalyDetector

# Initialize detector
detector = BatteryAnomalyDetector()

# Title and Instructions
st.title("🔋 Battery Cycle Anomaly Detector (Enhanced)")
st.markdown("""
Upload a CSV summary (e.g., cycle_summary.csv) and select a cycle to view its anomaly risk.
**NEW:** Now with extracted logic and better error handling!
""")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'cycle_id' not in df.columns:
            st.error("The file must contain a 'cycle_id' column.")
        else:
            selected_cycle = st.selectbox("Select a cycle:", df['cycle_id'].unique())
            selected_row = df[df['cycle_id'] == selected_cycle].iloc[0]

            # Use the new detector class
            score, is_anomaly = detector.detect_anomaly(
                selected_row['temp_max'],
                selected_row['current_mean'],
                selected_row['voltage_mean'],
                df
            )

            st.metric("Anomaly Score", f"{score:.2f}")
            
            if is_anomaly:
                st.error("⚠️ Anomaly Detected!")
            else:
                st.success("✅ No anomaly detected")

            # Show basic trip metrics
            st.subheader("Cycle Metrics")
            st.dataframe(selected_row.to_frame())
            
            # NEW: Show statistics
            st.subheader("Dataset Statistics")
            stats = detector.calculate_statistics(df)
            st.json(stats)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure your CSV has the required columns: cycle_id, temp_max, current_mean, voltage_mean")
        if st.checkbox("🤖 Run ML Anomaly Detection on All Cycles"):
            with st.spinner("Running machine learning analysis..."):
                ml_results = detector.detect_ml_anomalies(df)
            
            st.subheader("ML Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                zscore_count = ml_results['zscore_anomaly'].sum()
                st.metric("Z-Score Anomalies", zscore_count)
            
            with col2:
                ml_count = ml_results['ml_anomaly'].sum()
                st.metric("ML Anomalies", ml_count)
            
            # Show some results
            st.subheader("Anomaly Comparison")
            comparison = ml_results[['cycle_id', 'zscore_anomaly', 'ml_anomaly']].head(10)
            st.dataframe(comparison)
            
            # Download results
            csv = ml_results.to_csv(index=False)
            st.download_button(
                "📥 Download ML Results",
                csv,
                "ml_anomaly_results.csv",
                "text/csv"
            )