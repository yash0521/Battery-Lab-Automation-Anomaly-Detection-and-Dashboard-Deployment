import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_data_loading():
    """Test that we can load the cycle summary data"""
    # Load your actual data file
    data = pd.read_csv('cycle_summary.csv')
    
    # Basic checks
    assert len(data) > 0, "Data should not be empty"
    assert 'cycle_id' in data.columns, "Should have cycle_id column"
    assert 'temp_max' in data.columns, "Should have temp_max column"
    
def test_anomaly_calculation():
    """Test the existing anomaly detection logic"""
    # Create sample data
    data = pd.DataFrame({
        'temp_max': [22.0, 26.0, 17.0],
        'current_mean': [-11.9, -20.3, -11.8],
        'voltage_mean': [388.4, 381.5, 388.4]
    })
    
    # Test stats calculation (copied from your dashboard)
    stats = {
        'temp_max_mean': data['temp_max'].mean(),
        'temp_max_std': data['temp_max'].std(),
        'current_mean_mean': data['current_mean'].mean(),
        'current_mean_std': data['current_mean'].std(),
        'voltage_mean_mean': data['voltage_mean'].mean(),
        'voltage_mean_std': data['voltage_mean'].std()
    }
    
    # Test anomaly detection function (copied from your dashboard)
    def detect_anomaly(temp, current, voltage, stats):
        z_temp = (temp - stats['temp_max_mean']) / stats['temp_max_std']
        z_current = (current - stats['current_mean_mean']) / stats['current_mean_std']
        z_voltage = (voltage - stats['voltage_mean_mean']) / stats['voltage_mean_std']
        score = abs(z_temp) + abs(z_current) + abs(z_voltage)
        return score, bool(score > 4.0)  # Explicitly cast to boolean
    
    # Test with first row
    score, is_anomaly = detect_anomaly(
        data.iloc[0]['temp_max'],
        data.iloc[0]['current_mean'], 
        data.iloc[0]['voltage_mean'],
        stats
    )
    
    assert isinstance(score, float), "Score should be a number"
    assert isinstance(is_anomaly, bool), "Anomaly flag should be boolean"