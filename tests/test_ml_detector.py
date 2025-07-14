import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.anomaly_detector import BatteryAnomalyDetector

@pytest.fixture
def sample_data():
    """Create sample battery data for testing"""
    return pd.DataFrame({
        'cycle_id': ['Test01', 'Test02', 'Test03', 'Test04', 'Test05'],
        'temp_max': [22.0, 26.0, 17.0, 45.0, 21.0],  # 45.0 is an outlier
        'current_mean': [-11.9, -20.3, -11.8, -50.0, -12.0],  # -50.0 is an outlier
        'voltage_mean': [388.4, 381.5, 388.4, 350.0, 387.0]  # 350.0 is an outlier
    })

def test_ml_anomaly_detection(sample_data):
    """Test ML anomaly detection functionality"""
    detector = BatteryAnomalyDetector()
    
    results = detector.detect_ml_anomalies(sample_data)
    
    # Check that results have the right structure
    assert 'cycle_id' in results.columns
    assert 'ml_anomaly' in results.columns
    assert 'zscore_anomaly' in results.columns
    assert 'zscore_score' in results.columns
    
    # Check that we got results for all cycles
    assert len(results) == len(sample_data)
    
    # Check that outlier row (Test04) is detected by at least one method
    test04_results = results[results['cycle_id'] == 'Test04'].iloc[0]
    assert test04_results['ml_anomaly'] == 1 or test04_results['zscore_anomaly'] == 1

def test_traditional_anomaly_detection(sample_data):
    """Test the traditional Z-score method"""
    detector = BatteryAnomalyDetector()
    
    # Test with the outlier data point
    outlier_row = sample_data[sample_data['cycle_id'] == 'Test04'].iloc[0]
    
    score, is_anomaly = detector.detect_anomaly(
        outlier_row['temp_max'],
        outlier_row['current_mean'],
        outlier_row['voltage_mean'],
        sample_data
    )
    
    # The outlier should have a high score
    assert score > 2.0  # Should be well above normal
    # Might or might not be flagged as anomaly depending on threshold