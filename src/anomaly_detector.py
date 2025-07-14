# src/anomaly_detector.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class BatteryAnomalyDetector:
    
    def __init__(self):
        self.stats = {}
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate stats like in your original dashboard"""
        return {
            'temp_max_mean': data['temp_max'].mean(),
            'temp_max_std': data['temp_max'].std(),
            'current_mean_mean': data['current_mean'].mean(),
            'current_mean_std': data['current_mean'].std(),
            'voltage_mean_mean': data['voltage_mean'].mean(),
            'voltage_mean_std': data['voltage_mean'].std()
        }
    
    def detect_anomaly(self, temp: float, current: float, voltage: float, 
                      data: pd.DataFrame, threshold: float = 4.0) -> Tuple[float, bool]:
        """Your existing anomaly detection logic"""
        stats = self.calculate_statistics(data)
        
        z_temp = (temp - stats['temp_max_mean']) / stats['temp_max_std']
        z_current = (current - stats['current_mean_mean']) / stats['current_mean_std']
        z_voltage = (voltage - stats['voltage_mean_mean']) / stats['voltage_mean_std']
        
        score = abs(z_temp) + abs(z_current) + abs(z_voltage)
        is_anomaly = score > threshold
        
        return score, is_anomaly
    
    def detect_ml_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add machine learning anomaly detection"""
        # Prepare features
        features = ['temp_max', 'current_mean', 'voltage_mean']
        X = data[features].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        ml_anomalies = iso_forest.fit_predict(X_scaled)
        
        # Create results dataframe
        results = data[['cycle_id']].copy()
        results['ml_anomaly'] = (ml_anomalies == -1).astype(int)
        
        # Add traditional Z-score results
        zscore_results = []
        for idx, row in data.iterrows():
            score, is_anomaly = self.detect_anomaly(
                row['temp_max'], row['current_mean'], row['voltage_mean'], data
            )
            zscore_results.append({'score': score, 'anomaly': int(is_anomaly)})
        
        results['zscore_score'] = [r['score'] for r in zscore_results]
        results['zscore_anomaly'] = [r['anomaly'] for r in zscore_results]
        
        return results
