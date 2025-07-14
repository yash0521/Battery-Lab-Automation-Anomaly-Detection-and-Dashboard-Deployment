# 🔋 Battery Lab Automation - Enhanced Edition

Advanced analytics and anomaly detection for electric vehicle battery data with machine learning capabilities, comprehensive testing, and professional documentation.

## ✨ New Features

- **🤖 Machine Learning**: Isolation Forest anomaly detection
- **🧪 Testing Suite**: Comprehensive pytest test coverage  
- **📚 Documentation**: Auto-generated Sphinx documentation
- **📊 Enhanced Dashboard**: Multi-algorithm comparison and visualization
- **🔧 Modular Design**: Clean, testable, maintainable code

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start Jupyter Notebook
jupyter notebook battery.ipynb

# Run enhanced dashboard
streamlit run src/production_dashboard.py

# Run tests
pytest tests/ -v

# Build documentation
cd docs && make html
```

## 📖 Usage

### Data Preparation
Prepare your battery data in a CSV file or a similar format. Ensure the following columns exist:
- `cycle_id`: Unique identifier for each cycle.
- `temp_max`: Maximum temperature recorded.
- `current_mean`: Average current.
- `voltage_mean`: Average voltage.

### Running the Anomaly Detection
Use the provided `BatteryAnomalyDetector` class to detect anomalies in your data.

### Testing
Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

### Dashboard Deployment
The dashboard is implemented using a web framework (e.g., Flask or Streamlit). Run the dashboard:
```bash
streamlit run src/production_dashboard.py
```
Access the dashboard at [http://localhost:5000](http://localhost:5000).

## 🔮 Future Enhancements
- Add support for real-time data ingestion.
- Enhance ML model with deep learning methods.
- Deploy the dashboard as a cloud-based solution.