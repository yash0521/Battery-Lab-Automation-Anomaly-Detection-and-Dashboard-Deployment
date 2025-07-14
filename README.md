# 🔋 Battery Lab Automation - Anomaly Detection & Dashboard Deployment

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

## Dashboard Previews

<img width="1708" height="940" alt="Screenshot 2025-07-14 at 19 22 18" src="https://github.com/user-attachments/assets/4790d15d-c6b9-4339-a798-9fe2ede490dd" />


<img width="1710" height="941" alt="Screenshot 2025-07-14 at 19 22 05" src="https://github.com/user-attachments/assets/8cdd24fa-f213-4da0-a77b-15cb3e285c69" />


<img width="1709" height="945" alt="Screenshot 2025-07-14 at 19 21 05" src="https://github.com/user-attachments/assets/b8051503-31d3-49b2-b995-a9890f732670" />
