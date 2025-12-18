# Quick Start Guide

## 🚀 Getting Started

### 1. Setup Environment

```bash
cd fraud-detection-system
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Run Pipeline

Generate data, apply rules, train ML model, and compute risk scores:

```bash
python run_pipeline.py
```

This will:
- Generate 1000 synthetic beneficiaries (with 18% fraud)
- Apply rule-based detection
- Train Isolation Forest model
- Calculate risk scores
- Save results to `data/processed/final_results.csv`

### 3. Start API Server

In a new terminal:

```bash
cd fraud-detection-system
source venv/bin/activate
uvicorn src.api.main:app --reload
```

API will be available at: http://localhost:8000

### 4. Start Dashboard

In another terminal:

```bash
cd fraud-detection-system
source venv/bin/activate
streamlit run frontend/dashboard.py
```

Dashboard will open automatically in your browser.

## 📋 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /results` - Get all beneficiaries (with optional filters: `?risk_level=HIGH&limit=10`)
- `GET /explain/{beneficiary_id}` - Get explanation for a beneficiary (e.g., `/explain/B0001`)
- `POST /predict` - Predict risk for a new beneficiary

## 🧪 Test API

```bash
# Get all results
curl http://localhost:8000/results

# Get explanation
curl http://localhost:8000/explain/B0001

# Health check
curl http://localhost:8000/health
```

## 📊 Dashboard Features

1. **Overview**: Key metrics, risk distribution charts
2. **Beneficiaries**: Filterable table of all beneficiaries
3. **Risk Analysis**: Visualizations (income vs amount, rule scores)
4. **Explain Risk**: Get detailed explanations for any beneficiary

## 🔧 Troubleshooting

### API not connecting?
- Make sure API server is running: `uvicorn src.api.main:app --reload`
- Check if port 8000 is available

### No data in dashboard?
- Run pipeline first: `python run_pipeline.py`
- Check if `data/processed/final_results.csv` exists

### Import errors?
- Make sure you're in the project root directory
- Activate virtual environment
- Install all requirements: `pip install -r requirements.txt`

## 📁 Project Structure

```
fraud-detection-system/
├── data/
│   ├── raw/              # Raw beneficiary data
│   └── processed/       # Processed results
├── models/              # Saved ML models
├── src/                 # Source code
├── frontend/            # Streamlit dashboard
└── run_pipeline.py      # Main pipeline script
```


