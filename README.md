# Fraud Detection System

A comprehensive fraud detection system for government scheme beneficiaries using rule-based detection and ML anomaly detection.

## Project Structure

```
fraud-detection-system/
├── data/
│   ├── raw/              # Raw beneficiary data
│   └── processed/        # Processed results
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/            # Data generation
│   ├── rules/           # Rule-based detection
│   ├── models/          # ML models
│   ├── risk/            # Risk scoring
│   ├── explain/         # Explanation engine
│   ├── api/             # FastAPI backend
│   └── utils/           # Helper utilities
├── frontend/            # Streamlit dashboard
├── models/              # Saved ML models
└── run_pipeline.py      # Main pipeline script
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Pipeline
Generate data, apply rules, train ML model, and compute risk scores:
```bash
python run_pipeline.py
```

### 2. Start API Server
```bash
uvicorn src.api.main:app --reload
```

### 3. Start Dashboard
```bash
streamlit run frontend/dashboard.py
```

## Features

- **Rule-based Detection**: Deterministic fraud rules for interpretability
- **ML Anomaly Detection**: Isolation Forest for unknown patterns
- **Risk Scoring**: Combined rule + ML scores
- **Explainability**: Human-readable fraud explanations
- **REST API**: FastAPI backend for integration
- **Dashboard**: Streamlit visualization interface

## API Endpoints

- `GET /results` - Get all beneficiaries with risk scores
- `GET /explain/{beneficiary_id}` - Get explanation for a beneficiary
- `POST /predict` - Predict risk for a single beneficiary
- `GET /health` - Health check


