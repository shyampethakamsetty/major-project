from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Any, Dict, List
from pydantic import BaseModel

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# Import explanation engine
import sys
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from src.explain.explanation_engine import explain_beneficiary

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data at startup
model = None
df_results = None

@app.on_event("startup")
async def startup_event():
    """Load model and results on startup."""
    global model, df_results
    
    model_path = "models/anomaly_model.joblib"
    results_path = "data/processed/final_results.csv"
    
    # Load model if exists
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}")
    
    # Load results if exists
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        print(f"✓ Results loaded from {results_path}")
    else:
        print(f"⚠ Results not found at {results_path}. Run pipeline first.")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/results": "Get all beneficiaries with risk scores",
            "/explain/{beneficiary_id}": "Get explanation for a beneficiary",
            "/predict": "Predict risk for a single beneficiary",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "results_loaded": df_results is not None
    }

@app.get("/results")
async def get_results(risk_level: Optional[str] = None, limit: Optional[int] = None):
    """
    Get all beneficiaries with risk scores.
    
    Query parameters:
    - risk_level: Filter by risk level (HIGH, MEDIUM, LOW)
    - limit: Limit number of results
    """
    if df_results is None:
        raise HTTPException(status_code=404, detail="Results not found. Run pipeline first.")
    
    df = df_results.copy()
    
    # Filter by risk level if provided
    if risk_level:
        df = df[df["risk_level"] == risk_level.upper()]
    
    # Limit results if provided
    if limit:
        df = df.head(limit)
    
    # Convert to dict format and ensure native Python types
    results = df.to_dict(orient="records")
    results = convert_numpy_types(results)
    
    return {
        "total": len(results),
        "data": results
    }

@app.get("/explain/{beneficiary_id}")
async def explain(beneficiary_id: str):
    """
    Get explanation for a specific beneficiary.
    
    Args:
        beneficiary_id: ID of the beneficiary (e.g., "B0001")
    """
    if df_results is None:
        raise HTTPException(status_code=404, detail="Results not found. Run pipeline first.")
    
    explanation = explain_beneficiary(df_results, beneficiary_id)
    
    if "error" in explanation:
        raise HTTPException(status_code=404, detail=explanation["error"])
    
    return explanation

# Pydantic model for prediction input
class BeneficiaryInput(BaseModel):
    age: int
    state: str
    scheme_name: str
    income: int
    bank_account: int
    amount_requested: int
    past_claims: int
    claim_frequency: float
    created_date: Optional[str] = None

@app.post("/predict")
async def predict(beneficiary: BeneficiaryInput):
    """
    Predict risk for a single beneficiary.
    
    This endpoint uses the trained model for inference.
    Note: For full risk scoring, the pipeline should be run.
    """
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found. Train model first.")
    
    # Prepare features
    import numpy as np
    features = np.array([[
        beneficiary.income,
        beneficiary.amount_requested,
        beneficiary.past_claims,
        beneficiary.claim_frequency
    ]])
    
    # Predict anomaly
    anomaly_score = model.predict(features)[0]
    
    # Simple rule-based scoring (simplified)
    rule_score = 0
    if beneficiary.amount_requested > 200000 and beneficiary.income < 150000:
        rule_score += 40
    if beneficiary.claim_frequency > 3:
        rule_score += 20
    if beneficiary.past_claims > 5:
        rule_score += 20
    
    # Combine scores
    final_score = rule_score
    if anomaly_score == -1:
        final_score += 40
    
    # Determine risk level
    if final_score >= 70:
        risk_level = "HIGH"
    elif final_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "beneficiary_id": "NEW",
        "risk_level": risk_level,
        "final_score": final_score,
        "rule_score": rule_score,
        "anomaly_detected": anomaly_score == -1,
        "anomaly_score": int(anomaly_score)
    }

