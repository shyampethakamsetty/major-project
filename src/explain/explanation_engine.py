import pandas as pd
import numpy as np
from datetime import datetime

def convert_to_native_type(value):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def explain_risk(row, bank_account_counts=None):
    """
    Generate human-readable explanation for risk assessment.
    
    Args:
        row: Single row (Series) from DataFrame
        bank_account_counts: Dictionary of bank account frequencies (optional)
    
    Returns:
        List of explanation strings
    """
    reasons = []
    
    # Rule explanations
    if row["amount_requested"] > 200000 and row["income"] < 150000:
        reasons.append(f"High amount requested ({row['amount_requested']:,}) with low income ({row['income']:,})")
    
    if row["claim_frequency"] > 3:
        reasons.append(f"High claim frequency ({row['claim_frequency']:.2f} claims per year)")
    
    if row["past_claims"] > 5:
        reasons.append(f"Multiple past claims ({row['past_claims']} claims)")
    
    # Duplicate bank account check
    if bank_account_counts:
        bank_account = convert_to_native_type(row["bank_account"])
        count = bank_account_counts.get(bank_account, 0)
        if count > 1:
            reasons.append(f"Bank account {bank_account} is used by {count} beneficiaries")
    
    # Recently created with large payout
    if 'days_since_creation' in row:
        if row["days_since_creation"] < 30 and row["amount_requested"] > 200000:
            reasons.append(f"Recently created beneficiary ({row['days_since_creation']} days ago) requesting large amount")
    
    # ML anomaly explanation
    if row.get("anomaly_score") == -1:
        reasons.append("Unusual application pattern detected by ML model")
    
    if not reasons:
        reasons.append("No specific risk factors identified")
    
    return reasons

def explain_beneficiary(df, beneficiary_id):
    """
    Get explanation for a specific beneficiary.
    
    Args:
        df: DataFrame with all beneficiary data
        beneficiary_id: ID of the beneficiary
    
    Returns:
        Dictionary with beneficiary info and explanation
    """
    beneficiary = df[df["beneficiary_id"] == beneficiary_id]
    
    if beneficiary.empty:
        return {"error": "Beneficiary not found"}
    
    row = beneficiary.iloc[0]
    
    # Calculate bank account counts (convert keys to native types)
    bank_account_counts_raw = df['bank_account'].value_counts().to_dict()
    bank_account_counts = {convert_to_native_type(k): convert_to_native_type(v) for k, v in bank_account_counts_raw.items()}
    
    explanation = {
        "beneficiary_id": str(beneficiary_id),
        "risk_level": str(row.get("risk_level", "UNKNOWN")),
        "final_score": convert_to_native_type(row.get("final_score", 0)),
        "rule_score": convert_to_native_type(row.get("rule_score", 0)),
        "anomaly_score": "Anomaly" if convert_to_native_type(row.get("anomaly_score")) == -1 else "Normal",
        "reasons": explain_risk(row, bank_account_counts),
        "details": {
            "age": convert_to_native_type(row["age"]),
            "state": str(row["state"]),
            "scheme_name": str(row["scheme_name"]),
            "income": convert_to_native_type(row["income"]),
            "amount_requested": convert_to_native_type(row["amount_requested"]),
            "past_claims": convert_to_native_type(row["past_claims"]),
            "claim_frequency": convert_to_native_type(row["claim_frequency"])
        }
    }
    
    return explanation

