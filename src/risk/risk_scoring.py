import pandas as pd

def calculate_risk(df):
    """
    Calculate final risk levels by combining rule scores and ML anomaly scores.
    
    Risk Levels:
    - HIGH: score >= 70
    - MEDIUM: score >= 40
    - LOW: score < 40
    
    Args:
        df: DataFrame with rule_score and anomaly_score columns
    
    Returns:
        DataFrame with risk_level and final_score columns added
    """
    df = df.copy()
    risk_levels = []
    final_scores = []
    
    for _, row in df.iterrows():
        score = row["rule_score"]
        
        # Add ML anomaly contribution
        if row["anomaly_score"] == -1:  # Anomaly detected
            score += 40
        
        final_scores.append(score)
        
        # Classify risk level
        if score >= 70:
            risk_levels.append("HIGH")
        elif score >= 40:
            risk_levels.append("MEDIUM")
        else:
            risk_levels.append("LOW")
    
    df["final_score"] = final_scores
    df["risk_level"] = risk_levels
    
    return df


