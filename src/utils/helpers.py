import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(df, y_true_col="is_fraud", y_pred_col="risk_level"):
    """
    Calculate evaluation metrics (precision, recall, F1).
    
    Args:
        df: DataFrame with true labels and predictions
        y_true_col: Column name for true labels (0/1)
        y_pred_col: Column name for predictions (LOW/MEDIUM/HIGH)
    
    Returns:
        Dictionary with metrics
    """
    if y_true_col not in df.columns:
        return {"error": "True labels not found in dataframe"}
    
    # Convert risk levels to binary (HIGH/MEDIUM = fraud, LOW = normal)
    y_pred_binary = (df[y_pred_col].isin(["HIGH", "MEDIUM"])).astype(int)
    y_true = df[y_true_col].astype(int)
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "note": "Simulated evaluation with synthetic labels"
    }


