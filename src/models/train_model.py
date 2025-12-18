import pandas as pd
import joblib
import os
from src.models.anomaly_model import AnomalyDetector

def train_and_save_model(df, model_path="models/anomaly_model.joblib"):
    """
    Train the anomaly detection model and save it.
    
    Args:
        df: DataFrame with features
        model_path: Path to save the model
    
    Returns:
        Trained model and DataFrame with anomaly scores
    """
    # Select features for ML model
    feature_columns = ["income", "amount_requested", "past_claims", "claim_frequency"]
    features = df[feature_columns]
    
    # Initialize and train model
    model = AnomalyDetector(contamination=0.05, random_state=42)
    anomaly_scores = model.fit_predict(features)
    
    # Add anomaly scores to dataframe
    df = df.copy()
    df["anomaly_score"] = anomaly_scores
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved to {model_path}")
    print(f"Anomalies detected: {(anomaly_scores == -1).sum()} ({(anomaly_scores == -1).mean()*100:.1f}%)")
    
    return model, df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/processed/processed_data.csv")
    
    # Train and save model
    model, df_with_scores = train_and_save_model(df)
    
    # Save results
    df_with_scores.to_csv("data/processed/processed_data.csv", index=False)
    print("Results saved to data/processed/processed_data.csv")

