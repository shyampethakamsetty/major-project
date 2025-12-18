"""
Main pipeline script to run the complete fraud detection pipeline.

Pipeline order:
1. Generate raw data
2. Apply rule engine
3. Run ML anomaly detection
4. Compute final risk score
5. Generate explanations
6. Save final results
"""

import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_generator import generate_data
from src.rules.rule_engine import apply_rules
from src.models.train_model import train_and_save_model
from src.risk.risk_scoring import calculate_risk
from src.utils.helpers import calculate_metrics

def run_pipeline(n_beneficiaries=1000, fraud_percentage=0.18):
    """
    Run the complete fraud detection pipeline.
    
    Args:
        n_beneficiaries: Number of beneficiaries to generate
        fraud_percentage: Percentage of fraudulent cases
    """
    print("=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 60)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Generate data
    print("\n[1/5] Generating synthetic beneficiary data...")
    df = generate_data(n=n_beneficiaries, fraud_percentage=fraud_percentage)
    df.to_csv("data/raw/beneficiaries.csv", index=False)
    print(f"✓ Generated {len(df)} beneficiaries")
    print(f"  - Normal: {(df['is_fraud'] == 0).sum()}")
    print(f"  - Fraud: {df['is_fraud'].sum()}")
    
    # Step 2: Apply rules
    print("\n[2/5] Applying rule-based detection...")
    df = apply_rules(df)
    print(f"✓ Rule scores computed")
    print(f"  - High risk (score >= 70): {(df['rule_score'] >= 70).sum()}")
    print(f"  - Medium risk (40-69): {((df['rule_score'] >= 40) & (df['rule_score'] < 70)).sum()}")
    print(f"  - Low risk (< 40): {(df['rule_score'] < 40).sum()}")
    
    # Save intermediate results
    df.to_csv("data/processed/processed_data.csv", index=False)
    
    # Step 3: Train ML model and detect anomalies
    print("\n[3/5] Training ML anomaly detection model...")
    model, df = train_and_save_model(df, model_path="models/anomaly_model.joblib")
    print(f"✓ Model trained and saved")
    
    # Step 4: Calculate final risk scores
    print("\n[4/5] Calculating final risk scores...")
    df = calculate_risk(df)
    print(f"✓ Risk levels assigned")
    print(f"  - HIGH: {(df['risk_level'] == 'HIGH').sum()}")
    print(f"  - MEDIUM: {(df['risk_level'] == 'MEDIUM').sum()}")
    print(f"  - LOW: {(df['risk_level'] == 'LOW').sum()}")
    
    # Step 5: Calculate metrics (if labels available)
    print("\n[5/5] Calculating evaluation metrics...")
    if 'is_fraud' in df.columns:
        metrics = calculate_metrics(df)
        print(f"✓ Metrics computed:")
        print(f"  - Precision: {metrics.get('precision', 'N/A')}")
        print(f"  - Recall: {metrics.get('recall', 'N/A')}")
        print(f"  - F1-Score: {metrics.get('f1_score', 'N/A')}")
        print(f"  - Note: {metrics.get('note', '')}")
    
    # Save final results
    df.to_csv("data/processed/final_results.csv", index=False)
    print(f"\n✓ Final results saved to data/processed/final_results.csv")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    # Run pipeline with default parameters
    df = run_pipeline(n_beneficiaries=1000, fraud_percentage=0.18)
    
    # Print summary
    print("\n📊 SUMMARY:")
    print(f"Total beneficiaries: {len(df)}")
    print(f"High risk: {(df['risk_level'] == 'HIGH').sum()}")
    print(f"Medium risk: {(df['risk_level'] == 'MEDIUM').sum()}")
    print(f"Low risk: {(df['risk_level'] == 'LOW').sum()}")

