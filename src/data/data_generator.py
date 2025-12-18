import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(n=1000, fraud_percentage=0.15):
    """
    Generate synthetic beneficiary data with injected fraud patterns.
    
    Args:
        n: Total number of beneficiaries
        fraud_percentage: Percentage of fraudulent cases (15-20%)
    
    Returns:
        DataFrame with beneficiary data
    """
    data = []
    fraud_count = int(n * fraud_percentage)
    normal_count = n - fraud_count
    
    # Track bank accounts for duplication
    bank_accounts_pool = []
    duplicate_accounts = []
    
    # Generate some bank accounts that will be reused (5-10% duplication)
    num_duplicate_accounts = max(5, int(n * 0.07))
    for _ in range(num_duplicate_accounts):
        duplicate_accounts.append(np.random.randint(10**9, 10**10))
    
    # Generate normal beneficiaries (80-85%)
    for i in range(normal_count):
        income = np.random.randint(50000, 800000)
        amount = np.random.randint(10000, 300000)
        
        # Use unique bank account for most normal cases
        bank_account = np.random.randint(10**9, 10**10)
        bank_accounts_pool.append(bank_account)
        
        # Random creation date (within last 2 years)
        days_ago = np.random.randint(0, 730)
        created_date = datetime.now() - timedelta(days=days_ago)
        
        data.append({
            "beneficiary_id": f"B{i:04d}",
            "age": np.random.randint(18, 70),
            "state": random.choice(["AP", "TS", "KA", "TN", "MH", "GJ"]),
            "scheme_name": random.choice(["Farmer Aid", "Education Grant", "Health Support", "Housing Scheme"]),
            "income": income,
            "bank_account": bank_account,
            "amount_requested": amount,
            "past_claims": np.random.randint(0, 5),
            "claim_frequency": np.random.uniform(0, 2.5),
            "created_date": created_date.strftime("%Y-%m-%d"),
            "is_fraud": 0  # Label for evaluation only
        })
    
    # Generate fraudulent beneficiaries (15-20%)
    fraud_patterns = [
        "high_amount_low_income",
        "high_frequency",
        "duplicate_account",
        "recent_large_payout"
    ]
    
    for i in range(fraud_count):
        pattern = random.choice(fraud_patterns)
        
        if pattern == "high_amount_low_income":
            # High amount requested with very low income
            income = np.random.randint(50000, 120000)
            amount = np.random.randint(200000, 300000)
            past_claims = np.random.randint(0, 3)
            claim_frequency = np.random.uniform(0, 1.5)
            bank_account = np.random.randint(10**9, 10**10)
            days_ago = np.random.randint(30, 365)
            
        elif pattern == "high_frequency":
            # Very frequent claims
            income = np.random.randint(100000, 500000)
            amount = np.random.randint(50000, 200000)
            past_claims = np.random.randint(6, 15)
            claim_frequency = np.random.uniform(4, 8)
            bank_account = np.random.randint(10**9, 10**10)
            days_ago = np.random.randint(100, 500)
            
        elif pattern == "duplicate_account":
            # Reuse existing bank account
            income = np.random.randint(80000, 400000)
            amount = np.random.randint(100000, 250000)
            past_claims = np.random.randint(2, 8)
            claim_frequency = np.random.uniform(1, 4)
            # Reuse a duplicate account
            bank_account = random.choice(duplicate_accounts)
            days_ago = np.random.randint(50, 400)
            
        else:  # recent_large_payout
            # Recently created beneficiary with large payout
            income = np.random.randint(60000, 300000)
            amount = np.random.randint(250000, 300000)
            past_claims = np.random.randint(0, 2)
            claim_frequency = np.random.uniform(0, 0.5)
            bank_account = np.random.randint(10**9, 10**10)
            days_ago = np.random.randint(0, 30)  # Very recent
        
        created_date = datetime.now() - timedelta(days=days_ago)
        
        data.append({
            "beneficiary_id": f"B{normal_count + i:04d}",
            "age": np.random.randint(18, 70),
            "state": random.choice(["AP", "TS", "KA", "TN", "MH", "GJ"]),
            "scheme_name": random.choice(["Farmer Aid", "Education Grant", "Health Support", "Housing Scheme"]),
            "income": income,
            "bank_account": bank_account,
            "amount_requested": amount,
            "past_claims": past_claims,
            "claim_frequency": claim_frequency,
            "created_date": created_date.strftime("%Y-%m-%d"),
            "is_fraud": 1  # Label for evaluation only
        })
    
    # Shuffle the data
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_data(n=1000, fraud_percentage=0.18)
    
    # Save to CSV
    import os
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/beneficiaries.csv", index=False)
    
    print(f"Generated {len(df)} beneficiaries")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Saved to data/raw/beneficiaries.csv")

