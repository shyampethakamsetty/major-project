import pandas as pd
from datetime import datetime

def apply_rules(df):
    """
    Apply fraud detection rules to the dataset.
    
    Rules:
    1. High amount requested with low income
    2. High claim frequency
    3. Many past claims
    4. Duplicate bank account
    5. Recently created beneficiary with large payout
    
    Args:
        df: DataFrame with beneficiary data
    
    Returns:
        DataFrame with rule_score column added
    """
    df = df.copy()
    scores = []
    
    # Calculate bank account frequency for duplicate detection
    bank_account_counts = df['bank_account'].value_counts().to_dict()
    
    # Calculate days since creation
    if 'created_date' in df.columns:
        df['days_since_creation'] = pd.to_datetime(df['created_date']).apply(
            lambda x: (datetime.now() - x).days
        )
    else:
        df['days_since_creation'] = 365  # Default if not available
    
    for _, row in df.iterrows():
        score = 0
        
        # Rule 1: High amount + low income
        if row["amount_requested"] > 200000 and row["income"] < 150000:
            score += 40
        
        # Rule 2: High claim frequency
        if row["claim_frequency"] > 3:
            score += 20
        
        # Rule 3: Many past claims
        if row["past_claims"] > 5:
            score += 20
        
        # Rule 4: Duplicate bank account
        bank_account = row["bank_account"]
        if bank_account_counts.get(bank_account, 0) > 1:
            score += 30
        
        # Rule 5: Recently created beneficiary with large payout
        if row["days_since_creation"] < 30 and row["amount_requested"] > 200000:
            score += 25
        
        scores.append(score)
    
    df["rule_score"] = scores
    return df


