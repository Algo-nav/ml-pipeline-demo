"""
Synthetic Fraud Dataset Generator
=================================
Generates realistic credit card transaction data with configurable drift.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime


FEATURE_COLS = [
    "amount", "hour", "day_of_week", "distance_from_home",
    "distance_from_last_txn", "ratio_to_median", "num_txn_last_24h",
    "is_foreign", "merchant_risk_score", "card_age_months"
]


def generate_baseline_data(n_samples=5000, fraud_rate=0.05, random_state=42):
    """
    Generate baseline transaction dataset.
    
    Features:
    - amount: Transaction amount (log-normal distribution)
    - hour: Hour of transaction (0-23)
    - day_of_week: Day of week (0-6)
    - distance_from_home: Distance from cardholder home (exponential)
    - distance_from_last_txn: Distance from last transaction
    - ratio_to_median: Ratio of amount to median spending
    - num_txn_last_24h: Number of transactions in last 24 hours
    - is_foreign: Whether transaction is foreign (binary)
    - merchant_risk_score: Risk score of merchant (0-1)
    - card_age_months: Age of card in months
    """
    rng = np.random.RandomState(random_state)
    
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    # --- Legitimate transactions ---
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=3.5, sigma=1.0, size=n_legit),
        "hour": rng.choice(range(24), size=n_legit, p=_hour_probs_legit()),
        "day_of_week": rng.randint(0, 7, size=n_legit),
        "distance_from_home": rng.exponential(scale=10, size=n_legit),
        "distance_from_last_txn": rng.exponential(scale=5, size=n_legit),
        "ratio_to_median": rng.lognormal(mean=0, sigma=0.5, size=n_legit),
        "num_txn_last_24h": rng.poisson(lam=3, size=n_legit),
        "is_foreign": rng.binomial(1, 0.05, size=n_legit),
        "merchant_risk_score": rng.beta(2, 10, size=n_legit),
        "card_age_months": rng.randint(1, 120, size=n_legit),
        "is_fraud": 0
    })
    
    # --- Fraudulent transactions ---
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=5.0, sigma=1.5, size=n_fraud),
        "hour": rng.choice(range(24), size=n_fraud, p=_hour_probs_fraud()),
        "day_of_week": rng.randint(0, 7, size=n_fraud),
        "distance_from_home": rng.exponential(scale=50, size=n_fraud),
        "distance_from_last_txn": rng.exponential(scale=30, size=n_fraud),
        "ratio_to_median": rng.lognormal(mean=1.5, sigma=0.8, size=n_fraud),
        "num_txn_last_24h": rng.poisson(lam=8, size=n_fraud),
        "is_foreign": rng.binomial(1, 0.3, size=n_fraud),
        "merchant_risk_score": rng.beta(5, 3, size=n_fraud),
        "card_age_months": rng.randint(1, 120, size=n_fraud),
        "is_fraud": 1
    })
    
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def generate_new_batch(n_samples=1000, drift_level="none", random_state=None):
    """
    Generate a new batch of transactions simulating incoming production data.
    
    drift_level:
    - "none": Same distribution as baseline
    - "mild": Slight distribution shift (new fraud pattern emerging)
    - "severe": Major shift (fraud ring with different behavior)
    """
    if random_state is None:
        random_state = int(datetime.now().timestamp()) % 100000
    
    rng = np.random.RandomState(random_state)
    
    if drift_level == "none":
        fraud_rate = 0.05
        amount_mean_legit = 3.5
        amount_mean_fraud = 5.0
        distance_scale_fraud = 50
        foreign_rate_fraud = 0.3
    elif drift_level == "mild":
        fraud_rate = 0.08
        amount_mean_legit = 3.7
        amount_mean_fraud = 5.5
        distance_scale_fraud = 40
        foreign_rate_fraud = 0.4
    elif drift_level == "severe":
        fraud_rate = 0.15
        amount_mean_legit = 4.0
        amount_mean_fraud = 6.0
        distance_scale_fraud = 20
        foreign_rate_fraud = 0.6
    else:
        raise ValueError(f"Unknown drift_level: {drift_level}")
    
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=amount_mean_legit, sigma=1.0, size=n_legit),
        "hour": rng.choice(range(24), size=n_legit, p=_hour_probs_legit()),
        "day_of_week": rng.randint(0, 7, size=n_legit),
        "distance_from_home": rng.exponential(scale=10, size=n_legit),
        "distance_from_last_txn": rng.exponential(scale=5, size=n_legit),
        "ratio_to_median": rng.lognormal(mean=0, sigma=0.5, size=n_legit),
        "num_txn_last_24h": rng.poisson(lam=3, size=n_legit),
        "is_foreign": rng.binomial(1, 0.05, size=n_legit),
        "merchant_risk_score": rng.beta(2, 10, size=n_legit),
        "card_age_months": rng.randint(1, 120, size=n_legit),
        "is_fraud": 0
    })
    
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=amount_mean_fraud, sigma=1.5, size=n_fraud),
        "hour": rng.choice(range(24), size=n_fraud, p=_hour_probs_fraud()),
        "day_of_week": rng.randint(0, 7, size=n_fraud),
        "distance_from_home": rng.exponential(scale=distance_scale_fraud, size=n_fraud),
        "distance_from_last_txn": rng.exponential(scale=30, size=n_fraud),
        "ratio_to_median": rng.lognormal(mean=1.5, sigma=0.8, size=n_fraud),
        "num_txn_last_24h": rng.poisson(lam=8, size=n_fraud),
        "is_foreign": rng.binomial(1, foreign_rate_fraud, size=n_fraud),
        "merchant_risk_score": rng.beta(5, 3, size=n_fraud),
        "card_age_months": rng.randint(1, 120, size=n_fraud),
        "is_fraud": 1
    })
    
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def _hour_probs_legit():
    """Hour distribution for legitimate transactions (peaks at 10am, 2pm, 7pm)."""
    probs = np.array([
        1, 0.5, 0.3, 0.2, 0.2, 0.3, 0.8, 1.5, 2.5, 3.5,
        4.0, 3.8, 3.5, 3.8, 4.0, 3.5, 3.0, 2.8, 3.0, 3.5,
        3.0, 2.5, 2.0, 1.5
    ])
    return probs / probs.sum()


def _hour_probs_fraud():
    """Hour distribution for fraud (peaks late night / early morning)."""
    probs = np.array([
        3.0, 3.5, 4.0, 4.0, 3.5, 3.0, 2.0, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0,
        2.5, 2.5, 3.0, 3.0
    ])
    return probs / probs.sum()


def check_drift(reference_df, current_df, threshold=0.1):
    """
    Simple drift detection using Wasserstein distance.
    Returns drift report with per-feature scores.
    """
    from scipy.stats import wasserstein_distance
    
    feature_cols = [c for c in reference_df.columns if c != "is_fraud"]
    drift_scores = {}
    drifted_features = []
    
    for col in feature_cols:
        score = wasserstein_distance(
            reference_df[col].values,
            current_df[col].values
        )
        # Normalize by reference std to make scores comparable
        ref_std = reference_df[col].std()
        if ref_std > 0:
            normalized_score = score / ref_std
        else:
            normalized_score = score
        
        drift_scores[col] = round(normalized_score, 4)
        if normalized_score > threshold:
            drifted_features.append(col)
    
    drift_share = len(drifted_features) / len(feature_cols)
    dataset_drift = drift_share > 0.5
    
    return {
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4),
        "drifted_features": drifted_features,
        "feature_scores": drift_scores,
        "threshold": threshold
    }


if __name__ == "__main__":
    print("Generating baseline data...")
    baseline = generate_baseline_data()
    print(f"  Shape: {baseline.shape}")
    print(f"  Fraud rate: {baseline['is_fraud'].mean():.2%}")
    
    for level in ["none", "mild", "severe"]:
        print(f"\nGenerating batch (drift={level})...")
        batch = generate_new_batch(drift_level=level, random_state=99)
        print(f"  Shape: {batch.shape}")
        print(f"  Fraud rate: {batch['is_fraud'].mean():.2%}")
        
        drift = check_drift(baseline, batch)
        print(f"  Dataset drift: {drift['dataset_drift']}")
        print(f"  Drift share: {drift['drift_share']:.2%}")
        print(f"  Drifted features: {drift['drifted_features']}")
