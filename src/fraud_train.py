"""
Fraud Detection Model Training
==============================
Trains a gradient boosted classifier for fraud detection.
Uses scikit-learn's HistGradientBoostingClassifier (fast, handles imbalanced data).
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.ensemble import HistGradientBoostingClassifier

from fraud_data import generate_baseline_data


FEATURE_COLS = [
    "amount", "hour", "day_of_week", "distance_from_home",
    "distance_from_last_txn", "ratio_to_median", "num_txn_last_24h",
    "is_foreign", "merchant_risk_score", "card_age_months"
]


def train_model(X_train, y_train, random_state=42):
    """Train HistGradientBoosting classifier."""
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return comprehensive metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    return metrics


def save_artifacts(model, metrics, data_stats, output_dir="fraud_artifacts"):
    """Save model, metrics, and data stats."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    report = {
        "metrics": metrics,
        "data_stats": data_stats,
        "timestamp": datetime.now().isoformat(),
        "model_type": "HistGradientBoostingClassifier",
        "features": FEATURE_COLS
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    return report


def main(args):
    print("=" * 60)
    print("FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Generate or load data
    print("\nGenerating baseline data...")
    df = generate_baseline_data(n_samples=args.n_samples, random_state=args.random_state)
    print(f"  Total samples: {len(df)}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    
    X = df[FEATURE_COLS]
    y = df["is_fraud"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train
    print("\nTraining model...")
    model = train_model(X_train, y_train, random_state=args.random_state)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Check threshold
    if metrics["f1"] < args.min_f1:
        print(f"\n!! FAILED: F1 {metrics['f1']:.4f} < threshold {args.min_f1}")
        exit(1)
    
    print(f"  Passed F1 threshold ({args.min_f1})")
    
    # Save
    data_stats = {
        "n_samples": len(df),
        "fraud_rate": float(df["is_fraud"].mean()),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    report = save_artifacts(model, metrics, data_stats, args.output_dir)
    
    # Save baseline data for drift comparison
    baseline_path = os.path.join(args.output_dir, "baseline_data.pkl")
    with open(baseline_path, "wb") as f:
        pickle.dump(df, f)
    
    print(f"\nArtifacts saved to: {args.output_dir}/")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--min_f1", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="fraud_artifacts")
    args = parser.parse_args()
    main(args)
