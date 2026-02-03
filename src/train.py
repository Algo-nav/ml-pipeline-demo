"""
Model Training Script
====================
Trains a simple classifier on the Iris dataset.
Designed for CI/CD demonstration - fast training, reproducible results.
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(test_size=0.2, random_state=42):
    """Load and split the Iris dataset."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, 
        test_size=test_size, 
        random_state=random_state,
        stratify=iris.target
    )
    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return accuracy, report


def save_artifacts(model, metrics, output_dir="artifacts"):
    """Save model and metrics to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return model_path, metrics_path


def main(args):
    print("=" * 60)
    print("ML PIPELINE - MODEL TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Parameters: n_estimators={args.n_estimators}, test_size={args.test_size}")
    print("-" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data(
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n🔧 Training model...")
    model = train_model(X_train, y_train, n_estimators=args.n_estimators, random_state=args.random_state)
    print(f"   Model: RandomForestClassifier")
    print(f"   Trees: {args.n_estimators}")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    accuracy, report = evaluate_model(model, X_test, y_test, list(target_names))
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Check accuracy threshold
    if accuracy < args.min_accuracy:
        print(f"\n❌ FAILED: Accuracy {accuracy:.4f} < threshold {args.min_accuracy}")
        exit(1)
    
    print(f"   ✅ Passed accuracy threshold ({args.min_accuracy})")
    
    # Save artifacts
    print("\n💾 Saving artifacts...")
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "parameters": {
            "n_estimators": args.n_estimators,
            "test_size": args.test_size,
            "random_state": args.random_state
        },
        "timestamp": datetime.now().isoformat()
    }
    model_path, metrics_path = save_artifacts(model, metrics, args.output_dir)
    print(f"   Model: {model_path}")
    print(f"   Metrics: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--min_accuracy", type=float, default=0.9, help="Minimum accuracy threshold")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Output directory")
    
    args = parser.parse_args()
    main(args)
