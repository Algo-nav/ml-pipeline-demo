"""
Automated Retraining Pipeline
=============================
1. Load current champion model
2. Generate new data batch (simulates incoming production data)
3. Check for drift
4. Train challenger model on combined data
5. Compare champion vs challenger
6. Deploy if challenger wins
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from fraud_data import generate_new_batch, check_drift, generate_baseline_data
from fraud_train import train_model, evaluate_model, FEATURE_COLS


def load_champion(artifacts_dir="fraud_artifacts"):
    """Load current production model and its metrics."""
    model_path = os.path.join(artifacts_dir, "model.pkl")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    baseline_path = os.path.join(artifacts_dir, "baseline_data.pkl")
    
    if not os.path.exists(model_path):
        print("No champion model found. Training initial model...")
        return None, None, None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    baseline_df = None
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            baseline_df = pickle.load(f)
    
    return model, metrics, baseline_df


def run_retraining(args):
    print("=" * 60)
    print("AUTOMATED RETRAINING PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Drift level: {args.drift_level}")
    print("=" * 60)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "drift_level": args.drift_level,
        "decision": None,
        "reason": None,
    }
    
    # Step 1: Load champion
    print("\n--- STEP 1: Load Champion Model ---")
    champion_model, champion_report, baseline_df = load_champion(args.artifacts_dir)
    
    if champion_model is None:
        print("No existing model. Will train from scratch.")
        baseline_df = generate_baseline_data(n_samples=5000, random_state=42)
        champion_metrics = None
    else:
        champion_metrics = champion_report.get("metrics", {})
        print(f"  Champion F1: {champion_metrics.get('f1', 'N/A')}")
        print(f"  Champion AUC: {champion_metrics.get('roc_auc', 'N/A')}")
    
    # Step 2: Generate new data batch
    print("\n--- STEP 2: Generate New Data Batch ---")
    new_batch = generate_new_batch(
        n_samples=args.batch_size,
        drift_level=args.drift_level,
        random_state=args.random_state
    )
    print(f"  New batch size: {len(new_batch)}")
    print(f"  New batch fraud rate: {new_batch['is_fraud'].mean():.2%}")
    
    # Step 3: Check drift
    print("\n--- STEP 3: Drift Detection ---")
    drift_report = check_drift(baseline_df, new_batch, threshold=args.drift_threshold)
    print(f"  Dataset drift detected: {drift_report['dataset_drift']}")
    print(f"  Drift share: {drift_report['drift_share']:.2%}")
    print(f"  Drifted features: {drift_report['drifted_features']}")
    
    report["drift"] = drift_report
    
    # Step 4: Train challenger on combined data
    print("\n--- STEP 4: Train Challenger Model ---")
    combined_df = pd.concat([baseline_df, new_batch], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    print(f"  Combined dataset size: {len(combined_df)}")
    print(f"  Combined fraud rate: {combined_df['is_fraud'].mean():.2%}")
    
    # Split combined data
    from sklearn.model_selection import train_test_split
    X = combined_df[FEATURE_COLS]
    y = combined_df["is_fraud"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    
    challenger_model = train_model(X_train, y_train, random_state=args.random_state)
    challenger_metrics = evaluate_model(challenger_model, X_test, y_test)
    
    print(f"  Challenger F1:  {challenger_metrics['f1']:.4f}")
    print(f"  Challenger AUC: {challenger_metrics['roc_auc']:.4f}")
    
    report["challenger_metrics"] = challenger_metrics
    
    # Step 5: Champion vs Challenger comparison
    print("\n--- STEP 5: Champion vs Challenger ---")
    
    if champion_metrics is None:
        # No champion exists, challenger wins by default
        decision = "deploy"
        reason = "No existing champion model. Deploying initial model."
        print(f"  {reason}")
    else:
        # Evaluate champion on the SAME test set for fair comparison
        champion_metrics_on_new = evaluate_model(champion_model, X_test, y_test)
        report["champion_metrics_on_new_data"] = champion_metrics_on_new
        
        print(f"  Champion F1 (on new data):   {champion_metrics_on_new['f1']:.4f}")
        print(f"  Challenger F1 (on new data): {challenger_metrics['f1']:.4f}")
        
        f1_improvement = challenger_metrics["f1"] - champion_metrics_on_new["f1"]
        auc_improvement = challenger_metrics["roc_auc"] - champion_metrics_on_new["roc_auc"]
        
        print(f"  F1 improvement:  {f1_improvement:+.4f}")
        print(f"  AUC improvement: {auc_improvement:+.4f}")
        
        report["f1_improvement"] = round(f1_improvement, 4)
        report["auc_improvement"] = round(auc_improvement, 4)
        
        min_improvement = args.min_improvement
        
        if f1_improvement >= min_improvement:
            decision = "deploy"
            reason = f"Challenger F1 improved by {f1_improvement:+.4f} (threshold: {min_improvement})"
        elif f1_improvement >= 0 and drift_report["dataset_drift"]:
            decision = "deploy"
            reason = f"Drift detected and challenger maintains performance (F1 delta: {f1_improvement:+.4f})"
        else:
            decision = "skip"
            reason = f"Challenger did not improve sufficiently (F1 delta: {f1_improvement:+.4f}, threshold: {min_improvement})"
    
    report["decision"] = decision
    report["reason"] = reason
    
    print(f"\n  DECISION: {decision.upper()}")
    print(f"  REASON: {reason}")
    
    # Step 6: Save report and optionally update model
    print("\n--- STEP 6: Save Results ---")
    
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # Always save retraining report
    report_path = os.path.join(args.artifacts_dir, "retrain_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")
    
    if decision == "deploy":
        # Update champion model
        model_path = os.path.join(args.artifacts_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(challenger_model, f)
        
        # Update metrics
        metrics_report = {
            "metrics": challenger_metrics,
            "data_stats": {
                "n_samples": len(combined_df),
                "fraud_rate": float(combined_df["is_fraud"].mean()),
                "train_size": len(X_train),
                "test_size": len(X_test)
            },
            "timestamp": datetime.now().isoformat(),
            "model_type": "HistGradientBoostingClassifier",
            "features": FEATURE_COLS,
            "retrain_reason": reason
        }
        with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
            json.dump(metrics_report, f, indent=2)
        
        # Update baseline
        with open(os.path.join(args.artifacts_dir, "baseline_data.pkl"), "wb") as f:
            pickle.dump(combined_df, f)
        
        print("  Champion model UPDATED")
        
        # Write deploy flag
        with open(os.path.join(args.artifacts_dir, "deploy_flag"), "w") as f:
            f.write("true")
        print("  Deploy flag SET")
    else:
        # Write no-deploy flag
        with open(os.path.join(args.artifacts_dir, "deploy_flag"), "w") as f:
            f.write("false")
        print("  Deploy flag NOT set (champion retained)")
    
    print("\n" + "=" * 60)
    print(f"RETRAINING COMPLETE - Decision: {decision.upper()}")
    print("=" * 60)
    
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift_level", type=str, default="mild",
                        choices=["none", "mild", "severe"])
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--drift_threshold", type=float, default=0.1)
    parser.add_argument("--min_improvement", type=float, default=0.01)
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--artifacts_dir", type=str, default="fraud_artifacts")
    args = parser.parse_args()
    
    if args.random_state is None:
        args.random_state = int(datetime.now().timestamp()) % 100000
    
    run_retraining(args)
