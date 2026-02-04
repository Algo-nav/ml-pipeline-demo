"""
Tests for Fraud Detection Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_data import generate_baseline_data, generate_new_batch, check_drift
from fraud_train import train_model, evaluate_model, FEATURE_COLS


class TestDataGeneration:

    def test_baseline_shape(self):
        df = generate_baseline_data(n_samples=500, random_state=42)
        assert df.shape[0] == 500
        assert "is_fraud" in df.columns
        assert len(FEATURE_COLS) == 10
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_baseline_fraud_rate(self):
        df = generate_baseline_data(n_samples=1000, fraud_rate=0.05, random_state=42)
        actual_rate = df["is_fraud"].mean()
        assert 0.03 <= actual_rate <= 0.07, f"Fraud rate {actual_rate} outside expected range"

    def test_baseline_no_nulls(self):
        df = generate_baseline_data(n_samples=500, random_state=42)
        assert df.isnull().sum().sum() == 0

    def test_baseline_reproducible(self):
        df1 = generate_baseline_data(n_samples=100, random_state=42)
        df2 = generate_baseline_data(n_samples=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_new_batch_drift_levels(self):
        for level in ["none", "mild", "severe"]:
            batch = generate_new_batch(n_samples=200, drift_level=level, random_state=42)
            assert batch.shape[0] == 200
            assert "is_fraud" in batch.columns

    def test_severe_drift_higher_fraud_rate(self):
        none_batch = generate_new_batch(n_samples=2000, drift_level="none", random_state=42)
        severe_batch = generate_new_batch(n_samples=2000, drift_level="severe", random_state=42)
        assert severe_batch["is_fraud"].mean() > none_batch["is_fraud"].mean()


class TestDriftDetection:

    def test_no_drift_detected(self):
        baseline = generate_baseline_data(n_samples=2000, random_state=42)
        batch = generate_new_batch(n_samples=500, drift_level="none", random_state=99)
        result = check_drift(baseline, batch)
        assert isinstance(result, dict)
        assert "dataset_drift" in result
        assert "drift_share" in result

    def test_severe_drift_detected(self):
        baseline = generate_baseline_data(n_samples=2000, random_state=42)
        batch = generate_new_batch(n_samples=500, drift_level="severe", random_state=99)
        result = check_drift(baseline, batch)
        assert result["drift_share"] > 0, "Severe drift should flag some features"


class TestModelTraining:

    def test_train_and_evaluate(self):
        df = generate_baseline_data(n_samples=500, random_state=42)
        X = df[FEATURE_COLS]
        y = df["is_fraud"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["accuracy"] > 0.8, f"Accuracy too low: {metrics['accuracy']}"
        assert metrics["roc_auc"] > 0.7, f"AUC too low: {metrics['roc_auc']}"
        assert 0 <= metrics["f1"] <= 1

    def test_model_predicts_both_classes(self):
        df = generate_baseline_data(n_samples=1000, random_state=42)
        X = df[FEATURE_COLS]
        y = df["is_fraud"]

        model = train_model(X, y)
        preds = model.predict(X)

        assert 0 in preds, "Model should predict class 0"
        assert 1 in preds, "Model should predict class 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
