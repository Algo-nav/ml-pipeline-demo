"""
Data Validation Script
=====================
Validates data quality before training.
Checks for missing values, correct shapes, and expected distributions.
"""

import sys
import numpy as np
from sklearn.datasets import load_iris


def validate_iris_data():
    """Validate the Iris dataset."""
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Load data
    print("\n📊 Loading dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Check 1: No missing values
    print("\n🔍 Checking for missing values...")
    if np.isnan(X).any():
        errors.append("Features contain NaN values")
        print("   ❌ NaN values found in features")
    else:
        print("   ✅ No missing values in features")
    
    if np.isnan(y).any():
        errors.append("Target contains NaN values")
        print("   ❌ NaN values found in target")
    else:
        print("   ✅ No missing values in target")
    
    # Check 2: Expected shape
    print("\n🔍 Checking data shape...")
    expected_features = 4
    expected_samples = 150
    
    if X.shape[1] != expected_features:
        errors.append(f"Expected {expected_features} features, got {X.shape[1]}")
        print(f"   ❌ Wrong number of features: {X.shape[1]}")
    else:
        print(f"   ✅ Correct number of features: {X.shape[1]}")
    
    if X.shape[0] != expected_samples:
        warnings.append(f"Expected {expected_samples} samples, got {X.shape[0]}")
        print(f"   ⚠️ Unexpected number of samples: {X.shape[0]}")
    else:
        print(f"   ✅ Correct number of samples: {X.shape[0]}")
    
    # Check 3: Target classes
    print("\n🔍 Checking target classes...")
    unique_classes = np.unique(y)
    expected_classes = np.array([0, 1, 2])
    
    if not np.array_equal(unique_classes, expected_classes):
        errors.append(f"Unexpected classes: {unique_classes}")
        print(f"   ❌ Unexpected classes: {unique_classes}")
    else:
        print(f"   ✅ Correct classes: {unique_classes}")
    
    # Check 4: Class balance
    print("\n🔍 Checking class balance...")
    class_counts = np.bincount(y)
    imbalance_ratio = max(class_counts) / min(class_counts)
    
    if imbalance_ratio > 1.5:
        warnings.append(f"Class imbalance detected: ratio {imbalance_ratio:.2f}")
        print(f"   ⚠️ Class imbalance ratio: {imbalance_ratio:.2f}")
    else:
        print(f"   ✅ Classes balanced: {dict(enumerate(class_counts))}")
    
    # Check 5: Feature ranges
    print("\n🔍 Checking feature ranges...")
    for i, name in enumerate(iris.feature_names):
        min_val, max_val = X[:, i].min(), X[:, i].max()
        if min_val < 0:
            warnings.append(f"Feature {name} has negative values")
            print(f"   ⚠️ {name}: [{min_val:.2f}, {max_val:.2f}] (negative values)")
        else:
            print(f"   ✅ {name}: [{min_val:.2f}, {max_val:.2f}]")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ VALIDATION FAILED")
        print(f"   Errors: {len(errors)}")
        for e in errors:
            print(f"   - {e}")
        print("=" * 60)
        return False
    elif warnings:
        print("⚠️ VALIDATION PASSED WITH WARNINGS")
        print(f"   Warnings: {len(warnings)}")
        for w in warnings:
            print(f"   - {w}")
        print("=" * 60)
        return True
    else:
        print("✅ VALIDATION PASSED")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = validate_iris_data()
    sys.exit(0 if success else 1)
