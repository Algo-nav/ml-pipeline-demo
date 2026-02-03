"""
Unit Tests for ML Pipeline
=========================
Tests for data loading, model training, and predictions.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_data, train_model, evaluate_model


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_data_returns_correct_shapes(self):
        """Test that load_data returns correctly shaped arrays."""
        X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
        
        # Check shapes
        assert X_train.shape[1] == 4, "Features should have 4 columns"
        assert len(y_train) == len(X_train), "X and y should have same length"
        assert len(feature_names) == 4, "Should have 4 feature names"
        assert len(target_names) == 3, "Should have 3 target names"
    
    def test_load_data_stratified_split(self):
        """Test that the split is stratified."""
        _, _, y_train, y_test, _, _ = load_data(test_size=0.2, random_state=42)
        
        # Check class distribution is similar
        train_dist = np.bincount(y_train) / len(y_train)
        test_dist = np.bincount(y_test) / len(y_test)
        
        # Distributions should be close (within 10%)
        assert np.allclose(train_dist, test_dist, atol=0.1), "Split should be stratified"
    
    def test_load_data_reproducible(self):
        """Test that same random_state gives same split."""
        X1, _, y1, _, _, _ = load_data(random_state=42)
        X2, _, y2, _, _, _ = load_data(random_state=42)
        
        assert np.array_equal(X1, X2), "Same seed should give same data"
        assert np.array_equal(y1, y2), "Same seed should give same labels"


class TestModelTraining:
    """Tests for model training functionality."""
    
    def test_train_model_returns_fitted_model(self):
        """Test that train_model returns a fitted model."""
        X_train, _, y_train, _, _, _ = load_data()
        model = train_model(X_train, y_train, n_estimators=10)
        
        # Check model has been fitted
        assert hasattr(model, 'estimators_'), "Model should be fitted"
        assert len(model.estimators_) == 10, "Should have 10 trees"
    
    def test_train_model_can_predict(self):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, _, _, _ = load_data()
        model = train_model(X_train, y_train, n_estimators=10)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test), "Should predict for all test samples"
        assert all(p in [0, 1, 2] for p in predictions), "Predictions should be valid classes"
    
    def test_train_model_reproducible(self):
        """Test that same random_state gives same model."""
        X_train, X_test, y_train, _, _, _ = load_data()
        
        model1 = train_model(X_train, y_train, n_estimators=10, random_state=42)
        model2 = train_model(X_train, y_train, n_estimators=10, random_state=42)
        
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        assert np.array_equal(pred1, pred2), "Same seed should give same predictions"


class TestModelEvaluation:
    """Tests for model evaluation functionality."""
    
    def test_evaluate_returns_accuracy(self):
        """Test that evaluate_model returns accuracy."""
        X_train, X_test, y_train, y_test, _, target_names = load_data()
        model = train_model(X_train, y_train, n_estimators=10)
        
        accuracy, report = evaluate_model(model, X_test, y_test, list(target_names))
        
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert isinstance(report, dict), "Report should be a dictionary"
    
    def test_model_accuracy_threshold(self):
        """Test that model achieves minimum accuracy."""
        X_train, X_test, y_train, y_test, _, target_names = load_data()
        model = train_model(X_train, y_train, n_estimators=100)
        
        accuracy, _ = evaluate_model(model, X_test, y_test, list(target_names))
        
        # Iris is easy - should get >90% accuracy
        assert accuracy >= 0.9, f"Accuracy {accuracy} should be >= 0.9"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_small_n_estimators(self):
        """Test model works with small number of trees."""
        X_train, _, y_train, _, _, _ = load_data()
        model = train_model(X_train, y_train, n_estimators=1)
        
        assert len(model.estimators_) == 1, "Should work with 1 tree"
    
    def test_different_test_sizes(self):
        """Test different test set sizes."""
        for test_size in [0.1, 0.2, 0.3, 0.5]:
            X_train, X_test, _, _, _, _ = load_data(test_size=test_size)
            
            total = len(X_train) + len(X_test)
            actual_ratio = len(X_test) / total
            
            assert abs(actual_ratio - test_size) < 0.05, f"Test size should be ~{test_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
