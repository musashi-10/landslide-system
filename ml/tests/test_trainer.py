"""
Tests for ml.training.trainer — ModelTrainer.

Tests:
  - train_all returns list of results and best result
  - Best model is the one with the highest val_pr_auc
  - Fails gracefully if no positive samples are in training data
  - TrainingResult.summary_row returns correct keys
"""

import numpy as np
import pytest

from ml.training.trainer import ModelTrainer, TrainingResult


def make_dummy_data(n_samples=100, n_features=5):
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (n_samples, n_features))
    # Create somewhat separable data so models can actually learn
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    # Ensure at least some positives and negatives
    if y.sum() == 0:
        y[0] = 1
    elif y.sum() == n_samples:
        y[0] = 0
    return X, y


class TestModelTrainer:

    def test_train_all_returns_results(self):
        X, y = make_dummy_data(100)
        # Split into train/val/test
        X_train, y_train = X[:60], y[:60]
        X_val, y_val = X[60:80], y[60:80]
        X_test, y_test = X[80:], y[80:]
        
        feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
        
        trainer = ModelTrainer()
        all_results, best = trainer.train_all(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
        )
        
        assert isinstance(all_results, list)
        assert len(all_results) >= 2  # LR and RF at least
        assert isinstance(best, TrainingResult)
        
    def test_best_model_has_highest_val_pr_auc(self):
        X, y = make_dummy_data(100)
        X_train, y_train = X[:60], y[:60]
        X_val, y_val = X[60:80], y[60:80]
        X_test, y_test = X[80:], y[80:]
        feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
        
        trainer = ModelTrainer()
        all_results, best = trainer.train_all(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
        )
        
        max_pr_auc = max(r.val_metrics.pr_auc for r in all_results)
        assert best.val_metrics.pr_auc == max_pr_auc

    def test_fails_if_no_positive_samples(self):
        X, y = make_dummy_data(100)
        # Force all training targets to 0
        y_train = np.zeros(60, dtype=int)
        X_train = X[:60]
        X_val, y_val = X[60:80], y[60:80]
        X_test, y_test = X[80:], y[80:]
        feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
        
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="no positive samples"):
            trainer.train_all(
                X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
            )

    def test_summary_row_keys(self):
        X, y = make_dummy_data(50)
        X_train, y_train = X[:30], y[:30]
        X_val, y_val = X[30:40], y[30:40]
        X_test, y_test = X[40:], y[40:]
        feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
        
        trainer = ModelTrainer()
        all_results, best = trainer.train_all(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
        )
        
        row = best.summary_row()
        assert isinstance(row, dict)
        assert "model" in row
        assert "val_pr_auc" in row
        assert "test_pr_auc" in row
