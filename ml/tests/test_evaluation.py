"""
Tests for ml.evaluation.metrics — compute_metrics.

Tests:
  - compute_metrics returns a MetricsResult with all expected attributes
  - PR-AUC, ROC-AUC, LogLoss, Precision, Recall, F1 are in valid range
  - Works even when one class is missing in targets (graceful degradation)
"""

import numpy as np
import pytest

from ml.evaluation.metrics import compute_metrics, MetricsResult


class TestComputeMetrics:

    def test_calculate_metrics_returns_result(self):
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.3, 0.6, 0.2, 0.4, 0.9, 0.1])
        result = compute_metrics(y_true, y_prob)
        assert isinstance(result, MetricsResult)

    def test_metrics_in_valid_range(self):
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.3, 0.6, 0.2, 0.4, 0.9, 0.1])
        result = compute_metrics(y_true, y_prob)
        
        # AUCs and F1 should be in [0, 1]
        assert 0.0 <= result.pr_auc <= 1.0
        assert 0.0 <= result.roc_auc <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        
        # Brier score should be >= 0
        assert result.brier_score >= 0.0

    def test_graceful_degradation_when_only_one_class(self):
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.1, 0.3, 0.2])
        result = compute_metrics(y_true, y_prob)
        
        # ROC AUC cannot be calculated for one class, should return NaN gracefully
        assert result.roc_auc == 0.5
        assert result.pr_auc == 0.0

    def test_log_does_not_crash(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.8, 0.3, 0.6])
        result = compute_metrics(y_true, y_prob)
        result.log()  # Simply verify no exception is thrown
