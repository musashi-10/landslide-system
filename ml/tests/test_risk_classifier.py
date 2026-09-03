"""
Tests for ml.inference.risk_classifier — RiskLevel and RiskClassifier.

Tests:
  - All four risk levels are reachable
  - Boundary values are classified correctly
  - Probability outside [0, 1] raises ValueError
  - Default thresholds are valid
  - classify_many() works on a list
  - parse_risk_level() accepts valid values (case-insensitive)
  - parse_risk_level() rejects invalid values
  - threshold_summary() returns expected keys
"""

import pytest

from ml.inference.risk_classifier import RiskClassifier, RiskLevel, parse_risk_level
from ml.config.training_config import RiskThresholds


class TestRiskClassifier:

    def setup_method(self):
        self.clf = RiskClassifier()

    # ── Risk level coverage ────────────────────────────────────────────────────

    def test_low_risk(self):
        assert self.clf.classify(0.10) == RiskLevel.LOW

    def test_moderate_risk(self):
        assert self.clf.classify(0.40) == RiskLevel.MODERATE

    def test_high_risk(self):
        assert self.clf.classify(0.60) == RiskLevel.HIGH

    def test_critical_risk(self):
        assert self.clf.classify(0.90) == RiskLevel.CRITICAL

    # ── Boundary conditions ────────────────────────────────────────────────────

    def test_exactly_low_boundary(self):
        # probability == low_max → MODERATE
        t = self.clf.thresholds
        assert self.clf.classify(t.low_max) == RiskLevel.MODERATE

    def test_just_below_low_boundary(self):
        t = self.clf.thresholds
        assert self.clf.classify(t.low_max - 0.001) == RiskLevel.LOW

    def test_exactly_critical_boundary(self):
        t = self.clf.thresholds
        assert self.clf.classify(t.critical_min) == RiskLevel.CRITICAL

    def test_zero_is_low(self):
        assert self.clf.classify(0.0) == RiskLevel.LOW

    def test_one_is_critical(self):
        assert self.clf.classify(1.0) == RiskLevel.CRITICAL

    # ── Invalid probability ────────────────────────────────────────────────────

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="0.0, 1.0"):
            self.clf.classify(-0.01)

    def test_above_one_probability_raises(self):
        with pytest.raises(ValueError, match="0.0, 1.0"):
            self.clf.classify(1.01)

    # ── classify_many ──────────────────────────────────────────────────────────

    def test_classify_many(self):
        probs = [0.1, 0.4, 0.65, 0.9]
        expected = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]
        result = self.clf.classify_many(probs)
        assert result == expected

    def test_classify_many_empty(self):
        assert self.clf.classify_many([]) == []

    # ── Custom thresholds ──────────────────────────────────────────────────────

    def test_custom_thresholds(self):
        t = RiskThresholds(low_max=0.2, moderate_max=0.4, high_max=0.6, critical_min=0.6)
        clf = RiskClassifier(thresholds=t)
        assert clf.classify(0.15) == RiskLevel.LOW
        assert clf.classify(0.30) == RiskLevel.MODERATE
        assert clf.classify(0.50) == RiskLevel.HIGH
        assert clf.classify(0.70) == RiskLevel.CRITICAL

    def test_invalid_thresholds_raise(self):
        t = RiskThresholds(low_max=0.9, moderate_max=0.5, high_max=0.75, critical_min=0.75)
        with pytest.raises(ValueError):
            t.validate()

    # ── threshold_summary ──────────────────────────────────────────────────────

    def test_threshold_summary_has_all_levels(self):
        summary = self.clf.threshold_summary()
        for level in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
            assert level in summary

    def test_threshold_summary_marks_provisional(self):
        summary = self.clf.threshold_summary()
        assert "PROVISIONAL" in summary.get("status", "")


class TestParseRiskLevel:
    def test_valid_uppercase(self):
        assert parse_risk_level("HIGH") == RiskLevel.HIGH

    def test_valid_lowercase(self):
        assert parse_risk_level("low") == RiskLevel.LOW

    def test_valid_mixed_case(self):
        assert parse_risk_level("Moderate") == RiskLevel.MODERATE

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="not a valid risk level"):
            parse_risk_level("EXTREME")
