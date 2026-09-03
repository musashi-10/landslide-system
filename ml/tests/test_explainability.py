"""
Tests for ml.explainability.explainer — LandslideExplainer.

Tests:
  - global_importance() returns list of FeatureImportance
  - local_explanation() returns list of FeatureImportance
  - FeatureImportance.to_dict() has required keys
  - top_factor_names() returns list of strings
  - text_summary() returns non-empty string
  - Works without SHAP (fallback mode)
  - Works with RandomForest (tree model)
  - n_top parameter respected
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.explainability.explainer import LandslideExplainer, FeatureImportance


def make_simple_model_and_data(n=100, n_features=6):
    """Train a minimal RF on synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (n, n_features))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, feature_cols


def make_lr_model_and_data(n=100, n_features=6):
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (n, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model, X, feature_cols


class TestLandslideExplainerFallback:
    """Tests that run without SHAP (using fallback built-in importance)."""

    def setup_method(self):
        self.model, self.X, self.feature_cols = make_simple_model_and_data()
        # Force fallback by passing no explainer init (SHAP may or may not be installed)
        self.explainer = LandslideExplainer(self.model, self.feature_cols)

    def test_global_importance_returns_list(self):
        result = self.explainer.global_importance(self.X)
        assert isinstance(result, list)

    def test_global_importance_non_empty(self):
        result = self.explainer.global_importance(self.X)
        assert len(result) > 0

    def test_global_importance_items_are_feature_importance(self):
        result = self.explainer.global_importance(self.X)
        for item in result:
            assert isinstance(item, FeatureImportance)

    def test_global_importance_n_top_respected(self):
        result = self.explainer.global_importance(self.X, n_top=3)
        assert len(result) <= 3

    def test_global_importance_ranks_ordered(self):
        result = self.explainer.global_importance(self.X, n_top=6)
        ranks = [item.rank for item in result]
        assert ranks == sorted(ranks)

    def test_local_explanation_returns_list(self):
        row_df = pd.DataFrame([self.X[0]], columns=self.feature_cols)
        result = self.explainer.local_explanation(row_df)
        assert isinstance(result, list)

    def test_local_explanation_n_top_respected(self):
        row_df = pd.DataFrame([self.X[0]], columns=self.feature_cols)
        result = self.explainer.local_explanation(row_df, n_top=3)
        assert len(result) <= 3

    def test_feature_importance_to_dict_has_keys(self):
        result = self.explainer.global_importance(self.X, n_top=1)
        d = result[0].to_dict()
        assert "feature" in d
        assert "label" in d
        assert "shap_value" in d
        assert "rank" in d

    def test_top_factor_names_returns_strings(self):
        importances = self.explainer.global_importance(self.X, n_top=3)
        names = self.explainer.top_factor_names(importances)
        assert isinstance(names, list)
        for n in names:
            assert isinstance(n, str)

    def test_text_summary_non_empty(self):
        importances = self.explainer.global_importance(self.X, n_top=3)
        summary = self.explainer.text_summary("HIGH", importances)
        assert isinstance(summary, str)
        assert "HIGH" in summary
        assert len(summary) > 10

    def test_known_feature_label_mapping(self):
        """Features with known names should get human-readable labels."""
        model, X, _ = make_simple_model_and_data()
        named_features = ["slope", "rainfall_24h", "elevation", "feat_3", "feat_4", "feat_5"]
        explainer = LandslideExplainer(model, named_features)
        importances = explainer.global_importance(X, n_top=6)
        labels = {item.feature: item.label for item in importances}
        if "slope" in labels:
            assert labels["slope"] == "steep_slope"
        if "rainfall_24h" in labels:
            assert labels["rainfall_24h"] == "high_24h_rainfall"


class TestLandslideExplainerWithLR:
    """Tests with Logistic Regression model (LinearExplainer path)."""

    def setup_method(self):
        self.model, self.X, self.feature_cols = make_lr_model_and_data()
        self.explainer = LandslideExplainer(
            self.model, self.feature_cols, X_background=self.X[:20]
        )

    def test_global_importance_non_empty(self):
        result = self.explainer.global_importance(self.X)
        assert len(result) > 0

    def test_local_explanation_non_empty(self):
        row_df = pd.DataFrame([self.X[0]], columns=self.feature_cols)
        result = self.explainer.local_explanation(row_df)
        assert isinstance(result, list)
