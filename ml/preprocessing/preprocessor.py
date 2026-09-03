"""
Landslide Preprocessor
======================

Fits on training data; transforms train, validation, test, and inference data.

Components
----------
1. Categorical encoder — ordinal encoding with unknown-category handling
2. Numeric imputer — median imputation (documented, not silent zero-fill)
3. Numeric scaler — StandardScaler for logistic regression compatibility

Important: fit() must be called ONLY on training data.
           transform() may be called on any split.

Missing value policy
--------------------
Per data contract §12, missing values are NEVER silently converted to zero.
Instead: median imputation is applied with the median computed from the
training set only.  The imputation strategy is logged and saved in the
artifact metadata.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

_UNKNOWN_CATEGORY = "__UNKNOWN__"


class LandslidePreprocessor:
    """
    Fit-transform preprocessor for ML feature matrices.

    Usage
    -----
        pp = LandslidePreprocessor(
            numeric_cols=[...],
            categorical_cols=[...],
        )
        X_train = pp.fit_transform(train_df)
        X_val   = pp.transform(val_df)
        X_test  = pp.transform(test_df)

    Attributes
    ----------
    numeric_cols : list[str]
        Numeric feature columns to impute and scale.
    categorical_cols : list[str]
        Categorical feature columns to encode.
    scale_numeric : bool
        Whether to apply StandardScaler to numeric columns.
        Use True for Logistic Regression; may be False for tree models.
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        scale_numeric: bool = True,
        version: str = "1.0.0",
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scale_numeric = scale_numeric
        self.version = version
        self._is_fitted = False

        # Internal sklearn components — set in fit()
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._encoder: Optional[OrdinalEncoder] = None
        self._imputed_medians: dict[str, float] = {}
        self._category_maps: dict[str, list] = {}
        self._output_cols: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "LandslidePreprocessor":
        """
        Fit the preprocessor on training data.

        Must be called before transform().
        Must NOT be called on validation/test/inference data.
        """
        num_cols = [c for c in self.numeric_cols if c in df.columns]
        cat_cols = [c for c in self.categorical_cols if c in df.columns]

        # ── Categorical encoder ───────────────────────────────────────────────
        if cat_cols:
            # Fill NaN with a sentinel before encoding
            cat_data = df[cat_cols].fillna(_UNKNOWN_CATEGORY).astype(str)
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self._encoder.fit(cat_data)
            self._category_maps = {
                col: list(cats)
                for col, cats in zip(cat_cols, self._encoder.categories_)
            }

        # ── Numeric imputer (per-column) ──────────────────────────────────────
        # We impute column-by-column so that an all-NaN column during training
        # produces NaN in the output rather than crashing (sklearn silently drops
        # all-NaN columns when all imputers are batched together).
        if num_cols:
            self._col_imputers: dict[str, Optional[SimpleImputer]] = {}
            self._imputed_medians: dict[str, float] = {}
            for col in num_cols:
                col_data = df[[col]].astype(float)
                if col_data[col].notna().any():
                    imp = SimpleImputer(strategy="median")
                    imp.fit(col_data)
                    self._col_imputers[col] = imp
                    self._imputed_medians[col] = float(imp.statistics_[0])
                else:
                    # Entirely NaN column — cannot impute; leave as NaN
                    self._col_imputers[col] = None
                    self._imputed_medians[col] = float("nan")
                    logger.warning(
                        f"Column '{col}' is entirely NaN in training data. "
                        "It will remain NaN after imputation."
                    )
            # Keep a dummy _imputer for pickle compatibility
            self._imputer = None
            logger.info(
                "Numeric imputer fitted (strategy=median, per-column). "
                f"Medians: { {k: round(v, 4) for k, v in self._imputed_medians.items() if not __import__('math').isnan(v)} }"
            )

        # ── Scaler ────────────────────────────────────────────────────────────
        if num_cols and self.scale_numeric:
            # Build imputed data for fitting the scaler
            num_imputed_data = df[num_cols].astype(float).copy()
            for col in num_cols:
                imp = self._col_imputers.get(col)
                if imp is not None:
                    num_imputed_data[[col]] = imp.transform(num_imputed_data[[col]])
            self._scaler = StandardScaler()
            self._scaler.fit(num_imputed_data)

        self._fitted_num_cols = num_cols
        self._fitted_cat_cols = cat_cols
        self._output_cols = num_cols + cat_cols
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame using fitted parameters.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Processed feature matrix with the same column order as training.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        out = df.copy()

        # ── Categorical ───────────────────────────────────────────────────────
        cat_cols = self._fitted_cat_cols
        if cat_cols and self._encoder is not None:
            cat_data = out[cat_cols].fillna(_UNKNOWN_CATEGORY).astype(str)
            encoded = self._encoder.transform(cat_data)
            for i, col in enumerate(cat_cols):
                out[col] = encoded[:, i]

        # ── Numeric imputation (per-column) ───────────────────────────────────
        num_cols = self._fitted_num_cols
        col_imputers = getattr(self, "_col_imputers", {})
        if num_cols:
            for col in num_cols:
                # Graceful degradation: add column as NaN if absent at inference
                if col not in out.columns:
                    logger.warning(f"Column '{col}' absent during transform; filling with median.")
                    out[col] = np.nan
                out[col] = out[col].astype(float)
                imp = col_imputers.get(col)
                if imp is not None:
                    out[[col]] = imp.transform(out[[col]])

        # ── Scaling ───────────────────────────────────────────────────────────
        if num_cols and self.scale_numeric and self._scaler is not None:
            scaled = self._scaler.transform(out[num_cols])
            for i, col in enumerate(num_cols):
                out[col] = scaled[:, i]

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform in one call (for training data only)."""
        return self.fit(df).transform(df)

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return numpy array of (numeric + categorical) columns."""
        transformed = self.transform(df)
        all_cols = self._fitted_num_cols + self._fitted_cat_cols
        present = [c for c in all_cols if c in transformed.columns]
        return transformed[present].values.astype(float)

    def output_columns(self) -> list[str]:
        """Return the ordered list of output feature columns."""
        return list(self._output_cols)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> Path:
        """Pickle the fitted preprocessor to path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved → {p}")
        return p

    @classmethod
    def load(cls, path: str) -> "LandslidePreprocessor":
        """Load a pickled preprocessor from path."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LandslidePreprocessor, got {type(obj)}")
        return obj

    def metadata(self) -> dict:
        """Return preprocessing metadata for model artifact."""
        return {
            "version": self.version,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "scale_numeric": self.scale_numeric,
            "imputation_strategy": "median",
            "imputed_medians": self._imputed_medians,
            "category_maps": self._category_maps,
        }
