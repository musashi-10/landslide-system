"""
Spatial / Temporal Validation Strategy
=======================================

WHY NOT RANDOM TRAIN/TEST SPLIT?
─────────────────────────────────
Landslide data has two forms of autocorrelation that invalidate random splits:

1. SPATIAL autocorrelation: Nearby grid cells share the same terrain, geology,
   and rainfall exposure.  A random split would place highly similar cells in
   both train and test — artificially inflating performance metrics.

2. TEMPORAL autocorrelation: Consecutive time steps at the same location share
   the same antecedent conditions.  A random split would allow the model to
   "see" future-adjacent conditions during training.

The validation strategy here tests whether the model generalises to:
- UNSEEN TIME PERIODS (temporal holdout)
- UNSEEN SPATIAL REGIONS (spatial block holdout)

CHOSEN STRATEGY
───────────────
Primary: Temporal holdout
  - Sort all samples by timestamp_utc.
  - Hold out the last 20% of the timeline as the test set.
  - Hold out the preceding 10% as the validation set.
  - Train on the earliest 70%.
  Rationale: Simulates operational deployment — the model is trained on
  historical data and evaluated on future events it has never seen.
  This is the most realistic test of operational utility.

Secondary: Spatial block cross-validation (for hyperparameter tuning)
  - Divide the study area into N geographic blocks (strips by latitude).
  - In each fold, one block is held out for validation.
  - This tests whether the model generalises across terrain types.
  Rationale: Prevents learning location-specific patterns that won't
  transfer to new areas.

Combined approach: Temporal holdout for final test evaluation;
spatial block CV for model comparison and hyperparameter tuning.

KNOWN LIMITATION
────────────────
If the dataset covers only a short time period or a small area, either
strategy may produce very small test sets.  The dataset statistics module
reports this before training begins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationStrategy(str, Enum):
    """Available validation strategies."""

    TEMPORAL = "temporal"
    SPATIAL_BLOCK = "spatial_block"
    TEMPORAL_THEN_SPATIAL = "temporal_then_spatial"  # recommended


@dataclass
class SplitResult:
    """Indices for one train/val/test split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def summary(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


class SpatioTemporalSplitter:
    """
    Produces spatially- and temporally-honest train/val/test splits.

    Parameters
    ----------
    strategy : ValidationStrategy
        Which splitting strategy to use.
    temporal_test_fraction : float
        Fraction of the timeline to reserve for testing (most recent data).
    temporal_val_fraction : float
        Fraction of the timeline (from the end of training) for validation.
    n_spatial_folds : int
        Number of spatial folds for spatial block CV.
    timestamp_col : str
        Column name for timestamps.
    latitude_col : str
        Column name for latitude (used in spatial blocking).
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        strategy: ValidationStrategy = ValidationStrategy.TEMPORAL_THEN_SPATIAL,
        temporal_test_fraction: float = 0.20,
        temporal_val_fraction: float = 0.10,
        n_spatial_folds: int = 5,
        timestamp_col: str = "timestamp_utc",
        latitude_col: str = "latitude",
        random_seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.temporal_test_fraction = temporal_test_fraction
        self.temporal_val_fraction = temporal_val_fraction
        self.n_spatial_folds = n_spatial_folds
        self.timestamp_col = timestamp_col
        self.latitude_col = latitude_col
        self.random_seed = random_seed

    def split(self, df: pd.DataFrame) -> SplitResult:
        """
        Produce a single train/val/test split using the configured strategy.

        This is the primary split used for final model evaluation.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with timestamp_utc (and latitude if spatial strategy).

        Returns
        -------
        SplitResult
        """
        if self.strategy in (ValidationStrategy.TEMPORAL, ValidationStrategy.TEMPORAL_THEN_SPATIAL):
            return self._temporal_split(df)
        else:
            raise ValueError(
                f"Strategy {self.strategy} requires spatial_folds(), not split()."
            )

    def _temporal_split(self, df: pd.DataFrame) -> SplitResult:
        """Sort by time; hold out last fraction as test, preceding as val."""
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Column '{self.timestamp_col}' not found for temporal splitting. "
                f"Available: {list(df.columns)}"
            )

        ts = pd.to_datetime(df[self.timestamp_col], utc=True)
        sorted_idx = np.argsort(ts.values)
        n = len(sorted_idx)

        n_test = max(1, int(n * self.temporal_test_fraction))
        n_val = max(1, int(n * self.temporal_val_fraction))
        n_train = n - n_test - n_val

        if n_train <= 0:
            raise ValueError(
                f"Dataset too small for temporal split: {n} rows, "
                f"test={n_test}, val={n_val}, train would be {n_train}."
            )

        train_idx = sorted_idx[:n_train]
        val_idx = sorted_idx[n_train: n_train + n_val]
        test_idx = sorted_idx[n_train + n_val:]

        logger.info(
            f"Temporal split — train: {len(train_idx)}, "
            f"val: {len(val_idx)}, test: {len(test_idx)}"
        )
        _log_class_balance(df, "train", train_idx)
        _log_class_balance(df, "test", test_idx)

        return SplitResult(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

    def spatial_folds(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, val_idx) pairs for spatial block cross-validation.

        The study area is divided into N latitude strips.  Each strip is held
        out in turn.  This ensures the model is evaluated on terrain it has
        not seen during training.

        Yields
        ------
        (train_idx, val_idx) : tuple of np.ndarray
        """
        if self.latitude_col not in df.columns:
            raise ValueError(
                f"Column '{self.latitude_col}' not found for spatial CV. "
                f"Available: {list(df.columns)}"
            )

        lat = df[self.latitude_col].values
        lat_min, lat_max = lat.min(), lat.max()
        boundaries = np.linspace(lat_min, lat_max, self.n_spatial_folds + 1)

        for i in range(self.n_spatial_folds):
            lo, hi = boundaries[i], boundaries[i + 1]
            in_fold = (lat >= lo) & (lat < hi if i < self.n_spatial_folds - 1 else lat <= hi)
            val_idx = np.where(in_fold)[0]
            train_idx = np.where(~in_fold)[0]

            if len(val_idx) == 0:
                logger.warning(f"Spatial fold {i} is empty — no samples in latitude [{lo:.3f}, {hi:.3f}]")
                continue

            logger.info(
                f"Spatial fold {i + 1}/{self.n_spatial_folds} "
                f"lat [{lo:.3f}, {hi:.3f}] — train: {len(train_idx)}, val: {len(val_idx)}"
            )
            yield train_idx, val_idx

    def describe(self) -> dict:
        """Return strategy metadata for logging and artifact saving."""
        return {
            "strategy": self.strategy.value,
            "temporal_test_fraction": self.temporal_test_fraction,
            "temporal_val_fraction": self.temporal_val_fraction,
            "n_spatial_folds": self.n_spatial_folds,
            "rationale": (
                "Temporal holdout prevents future-data leakage. "
                "Spatial block CV prevents spatial autocorrelation from "
                "inflating cross-validation scores."
            ),
        }


def _log_class_balance(df: pd.DataFrame, split_name: str, idx: np.ndarray) -> None:
    """Log positive/negative balance for a data split."""
    target_col = next(
        (c for c in ["landslide_occurred", "historical_landslide"] if c in df.columns),
        None,
    )
    if target_col is None:
        return
    subset = df.iloc[idx][target_col]
    n_pos = int((subset == 1).sum())
    n_neg = int((subset == 0).sum())
    logger.info(f"  {split_name} — pos: {n_pos}, neg: {n_neg}")
