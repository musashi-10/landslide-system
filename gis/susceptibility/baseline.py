"""
Baseline susceptibility computation.

WARNING
-------
The weights used here are NOT scientifically validated.
They are configurable heuristic placeholders used for:
  * Visual sanity checks during development
  * Providing Engineer 5 (ML) a simple non-trivial baseline to beat
  * Populating the risk-map dashboard for integration testing

The final operational model belongs entirely to Engineer 5.

Algorithm
---------
1. Normalise each numeric conditioning factor to [0, 1] using min-max
   scaling across the entire grid.
2. Encode categorical factors to a numeric proxy (land_cover, soil_type,
   geology) using configurable lookup tables.  Unknown categories map to 0.5.
3. Weight-sum the normalised factors using ``GISSettings.susceptibility_weights``.
4. Clip the result to [0, 1].

Missing values are excluded from the normalisation denominator; cells with
no valid data receive NaN susceptibility (never silently 0).
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Categorical proxy encodings
# These map textual categories to a numeric vulnerability proxy [0, 1].
# They are explicitly documented and configurable.
# --------------------------------------------------------------------------- #

LAND_COVER_PROXY: Dict[str, float] = {
    "bare":        1.0,   # highest susceptibility (no root reinforcement)
    "shrub":       0.7,
    "grassland":   0.6,
    "forest":      0.3,   # lowest (root reinforcement, canopy interception)
    "cropland":    0.5,
    "urban":       0.4,
    "water":       0.0,
}

SOIL_TYPE_PROXY: Dict[str, float] = {
    "clay":        0.8,
    "silt":        0.7,
    "loam":        0.5,
    "sandy_loam":  0.4,
    "sand":        0.3,
    "rock":        0.1,
}

GEOLOGY_PROXY: Dict[str, float] = {
    "schist":      0.9,
    "phyllite":    0.85,
    "shale":       0.8,
    "sandstone":   0.6,
    "limestone":   0.5,
    "granite":     0.2,
    "gneiss":      0.3,
}


def _normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; return NaN where input is NaN."""
    lo = series.min(skipna=True)
    hi = series.max(skipna=True)
    if hi == lo:
        # Constant column – assign 0.5 to all non-NaN values
        result = series.copy()
        result[series.notna()] = 0.5
        return result
    return (series - lo) / (hi - lo)


def _encode_categorical(
    series: pd.Series,
    proxy_table: Dict[str, float],
    default: float = 0.5,
) -> pd.Series:
    """Map string categories to float using *proxy_table*."""
    return series.map(lambda v: proxy_table.get(str(v).lower(), default) if pd.notna(v) else np.nan)


def compute_baseline_susceptibility(
    features: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute a weighted-sum baseline susceptibility score per grid cell.

    Parameters
    ----------
    features : DataFrame containing static feature columns.
    weights  : Dict mapping feature name → weight (should sum to 1.0).

    Returns
    -------
    pd.Series named ``susceptibility`` with values in [0, 1] or NaN.
    """
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=0.01):
        logger.warning(
            "Susceptibility weights sum to %.4f, not 1.0.  "
            "Scores will be scaled accordingly.",
            total_weight,
        )

    weighted_sum = pd.Series(0.0, index=features.index)
    effective_weight = pd.Series(0.0, index=features.index)

    for feature, weight in weights.items():
        if feature not in features.columns:
            logger.debug("Weight defined for '%s' but column absent; skipping.", feature)
            continue

        raw = features[feature].copy()

        # Encode categoricals
        if feature == "land_cover":
            encoded = _encode_categorical(raw, LAND_COVER_PROXY)
        elif feature == "soil_type":
            encoded = _encode_categorical(raw, SOIL_TYPE_PROXY)
        elif feature == "geology":
            encoded = _encode_categorical(raw, GEOLOGY_PROXY)
        elif feature == "historical_landslide":
            # Binary 0/1 presence (or count clamped to 1)
            encoded = (raw > 0).astype(float)
            encoded[raw.isna()] = np.nan
        else:
            # Numeric – normalise
            encoded = _normalise(raw.astype(float))

        valid_mask = encoded.notna()
        weighted_sum[valid_mask]    += encoded[valid_mask] * weight
        effective_weight[valid_mask] += weight

    # Avoid division by zero for cells with no valid features
    result = pd.Series(np.nan, index=features.index, name="susceptibility")
    valid = effective_weight > 0
    result[valid] = (weighted_sum[valid] / effective_weight[valid]).clip(0.0, 1.0)

    n_valid = valid.sum()
    n_nan   = (~valid).sum()
    logger.info(
        "Baseline susceptibility: %d valid cells, %d NaN cells. "
        "Min=%.3f, Max=%.3f, Mean=%.3f",
        n_valid, n_nan,
        result.min(skipna=True),
        result.max(skipna=True),
        result.mean(skipna=True),
    )
    return result
