"""
Feature Engineering
===================

Reproducible, justified feature construction for the landslide risk model.

Design principles
-----------------
- Only features with a physical / scientific justification are created.
- No feature is created "just in case" — each one is documented below.
- All operations are deterministic: same input → same output.
- Missing values are propagated as NaN, never silently filled with zero.
- The FeatureEngineer class is stateless: it holds no fit state, so it can
  be applied identically to train and inference data.

Engineered features
-------------------
1. rainfall_intensity
   Definition: rainfall_1h / (rainfall_6h / 6 + ε)
   Rationale: Captures how concentrated recent rainfall is relative to the
   6-hour average rate.  A sudden intense burst on a steep slope is more
   likely to trigger a landslide than the same 6-hour total spread evenly.

2. antecedent_rainfall
   Definition: rainfall_3d + rainfall_7d
   Rationale: Antecedent moisture is a well-documented landslide trigger
   (Caine 1980; Guzzetti et al. 2008).  Saturated soil provides less
   resistance even at moderate rainfall intensities.

3. static_susceptibility_score
   Definition: 0.4 × slope_norm + 0.3 × elevation_norm + 0.3 × historical_landslide
   Rationale: A composite of the three strongest static predictors, providing
   a single "how landslide-prone is this place?" feature.  Weights are
   heuristic starting values; they should be reviewed after model training.
   ⚠️  These weights are not scientifically validated — they are provisional.

4. rain_x_slope
   Definition: rainfall_24h × slope
   Rationale: The interaction between rainfall loading and slope angle is the
   core physical mechanism of shallow translational landslides.  Explicitly
   encoding this interaction helps tree-based models find the threshold.

5. rain_x_susceptibility
   Definition: rainfall_24h × static_susceptibility_score
   Rationale: Generalises rain_x_slope to account for the full static context.
   A highly susceptible location needs less rainfall to fail than a stable one.

Features deliberately NOT created
----------------------------------
- rainfall_7d / rainfall_3d ratio: collinear with antecedent_rainfall, minimal
  additional information over the individual columns.
- NDVI × slope: satellite features are optional; interaction terms with optional
  features would break the base model.  Satellite interaction features can be
  added in the enhanced model configuration.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPSILON = 1e-6  # prevent division by zero


class FeatureEngineer:
    """
    Stateless feature engineering transformer.

    Methods
    -------
    transform(df) → pd.DataFrame
        Add engineered columns to a copy of df.  Does not modify input.

    engineered_column_names() → list[str]
        Return the list of columns this class will add.
    """

    ENGINEERED_COLS = [
        "rainfall_intensity",
        "antecedent_rainfall",
        "static_susceptibility_score",
        "rain_x_slope",
        "rain_x_susceptibility",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to a copy of df.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame from the data adapter.  Must contain at least some
            of the raw feature columns; missing inputs produce NaN outputs.

        Returns
        -------
        pd.DataFrame
            Copy of df with additional engineered columns appended.
        """
        out = df.copy()

        # 1. Rainfall intensity
        if "rainfall_1h" in out.columns and "rainfall_6h" in out.columns:
            hourly_rate_6h = out["rainfall_6h"] / 6.0
            out["rainfall_intensity"] = out["rainfall_1h"] / (
                hourly_rate_6h + _EPSILON
            )
        else:
            out["rainfall_intensity"] = np.nan
            logger.debug("rainfall_intensity set to NaN: missing rainfall_1h or rainfall_6h")

        # 2. Antecedent rainfall
        if "rainfall_3d" in out.columns and "rainfall_7d" in out.columns:
            out["antecedent_rainfall"] = out["rainfall_3d"] + out["rainfall_7d"]
        else:
            out["antecedent_rainfall"] = np.nan
            logger.debug("antecedent_rainfall set to NaN: missing rainfall_3d or rainfall_7d")

        # 3. Static susceptibility score
        out = _add_susceptibility_score(out)

        # 4. rain_x_slope
        if "rainfall_24h" in out.columns and "slope" in out.columns:
            out["rain_x_slope"] = out["rainfall_24h"] * out["slope"]
        else:
            out["rain_x_slope"] = np.nan
            logger.debug("rain_x_slope set to NaN: missing rainfall_24h or slope")

        # 5. rain_x_susceptibility
        if "rainfall_24h" in out.columns and "static_susceptibility_score" in out.columns:
            out["rain_x_susceptibility"] = (
                out["rainfall_24h"] * out["static_susceptibility_score"]
            )
        else:
            out["rain_x_susceptibility"] = np.nan
            logger.debug("rain_x_susceptibility set to NaN: missing prerequisite features")

        return out

    def engineered_column_names(self) -> list[str]:
        """Return the list of column names added by transform()."""
        return list(self.ENGINEERED_COLS)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: apply FeatureEngineer.transform() to df.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        df with engineered columns added.
    """
    return FeatureEngineer().transform(df)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _add_susceptibility_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute static_susceptibility_score.

    Score = 0.4 × slope_norm + 0.3 × elevation_norm + 0.3 × historical_landslide

    Normalisations are min-max within the provided batch.

    ⚠️  These weights are provisional (not scientifically validated).
    They should be reviewed after feature importance analysis.
    """
    has_slope = "slope" in df.columns
    has_elevation = "elevation" in df.columns
    has_historical = "historical_landslide" in df.columns

    if not any([has_slope, has_elevation, has_historical]):
        df["static_susceptibility_score"] = np.nan
        return df

    score = pd.Series(0.0, index=df.index)

    if has_slope:
        score += 0.4 * _minmax_norm(df["slope"])

    if has_elevation:
        # Higher elevation → generally more susceptible in mountain terrain
        score += 0.3 * _minmax_norm(df["elevation"])

    if has_historical:
        hl = df["historical_landslide"].fillna(0).astype(float)
        score += 0.3 * hl.clip(0, 1)

    df["static_susceptibility_score"] = score
    return df


def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]; returns 0.5 if constant."""
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)
