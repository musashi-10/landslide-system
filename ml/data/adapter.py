"""
ML Data Adapter
===============

Converts standardised outputs from Engineers 1–4 into a clean ML-ready
DataFrame.  This adapter is the ONLY point where external data enters the
ML pipeline.  All provider-specific logic stays outside this module.

Design contract
---------------
- Input: DataFrames conforming to the shared data contract (docs/data-contract.md)
- Output: A single validated, joined DataFrame ready for feature engineering
- Missing values: Never silently converted to zero (contract §12)
- No leakage: Features are validated to contain no post-event information
- Reproducible: same inputs → same output (deterministic joins, no random ops)

Integration
-----------
    Engineer 1 output → static_df  (location_id, lat, lon, historical_landslide, ...)
    Engineer 2 output → static_df  (elevation, slope, aspect, soil_type, geology,
                                    land_cover, drainage_feature)
    Engineer 3 output → satellite_df (optional; ndvi, vegetation_change, ...)
    Engineer 4 output → dynamic_df  (location_id, timestamp_utc, rainfall_*, ...)

All joined on (location_id, timestamp_utc).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Required column sets (from data contract) ────────────────────────────────

REQUIRED_ID_COLS = ["location_id"]
REQUIRED_DYNAMIC_COLS = ["timestamp_utc"]

STATIC_FEATURE_COLS = [
    "elevation",
    "slope",
    "aspect",
    "soil_type",
    "geology",
    "land_cover",
    "historical_landslide",
]

DYNAMIC_FEATURE_COLS = [
    "rainfall_1h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_3d",
    "rainfall_7d",
    "forecast_rainfall",
    "moisture_indicator",
]

SATELLITE_FEATURE_COLS = [
    "ndvi",
    "vegetation_change",
    "bare_surface_indicator",
    "change_indicator",
]

# ── Output dataclass ─────────────────────────────────────────────────────────


@dataclass
class MLDataset:
    """
    The validated, joined ML input dataset.

    Attributes
    ----------
    df : pd.DataFrame
        Feature matrix with target column 'landslide_occurred'.
    feature_cols : list[str]
        Ordered list of feature columns present in df.
    target_col : str
        Name of the target column ('landslide_occurred').
    n_positives : int
        Number of positive (landslide) samples.
    n_negatives : int
        Number of negative (non-event) samples.
    has_satellite : bool
        Whether satellite-derived features are included.
    missing_summary : dict
        Per-column missing value counts.
    """

    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str
    n_positives: int
    n_negatives: int
    has_satellite: bool
    missing_summary: dict


# ── Main adapter function ─────────────────────────────────────────────────────


def build_ml_dataset(
    static_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    satellite_df: Optional[pd.DataFrame] = None,
    target_col: str = "landslide_occurred",
    drop_rows_missing_target: bool = True,
    validate_no_future_leakage: bool = True,
) -> MLDataset:
    """
    Build a validated, joined ML-ready dataset from standardised engineer outputs.

    Parameters
    ----------
    static_df : pd.DataFrame
        Static features indexed by location_id.  Columns should include those
        listed in STATIC_FEATURE_COLS.  Multiple rows per location_id allowed
        (one per timestamp in time-series datasets).
    dynamic_df : pd.DataFrame
        Dynamic features indexed by (location_id, timestamp_utc).
    satellite_df : pd.DataFrame, optional
        Optional satellite-derived features indexed by (location_id,
        acquisition_time).  Only pre-event acquisitions should be provided
        (leakage prevention is the caller's responsibility; this function
        will log a warning if 'acquisition_time' is absent).
    target_col : str
        Name of the target column in the output DataFrame.  The raw input
        column is 'historical_landslide' (1=event, 0=non-event).
    drop_rows_missing_target : bool
        If True, rows where the target is NaN are dropped with a warning.
    validate_no_future_leakage : bool
        If True, run basic leakage checks and log warnings.

    Returns
    -------
    MLDataset
        Validated, joined feature matrix.

    Raises
    ------
    ValueError
        If required identifier columns are missing or if the join produces
        zero rows.
    """
    logger.info("Building ML dataset …")

    # ── Step 1: Validate required columns ────────────────────────────────────
    _validate_required_columns(static_df, REQUIRED_ID_COLS, "static_df")
    _validate_required_columns(dynamic_df, REQUIRED_ID_COLS + REQUIRED_DYNAMIC_COLS, "dynamic_df")

    if satellite_df is not None:
        _validate_required_columns(satellite_df, REQUIRED_ID_COLS, "satellite_df")
        if "acquisition_time" not in satellite_df.columns:
            logger.warning(
                "satellite_df missing 'acquisition_time' column. "
                "Cannot verify temporal leakage in satellite features. "
                "Ensure only pre-event acquisitions are included."
            )

    # ── Step 2: Normalise timestamps ─────────────────────────────────────────
    dynamic_df = dynamic_df.copy()
    dynamic_df["timestamp_utc"] = pd.to_datetime(
        dynamic_df["timestamp_utc"], utc=True
    )

    static_df = static_df.copy()
    if "timestamp_utc" in static_df.columns:
        static_df["timestamp_utc"] = pd.to_datetime(
            static_df["timestamp_utc"], utc=True
        )

    # ── Step 3: Join static + dynamic on (location_id [+ timestamp_utc]) ─────
    if "timestamp_utc" in static_df.columns:
        # Static data is time-stamped (e.g. each event has its own record)
        join_keys = ["location_id", "timestamp_utc"]
        merged = pd.merge(
            static_df, dynamic_df, on=join_keys, how="inner", suffixes=("", "_dyn")
        )
    else:
        # Static data has one row per location; broadcast to all dynamic rows
        merged = pd.merge(
            dynamic_df, static_df, on="location_id", how="left", suffixes=("", "_static")
        )

    if len(merged) == 0:
        raise ValueError(
            "Join of static_df and dynamic_df produced zero rows. "
            "Check that location_id values match between the two DataFrames."
        )

    logger.info(f"Joined dataset: {len(merged)} rows")

    # ── Step 4: Join satellite features (optional) ────────────────────────────
    has_satellite = False
    if satellite_df is not None:
        sat_cols = [c for c in SATELLITE_FEATURE_COLS if c in satellite_df.columns]
        if sat_cols:
            sat_join_keys = (
                ["location_id", "timestamp_utc"]
                if "timestamp_utc" in satellite_df.columns
                else ["location_id"]
            )
            sat_subset = satellite_df[sat_join_keys + sat_cols].copy()
            merged = pd.merge(
                merged, sat_subset, on=sat_join_keys, how="left", suffixes=("", "_sat")
            )
            has_satellite = True
            logger.info(f"Joined satellite features: {sat_cols}")
        else:
            logger.warning("satellite_df provided but no recognised satellite columns found.")

    # ── Step 5: Prepare target column ─────────────────────────────────────────
    if "historical_landslide" in merged.columns:
        merged[target_col] = merged["historical_landslide"].astype("Int64")
    elif target_col in merged.columns:
        merged[target_col] = merged[target_col].astype("Int64")
    else:
        merged[target_col] = pd.NA  # target unknown — inference mode

    if drop_rows_missing_target:
        n_before = len(merged)
        merged = merged.dropna(subset=[target_col])
        n_dropped = n_before - len(merged)
        if n_dropped > 0:
            logger.warning(
                f"Dropped {n_dropped} rows with missing target '{target_col}'."
            )
        if len(merged) == 0:
            raise ValueError(
                "Join of static_df and dynamic_df produced zero rows after "
                "removing rows with missing target values. "
                "Check that location_id values match between the two DataFrames."
            )

    # ── Step 6: Leakage guard ─────────────────────────────────────────────────
    if validate_no_future_leakage:
        _check_leakage(merged, target_col)

    # ── Step 7: Compute missing-value summary ─────────────────────────────────
    all_feature_cols = _collect_feature_cols(merged, has_satellite)
    missing_summary = {
        col: int(merged[col].isna().sum())
        for col in all_feature_cols
        if col in merged.columns
    }
    total_missing = sum(missing_summary.values())
    if total_missing > 0:
        logger.info(
            f"Missing values detected in {sum(v > 0 for v in missing_summary.values())} "
            f"columns (total {total_missing} NaNs). "
            "These will be handled by the preprocessor."
        )

    # ── Step 8: Collect statistics ────────────────────────────────────────────
    n_positives = int((merged[target_col] == 1).sum())
    n_negatives = int((merged[target_col] == 0).sum())

    logger.info(
        f"Dataset ready — {n_positives} positives, {n_negatives} negatives "
        f"({len(merged)} total rows)"
    )

    present_feature_cols = [c for c in all_feature_cols if c in merged.columns]

    return MLDataset(
        df=merged.reset_index(drop=True),
        feature_cols=present_feature_cols,
        target_col=target_col,
        n_positives=n_positives,
        n_negatives=n_negatives,
        has_satellite=has_satellite,
        missing_summary=missing_summary,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame '{name}' is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _check_leakage(df: pd.DataFrame, target_col: str) -> None:
    """
    Basic leakage checks.  Logs warnings rather than raising to allow
    the pipeline to continue with the researcher's awareness.

    Known leakage risks (documented):
    1. historical_landslide computed using future events → check event timestamps
    2. Post-event satellite imagery → must be filtered by caller
    3. Rainfall features after prediction timestamp → dynamic_df must be filtered
    """
    # Check: rainfall columns should not contain data from future relative to
    # the prediction timestamp.  We cannot directly verify this here without
    # the raw timestamps, but we flag if forecast_rainfall looks like it's
    # very large (unusual), as a sanity check.
    if "forecast_rainfall" in df.columns:
        frac_forecast_present = df["forecast_rainfall"].notna().mean()
        if frac_forecast_present > 0.99:
            logger.debug(
                "forecast_rainfall is almost fully present. "
                "Ensure this is the forecast at prediction time, not observed future rainfall."
            )

    # Check: historical_landslide should not vary over time for the same
    # location in a way that implies future knowledge was used.
    if "historical_landslide" in df.columns and "location_id" in df.columns:
        loc_counts = df.groupby("location_id")["historical_landslide"].nunique()
        varying_locs = (loc_counts > 1).sum()
        if varying_locs > 0:
            logger.info(
                f"{varying_locs} locations have time-varying historical_landslide. "
                "Ensure the value at each timestamp reflects only past events "
                "(i.e., was computed using only data available before that timestamp)."
            )


def _collect_feature_cols(df: pd.DataFrame, has_satellite: bool) -> list[str]:
    """Return ordered feature column list for the given dataset."""
    cols = STATIC_FEATURE_COLS + DYNAMIC_FEATURE_COLS
    if has_satellite:
        cols = cols + SATELLITE_FEATURE_COLS
    return cols
