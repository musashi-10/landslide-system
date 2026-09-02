"""
Data quality pipeline.

Runs all validation checks over a raw DataFrame and returns:
  * a cleaned DataFrame (records are never silently dropped)
  * a :class:`~data_engineering.schemas.ValidationReport`

The pipeline marks invalid records with a ``_valid`` boolean column and
populates the ``is_duplicate`` column.  Callers decide whether to filter
or retain invalid rows.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data_engineering.schemas import ValidationReport
from data_engineering.validation.coordinate_validator import (
    is_valid_coordinate_pair,
)
from data_engineering.validation.timestamp_validator import normalize_timestamp, TimestampError
from data_engineering.validation.duplicate_detector import detect_duplicates

logger = logging.getLogger(__name__)


def run_quality_pipeline(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "timestamp_utc",
    required_feature_cols: Optional[list[str]] = None,
    coord_precision: int = 4,
    reject_null_island: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Run the full data-quality pipeline on a raw DataFrame.

    Steps performed (in order):
      1. Coordinate presence and range validation
      2. Timestamp normalisation and validation
      3. Missing required feature detection
      4. Duplicate detection

    Records are annotated with ``_valid`` and ``is_duplicate`` columns.
    Missing values are **never** silently replaced with zero.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    lat_col, lon_col : str
        Column names for latitude and longitude.
    time_col : str
        Column name for the timestamp field.
    required_feature_cols : list[str] | None
        Additional columns that must not be all-None/NaN for a record to
        be considered complete.  Pass ``None`` to skip feature checks.
    coord_precision : int
        Decimal places for duplicate coordinate comparison.
    reject_null_island : bool
        Treat exact (0.0, 0.0) as invalid.

    Returns
    -------
    tuple[pd.DataFrame, ValidationReport]
        * Annotated DataFrame with ``_valid``, ``_invalid_reason``, and
          ``is_duplicate`` columns added.
        * :class:`~data_engineering.schemas.ValidationReport` summary.
    """
    report = ValidationReport(records_total=len(df))
    result = df.copy()

    # Initialise annotation columns
    result["_valid"] = True
    result["_invalid_reason"] = None

    def _mark_invalid(idx: int, reason: str) -> None:
        result.at[idx, "_valid"] = False
        existing = result.at[idx, "_invalid_reason"]
        if existing:
            result.at[idx, "_invalid_reason"] = f"{existing}; {reason}"
        else:
            result.at[idx, "_invalid_reason"] = reason

    # ------------------------------------------------------------------ #
    # 1. Coordinate validation
    # ------------------------------------------------------------------ #
    for idx in result.index:
        lat = result.at[idx, lat_col] if lat_col in result.columns else None
        lon = result.at[idx, lon_col] if lon_col in result.columns else None

        # Missing
        if pd.isna(lat) or pd.isna(lon) or lat is None or lon is None:
            _mark_invalid(idx, "missing_coordinates")
            report.missing_coordinates += 1
            continue

        # Out-of-range / type issues
        if not is_valid_coordinate_pair(lat, lon, reject_null_island=reject_null_island):
            _mark_invalid(idx, "invalid_coordinates")
            report.invalid_coordinates += 1

    # ------------------------------------------------------------------ #
    # 2. Timestamp normalisation
    # ------------------------------------------------------------------ #
    normalised_times: list[Optional[str]] = []
    for idx in result.index:
        raw_ts = result.at[idx, time_col] if time_col in result.columns else None
        try:
            norm = normalize_timestamp(raw_ts)
            normalised_times.append(norm)
        except TimestampError as exc:
            normalised_times.append(None)
            _mark_invalid(idx, "missing_or_invalid_timestamp")
            report.missing_timestamps += 1
            logger.debug("Row %d: timestamp error: %s", idx, exc)

    result[time_col] = normalised_times

    # ------------------------------------------------------------------ #
    # 3. Required feature checks
    # ------------------------------------------------------------------ #
    if required_feature_cols:
        for idx in result.index:
            for col in required_feature_cols:
                val = result.at[idx, col] if col in result.columns else None
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    _mark_invalid(idx, f"missing_feature:{col}")
                    report.missing_feature_values += 1
                    break  # one flag per row

    # ------------------------------------------------------------------ #
    # 4. Duplicate detection (only over valid rows with real timestamps)
    # ------------------------------------------------------------------ #
    valid_mask = result["_valid"]
    if valid_mask.any():
        valid_subset = result.loc[valid_mask].copy()
        valid_subset = detect_duplicates(
            valid_subset,
            lat_col=lat_col,
            lon_col=lon_col,
            time_col=time_col,
            coord_precision=coord_precision,
        )
        result.loc[valid_mask, "is_duplicate"] = valid_subset["is_duplicate"].values
        report.duplicate_records = int(result.get("is_duplicate", pd.Series([])).sum())
    else:
        result["is_duplicate"] = False
        report.duplicate_records = 0

    # ------------------------------------------------------------------ #
    # Final counts
    # ------------------------------------------------------------------ #
    report.records_invalid = int((~result["_valid"]).sum())
    report.records_valid = report.records_total - report.records_invalid

    logger.info(
        "Quality pipeline complete: %d/%d valid, %d duplicates",
        report.records_valid,
        report.records_total,
        report.duplicate_records,
    )

    return result, report
