"""
Duplicate detection for landslide event records.

A duplicate is defined as two or more records that share the same:
  * rounded latitude  (to ``coord_precision`` decimal places)
  * rounded longitude (to ``coord_precision`` decimal places)
  * timestamp_utc     (exact ISO string match after normalisation)

Duplicates are **flagged**, never silently dropped.  The caller decides
what to do with them (e.g. keep first, discard rest, or retain all with a
``is_duplicate`` column for audit).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def detect_duplicates(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "timestamp_utc",
    coord_precision: int = 4,
) -> pd.DataFrame:
    """
    Add an ``is_duplicate`` boolean column to *df* marking duplicate rows.

    Two rows are considered duplicates when, after rounding coordinates to
    *coord_precision* decimal places, the ``(lat, lon, timestamp)`` triplet
    matches that of a previous row (first occurrence is NOT marked).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.  Must contain *lat_col*, *lon_col*, and *time_col*.
    lat_col : str
        Column name for latitude.
    lon_col : str
        Column name for longitude.
    time_col : str
        Column name for the normalised UTC timestamp string.
    coord_precision : int
        Number of decimal places used when comparing coordinates.
        At 4 d.p., two locations < ~11 m apart are treated as identical.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an ``is_duplicate`` column added.

    Notes
    -----
    Records with missing coordinates or timestamps are **never** classified
    as duplicates; they are handled by the coordinate/timestamp validators.
    """
    result = df.copy()

    # Build a rounded key for spatial comparison
    key_lat = result[lat_col].round(coord_precision).astype(str)
    key_lon = result[lon_col].round(coord_precision).astype(str)
    key_time = result[time_col].fillna("").astype(str)

    composite_key = key_lat + "|" + key_lon + "|" + key_time

    # Mark all rows whose composite key has appeared before
    result["is_duplicate"] = composite_key.duplicated(keep="first")

    n_dups = int(result["is_duplicate"].sum())
    if n_dups:
        logger.warning("Detected %d duplicate record(s).", n_dups)
    else:
        logger.info("No duplicates detected.")

    return result


def count_duplicates(df: pd.DataFrame, **kwargs) -> int:
    """
    Return the number of duplicate rows detected in *df*.

    Accepts the same keyword arguments as :func:`detect_duplicates`.
    """
    marked = detect_duplicates(df, **kwargs)
    return int(marked["is_duplicate"].sum())
