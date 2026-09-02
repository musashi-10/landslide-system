"""
Historical landslide record standardiser.

Converts a raw pandas DataFrame (from any loader) into a standardised
DataFrame whose schema matches the data contract:

Required output columns (always present):
  * location_id
  * latitude
  * longitude
  * geometry        (WKT string or None)
  * timestamp_utc   (ISO 8601 UTC string)
  * historical_landslide (int: 1)
  * source

Optional output columns (present when available):
  * grid_id
  * elevation, slope, aspect, soil_type, geology, land_cover
  * (and other passthrough columns via ``extra_cols``)

Column-name mapping is fully configurable via the ``column_map`` parameter
so that differently-named raw sources can be ingested without modifying this
module.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data_engineering.schemas import ValidationReport
from data_engineering.validation.coordinate_validator import (
    validate_coordinates,
    CoordinateError,
)
from data_engineering.validation.timestamp_validator import (
    normalize_timestamp,
    TimestampError,
)
from data_engineering.spatial.location_id import generate_location_id

logger = logging.getLogger(__name__)

# Default column map: maps contract names → raw column names
# Override this per source using the ``column_map`` parameter.
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "latitude": "latitude",
    "longitude": "longitude",
    "timestamp": "date",
    "source": "source",
}

# Passthrough static-feature columns (kept if present in raw data)
STATIC_FEATURE_COLS = [
    "elevation",
    "slope",
    "aspect",
    "soil_type",
    "geology",
    "land_cover",
    "drainage_feature",
]


def standardize_landslide_records(
    df: pd.DataFrame,
    *,
    column_map: Optional[dict[str, str]] = None,
    source_name: Optional[str] = None,
    extra_cols: Optional[list[str]] = None,
    reject_null_island: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Standardise a raw landslide record DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input — typically from one of the ingestion loaders.
    column_map : dict[str, str] | None
        Mapping ``{contract_name: raw_column_name}``.  Defaults to
        :data:`DEFAULT_COLUMN_MAP`.
    source_name : str | None
        Override for the ``source`` column.  If None, uses the value from
        the DataFrame's ``source`` column.
    extra_cols : list[str] | None
        Additional raw columns to preserve in the output.
    reject_null_island : bool
        Treat (0.0, 0.0) as invalid (see coordinate validator).

    Returns
    -------
    tuple[pd.DataFrame, ValidationReport]
        * Standardised DataFrame — only valid records.
        * Validation report with counts for the caller.

    Notes
    -----
    Records that fail coordinate or timestamp validation are excluded from
    the output DataFrame.  They are counted in the ``ValidationReport`` and
    logged at WARNING level.  Missing values are **never** silently set to
    zero.
    """
    col_map = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    report = ValidationReport(records_total=len(df))

    lat_col = col_map.get("latitude", "latitude")
    lon_col = col_map.get("longitude", "longitude")
    ts_col = col_map.get("timestamp", "date")
    src_col = col_map.get("source", "source")

    output_rows: list[dict] = []

    for idx, row in df.iterrows():
        # ---- Coordinates ------------------------------------------------
        raw_lat = row.get(lat_col)
        raw_lon = row.get(lon_col)

        if pd.isna(raw_lat) or pd.isna(raw_lon) or raw_lat is None or raw_lon is None:
            report.missing_coordinates += 1
            report.records_invalid += 1
            report.details.append({"row": idx, "reason": "missing_coordinates"})
            logger.warning("Row %d: missing coordinates, skipping.", idx)
            continue

        try:
            lat, lon = validate_coordinates(
                raw_lat, raw_lon, reject_null_island=reject_null_island
            )
        except CoordinateError as exc:
            report.invalid_coordinates += 1
            report.records_invalid += 1
            report.details.append({"row": idx, "reason": f"invalid_coordinates: {exc}"})
            logger.warning("Row %d: %s, skipping.", idx, exc)
            continue

        # ---- Timestamp --------------------------------------------------
        raw_ts = row.get(ts_col)
        try:
            ts_utc = normalize_timestamp(raw_ts)
        except TimestampError as exc:
            report.missing_timestamps += 1
            report.records_invalid += 1
            report.details.append({"row": idx, "reason": f"invalid_timestamp: {exc}"})
            logger.warning("Row %d: %s, skipping.", idx, exc)
            continue

        # ---- Source -----------------------------------------------------
        source = source_name or (
            row.get(src_col) if src_col in row.index else "unknown"
        )

        # ---- Geometry (WKT) ---------------------------------------------
        geometry_wkt: Optional[str] = None
        if "geometry" in row.index and row["geometry"] is not None:
            try:
                geom = row["geometry"]
                geometry_wkt = geom.wkt if hasattr(geom, "wkt") else str(geom)
            except Exception:
                geometry_wkt = None
        else:
            from shapely.geometry import Point
            geometry_wkt = Point(lon, lat).wkt

        # ---- location_id ------------------------------------------------
        location_id = generate_location_id(lat, lon)

        # ---- Build output row ------------------------------------------
        out: dict = {
            "location_id": location_id,
            "latitude": lat,
            "longitude": lon,
            "geometry": geometry_wkt,
            "timestamp_utc": ts_utc,
            "historical_landslide": 1,
            "source": source,
        }

        # Passthrough static features
        for feat in STATIC_FEATURE_COLS:
            if feat in row.index:
                val = row[feat]
                out[feat] = None if pd.isna(val) else val

        # User-requested extra columns
        if extra_cols:
            for col in extra_cols:
                if col in row.index:
                    val = row[col]
                    out[col] = None if pd.isna(val) else val

        output_rows.append(out)

    result_df = pd.DataFrame(output_rows)

    # Deduplicate (flag, don't drop)
    if not result_df.empty:
        from data_engineering.validation.duplicate_detector import detect_duplicates
        result_df = detect_duplicates(
            result_df,
            lat_col="latitude",
            lon_col="longitude",
            time_col="timestamp_utc",
        )
        report.duplicate_records = int(result_df["is_duplicate"].sum())

    report.records_valid = len(result_df)
    report.records_invalid = report.records_total - report.records_valid

    logger.info(
        "Standardisation complete: %d/%d valid, %d duplicates.",
        report.records_valid,
        report.records_total,
        report.duplicate_records,
    )

    return result_df, report
