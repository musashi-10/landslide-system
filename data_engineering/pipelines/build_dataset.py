"""
Primary pipeline: raw data → standardised, validated, spatially indexed
training-ready dataset.

This is the main entry point for all downstream engineers.

Usage example
-------------
::

    from data_engineering.pipelines import build_dataset, DatasetConfig
    from data_engineering.spatial import GridConfig

    dataset, report, provenance = build_dataset(
        source_path="data/raw/landslides.csv",
        grid_config=GridConfig(
            min_lat=26.0, max_lat=30.0,
            min_lon=85.0, max_lon=92.0,
            resolution_deg=0.01,
        ),
        column_map={"latitude": "lat", "longitude": "lon", "timestamp": "event_date"},
        output_parquet="data/processed/landslide_events.parquet",
    )

Output schema (Parquet / DataFrame)
-------------------------------------
Required columns (always present in valid output):

+------------------------+---------+--------------------------------------------+
| Column                 | Type    | Description                                |
+========================+=========+============================================+
| location_id            | str     | Deterministic spatial identifier           |
| latitude               | float   | WGS-84 latitude (decimal degrees)          |
| longitude              | float   | WGS-84 longitude (decimal degrees)         |
| geometry               | str     | WKT point geometry                         |
| timestamp_utc          | str     | ISO 8601 UTC string                        |
| historical_landslide   | int     | 1 = event, 0 = non-event                   |
| source                 | str     | Data source identifier                     |
| grid_id                | str     | Spatial grid cell (optional)               |
| is_duplicate           | bool    | Duplicate flag (never silently dropped)    |
+------------------------+---------+--------------------------------------------+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from data_engineering.ingestion import load_csv, load_geojson, load_shapefile
from data_engineering.preprocessing import standardize_landslide_records
from data_engineering.schemas import DataProvenance, ValidationReport
from data_engineering.spatial import GridConfig, assign_grid_id

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    Configuration for the :func:`build_dataset` pipeline.

    Attributes
    ----------
    source_path : str | Path
        Path to the input file (CSV, GeoJSON, or Shapefile/GeoPackage).
    grid_config : GridConfig | None
        Spatial grid configuration.  If ``None``, grid assignment is skipped.
    column_map : dict[str, str]
        Maps contract column names to raw column names.
        E.g. ``{"latitude": "lat", "longitude": "lng", "timestamp": "date"}``.
    source_name : str | None
        Human-readable source name for provenance tracking.
    output_parquet : str | Path | None
        If set, the standardised dataset is written to this Parquet file.
    output_geojson : str | Path | None
        If set, writes a GeoJSON output (geometry required).
    reject_null_island : bool
        Treat (0.0, 0.0) as invalid coordinates.
    extra_cols : list[str]
        Extra raw columns to preserve in output.
    acquisition_datetime : str | None
        ISO 8601 datetime when the source data was obtained.
    geographic_coverage : str | None
        Free-text description of geographic extent.
    preprocessing_version : str
        Version tag for this pipeline run.
    """

    source_path: str | Path = ""
    grid_config: Optional[GridConfig] = None
    column_map: dict[str, str] = field(default_factory=dict)
    source_name: Optional[str] = None
    output_parquet: Optional[str | Path] = None
    output_geojson: Optional[str | Path] = None
    reject_null_island: bool = False
    extra_cols: list[str] = field(default_factory=list)
    acquisition_datetime: Optional[str] = None
    geographic_coverage: Optional[str] = None
    preprocessing_version: str = "1.0.0"


def build_dataset(
    source_path: Optional[str | Path] = None,
    *,
    grid_config: Optional[GridConfig] = None,
    column_map: Optional[dict[str, str]] = None,
    source_name: Optional[str] = None,
    output_parquet: Optional[str | Path] = None,
    output_geojson: Optional[str | Path] = None,
    reject_null_island: bool = False,
    extra_cols: Optional[list[str]] = None,
    acquisition_datetime: Optional[str] = None,
    geographic_coverage: Optional[str] = None,
    preprocessing_version: str = "1.0.0",
    config: Optional[DatasetConfig] = None,
) -> tuple[pd.DataFrame, ValidationReport, DataProvenance]:
    """
    Run the full data engineering pipeline on one source file.

    Parameters
    ----------
    source_path : str | Path
        Input file.  Supported: ``.csv``, ``.geojson``, ``.json``,
        ``.shp``, ``.gpkg``.
    grid_config : GridConfig | None
        Spatial grid for assigning ``grid_id``.
    column_map : dict | None
        Raw-to-contract column name mapping.
    source_name : str | None
        Source provenance label.
    output_parquet : str | Path | None
        Write standardised data to Parquet.
    output_geojson : str | Path | None
        Write standardised data to GeoJSON.
    reject_null_island : bool
        Treat (0.0, 0.0) as invalid.
    extra_cols : list[str] | None
        Extra columns to preserve from the raw data.
    acquisition_datetime : str | None
        When the data was obtained.
    geographic_coverage : str | None
        Textual geographic coverage description.
    preprocessing_version : str
        Version string for this pipeline run.
    config : DatasetConfig | None
        Alternative: pass all options as a :class:`DatasetConfig` object.

    Returns
    -------
    tuple[pd.DataFrame, ValidationReport, DataProvenance]
        * Standardised DataFrame ready for Engineer 2/4/5 consumption.
        * :class:`~data_engineering.schemas.ValidationReport` with quality metrics.
        * :class:`~data_engineering.schemas.DataProvenance` metadata.

    Raises
    ------
    ValueError
        When no source path is provided.
    """
    # Allow config object to override kwargs
    if config is not None:
        source_path = source_path or config.source_path
        grid_config = grid_config or config.grid_config
        column_map = column_map or config.column_map
        source_name = source_name or config.source_name
        output_parquet = output_parquet or config.output_parquet
        output_geojson = output_geojson or config.output_geojson
        reject_null_island = reject_null_island or config.reject_null_island
        extra_cols = extra_cols or config.extra_cols
        acquisition_datetime = acquisition_datetime or config.acquisition_datetime
        geographic_coverage = geographic_coverage or config.geographic_coverage
        preprocessing_version = preprocessing_version or config.preprocessing_version

    if not source_path:
        raise ValueError("source_path is required")

    path = Path(source_path)
    suffix = path.suffix.lower()

    # ------------------------------------------------------------------ #
    # 1. Ingest raw data
    # ------------------------------------------------------------------ #
    logger.info("Pipeline: ingesting %s", path)

    if suffix == ".csv":
        raw_df = load_csv(path, source_name=source_name)
    elif suffix in {".geojson", ".json"}:
        gdf = load_geojson(path, source_name=source_name)
        raw_df = _geodataframe_to_dataframe(gdf)
    elif suffix in {".shp", ".gpkg"}:
        gdf = load_shapefile(path, source_name=source_name)
        raw_df = _geodataframe_to_dataframe(gdf)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. "
            "Supported: .csv, .geojson, .json, .shp, .gpkg"
        )

    # ------------------------------------------------------------------ #
    # 2. Standardise records
    # ------------------------------------------------------------------ #
    logger.info("Pipeline: standardising records")
    std_df, report = standardize_landslide_records(
        raw_df,
        column_map=column_map,
        source_name=source_name,
        extra_cols=extra_cols or [],
        reject_null_island=reject_null_island,
    )

    if std_df.empty:
        logger.warning("Pipeline: no valid records after standardisation.")
        provenance = _make_provenance(
            path, source_name, acquisition_datetime,
            geographic_coverage, preprocessing_version,
        )
        return std_df, report, provenance

    # ------------------------------------------------------------------ #
    # 3. Assign grid IDs (optional)
    # ------------------------------------------------------------------ #
    if grid_config is not None:
        logger.info("Pipeline: assigning grid IDs")
        std_df = assign_grid_id(std_df, grid_config)

    # ------------------------------------------------------------------ #
    # 4. Write outputs
    # ------------------------------------------------------------------ #
    if output_parquet:
        _write_parquet(std_df, output_parquet)

    if output_geojson:
        _write_geojson(std_df, output_geojson)

    # ------------------------------------------------------------------ #
    # 5. Provenance
    # ------------------------------------------------------------------ #
    provenance = _make_provenance(
        path, source_name, acquisition_datetime,
        geographic_coverage, preprocessing_version,
    )

    logger.info(
        "Pipeline complete: %d valid records from %s.", len(std_df), path.name
    )
    return std_df, report, provenance


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _geodataframe_to_dataframe(gdf) -> pd.DataFrame:
    """
    Convert a GeoDataFrame to a plain DataFrame with explicit lat/lon columns.

    If the GeoDataFrame already has ``latitude`` / ``longitude`` columns they
    are preserved.  Otherwise, point coordinates are extracted from the
    ``geometry`` column (centroid for non-point geometries).

    The geometry column is kept as WKT strings so the standardiser can use it.
    """
    import geopandas as gpd

    df = pd.DataFrame(gdf)

    # Extract lat/lon from geometry when not already present
    if "latitude" not in df.columns or "longitude" not in df.columns:
        try:
            # Use representative_point() to avoid geographic-CRS centroid warnings;
            # for point geometries this returns the point itself.
            rep_points = gdf.geometry.representative_point()
            if "latitude" not in df.columns:
                df["latitude"] = rep_points.y
            if "longitude" not in df.columns:
                df["longitude"] = rep_points.x
        except Exception as exc:
            logger.warning("Could not extract lat/lon from geometry: %s", exc)

    # Convert geometry column to WKT strings (Shapely objects are not JSON-serialisable)
    if "geometry" in df.columns:
        df["geometry"] = gdf.geometry.apply(
            lambda g: g.wkt if g is not None and not (hasattr(g, "is_empty") and g.is_empty) else None
        )

    return df


def _write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write *df* to a Parquet file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote Parquet: %s (%d rows)", path, len(df))


def _write_geojson(df: pd.DataFrame, path: str | Path) -> None:
    """Write *df* to a GeoJSON file using the ``geometry`` WKT column."""
    import geopandas as gpd
    from shapely import wkt as shapely_wkt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "geometry" not in df.columns:
        logger.warning("No geometry column; skipping GeoJSON output.")
        return

    try:
        geometries = df["geometry"].apply(
            lambda g: shapely_wkt.loads(g) if g else None
        )
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        gdf.to_file(str(path), driver="GeoJSON")
        logger.info("Wrote GeoJSON: %s (%d features)", path, len(gdf))
    except Exception as exc:
        logger.error("Failed to write GeoJSON: %s", exc)


def _make_provenance(
    path: Path,
    source_name: Optional[str],
    acquisition_datetime: Optional[str],
    geographic_coverage: Optional[str],
    preprocessing_version: str,
) -> DataProvenance:
    return DataProvenance(
        source=source_name or path.name,
        acquisition_datetime=acquisition_datetime,
        geographic_coverage=geographic_coverage,
        preprocessing_version=preprocessing_version,
    )
