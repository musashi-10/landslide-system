"""
satellite/spatial/alignment.py
--------------------------------
Converts raster feature arrays into spatially-referenced tabular records
that can be joined with Engineer 2's GIS layer on ``location_id``.

Location ID convention
----------------------
``location_id`` is generated as:

    LOC_LAT{lat:.4f}_LON{lon:.4f}

Examples
    LOC_LAT27.1230_LON88.4560
    LOC_LAT-1.5000_LON37.2500

This format is:
* Deterministic (same lat/lon always → same ID)
* Human-readable
* Sortable
* Unique at 4 decimal-place precision (~11 m resolution)

⚠ Engineer 2 must use the same ``location_id`` convention.  If Engineer 2
uses a different scheme, both teams must agree on a join key before
integration.  This convention is documented in ``SATELLITE_PIPELINE.md``.

CRS notes
---------
Source CRS  : recorded from the raster (e.g. EPSG:32645 for Sentinel-2 UTM)
Processing  : pixel centroids computed in source CRS, then reprojected to
              EPSG:4326 using pyproj.
Output CRS  : EPSG:4326 (WGS-84 geographic coordinates)

All outputs carry the output CRS explicitly; no silent CRS mixing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_CRS = "EPSG:4326"


def _pixel_centroids(
    array_shape: tuple[int, int],
    transform: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the (x, y) centroids of all pixels in source CRS.

    Parameters
    ----------
    array_shape : (rows, cols)
    transform : affine.Affine or compatible
        Affine geotransform mapping pixel indices to coordinates.

    Returns
    -------
    xs, ys : np.ndarray of shape (rows * cols,)
        Flattened x and y coordinate arrays.
    """
    rows, cols = array_shape
    col_idx, row_idx = np.meshgrid(
        np.arange(cols, dtype=np.float64),
        np.arange(rows, dtype=np.float64),
    )
    # Pixel centroid: offset by 0.5
    col_c = col_idx.ravel() + 0.5
    row_c = row_idx.ravel() + 0.5

    # Apply affine transform: (x, y) = transform * (col, row)
    xs = transform.c + col_c * transform.a + row_c * transform.b
    ys = transform.f + col_c * transform.d + row_c * transform.e
    return xs, ys


def _reproject_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    source_crs: str,
    target_crs: str = OUTPUT_CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject coordinate arrays from source_crs to target_crs.

    Uses pyproj.Transformer for accuracy.

    Parameters
    ----------
    xs, ys : np.ndarray
        Flattened coordinate arrays in source_crs.
    source_crs : str
        EPSG string or WKT of the source CRS.
    target_crs : str
        Target CRS (default EPSG:4326).

    Returns
    -------
    lons, lats : np.ndarray
        Longitude and latitude arrays in target_crs.
    """
    try:
        from pyproj import Transformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pyproj is required for CRS reprojection.  pip install pyproj"
        ) from exc

    if source_crs == target_crs or source_crs in ("EPSG:4326", "WGS84"):
        # Already in target CRS — return as-is (xs=lons, ys=lats for 4326)
        return xs, ys

    transformer = Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    lons, lats = transformer.transform(xs, ys)
    return lons, lats


def make_location_id(lat: float, lon: float) -> str:
    """Generate a deterministic ``location_id`` from lat/lon.

    Parameters
    ----------
    lat : float   Latitude in degrees (EPSG:4326).
    lon : float   Longitude in degrees (EPSG:4326).

    Returns
    -------
    str   e.g. ``"LOC_LAT27.1230_LON88.4560"``
    """
    return f"LOC_LAT{lat:.4f}_LON{lon:.4f}"


def raster_to_points(
    feature_arrays: dict[str, np.ma.MaskedArray],
    transform: Any,
    source_crs: str,
    acquisition_time: str,
    source: str = "Sentinel-2",
    spatial_resolution_m: int = 10,
    processing_version: str = "v1",
) -> list[dict]:
    """Convert raster feature arrays to a list of point records.

    Each valid (unmasked) pixel becomes one record with:
    - spatial coordinates (lat, lon, location_id)
    - all feature values for that pixel
    - provenance metadata

    Masked pixels are excluded (not included as NaN rows).

    Parameters
    ----------
    feature_arrays : dict
        Mapping of feature name → 2-D masked array.
        All arrays must have the same shape.
    transform : affine.Affine
        Affine geotransform in source_crs.
    source_crs : str
        CRS of the raster arrays.
    acquisition_time : str
        ISO 8601 UTC acquisition timestamp.
    source : str
        Data source label (e.g. "Sentinel-2").
    spatial_resolution_m : int
        Pixel resolution in metres.
    processing_version : str
        Processing pipeline version tag.

    Returns
    -------
    list of dict
        One dict per valid pixel.  Each dict matches :class:`SatelliteFeatureRecord`.
    """
    if not feature_arrays:
        return []

    # All arrays must have the same shape
    shapes = {name: arr.shape for name, arr in feature_arrays.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise ValueError(f"Feature arrays have inconsistent shapes: {shapes}")

    ref_shape = next(iter(shapes.values()))
    rows, cols = ref_shape
    n_pixels = rows * cols

    # Compute pixel centroids and reproject to EPSG:4326
    xs, ys = _pixel_centroids(ref_shape, transform)
    lons, lats = _reproject_xy(xs, ys, source_crs, OUTPUT_CRS)

    # Build combined validity mask (True = valid in ALL arrays)
    combined_valid = np.ones(n_pixels, dtype=bool)
    flat_features: dict[str, np.ndarray] = {}

    for name, arr in feature_arrays.items():
        flat = arr.ravel() if not isinstance(arr, np.ma.MaskedArray) else arr.filled(np.nan).ravel()
        mask = np.ma.getmaskarray(arr).ravel() if isinstance(arr, np.ma.MaskedArray) else np.zeros(n_pixels, dtype=bool)
        combined_valid &= ~mask
        flat_features[name] = flat

    logger.info(
        "raster_to_points: %d / %d pixels valid after masking",
        int(np.sum(combined_valid)), n_pixels,
    )

    records = []
    valid_indices = np.where(combined_valid)[0]

    for idx in valid_indices:
        lat = float(lats[idx])
        lon = float(lons[idx])
        record: dict[str, Any] = {
            "location_id": make_location_id(lat, lon),
            "latitude": lat,
            "longitude": lon,
            "geometry": f"POINT ({lon:.6f} {lat:.6f})",
            "acquisition_time": acquisition_time,
            "source": source,
            "spatial_resolution_m": spatial_resolution_m,
            "processing_version": processing_version,
            "source_crs": source_crs,
            "output_crs": OUTPUT_CRS,
        }
        for name, flat in flat_features.items():
            val = flat[idx]
            record[name] = None if np.isnan(val) else float(val)

        records.append(record)

    return records


def assign_location_id(
    records: list[dict],
) -> list[dict]:
    """Ensure every record has a ``location_id`` field.

    If ``location_id`` is already present it is kept unchanged.
    Otherwise it is generated from ``latitude`` and ``longitude``.

    Parameters
    ----------
    records : list of dict
        Feature records (output of :func:`raster_to_points`).

    Returns
    -------
    list of dict  (same list, mutated in place for efficiency)
    """
    for rec in records:
        if "location_id" not in rec or rec["location_id"] is None:
            rec["location_id"] = make_location_id(rec["latitude"], rec["longitude"])
    return records


def records_to_geodataframe(records: list[dict]) -> Any:
    """Convert a list of feature records to a GeoDataFrame.

    Requires ``geopandas`` and ``shapely``.

    Parameters
    ----------
    records : list of dict

    Returns
    -------
    geopandas.GeoDataFrame
        CRS set to EPSG:4326.
    """
    try:
        import geopandas as gpd  # noqa: PLC0415
        import pandas as pd      # noqa: PLC0415
        from shapely import wkt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "geopandas, pandas, and shapely are required.  "
            "pip install geopandas pandas shapely"
        ) from exc

    df = pd.DataFrame(records)
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=OUTPUT_CRS)
    else:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=OUTPUT_CRS,
        )
    return gdf
