"""
satellite/ingestion/loader.py
------------------------------
Raster loading utilities for the satellite pipeline.

All I/O is isolated here so the rest of the pipeline works with plain
NumPy masked arrays + affine transform metadata.  This makes unit
testing possible without real GeoTIFF files.

Data source notes
-----------------
* Sentinel-2 L2A products are distributed as multi-band GeoTIFFs or
  separate JP2 files (one per band) in the SAFE archive format.
* Band naming convention used here:
    B04 → Red     (~665 nm)   10 m native
    B08 → NIR     (~842 nm)   10 m native
    B11 → SWIR1   (~1610 nm)  20 m native (up-sampled to 10 m on load)
* CRS: Sentinel-2 tiles are in UTM (e.g. EPSG:32645 for tile 45R).
  We record the source CRS and reproject to EPSG:4326 at the spatial
  alignment stage so that all outputs share a common reference frame.

Processing CRS: recorded per scene from rasterio metadata.
Output CRS:     EPSG:4326 (set during spatial alignment).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Bands of interest for Sentinel-2 L2A
SENTINEL2_BANDS = {
    "red": "B04",
    "nir": "B08",
    "swir": "B11",
}

# Sentinel-2 reflectance scale factor (DN → [0, 1] surface reflectance)
SENTINEL2_SCALE_FACTOR: float = 10_000.0
SENTINEL2_NODATA: int = 0


def load_raster_band(
    path: str | Path,
    band_index: int = 1,
    nodata: float | None = None,
) -> tuple[np.ma.MaskedArray, Any, str]:
    """Load a single band from a GeoTIFF / JP2 raster file.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the raster file.
    band_index : int
        1-based band index to read (rasterio convention).  Default: 1.
    nodata : float or None
        Override nodata value.  If None, uses the value embedded in the file.

    Returns
    -------
    array : np.ma.MaskedArray
        2-D masked array with invalid pixels masked.
    transform : affine.Affine
        Affine geotransform of the raster.
    crs : str
        WKT or EPSG string of the source coordinate reference system.

    Raises
    ------
    ImportError
        If ``rasterio`` is not installed.
    FileNotFoundError
        If the raster file does not exist.
    """
    try:
        import rasterio  # noqa: PLC0415  (optional heavy dep)
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for raster loading.  "
            "Install it with:  pip install rasterio"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(band_index)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "UNKNOWN"
        _nodata = nodata if nodata is not None else src.nodata

    logger.debug(
        "Loaded band %d from %s  shape=%s  crs=%s",
        band_index, path.name, data.shape, crs,
    )

    masked = np.ma.masked_equal(data, _nodata) if _nodata is not None else np.ma.MaskedArray(data)
    return masked, transform, crs


def load_scene(
    scene_dir: str | Path,
    band_map: dict[str, str] | None = None,
    scale: float = SENTINEL2_SCALE_FACTOR,
    nodata: int = SENTINEL2_NODATA,
) -> dict[str, Any]:
    """Load a multi-band Sentinel-2 scene from a directory.

    Expects one GeoTIFF (or JP2) per band, named ``*B04*.tif``, etc.
    File discovery is done by glob pattern matching the band suffix.

    Parameters
    ----------
    scene_dir : str or Path
        Directory containing the per-band raster files.
    band_map : dict mapping logical names to band suffixes
        Defaults to ``SENTINEL2_BANDS`` (red/nir/swir).
    scale : float
        Divide raw DN by this value to get surface reflectance.
    nodata : int
        Pixel value indicating missing/invalid data.

    Returns
    -------
    scene : dict with keys
        ``"bands"``   → dict[str, np.ma.MaskedArray]  (scaled to [0, 1])
        ``"transform"`` → affine.Affine  (from the first band loaded)
        ``"crs"``      → str
        ``"scene_dir"`` → str
    """
    scene_dir = Path(scene_dir)
    band_map = band_map or SENTINEL2_BANDS

    loaded_bands: dict[str, np.ma.MaskedArray] = {}
    transform = None
    crs = "UNKNOWN"

    for logical_name, suffix in band_map.items():
        matches = sorted(scene_dir.glob(f"*{suffix}*.tif")) + sorted(
            scene_dir.glob(f"*{suffix}*.jp2")
        )
        if not matches:
            logger.warning(
                "No file found for band %s (%s) in %s — skipping",
                logical_name, suffix, scene_dir,
            )
            continue

        band_path = matches[0]
        arr, xform, _crs = load_raster_band(band_path, band_index=1, nodata=nodata)

        # Scale reflectance values
        arr = arr.astype(np.float32) / scale

        loaded_bands[logical_name] = arr
        if transform is None:
            transform = xform
            crs = _crs

    logger.info(
        "Loaded scene from %s  bands=%s  crs=%s",
        scene_dir, list(loaded_bands.keys()), crs,
    )

    return {
        "bands": loaded_bands,
        "transform": transform,
        "crs": crs,
        "scene_dir": str(scene_dir),
    }


def scene_from_arrays(
    band_arrays: dict[str, np.ndarray],
    transform: Any,
    crs: str = "EPSG:4326",
) -> dict[str, Any]:
    """Construct a scene dict from in-memory NumPy arrays.

    This is the preferred entry point for unit tests — avoids any file I/O.

    Parameters
    ----------
    band_arrays : dict[str, np.ndarray]
        Mapping of logical band name to 2-D array.
        Values should already be surface reflectance (float, [0, 1]).
    transform : affine.Affine or compatible object
        Affine geotransform.
    crs : str
        CRS string (default EPSG:4326 for synthetic fixtures).

    Returns
    -------
    scene : dict  (same structure as :func:`load_scene`)
    """
    masked_bands = {
        name: np.ma.MaskedArray(arr.astype(np.float32))
        for name, arr in band_arrays.items()
    }
    return {
        "bands": masked_bands,
        "transform": transform,
        "crs": crs,
        "scene_dir": "synthetic",
    }
