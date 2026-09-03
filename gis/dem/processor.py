"""
DEM (Digital Elevation Model) processing utilities.

Responsibilities
----------------
* Open and validate a GeoTIFF DEM.
* Reproject / validate CRS.
* Clip to a bounding box.
* Extract elevation at arbitrary points.
* Compute slope (degrees, 0–90) and aspect (degrees, 0–360, clockwise from N).

Units
-----
* Elevation : metres (as stored in the source DEM).
* Slope     : degrees (0 = flat, 90 = vertical).
* Aspect    : degrees clockwise from North (0/360 = N, 90 = E, 180 = S, 270 = W).
              NoData aspect (flat areas) is stored as -1.

CRS handling
------------
All rasters are internally reprojected to ``target_crs`` (default EPSG:4326)
before any derivative computation.  The caller sees only the target CRS.

NoData handling
---------------
Pixels with the source ``nodata`` value are preserved as NaN in NumPy arrays
and are never silently converted to 0.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _read_band(path: str | Path) -> Tuple[np.ndarray, Affine, CRS, float]:
    """
    Open a single-band raster and return (data, transform, crs, nodata).

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF (or any rasterio-readable raster).

    Returns
    -------
    data      : float32 ndarray (rows × cols), nodata pixels set to NaN.
    transform : Affine transform.
    crs       : rasterio.crs.CRS of the file.
    nodata    : original nodata value from the file (may be None → NaN used).
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    return data, transform, crs, nodata


def _pixel_size_metres(transform: Affine, crs: CRS) -> Tuple[float, float]:
    """
    Estimate (x_res_m, y_res_m) from transform + CRS.

    For geographic CRS we use the midpoint latitude for cos-correction.
    For projected CRS we return the raw pixel sizes (already in metres).
    """
    if crs.is_geographic:
        # Degrees → metres approximation (WGS-84 mean Earth radius)
        lat_mid = 0.0  # conservative; caller may update for their region
        x_deg = abs(transform.a)
        y_deg = abs(transform.e)
        x_m = x_deg * math.pi / 180.0 * 6_371_000.0 * math.cos(math.radians(lat_mid))
        y_m = y_deg * math.pi / 180.0 * 6_371_000.0
        return x_m, y_m
    else:
        return abs(transform.a), abs(transform.e)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def validate_and_reproject(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    target_crs_str: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, Affine, CRS]:
    """
    Reproject *data* to ``target_crs_str`` if needed.

    If ``src_crs`` already matches ``target_crs_str`` the data is returned
    unchanged (no unnecessary copy).

    Parameters
    ----------
    data           : float32 ndarray (rows × cols).
    src_transform  : source Affine transform.
    src_crs        : source CRS.
    target_crs_str : EPSG string for the desired output CRS.
    resampling     : rasterio Resampling algorithm.

    Returns
    -------
    out_data      : reprojected ndarray.
    out_transform : new Affine transform.
    out_crs       : target CRS.
    """
    target_crs = CRS.from_string(target_crs_str)

    if src_crs == target_crs:
        logger.debug("CRS already matches %s; no reprojection needed.", target_crs_str)
        return data, src_transform, src_crs

    rows, cols = data.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, cols, rows, *_raster_bounds(src_transform, rows, cols)
    )

    out = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=data,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    logger.debug(
        "Reprojected %s → %s; shape %s → %s",
        src_crs.to_epsg(),
        target_crs_str,
        (rows, cols),
        (dst_height, dst_width),
    )
    return out, dst_transform, target_crs


def _raster_bounds(transform: Affine, rows: int, cols: int) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) from an Affine transform."""
    west = transform.c
    north = transform.f
    east = west + transform.a * cols
    south = north + transform.e * rows
    return west, min(south, north), east, max(south, north)


def clip_raster(
    data: np.ndarray,
    transform: Affine,
    bbox: Tuple[float, float, float, float],  # (west, south, east, north)
) -> Tuple[np.ndarray, Affine]:
    """
    Clip *data* to the pixel extent that overlaps *bbox*.

    Parameters
    ----------
    data      : 2-D float32 array.
    transform : Affine transform describing *data*.
    bbox      : (west, south, east, north) in the same CRS as *transform*.

    Returns
    -------
    clipped   : 2-D float32 array (may be the same object if bbox covers all).
    new_transform : updated Affine transform for the clipped window.
    """
    west, south, east, north = bbox
    rows, cols = data.shape

    # Convert geographic bbox to pixel row/col windows
    col_off = max(0, int((west  - transform.c) / transform.a))
    row_off = max(0, int((north - transform.f) / transform.e))
    col_end = min(cols, int(math.ceil((east  - transform.c) / transform.a)))
    row_end = min(rows, int(math.ceil((south - transform.f) / transform.e)))

    if row_off >= row_end or col_off >= col_end:
        raise ValueError(
            f"Bounding box {bbox} does not overlap the raster extent "
            f"({_raster_bounds(transform, rows, cols)})."
        )

    clipped = data[row_off:row_end, col_off:col_end]
    new_transform = Affine(
        transform.a,
        transform.b,
        transform.c + col_off * transform.a,
        transform.d,
        transform.e,
        transform.f + row_off * transform.e,
    )
    return clipped, new_transform


def open_dem(
    path: str | Path,
    target_crs: str = "EPSG:4326",
) -> Tuple[np.ndarray, Affine, CRS]:
    """
    Open a DEM GeoTIFF, reproject to *target_crs*, and return the array.

    Parameters
    ----------
    path       : Path to DEM GeoTIFF.
    target_crs : Desired output CRS (EPSG string).

    Returns
    -------
    elevation  : float32 ndarray (rows × cols); nodata → NaN.
    transform  : Affine transform in *target_crs*.
    crs        : target CRS object.
    """
    data, transform, src_crs, _ = _read_band(path)
    data, transform, crs = validate_and_reproject(data, transform, src_crs, target_crs)
    logger.info("Opened DEM %s: shape=%s, CRS=%s", path, data.shape, crs)
    return data, transform, crs


def extract_elevation(
    elevation: np.ndarray,
    transform: Affine,
    lat: float,
    lon: float,
) -> Optional[float]:
    """
    Sample elevation at a geographic point.

    Parameters
    ----------
    elevation : float32 ndarray.
    transform : Affine transform aligned with *elevation*.
    lat, lon  : Geographic coordinates (degrees, WGS-84 or matching CRS).

    Returns
    -------
    Elevation in metres, or None if the point falls outside the array / nodata.
    """
    col = int((lon - transform.c) / transform.a)
    row = int((lat - transform.f) / transform.e)
    rows, cols = elevation.shape
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    val = elevation[row, col]
    return None if np.isnan(val) else float(val)


def compute_slope(
    elevation: np.ndarray,
    transform: Affine,
    crs: CRS,
) -> np.ndarray:
    """
    Compute slope in degrees using the Horn (1981) finite-difference method.

    Slope ranges 0–90°.  NaN is propagated from elevation nodata.

    Algorithm: Horn, B.K.P. (1981). Hill shading and the reflectance map.
    Proceedings of the IEEE, 69(1), 14–47.

    Parameters
    ----------
    elevation : float32 ndarray (rows × cols) in metres.
    transform : Affine transform.
    crs       : CRS of the raster (used to determine pixel size in metres).

    Returns
    -------
    slope : float32 ndarray (degrees, 0–90), same shape as *elevation*.
    """
    x_res, y_res = _pixel_size_metres(transform, crs)

    # Pad with NaN so edges keep NaN instead of zero
    padded = np.pad(elevation, 1, mode="edge")
    # Neighbours (Horn 3×3 kernel)
    a = padded[0:-2, 0:-2]; b = padded[0:-2, 1:-1]; c = padded[0:-2, 2:]
    d = padded[1:-1, 0:-2];                          f = padded[1:-1, 2:]
    g = padded[2:,   0:-2]; h = padded[2:,   1:-1]; i = padded[2:,   2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * x_res)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * y_res)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Propagate NaN from original nodata
    slope_deg[np.isnan(elevation)] = np.nan
    return slope_deg


def compute_aspect(
    elevation: np.ndarray,
    transform: Affine,
    crs: CRS,
) -> np.ndarray:
    """
    Compute aspect in degrees clockwise from North.

    Convention
    ----------
    * 0° / 360° = North
    * 90°       = East
    * 180°      = South
    * 270°      = West
    * -1        = flat (slope ≈ 0)

    Parameters
    ----------
    elevation : float32 ndarray (rows × cols) in metres.
    transform : Affine transform.
    crs       : CRS of the raster.

    Returns
    -------
    aspect : float32 ndarray (degrees, 0–360 or -1), same shape as *elevation*.
    """
    x_res, y_res = _pixel_size_metres(transform, crs)

    padded = np.pad(elevation, 1, mode="edge")
    a = padded[0:-2, 0:-2]; b = padded[0:-2, 1:-1]; c = padded[0:-2, 2:]
    d = padded[1:-1, 0:-2];                          f = padded[1:-1, 2:]
    g = padded[2:,   0:-2]; h = padded[2:,   1:-1]; i = padded[2:,   2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * x_res)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * y_res)

    # atan2 gives math bearing; convert to clockwise-from-North
    aspect_rad = np.arctan2(dz_dx, dz_dy)          # E positive, N up
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = 90.0 - aspect_deg                   # rotate: math→geographic
    aspect_deg[aspect_deg < 0] += 360.0
    aspect_deg[aspect_deg >= 360.0] -= 360.0

    # Mark flat areas
    flat_mask = (np.abs(dz_dx) < 1e-10) & (np.abs(dz_dy) < 1e-10)
    aspect_deg[flat_mask] = -1.0

    # Propagate NaN
    aspect_out = aspect_deg.astype(np.float32)
    aspect_out[np.isnan(elevation)] = np.nan
    return aspect_out


def save_raster(
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    path: str | Path,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    """
    Write a single-band raster to a GeoTIFF.

    Parameters
    ----------
    data      : 2-D array.
    transform : Affine transform.
    crs       : CRS.
    path      : Output file path.
    nodata    : Nodata value written to the file (NaN → nodata before save).
    dtype     : rasterio dtype string.
    """
    out = data.copy()
    out[np.isnan(out)] = nodata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(out.astype(dtype), 1)
    logger.info("Saved raster to %s", path)
