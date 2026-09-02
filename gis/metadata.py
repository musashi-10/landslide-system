"""
Dataset metadata generation.

Every output dataset must carry provenance metadata per docs/data-contract.md
Section 11 (Data Provenance).

The metadata is written as a JSON sidecar alongside each output file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def build_metadata(
    source: str,
    crs: str,
    spatial_resolution: str,
    processing_version: str,
    nodata_handling: str,
    geographic_coverage: str = "configured AOI",
    temporal_resolution: str = "static",
    acquisition_date: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a provenance metadata dictionary.

    Parameters
    ----------
    source               : Description of input data source.
    crs                  : EPSG string of output.
    spatial_resolution   : Human-readable resolution (e.g. "0.01 degrees (~1 km)").
    processing_version   : GIS pipeline version string.
    nodata_handling      : How nodata is handled (e.g. "NaN preserved; never zero").
    geographic_coverage  : Region description.
    temporal_resolution  : "static" for static features.
    acquisition_date     : ISO-8601 date of source data (if known).
    extra                : Additional metadata key-value pairs.

    Returns
    -------
    Dict suitable for JSON serialisation.
    """
    meta: Dict[str, Any] = {
        "source":               source,
        "crs":                  crs,
        "spatial_resolution":   spatial_resolution,
        "temporal_resolution":  temporal_resolution,
        "processing_version":   processing_version,
        "processed_at_utc":     datetime.now(timezone.utc).isoformat(),
        "nodata_handling":      nodata_handling,
        "geographic_coverage":  geographic_coverage,
    }
    if acquisition_date:
        meta["acquisition_date"] = acquisition_date
    if extra:
        meta.update(extra)
    return meta


def save_metadata(metadata: Dict[str, Any], path: str | Path) -> None:
    """Write metadata dict to a JSON file alongside the dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved to %s", path)
