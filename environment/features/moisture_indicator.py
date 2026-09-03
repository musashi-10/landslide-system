"""
environment/features/moisture_indicator.py

Estimated environmental moisture indicator.

IMPORTANT DISCLAIMER
--------------------
The ``moisture_indicator`` field is an ESTIMATED PROXY, not a direct
physical soil moisture measurement.

We do NOT claim: "we measure soil moisture at every location hourly."

What it is:
  A dimensionless index in [0, 1] derived from recent rainfall history.
  Higher values indicate wetter antecedent conditions.
  It is a simplified proxy for environmental moisture state.

What it is NOT:
  - Not a physical gravimetric soil moisture (% by weight)
  - Not a volumetric water content (m³/m³)
  - Not a remotely-sensed soil moisture (e.g. SMAP, Sentinel-1 SAR)

Source
------
  Derived from rainfall history (Open-Meteo ERA5-Land or mock)
  using a sigmoid-normalised antecedent rainfall index.

Spatial resolution
------------------
  Same as the rainfall source (~1 km for ERA5-Land).

Temporal resolution
-------------------
  Computed per-hour from hourly rainfall history.

Update frequency
----------------
  Updated each time the pipeline runs (pipeline update frequency determines
  how often this value changes).

Formula
-------
  API = antecedent rainfall index (from rainfall_windows.calculate_antecedent_rainfall)
  moisture_indicator = sigmoid(API / saturation_scale)

  sigmoid(x) = 1 / (1 + exp(-x))   (logistic function)

  ``saturation_scale`` is configurable (default: 100 mm).  At this value
  the sigmoid output is ~0.73.  This provides a smooth transition from
  dry (0) to wet (1) conditions without a hard threshold.

Limitations
-----------
  1. This is a rainfall-derived proxy, not a soil physics model.
  2. Spatial variability in soil properties is not captured.
  3. Evapotranspiration, drainage, and runoff are not modelled.
  4. The proxy cannot detect dry conditions caused by vegetation uptake or
     drainage without antecedent rainfall data.
  5. For production use, a physically-based soil moisture model or
     satellite-derived product (e.g. SMAP, Copernicus CGLS) should be used.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Default saturation scale (mm): at this API value, moisture_indicator ≈ 0.73
_DEFAULT_SATURATION_SCALE_MM = 100.0


def estimate_moisture_indicator(
    antecedent_rainfall: Optional[float],
    saturation_scale_mm: float = _DEFAULT_SATURATION_SCALE_MM,
) -> Optional[float]:
    """
    Estimate a dimensionless environmental moisture proxy from antecedent rainfall.

    Parameters
    ----------
    antecedent_rainfall : float | None
        Antecedent rainfall index (mm-equivalent) from
        ``rainfall_windows.calculate_antecedent_rainfall``.
        None if insufficient history.
    saturation_scale_mm : float
        Scale parameter controlling the sigmoid midpoint.
        At ``antecedent_rainfall == saturation_scale_mm``, the indicator is ~0.73.
        Default: 100 mm.

    Returns
    -------
    float | None
        Moisture proxy in [0, 1].
        0 → very dry antecedent conditions.
        1 → fully saturated antecedent conditions (asymptotic).
        None → insufficient history to estimate.

    Notes
    -----
    This is NOT a physical soil moisture measurement.
    See module docstring for full limitations.
    """
    if antecedent_rainfall is None:
        logger.debug("antecedent_rainfall is None; moisture_indicator = None")
        return None

    if antecedent_rainfall < 0:
        logger.warning("antecedent_rainfall = %.3f < 0; clamping to 0", antecedent_rainfall)
        antecedent_rainfall = 0.0

    if saturation_scale_mm <= 0:
        raise ValueError(f"saturation_scale_mm must be > 0, got {saturation_scale_mm}")

    # Sigmoid normalisation
    # x is centred so that x=0 at antecedent_rainfall=0 → sigmoid(negative) → near 0
    # Shift x so that API=0 → indicator≈0 and API=saturation_scale → indicator≈0.73
    x = (antecedent_rainfall / saturation_scale_mm) * 2 - 2  # maps [0, scale] → [-2, 0]
    indicator = 1.0 / (1.0 + math.exp(-x))

    # Clamp to [0, 1] for numerical safety
    indicator = max(0.0, min(1.0, indicator))
    return round(indicator, 4)


def estimate_moisture_from_rainfall_windows(
    rainfall_7d: Optional[float],
    rainfall_24h: Optional[float],
    saturation_scale_mm: float = _DEFAULT_SATURATION_SCALE_MM,
) -> Optional[float]:
    """
    Alternative estimator using windowed rainfall totals directly.

    This can be used when the full observation series is not available but
    the aggregated window values are.

    Parameters
    ----------
    rainfall_7d : float | None
        7-day accumulated rainfall (mm).
    rainfall_24h : float | None
        24-hour accumulated rainfall (mm).
    saturation_scale_mm : float
        Scale parameter.

    Returns
    -------
    float | None
        Moisture proxy in [0, 1].  None if both inputs are None.
    """
    if rainfall_7d is None and rainfall_24h is None:
        return None

    # Weighted proxy: 7d has more weight (antecedent memory), 24h captures current event
    r7d = rainfall_7d if rainfall_7d is not None else 0.0
    r24h = rainfall_24h if rainfall_24h is not None else 0.0

    # Weighted antecedent proxy (weights are heuristic, not calibrated)
    proxy = 0.6 * r7d / 7.0 + 0.4 * r24h  # per-day 7d + full 24h
    return estimate_moisture_indicator(proxy, saturation_scale_mm=saturation_scale_mm)
