"""
environment/config.py

Configuration for the dynamic environmental-condition pipeline.

All window sizes, cache settings, and provider choices are centralised here.
Nothing in the pipeline should hard-code these values.

Configuration is loaded from environment variables (via shared Settings) and
can also be overridden programmatically for tests.

Rainfall window sizes (hours)
-----------------------------
window_1h  : 1   — captures intense short bursts
window_6h  : 6   — sub-daily storm accumulation
window_24h : 24  — daily accumulation (primary landslide trigger window)
window_3d  : 72  — multi-day saturation
window_7d  : 168 — antecedent wetness / deep infiltration proxy

These sizes come from published landslide-rainfall threshold literature
(e.g. Guzzetti et al., 2007; Caine, 1980) and are kept configurable so
the ML team can experiment with different windows.

Units: all rainfall in millimetres (mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EnvironmentConfig:
    """
    Pipeline-wide configuration object.

    Attributes
    ----------
    window_hours : dict
        Mapping of feature name -> window size in hours.
        Changing these values changes what gets computed — update tests accordingly.
    forecast_horizon_hours : int
        How many hours of forecast to accumulate for ``forecast_rainfall``.
    cache_ttl_seconds : int
        Time-to-live for cached provider responses.
        Set to 0 to disable caching (useful in tests).
    provider : str
        Default provider name.  Supported values: 'open_meteo', 'mock'.
        Resolved at runtime by the provider factory.
    moisture_decay_factor : float
        Exponential decay factor for the antecedent rainfall index.
        Higher values give more weight to recent rainfall.
        Range: (0, 1).  Default: 0.85 (moderate memory).
    max_valid_hourly_rainfall_mm : float
        Upper bound for a plausible hourly rainfall value (mm).
        Values above this are flagged as suspect and logged.
        This is NOT used to discard data, only to log warnings.
        Reference: world record ~300 mm/h; practical limit ~100 mm/h for
        large-scale grid products.
    preprocessing_version : str
        Version string stamped on every output record.
    """

    # Rainfall accumulation windows
    window_hours: Dict[str, int] = field(
        default_factory=lambda: {
            "1h": 1,
            "6h": 6,
            "24h": 24,
            "3d": 72,
            "7d": 168,
        }
    )

    # Forecast accumulation horizon
    forecast_horizon_hours: int = 24

    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour

    # Provider
    provider: str = "open_meteo"

    # Moisture indicator
    moisture_decay_factor: float = 0.85

    # Validation / QC
    max_valid_hourly_rainfall_mm: float = 200.0  # mm/hour — flag if exceeded

    # Versioning
    preprocessing_version: str = "1.0.0"


# Default singleton used by the pipeline when no explicit config is passed
DEFAULT_CONFIG = EnvironmentConfig()
