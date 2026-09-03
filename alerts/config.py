"""
Alert engine configuration.

All thresholds are configurable — not scientifically validated values.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlertConfig:
    """
    Parameters that control when the alert engine fires.

    Attributes
    ----------
    min_risk_level : str
        Minimum risk level that triggers an alert.
        One of: LOW | MODERATE | HIGH | CRITICAL.
    cooldown_seconds : int
        Minimum seconds between two alerts for the same location.
        Prevents alert storms from repeated high-risk predictions.
    persistence_count : int
        Number of consecutive predictions at or above min_risk_level
        required before an alert is created.  Set to 1 to alert on
        the first qualifying prediction.
    enabled : bool
        Master switch. When False, no alerts are created.
    """

    min_risk_level: str = "HIGH"
    cooldown_seconds: int = 3600
    persistence_count: int = 1
    enabled: bool = True
