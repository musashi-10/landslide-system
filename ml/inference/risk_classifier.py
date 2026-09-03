"""
Risk Level Classifier
=====================

Converts a continuous risk probability in [0, 1] to a discrete risk level:
  LOW | MODERATE | HIGH | CRITICAL

Threshold policy
----------------
⚠️  The default thresholds below are PROVISIONAL STARTING VALUES.

They have NOT been scientifically validated against operational data.
Before operational deployment, thresholds should be determined by:

1. Plotting the precision-recall curve on the validation set.
2. Selecting recall targets based on acceptable miss-rate constraints
   (e.g. "we want to catch at least 80% of HIGH events").
3. Consulting domain experts about the cost ratio of false alarms
   vs missed events.
4. Calibrating against historical alert verification data.

Until this is done, document all thresholds as "provisional".

Default thresholds
------------------
  probability < 0.30           → LOW
  0.30 ≤ probability < 0.55   → MODERATE
  0.55 ≤ probability < 0.75   → HIGH
  probability ≥ 0.75           → CRITICAL
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from ml.config.training_config import RiskThresholds


class RiskLevel(str, Enum):
    """Discrete risk level values."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


_VALID_LEVELS = {rl.value for rl in RiskLevel}


class RiskClassifier:
    """
    Maps probability → RiskLevel using configurable thresholds.

    Parameters
    ----------
    thresholds : RiskThresholds, optional
        Risk level boundary values.  Uses defaults if not provided.

    Example
    -------
        clf = RiskClassifier()
        level = clf.classify(0.82)   # → RiskLevel.CRITICAL
        level = clf.classify(0.40)   # → RiskLevel.MODERATE
    """

    def __init__(self, thresholds: Optional[RiskThresholds] = None) -> None:
        self.thresholds = thresholds or RiskThresholds()
        self.thresholds.validate()

    def classify(self, probability: float) -> RiskLevel:
        """
        Convert a risk probability to a risk level.

        Parameters
        ----------
        probability : float
            Risk probability in [0.0, 1.0].

        Returns
        -------
        RiskLevel

        Raises
        ------
        ValueError : if probability is outside [0, 1].
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError(
                f"Risk probability must be in [0.0, 1.0], got {probability}."
            )

        t = self.thresholds
        if probability < t.low_max:
            return RiskLevel.LOW
        elif probability < t.moderate_max:
            return RiskLevel.MODERATE
        elif probability < t.high_max:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def classify_many(self, probabilities: list[float]) -> list[RiskLevel]:
        """Classify a list of probabilities."""
        return [self.classify(p) for p in probabilities]

    def threshold_summary(self) -> dict:
        """Return threshold values as a dict for serialisation."""
        t = self.thresholds
        return {
            "LOW": f"< {t.low_max}",
            "MODERATE": f"{t.low_max} – {t.moderate_max}",
            "HIGH": f"{t.moderate_max} – {t.high_max}",
            "CRITICAL": f">= {t.critical_min}",
            "status": "PROVISIONAL — not scientifically validated",
        }


def parse_risk_level(value: str) -> RiskLevel:
    """
    Parse a string to a RiskLevel, raising ValueError if invalid.

    Parameters
    ----------
    value : str  (case-insensitive)
    """
    upper = value.upper()
    if upper not in _VALID_LEVELS:
        raise ValueError(
            f"'{value}' is not a valid risk level. Expected one of: {sorted(_VALID_LEVELS)}"
        )
    return RiskLevel(upper)
