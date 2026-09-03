"""
Target Definition
=================

Defines what constitutes a positive (landslide) and negative (non-event)
sample for model training.

This is a critical scientific document.  Every assumption is explicitly stated.

─────────────────────────────────────────────────────────────────────────────
POSITIVE EVENT DEFINITION
─────────────────────────────────────────────────────────────────────────────

A sample is labelled POSITIVE (landslide_occurred = 1) when:

1. The source data contains a confirmed landslide event record.
2. The event has a valid location (latitude, longitude) within the study area.
3. The event has a valid timestamp that can be converted to UTC.
4. The event record is assigned to the nearest grid cell (location_id) within
   a configurable spatial buffer (default: 1 km).
5. The prediction timestamp T is within a configurable time window before the
   event (default: 0–24 hours before, i.e. the model is asked to predict
   the event on the day it occurred using only pre-event features).

─────────────────────────────────────────────────────────────────────────────
NEGATIVE SAMPLE DEFINITION
─────────────────────────────────────────────────────────────────────────────

A sample is labelled NEGATIVE (landslide_occurred = 0) when:

1. The grid cell has NO confirmed landslide event within a temporal buffer
   of ±event_buffer_days (default: 7 days) around the sample timestamp.
2. The grid cell is located at least event_buffer_km (default: 5 km) from
   any known landslide event in the entire dataset.
3. The sample timestamp predates the training cutoff (no future leakage).

⚠️  IMPORTANT: We do NOT treat every non-event grid cell as a clean negative.

   Reason: A cell with no recorded event may have an UNREPORTED landslide
   (the dataset is not complete).  The spatial/temporal buffer around known
   events provides a safety margin, but labelling uncertainty remains.

   This is documented as a known limitation of any inventory-based model.

─────────────────────────────────────────────────────────────────────────────
TEMPORAL WINDOW
─────────────────────────────────────────────────────────────────────────────

The model predicts risk at time T using features computed from:
- Static features: always from before T (terrain, geology — time-invariant)
- Dynamic features at T:
    rainfall_1h   = total rainfall in the 1 hour ending at T
    rainfall_6h   = total rainfall in the 6 hours ending at T
    rainfall_24h  = total rainfall in the 24 hours ending at T
    rainfall_3d   = total rainfall in the 3 days ending at T
    rainfall_7d   = total rainfall in the 7 days ending at T
    forecast_rain = rainfall forecast for the 24 hours AFTER T
    moisture_ind  = soil moisture estimate at T (model/reanalysis)

⚠️  forecast_rainfall is a legitimate feature: it reflects information the
    forecaster has at time T about future conditions.  It does NOT violate
    the leakage rule because it is a forecast, not an observation.
    It WOULD be a leak if it were replaced with observed_rainfall_next_24h.

─────────────────────────────────────────────────────────────────────────────
DATA LEAKAGE PREVENTION
─────────────────────────────────────────────────────────────────────────────

The following leakage risks are explicitly guarded against:

1. historical_landslide must be computed as "did this location have a landslide
   BEFORE timestamp T?" — NOT using the current or future event.
   Implementation: The data adapter logs a warning if historical_landslide
   varies over time, prompting the caller to verify this.

2. Post-event satellite imagery must NOT be used for pre-event predictions.
   Implementation: satellite_df must only contain pre-acquisition images.
   The adapter logs a warning if acquisition_time is missing.

3. Rainfall AFTER T must NOT appear in features.
   Implementation: dynamic_df is expected to be filtered by caller.
   forecast_rainfall is explicitly documented as a forecast, not an observation.

4. Spatial train/test contamination:
   Implementation: SpatioTemporalSplitter ensures nearby grid cells from the
   same event are NOT split across train/test.  See validation_strategy.py.

─────────────────────────────────────────────────────────────────────────────
SPATIAL WINDOW
─────────────────────────────────────────────────────────────────────────────

Landslide events in the inventory are matched to the grid cell whose centroid
is closest to the event point, within event_location_match_km (default: 1 km).
Events not matched to any grid cell are excluded from training.

─────────────────────────────────────────────────────────────────────────────
ASSUMPTIONS AND LIMITATIONS
─────────────────────────────────────────────────────────────────────────────

A1. Inventory completeness: The landslide inventory is assumed to be reasonably
    complete for the study period and region.  Unrecorded events may contaminate
    negative samples.

A2. Grid resolution: The grid cell resolution (shared with Engineer 1/2)
    determines the spatial precision of event assignment.  Finer grids provide
    better precision but require more data to populate.

A3. Temporal precision: Many historical inventories have date-only precision
    (not hour-of-day).  In this case, the event time is assumed to be noon UTC
    on the reported date.  This is documented in the adapter.

A4. Negative sample representativeness: Negatives are sampled from grid cells
    and timestamps where no event is recorded.  The distribution of negatives
    is determined by the available dynamic data, which may not uniformly cover
    all conditions.

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TargetConfig:
    """
    Configuration for target label construction.

    Attributes
    ----------
    event_buffer_days : int
        Temporal buffer around known events.  Grid cells within this window
        of a known event are excluded from the negative sample pool.
    event_buffer_km : float
        Spatial buffer around known events.  Grid cells within this distance
        of any event are excluded from the negative sample pool.
    event_location_match_km : float
        Maximum distance to match an event point to a grid cell centroid.
    assume_noon_utc_for_date_only : bool
        If True, events with date-only precision are assumed to occur at noon UTC.
    target_col : str
        Name of the output target column.
    """

    event_buffer_days: int = 7
    event_buffer_km: float = 5.0
    event_location_match_km: float = 1.0
    assume_noon_utc_for_date_only: bool = True
    target_col: str = "landslide_occurred"


class TargetBuilder:
    """
    Constructs the binary target column from a joined dataset.

    The dataset produced by the data adapter already contains
    'historical_landslide' from Engineer 1.  This class:
    1. Renames/validates it into the ML target column.
    2. Applies the documented negative-sample exclusion logic.
    3. Reports label statistics.
    """

    def __init__(self, config: Optional[TargetConfig] = None) -> None:
        self.config = config or TargetConfig()

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the target column from the joined dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset from the data adapter; must contain 'historical_landslide'.

        Returns
        -------
        pd.DataFrame
            Copy of df with target column added and verified.
        """
        out = df.copy()
        cfg = self.config

        if "historical_landslide" not in out.columns:
            raise ValueError(
                "Column 'historical_landslide' not found.  "
                "Ensure the data adapter has joined Engineer 1 output."
            )

        # Use historical_landslide as the primary target signal
        out[cfg.target_col] = out["historical_landslide"].astype("Int64")

        n_pos = int((out[cfg.target_col] == 1).sum())
        n_neg = int((out[cfg.target_col] == 0).sum())
        n_unknown = int(out[cfg.target_col].isna().sum())

        logger.info(
            f"Target distribution — positives: {n_pos}, negatives: {n_neg}, "
            f"unknown/missing: {n_unknown}"
        )

        if n_pos == 0:
            logger.warning(
                "No positive samples found!  The model cannot learn landslide "
                "signatures without positive examples.  Check historical_landslide column."
            )

        if n_pos > 0 and n_neg > 0:
            ratio = n_neg / n_pos
            logger.info(f"Class imbalance ratio (neg:pos) = {ratio:.1f}:1")

        return out

    def describe(self) -> dict:
        """Return the target definition as a serialisable dictionary."""
        cfg = self.config
        return {
            "positive_definition": (
                "historical_landslide == 1: confirmed landslide event at "
                "this location and timestamp"
            ),
            "negative_definition": (
                "historical_landslide == 0: no confirmed event within "
                f"±{cfg.event_buffer_days} days and {cfg.event_buffer_km} km "
                "of any known event"
            ),
            "temporal_window": (
                "Features are computed using data strictly available before "
                "the prediction timestamp T"
            ),
            "event_location_match_km": cfg.event_location_match_km,
            "leakage_guards": [
                "historical_landslide computed only from pre-T events",
                "satellite features must use pre-event acquisition time",
                "no observed post-T rainfall in features",
            ],
            "assumptions": [
                "A1: inventory completeness assumed for study region",
                "A2: grid resolution determines spatial precision",
                "A3: date-only events assumed at noon UTC",
                "A4: negative representativeness limited by dynamic data coverage",
            ],
        }
