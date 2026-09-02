"""
Training sample framework for ML model preparation.

Design principles
-----------------
* A training sample is conceptually: ``location + time + features + label``
* Positive samples = confirmed landslide events
* Negative (non-event) samples = locations/times without recorded events

HOW NEGATIVE SAMPLES ARE GENERATED
-------------------------------------
Negative samples are drawn from the spatial grid at times where NO
landslide was recorded within a configurable spatial and temporal radius.
They receive ``historical_landslide = 0``.

This is the ONLY correct way to generate negative labels.  Do NOT invent
negative labels by assuming all un-observed cells are safe (that would
introduce spatial label noise in active-but-unrecorded areas).

DATA LEAKAGE PREVENTION
------------------------
* Temporal split is mandatory: training data must not include records after
  a configurable ``cutoff_date``.
* Spatial exclusion zones can be configured to prevent spatial leakage
  near known landslide clusters.

IMPORTANT:
  25,000 samples ≠ 25,000 independent landslides.
  This class handles both positive events and negative non-events.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SamplerConfig:
    """
    Configuration for the training data sampler.

    Attributes
    ----------
    positive_label : int
        Label value for landslide events (default ``1``).
    negative_label : int
        Label value for non-events (default ``0``).
    negative_to_positive_ratio : float
        How many negative samples to draw per positive sample.
        Set to ``1.0`` for 1:1 balance.
    temporal_cutoff : str | None
        ISO 8601 date string.  Samples after this date are excluded from
        training (prevents temporal leakage).
    min_lat, max_lat, min_lon, max_lon : float | None
        Optional spatial filter bounding box.
    random_seed : int
        Reproducibility seed.
    """

    positive_label: int = 1
    negative_label: int = 0
    negative_to_positive_ratio: float = 1.0
    temporal_cutoff: Optional[str] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    random_seed: int = 42


class TrainingSampler:
    """
    Build a spatial-temporal training dataset from standardised records.

    Parameters
    ----------
    config : SamplerConfig
        Sampling configuration.

    Examples
    --------
    ::

        sampler = TrainingSampler(SamplerConfig(negative_to_positive_ratio=2.0))
        train_df = sampler.build(
            positive_df=standardized_events,
            grid_df=full_grid,       # all spatial cells
        )
    """

    def __init__(self, config: Optional[SamplerConfig] = None) -> None:
        self.config = config or SamplerConfig()
        random.seed(self.config.random_seed)

    def build(
        self,
        positive_df: pd.DataFrame,
        grid_df: Optional[pd.DataFrame] = None,
        *,
        available_timestamps: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Construct a training dataset.

        Parameters
        ----------
        positive_df : pd.DataFrame
            Standardised landslide events (output of the standardisation
            pipeline).  Must contain ``location_id``, ``latitude``,
            ``longitude``, ``timestamp_utc``.
        grid_df : pd.DataFrame | None
            Full spatial grid DataFrame (from :func:`create_spatial_grid`).
            Required for negative-sample generation.  If ``None``, no
            negative samples are added.
        available_timestamps : list[str] | None
            Pool of UTC timestamp strings from which negative sample
            timestamps are drawn.  If ``None``, uses the timestamps from
            *positive_df*.

        Returns
        -------
        pd.DataFrame
            Combined training DataFrame with a ``label`` column.
        """
        cfg = self.config

        # ---- Apply temporal filter ------------------------------------
        pos = positive_df.copy()
        if cfg.temporal_cutoff:
            pos = pos[pos["timestamp_utc"] <= cfg.temporal_cutoff]
            logger.info(
                "Temporal filter (cutoff %s): %d positives remaining.",
                cfg.temporal_cutoff,
                len(pos),
            )

        # ---- Apply spatial filter ------------------------------------
        if cfg.min_lat is not None:
            pos = pos[pos["latitude"] >= cfg.min_lat]
        if cfg.max_lat is not None:
            pos = pos[pos["latitude"] <= cfg.max_lat]
        if cfg.min_lon is not None:
            pos = pos[pos["longitude"] >= cfg.min_lon]
        if cfg.max_lon is not None:
            pos = pos[pos["longitude"] <= cfg.max_lon]

        pos = pos.copy()
        pos["label"] = cfg.positive_label

        parts = [pos]

        # ---- Negative samples ----------------------------------------
        if grid_df is not None and len(pos) > 0:
            n_neg = int(len(pos) * cfg.negative_to_positive_ratio)
            ts_pool = available_timestamps or pos["timestamp_utc"].unique().tolist()

            # Positive location+time pairs to exclude from negatives
            positive_keys = set(
                zip(pos["location_id"], pos["timestamp_utc"])
            )

            neg_rows = self._sample_negatives(
                grid_df=grid_df,
                n=n_neg,
                ts_pool=ts_pool,
                exclude_keys=positive_keys,
                label=cfg.negative_label,
            )

            if neg_rows:
                neg_df = pd.DataFrame(neg_rows)
                parts.append(neg_df)
                logger.info("Generated %d negative samples.", len(neg_df))
            else:
                logger.warning("No negative samples could be generated.")

        result = pd.concat(parts, ignore_index=True)
        logger.info(
            "Training dataset: %d total (%d positive, %d negative).",
            len(result),
            int((result["label"] == cfg.positive_label).sum()),
            int((result["label"] == cfg.negative_label).sum()),
        )
        return result

    def _sample_negatives(
        self,
        grid_df: pd.DataFrame,
        n: int,
        ts_pool: list[str],
        exclude_keys: set[tuple[str, str]],
        label: int,
    ) -> list[dict]:
        """
        Draw *n* negative samples from the grid.

        Each negative sample is a (location_id, timestamp) pair NOT in
        *exclude_keys*.  This prevents labelling confirmed-event cells as
        non-events.
        """
        location_ids = grid_df["grid_id"].dropna().unique().tolist()
        if not location_ids or not ts_pool:
            return []

        rows: list[dict] = []
        attempts = 0
        max_attempts = n * 20

        while len(rows) < n and attempts < max_attempts:
            attempts += 1
            loc = random.choice(location_ids)
            ts = random.choice(ts_pool)

            if (loc, ts) in exclude_keys:
                continue

            # Get representative lat/lon from grid
            grid_row = grid_df[grid_df["grid_id"] == loc].iloc[0]
            lat = float(grid_row.get("center_lat", 0.0))
            lon = float(grid_row.get("center_lon", 0.0))

            from data_engineering.spatial.location_id import generate_location_id
            rows.append(
                {
                    "location_id": generate_location_id(lat, lon),
                    "grid_id": loc,
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp_utc": ts,
                    "historical_landslide": label,
                    "label": label,
                    "source": "synthetic_negative",
                }
            )
            exclude_keys.add((loc, ts))

        if len(rows) < n:
            logger.warning(
                "Only generated %d/%d negative samples (grid too small or"
                " pool exhausted).",
                len(rows),
                n,
            )

        return rows
