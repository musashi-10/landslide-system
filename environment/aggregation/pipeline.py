"""
environment/aggregation/pipeline.py

Batch dynamic feature pipeline.

Processes multiple locations at a given timestamp, returning a list of
DynamicRecord objects.  Handles per-location errors without failing the
entire batch — a failing location is logged and skipped.

This is the entry point for batch processing (e.g. nightly feature generation
for all monitored locations).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from environment.aggregation.feature_builder import build_dynamic_features, dynamic_record_to_dict
from environment.caching.local_cache import LocalCache
from environment.config import DEFAULT_CONFIG, EnvironmentConfig
from environment.providers.base import WeatherProvider
from environment.schemas.dynamic_record import DataProvenance, DynamicRecord

logger = logging.getLogger(__name__)


@dataclass
class LocationSpec:
    """Input specification for one location."""

    location_id: str
    latitude: float
    longitude: float


@dataclass
class PipelineResult:
    """Output from a batch pipeline run."""

    records: List[DynamicRecord]
    failed_locations: List[Tuple[str, str]]  # (location_id, error_message)
    provenance: DataProvenance
    timestamp_utc: str

    @property
    def success_count(self) -> int:
        return len(self.records)

    @property
    def failure_count(self) -> int:
        return len(self.failed_locations)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a pandas DataFrame for ML consumption."""
        if not self.records:
            return pd.DataFrame()
        rows = [dynamic_record_to_dict(r) for r in self.records]
        return pd.DataFrame(rows)


class DynamicFeaturePipeline:
    """
    Batch processor: compute dynamic features for multiple locations.

    Parameters
    ----------
    provider : WeatherProvider
        The data provider to use for all locations.
    config : EnvironmentConfig | None
        Pipeline configuration.
    cache : LocalCache | None
        Cache instance (shared across all location calls).
    """

    def __init__(
        self,
        provider: WeatherProvider,
        config: Optional[EnvironmentConfig] = None,
        cache: Optional[LocalCache] = None,
    ) -> None:
        self._provider = provider
        self._config = config or DEFAULT_CONFIG
        self._cache = cache or LocalCache(ttl_seconds=self._config.cache_ttl_seconds)

    def run(
        self,
        locations: List[LocationSpec],
        timestamp_utc: str,
    ) -> PipelineResult:
        """
        Process all locations at the given timestamp.

        Parameters
        ----------
        locations : list[LocationSpec]
            Locations to process.
        timestamp_utc : str
            ISO 8601 UTC timestamp for "now".

        Returns
        -------
        PipelineResult
            Contains successful records, failures, and provenance.
        """
        logger.info(
            "DynamicFeaturePipeline: processing %d locations at %s with provider='%s'",
            len(locations), timestamp_utc, self._provider.name,
        )

        records: List[DynamicRecord] = []
        failed: List[Tuple[str, str]] = []

        for loc in locations:
            try:
                record = build_dynamic_features(
                    location_id=loc.location_id,
                    latitude=loc.latitude,
                    longitude=loc.longitude,
                    timestamp_utc=timestamp_utc,
                    provider=self._provider,
                    config=self._config,
                    cache=self._cache,
                )
                records.append(record)
            except Exception as exc:
                logger.error(
                    "Pipeline failed for location %s (%.4f, %.4f): %s",
                    loc.location_id, loc.latitude, loc.longitude, exc,
                )
                failed.append((loc.location_id, str(exc)))

        provenance = self._provider.provenance

        logger.info(
            "Pipeline complete: %d succeeded, %d failed",
            len(records), len(failed),
        )
        return PipelineResult(
            records=records,
            failed_locations=failed,
            provenance=provenance,
            timestamp_utc=timestamp_utc,
        )
