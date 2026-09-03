"""
environment/providers/base.py

Abstract base class for all weather / rainfall data providers.

Every provider must implement the two abstract methods:
  - get_observations() — historical hourly rainfall
  - get_forecast()     — future hourly rainfall

The pipeline depends ONLY on this interface.  Adding a new provider
requires only implementing this class; no pipeline code changes.

Error contract
--------------
Providers must raise ``ProviderError`` (or a subclass) for all failure modes.
They must NOT raise bare exceptions that would propagate uncaught.
The pipeline catches ``ProviderError`` and handles graceful degradation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from environment.schemas.dynamic_record import (
    DataProvenance,
    ForecastRecord,
    RawObservation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base exception for all provider failures."""


class ProviderTimeoutError(ProviderError):
    """Raised when the provider API times out."""


class ProviderUnavailableError(ProviderError):
    """Raised when the provider API is unreachable or returns HTTP 5xx."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider returns HTTP 429 (rate limit exceeded)."""


class ProviderInvalidResponseError(ProviderError):
    """Raised when the provider response cannot be parsed or is malformed."""


class ProviderMissingDataError(ProviderError):
    """Raised when the provider has no data for the requested location / period."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class WeatherProvider(ABC):
    """
    Abstract interface for weather / rainfall data providers.

    All concrete providers (Open-Meteo, mock, etc.) must implement this
    interface.  The pipeline code depends only on ``WeatherProvider``.

    Spatial grid note
    -----------------
    Providers may have their own spatial grid that differs from the project
    grid.  Spatial mapping (provider grid -> project location_id) is handled
    in ``environment.ingestion.rainfall_ingestor`` — providers just return
    data at the requested coordinates.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider, e.g. 'open_meteo', 'mock'."""

    @property
    @abstractmethod
    def provenance(self) -> DataProvenance:
        """
        Return static provenance metadata for this provider.

        The pipeline attaches this to every batch of output records.
        """

    @abstractmethod
    def get_observations(
        self,
        latitude: float,
        longitude: float,
        start_utc: str,
        end_utc: str,
    ) -> List[RawObservation]:
        """
        Fetch hourly rainfall observations for a coordinate pair.

        Parameters
        ----------
        latitude : float
            WGS-84 latitude in decimal degrees.
        longitude : float
            WGS-84 longitude in decimal degrees.
        start_utc : str
            Start of the observation window (ISO 8601 UTC, inclusive).
        end_utc : str
            End of the observation window (ISO 8601 UTC, inclusive).

        Returns
        -------
        List[RawObservation]
            Hourly observations ordered chronologically.
            Missing hours are represented as ``RawObservation(..., rainfall_mm=None)``,
            NOT omitted from the list.

        Raises
        ------
        ProviderError (or subclass)
            On any failure.  The pipeline catches this and handles gracefully.
        """

    @abstractmethod
    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        horizon_hours: int,
    ) -> List[ForecastRecord]:
        """
        Fetch hourly rainfall forecast for a coordinate pair.

        Parameters
        ----------
        latitude : float
            WGS-84 latitude in decimal degrees.
        longitude : float
            WGS-84 longitude in decimal degrees.
        horizon_hours : int
            How many future hours to fetch.

        Returns
        -------
        List[ForecastRecord]
            Hourly forecast records ordered chronologically.
            Returns an empty list (not None) if no forecast is available.
            Missing hours are ``ForecastRecord(..., forecast_rainfall_mm=None)``.

        Raises
        ------
        ProviderError (or subclass)
            On any failure.  The pipeline catches this and handles gracefully.
        """
