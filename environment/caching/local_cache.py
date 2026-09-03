"""
environment/caching/local_cache.py

Simple in-memory TTL (time-to-live) cache for weather provider responses.

Purpose
-------
Avoid repeatedly requesting the same observation window from a provider
within a short time period.  The cache stores provider responses keyed by
(latitude, longitude, start_utc, end_utc).

Design decisions
----------------
- In-memory only (not persistent across restarts).  Sufficient for prototype.
- Thread-safe via threading.Lock.
- TTL is configurable; set to 0 to disable caching (useful in tests).
- The cache stores DataFrames (observation data) and lists (forecast records).
- Cache eviction is lazy (entries are not evicted until accessed or cleared).

For production use
------------------
Replace with a distributed cache (Redis, Memcached) or a time-series database
(InfluxDB, TimescaleDB).  The LocalCache interface is designed to allow this
replacement without changing pipeline code.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache key type
_Key = Tuple[str, float, float, str, str]


class LocalCache:
    """
    Thread-safe in-memory TTL cache.

    Parameters
    ----------
    ttl_seconds : int
        Time-to-live for cache entries, in seconds.
        Set to 0 or negative to disable caching entirely.
    max_size : int
        Maximum number of entries.  Oldest entries are evicted when full.
        Default: 1000.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._lock = threading.Lock()
        logger.debug("LocalCache initialised: ttl=%ds, max_size=%d", ttl_seconds, max_size)

    @property
    def enabled(self) -> bool:
        """True if caching is active (TTL > 0)."""
        return self._ttl > 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value, or None if missing / expired.

        Parameters
        ----------
        key : str
            Cache key (use ``make_key`` to build standard keys).

        Returns
        -------
        Any | None
            The cached value, or None on cache miss or expiry.
        """
        if not self.enabled:
            return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.debug("Cache MISS: %s", key)
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                logger.debug("Cache EXPIRED: %s", key)
                return None
            logger.debug("Cache HIT: %s", key)
            return value

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache with the configured TTL.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.  Stored by reference — do not mutate after caching.
        """
        if not self.enabled:
            return

        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._store) >= self._max_size:
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]
                logger.debug("Cache evicted oldest entry: %s", oldest_key)

            expiry = time.monotonic() + self._ttl
            self._store[key] = (value, expiry)
            logger.debug("Cache SET: %s (TTL=%ds)", key, self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()
        logger.debug("Cache cleared")

    def size(self) -> int:
        """Return the number of entries currently in the cache."""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Key builders
    # ------------------------------------------------------------------

    @staticmethod
    def make_observation_key(
        latitude: float,
        longitude: float,
        start_utc: str,
        end_utc: str,
        provider: str,
    ) -> str:
        """Build a cache key for an observation request."""
        return f"obs:{provider}:{latitude:.6f}:{longitude:.6f}:{start_utc}:{end_utc}"

    @staticmethod
    def make_forecast_key(
        latitude: float,
        longitude: float,
        horizon_hours: int,
        provider: str,
    ) -> str:
        """Build a cache key for a forecast request."""
        return f"fcast:{provider}:{latitude:.6f}:{longitude:.6f}:{horizon_hours}h"

    @staticmethod
    def make_feature_key(
        location_id: str,
        timestamp_utc: str,
    ) -> str:
        """Build a cache key for a computed DynamicRecord."""
        return f"features:{location_id}:{timestamp_utc}"
