"""
environment/tests/test_cache.py

Tests for LocalCache — hit, miss, TTL expiry, thread safety, key builders.
"""

from __future__ import annotations

import time
import threading

import pytest

from environment.caching.local_cache import LocalCache


class TestLocalCacheBasics:
    def test_cache_miss_returns_none(self):
        cache = LocalCache(ttl_seconds=60)
        assert cache.get("nonexistent_key") is None

    def test_cache_set_and_get(self):
        cache = LocalCache(ttl_seconds=60)
        cache.set("key1", {"data": 42})
        result = cache.get("key1")
        assert result == {"data": 42}

    def test_cache_hit(self):
        cache = LocalCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_ttl_expiry(self):
        """Entry should expire after TTL."""
        cache = LocalCache(ttl_seconds=1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_zero_ttl_disables_cache(self):
        cache = LocalCache(ttl_seconds=0)
        assert not cache.enabled
        cache.set("key1", "value1")
        assert cache.get("key1") is None

    def test_negative_ttl_disables_cache(self):
        cache = LocalCache(ttl_seconds=-1)
        assert not cache.enabled

    def test_clear_removes_all(self):
        cache = LocalCache(ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_invalidate_specific_key(self):
        cache = LocalCache(ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate("a")
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_size_count(self):
        cache = LocalCache(ttl_seconds=60)
        assert cache.size() == 0
        cache.set("a", 1)
        assert cache.size() == 1
        cache.set("b", 2)
        assert cache.size() == 2

    def test_max_size_eviction(self):
        cache = LocalCache(ttl_seconds=60, max_size=3)
        for i in range(5):
            cache.set(f"key{i}", i)
        # Should not exceed max_size
        assert cache.size() <= 3

    def test_overwrite_same_key(self):
        cache = LocalCache(ttl_seconds=60)
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"


class TestLocalCacheThreadSafety:
    def test_concurrent_writes(self):
        """Multiple threads writing to cache should not raise."""
        cache = LocalCache(ttl_seconds=60)
        errors = []

        def writer(i):
            try:
                for j in range(100):
                    cache.set(f"key_{i}_{j}", i * j)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


class TestLocalCacheKeyBuilders:
    def test_observation_key_format(self):
        key = LocalCache.make_observation_key(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "mock")
        assert "obs:" in key
        assert "mock" in key
        assert "27.123" in key

    def test_forecast_key_format(self):
        key = LocalCache.make_forecast_key(27.123, 88.456, 24, "open_meteo")
        assert "fcast:" in key
        assert "24h" in key
        assert "open_meteo" in key

    def test_feature_key_format(self):
        key = LocalCache.make_feature_key("LOC_001", "2026-09-02T12:00:00Z")
        assert "features:" in key
        assert "LOC_001" in key

    def test_observation_key_is_deterministic(self):
        k1 = LocalCache.make_observation_key(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "mock")
        k2 = LocalCache.make_observation_key(27.123, 88.456, "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "mock")
        assert k1 == k2

    def test_different_providers_produce_different_keys(self):
        k1 = LocalCache.make_observation_key(27.0, 88.0, "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "mock")
        k2 = LocalCache.make_observation_key(27.0, 88.0, "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "open_meteo")
        assert k1 != k2
