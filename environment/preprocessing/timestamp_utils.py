"""
environment/preprocessing/timestamp_utils.py

Timestamp normalisation for the dynamic environment pipeline.

Wraps Engineer 1's ``data_engineering.validation.normalize_timestamp`` to
avoid duplicating logic.  Falls back to a local implementation if the
data_engineering module is unavailable (e.g. during isolated unit tests).

All timestamps produced by this module are in UTC ISO 8601 format:
    YYYY-MM-DDTHH:MM:SSZ

Examples
--------
>>> normalize_to_utc("2026-09-01T06:00:00Z")
'2026-09-01T06:00:00Z'
>>> normalize_to_utc("2026-09-01T06:00:00+05:30")
'2026-09-01T00:30:00Z'
>>> normalize_to_utc(None)  # raises TimestampNormalisationError
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class TimestampNormalisationError(ValueError):
    """Raised when a timestamp cannot be normalised to UTC."""


def normalize_to_utc(value: Optional[str]) -> str:
    """
    Parse *value* and return a canonical UTC ISO 8601 string.

    Parameters
    ----------
    value : str | None
        Any date/datetime string.  Timezone-naive strings are treated as UTC.

    Returns
    -------
    str
        UTC ISO 8601 string, e.g. ``"2026-09-02T12:00:00Z"``.

    Raises
    ------
    TimestampNormalisationError
        When *value* is None, empty, or cannot be parsed.
    """
    if value is None or str(value).strip() in ("", "nan", "nat", "null", "none"):
        raise TimestampNormalisationError("Timestamp value is missing or empty")

    # Attempt to reuse Engineer 1's validator
    try:
        from data_engineering.validation.timestamp_validator import normalize_timestamp
        return normalize_timestamp(str(value))
    except ImportError:
        pass  # data_engineering not available — use local implementation
    except Exception as exc:
        raise TimestampNormalisationError(f"normalize_timestamp failed: {exc}") from exc

    # Local fallback implementation
    return _local_normalize(str(value).strip())


def _local_normalize(raw: str) -> str:
    """Local fallback — parses common ISO 8601 formats."""
    clean = raw.strip()
    # Replace Z suffix for fromisoformat compatibility
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(clean)
    except ValueError as exc:
        raise TimestampNormalisationError(f"Cannot parse timestamp {raw!r}: {exc}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_datetime(ts: str) -> datetime:
    """
    Parse an ISO 8601 UTC string to a timezone-aware ``datetime``.

    Parameters
    ----------
    ts : str
        ISO 8601 UTC string (e.g. "2026-09-02T12:00:00Z").

    Returns
    -------
    datetime
        Timezone-aware (UTC) datetime object.
    """
    normalised = normalize_to_utc(ts)
    # At this point normalised ends with Z, which Python 3.11+ handles
    clean = normalised[:-1] + "+00:00"
    return datetime.fromisoformat(clean)
