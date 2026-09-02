"""
Timestamp normalisation utilities.

Converts a wide variety of date/datetime string formats into a canonical
ISO 8601 UTC timestamp (``YYYY-MM-DDTHH:MM:SSZ``).

Supported input formats (partial list — pandas fallback handles many more):

  * ``2026-09-02``
  * ``2026-09-02T12:00:00``
  * ``2026-09-02T12:00:00Z``
  * ``2026-09-02T12:00:00+05:30``
  * ``02/09/2026``
  * ``09/02/2026``  (US format)
  * ``2026-09-02 12:00:00``

Timezone-naive inputs are treated as UTC.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered list of explicit format strings to try before falling back to pandas
_EXPLICIT_FORMATS: list[str] = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%Y/%m/%d",
]


class TimestampError(ValueError):
    """Raised when a timestamp cannot be parsed or is absent."""


def normalize_timestamp(value: Optional[str]) -> str:
    """
    Parse *value* and return a canonical UTC ISO 8601 timestamp string.

    Parameters
    ----------
    value : str | None
        Raw date/datetime string from a source dataset.

    Returns
    -------
    str
        UTC ISO 8601 string, e.g. ``"2026-09-02T12:00:00Z"``.

    Raises
    ------
    TimestampError
        When *value* is None, empty, or cannot be parsed by any strategy.
    """
    if value is None or str(value).strip() == "" or str(value).lower() in ("nan", "nat", "null"):
        raise TimestampError("Timestamp value is missing or empty")

    raw = str(value).strip()

    # 1. Try explicit formats first (fast path)
    for fmt in _EXPLICIT_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return _to_utc_string(dt)
        except ValueError:
            continue

    # 2. Attempt pandas parsing (handles timezone-aware strings via utc=True)
    try:
        parsed = pd.to_datetime(raw, utc=True)
        dt = parsed.to_pydatetime()
        return _to_utc_string(dt)
    except Exception as exc:
        raise TimestampError(
            f"Cannot parse timestamp {raw!r}: {exc}"
        ) from exc


def _to_utc_string(dt: datetime) -> str:
    """Convert a datetime to a UTC ISO 8601 string."""
    if dt.tzinfo is None:
        # Naive → assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
