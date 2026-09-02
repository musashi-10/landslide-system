"""
Unit tests for timestamp normalisation.
"""

import pytest
from data_engineering.validation.timestamp_validator import (
    normalize_timestamp,
    TimestampError,
)


class TestNormalizeTimestamp:
    def test_iso_date_only(self):
        result = normalize_timestamp("2022-06-15")
        assert result == "2022-06-15T00:00:00Z"

    def test_iso_datetime(self):
        result = normalize_timestamp("2026-09-02T12:00:00")
        assert result == "2026-09-02T12:00:00Z"

    def test_iso_datetime_with_z(self):
        result = normalize_timestamp("2026-09-02T12:00:00Z")
        assert result == "2026-09-02T12:00:00Z"

    def test_iso_datetime_with_offset(self):
        """Timezone-aware inputs must be converted to UTC."""
        result = normalize_timestamp("2026-09-02T17:30:00+05:30")
        assert result == "2026-09-02T12:00:00Z"

    def test_datetime_with_space(self):
        result = normalize_timestamp("2026-09-02 12:00:00")
        assert result == "2026-09-02T12:00:00Z"

    def test_slash_dmy_format(self):
        result = normalize_timestamp("15/06/2022")
        assert result == "2022-06-15T00:00:00Z"

    def test_none_raises(self):
        with pytest.raises(TimestampError, match="missing or empty"):
            normalize_timestamp(None)

    def test_empty_string_raises(self):
        with pytest.raises(TimestampError, match="missing or empty"):
            normalize_timestamp("")

    def test_whitespace_raises(self):
        with pytest.raises(TimestampError, match="missing or empty"):
            normalize_timestamp("   ")

    def test_nan_string_raises(self):
        with pytest.raises(TimestampError, match="missing or empty"):
            normalize_timestamp("nan")

    def test_invalid_string_raises(self):
        with pytest.raises(TimestampError):
            normalize_timestamp("invalid_date")

    def test_output_always_utc(self):
        """All outputs must end with Z (UTC marker)."""
        result = normalize_timestamp("2023-01-01")
        assert result.endswith("Z")
