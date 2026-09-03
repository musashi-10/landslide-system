"""
environment/tests/test_timestamp_utils.py

Tests for UTC timestamp normalisation utilities.
"""

from __future__ import annotations

import pytest

from environment.preprocessing.timestamp_utils import (
    TimestampNormalisationError,
    normalize_to_utc,
    parse_utc_datetime,
)


class TestNormalizeToUtc:
    def test_already_utc_z(self):
        result = normalize_to_utc("2026-09-02T12:00:00Z")
        assert result == "2026-09-02T12:00:00Z"

    def test_with_offset_converted_to_utc(self):
        # +05:30 = IST -> subtract 5h30m -> 06:30 UTC
        result = normalize_to_utc("2026-09-02T12:00:00+05:30")
        assert result == "2026-09-02T06:30:00Z"

    def test_naive_treated_as_utc(self):
        result = normalize_to_utc("2026-09-02T12:00:00")
        assert result == "2026-09-02T12:00:00Z"

    def test_date_only(self):
        result = normalize_to_utc("2026-09-02")
        assert result == "2026-09-02T00:00:00Z"

    def test_none_raises(self):
        with pytest.raises(TimestampNormalisationError):
            normalize_to_utc(None)

    def test_empty_string_raises(self):
        with pytest.raises(TimestampNormalisationError):
            normalize_to_utc("")

    def test_nan_string_raises(self):
        with pytest.raises(TimestampNormalisationError):
            normalize_to_utc("nan")

    def test_null_string_raises(self):
        with pytest.raises(TimestampNormalisationError):
            normalize_to_utc("null")

    def test_invalid_format_raises(self):
        with pytest.raises(TimestampNormalisationError):
            normalize_to_utc("not-a-date")

    def test_output_ends_with_z(self):
        result = normalize_to_utc("2026-09-01T00:00:00Z")
        assert result.endswith("Z")

    def test_output_iso_format(self):
        result = normalize_to_utc("2026-09-01T00:00:00Z")
        # Should be exactly YYYY-MM-DDTHH:MM:SSZ
        assert len(result) == 20
        assert result[10] == "T"
        assert result[19] == "Z"


class TestParseUtcDatetime:
    def test_basic_parse(self):
        from datetime import timezone
        dt = parse_utc_datetime("2026-09-02T12:00:00Z")
        assert dt.year == 2026
        assert dt.month == 9
        assert dt.day == 2
        assert dt.hour == 12
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_offset_converted(self):
        dt = parse_utc_datetime("2026-09-02T12:00:00+05:30")
        assert dt.hour == 6
        assert dt.minute == 30
