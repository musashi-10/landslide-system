"""
satellite/tests/test_indices.py
Tests for satellite/features/indices.py
"""

import numpy as np
import pytest

from satellite.features.indices import compute_ndvi, compute_bsi, classify_land_cover


class TestComputeNDVI:
    def test_formula_correct(self):
        """NDVI = (NIR - Red) / (NIR + Red)."""
        red = np.ma.array([[0.1, 0.2]], dtype=np.float32)
        nir = np.ma.array([[0.3, 0.4]], dtype=np.float32)
        result = compute_ndvi(red, nir)
        expected_0 = (0.3 - 0.1) / (0.3 + 0.1)
        expected_1 = (0.4 - 0.2) / (0.4 + 0.2)
        assert abs(float(result[0, 0]) - expected_0) < 1e-5
        assert abs(float(result[0, 1]) - expected_1) < 1e-5

    def test_output_range_minus_one_to_one(self, red_band, nir_band):
        """All NDVI values must be in [−1, 1]."""
        result = compute_ndvi(red_band, nir_band)
        assert float(result.min()) >= -1.0 - 1e-6
        assert float(result.max()) <= 1.0 + 1e-6

    def test_zero_denominator_masked(self, zero_denominator_red, zero_nir):
        """Zero denominator (NIR=0, Red=0) must produce masked pixels."""
        result = compute_ndvi(zero_denominator_red, zero_nir)
        assert np.all(result.mask)

    def test_masked_pixels_propagated(self, red_band_with_masked, nir_band):
        """Masked red pixels must propagate to masked NDVI pixels."""
        result = compute_ndvi(red_band_with_masked, nir_band)
        # Top-left 2×2 was masked in red_band_with_masked
        assert result.mask[0, 0] == True
        assert result.mask[0, 1] == True

    def test_returns_masked_array(self, red_band, nir_band):
        result = compute_ndvi(red_band, nir_band)
        assert isinstance(result, np.ma.MaskedArray)

    def test_vegetation_positive_ndvi(self):
        """Dense vegetation: NIR >> Red → positive NDVI."""
        red = np.ma.array([[0.05]], dtype=np.float32)
        nir = np.ma.array([[0.45]], dtype=np.float32)
        result = compute_ndvi(red, nir)
        assert float(result[0, 0]) > 0.0

    def test_water_negative_ndvi(self):
        """Water: Red > NIR → negative NDVI."""
        red = np.ma.array([[0.30]], dtype=np.float32)
        nir = np.ma.array([[0.05]], dtype=np.float32)
        result = compute_ndvi(red, nir)
        assert float(result[0, 0]) < 0.0

    def test_no_nan_in_valid_output(self, red_band, nir_band):
        """No NaN values should appear in valid (unmasked) pixels."""
        result = compute_ndvi(red_band, nir_band)
        filled = result.filled(0.0)
        assert not np.any(np.isnan(filled))
        assert not np.any(np.isinf(filled))


class TestComputeBSI:
    def test_formula_correct(self):
        """BSI = (SWIR + Red - NIR) / (SWIR + Red + NIR)."""
        red = np.ma.array([[0.10]], dtype=np.float32)
        nir = np.ma.array([[0.30]], dtype=np.float32)
        swir = np.ma.array([[0.20]], dtype=np.float32)
        result = compute_bsi(red, nir, swir)
        num = 0.20 + 0.10 - 0.30
        den = 0.20 + 0.10 + 0.30
        expected = num / den
        assert abs(float(result[0, 0]) - expected) < 1e-5

    def test_output_range(self, red_band, nir_band, swir_band):
        """BSI values must be in [−1, 1]."""
        result = compute_bsi(red_band, nir_band, swir_band)
        assert float(result.min()) >= -1.0 - 1e-6
        assert float(result.max()) <= 1.0 + 1e-6

    def test_zero_denominator_masked(self):
        """All-zero bands → zero denominator → masked."""
        z = np.ma.zeros((3, 3), dtype=np.float32)
        result = compute_bsi(z, z, z)
        assert np.all(result.mask)

    def test_high_swir_gives_positive_bsi(self):
        """High SWIR relative to NIR → positive BSI (bare surface)."""
        red = np.ma.array([[0.15]], dtype=np.float32)
        nir = np.ma.array([[0.10]], dtype=np.float32)
        swir = np.ma.array([[0.40]], dtype=np.float32)
        result = compute_bsi(red, nir, swir)
        assert float(result[0, 0]) > 0.0

    def test_returns_masked_array(self, red_band, nir_band, swir_band):
        result = compute_bsi(red_band, nir_band, swir_band)
        assert isinstance(result, np.ma.MaskedArray)


class TestClassifyLandCover:
    def test_high_ndvi_is_forest(self, vegetation_ndvi, high_bsi):
        """NDVI = 0.70 → forest."""
        # Use low BSI so only NDVI determines class
        low_bsi = np.ma.array(np.full(vegetation_ndvi.shape, -0.1, dtype=np.float32))
        result = classify_land_cover(vegetation_ndvi, low_bsi)
        assert np.all(result == "forest")

    def test_low_ndvi_high_bsi_is_bare(self, bare_ndvi, high_bsi):
        """NDVI = 0.05, BSI = 0.30 → bare."""
        result = classify_land_cover(bare_ndvi, high_bsi)
        assert np.all(result == "bare")

    def test_negative_ndvi_is_water(self):
        """NDVI < −0.05 → water."""
        ndvi = np.ma.array(np.full((4, 4), -0.15, dtype=np.float32))
        bsi = np.ma.array(np.full((4, 4), 0.0, dtype=np.float32))
        result = classify_land_cover(ndvi, bsi)
        assert np.all(result == "water")

    def test_masked_pixels_are_unknown(self):
        """Masked NDVI pixels → 'unknown' land cover."""
        ndvi = np.ma.array([[0.7, 0.1]], dtype=np.float32, mask=[[True, False]])
        bsi = np.ma.array([[0.0, 0.3]], dtype=np.float32, mask=[[True, False]])
        result = classify_land_cover(ndvi, bsi)
        assert result[0, 0] == "unknown"

    def test_shape_mismatch_raises(self, vegetation_ndvi):
        """Mismatched shapes must raise ValueError."""
        wrong_bsi = np.ma.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            classify_land_cover(vegetation_ndvi, wrong_bsi)

    def test_output_is_string_array(self, vegetation_ndvi, high_bsi):
        result = classify_land_cover(vegetation_ndvi, high_bsi)
        assert result.dtype == object

    def test_all_categories_reachable(self):
        """All expected categories should be producible."""
        # Build a 4×1 array covering each class
        ndvi = np.ma.array([[0.70], [0.25], [0.10], [-0.10]], dtype=np.float32)
        bsi = np.ma.array([[0.0], [-0.1], [0.20], [0.0]], dtype=np.float32)
        result = classify_land_cover(ndvi, bsi)
        categories = set(result.ravel().tolist())
        assert "forest" in categories
        assert "water" in categories
