"""
satellite/tests/test_preprocessing.py
Tests for satellite/preprocessing/pipeline.py
"""

import numpy as np
import pytest

from satellite.preprocessing.pipeline import (
    mask_invalid,
    normalize_band,
    apply_cloud_mask,
    preprocess_scene,
)


class TestMaskInvalid:
    def test_valid_pixels_not_masked(self, red_band):
        """All in-range pixels should remain unmasked."""
        result = mask_invalid(red_band)
        # Our red_band fixture is in [0.05, 0.25] — all valid
        assert result.count() == red_band.size

    def test_values_below_zero_masked(self):
        """Values below 0.0 should be masked."""
        data = np.ma.array([-0.1, 0.5, 0.8], dtype=np.float32)
        result = mask_invalid(data)
        assert result.mask[0] == True
        assert result.mask[1] == False
        assert result.mask[2] == False

    def test_values_above_one_masked(self):
        """Values above 1.0 should be masked."""
        data = np.ma.array([0.5, 1.1, 2.0], dtype=np.float32)
        result = mask_invalid(data)
        assert result.mask[0] == False
        assert result.mask[1] == True
        assert result.mask[2] == True

    def test_nan_values_masked(self):
        """NaN values should be masked."""
        data = np.ma.array([0.5, np.nan, 0.3], dtype=np.float32)
        result = mask_invalid(data)
        assert result.mask[1] == True

    def test_inf_values_masked(self):
        """Inf values should be masked."""
        data = np.ma.array([0.5, np.inf, -np.inf], dtype=np.float32)
        result = mask_invalid(data)
        assert result.mask[1] == True
        assert result.mask[2] == True

    def test_existing_mask_preserved(self, red_band_with_masked):
        """Pre-existing masked pixels must stay masked."""
        result = mask_invalid(red_band_with_masked)
        # The top-left 2×2 was masked in the fixture
        assert result.mask[0, 0] == True
        assert result.mask[0, 1] == True
        assert result.mask[1, 0] == True
        assert result.mask[1, 1] == True

    def test_output_type_is_masked_array(self, red_band):
        """Output must be a masked array."""
        result = mask_invalid(red_band)
        assert isinstance(result, np.ma.MaskedArray)

    def test_missing_never_silently_zero(self):
        """Masked pixels must not become 0.0 in the data."""
        data = np.ma.array([-999.0, 0.5], dtype=np.float32)
        result = mask_invalid(data, low=0.0, high=1.0)
        # The masked pixel should be masked, not filled with 0
        assert result.mask[0] == True


class TestNormalizeBand:
    def test_output_range_zero_to_one(self, red_band):
        """Normalized band must be in [0, 1]."""
        result = normalize_band(red_band)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_masked_pixels_remain_masked(self, red_band_with_masked):
        """Masked pixels must remain masked after normalization."""
        result = normalize_band(red_band_with_masked)
        assert result.mask[0, 0] == True

    def test_equal_low_high_returns_zeros(self):
        """Edge case: low == high should not raise, returns 0s."""
        data = np.ma.array([0.5, 0.5], dtype=np.float32)
        result = normalize_band(data, low=0.5, high=0.5)
        assert float(result[0]) == 0.0

    def test_output_is_float32(self, red_band):
        result = normalize_band(red_band)
        assert result.dtype == np.float32


class TestApplyCloudMask:
    def test_no_cloud_mask_returns_original(self, red_band):
        """Without a cloud mask the array is returned unchanged."""
        result = apply_cloud_mask(red_band, cloud_mask=None)
        assert result.count() == red_band.count()

    def test_cloud_pixels_masked(self, red_band):
        """Pixels with cloud_mask=True must be masked in output."""
        cloud = np.zeros(red_band.shape, dtype=bool)
        cloud[0, 0] = True
        cloud[0, 1] = True
        result = apply_cloud_mask(red_band, cloud_mask=cloud)
        assert result.mask[0, 0] == True
        assert result.mask[0, 1] == True

    def test_non_cloud_pixels_unchanged(self, red_band):
        """Non-cloud pixels must not be affected."""
        cloud = np.zeros(red_band.shape, dtype=bool)
        cloud[0, 0] = True
        result = apply_cloud_mask(red_band, cloud_mask=cloud)
        # Count should decrease by exactly 1
        assert result.count() == red_band.count() - 1

    def test_shape_mismatch_raises(self, red_band):
        """Shape mismatch between array and cloud_mask should raise."""
        bad_cloud = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            apply_cloud_mask(red_band, cloud_mask=bad_cloud)

    def test_existing_mask_combined_with_cloud(self, red_band_with_masked):
        """Cloud mask should be ORed with existing mask."""
        cloud = np.zeros(red_band_with_masked.shape, dtype=bool)
        cloud[5, 5] = True  # a new pixel not previously masked
        result = apply_cloud_mask(red_band_with_masked, cloud_mask=cloud)
        # Original masked pixels still masked
        assert result.mask[0, 0] == True
        # New cloud pixel masked
        assert result.mask[5, 5] == True


class TestPreprocessScene:
    def test_all_bands_preprocessed(self, synthetic_scene):
        """All bands in scene must be preprocessed."""
        result = preprocess_scene(synthetic_scene)
        assert set(result["bands"].keys()) == {"red", "nir", "swir"}

    def test_transform_crs_preserved(self, synthetic_scene):
        """CRS and transform must be unchanged after preprocessing."""
        result = preprocess_scene(synthetic_scene)
        assert result["crs"] == synthetic_scene["crs"]
        assert result["transform"] is synthetic_scene["transform"]

    def test_cloud_mask_applied(self, synthetic_scene):
        """Cloud mask should reduce valid pixel count."""
        cloud = np.zeros((10, 10), dtype=bool)
        cloud[0, 0] = True
        result = preprocess_scene(synthetic_scene, cloud_masks={"red": cloud})
        assert result["bands"]["red"].mask[0, 0] == True
