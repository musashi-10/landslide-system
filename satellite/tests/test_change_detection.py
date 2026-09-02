"""
satellite/tests/test_change_detection.py
Tests for satellite/change_detection/detector.py
"""

import numpy as np
import pytest

from satellite.change_detection.detector import compute_ndvi_change, detect_disturbance


class TestComputeNDVIChange:
    def test_delta_is_t2_minus_t1(self):
        """ΔNDVI = NDVI(t2) - NDVI(t1)."""
        t1 = np.ma.array([[0.3, 0.5]], dtype=np.float32)
        t2 = np.ma.array([[0.1, 0.6]], dtype=np.float32)
        result = compute_ndvi_change(t1, t2)
        assert abs(float(result[0, 0]) - (-0.2)) < 1e-5
        assert abs(float(result[0, 1]) - 0.1) < 1e-5

    def test_identical_images_produce_zero(self):
        """Two identical images → zero change everywhere."""
        ndvi = np.ma.array(
            np.random.default_rng(99).uniform(-1, 1, (10, 10)).astype(np.float32)
        )
        result = compute_ndvi_change(ndvi, ndvi)
        assert float(np.abs(result).max()) < 1e-5

    def test_shape_mismatch_raises(self):
        """Different shaped arrays must raise ValueError."""
        t1 = np.ma.zeros((5, 5), dtype=np.float32)
        t2 = np.ma.zeros((6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            compute_ndvi_change(t1, t2)

    def test_masked_in_t1_propagated(self):
        """Pixel masked in t1 must be masked in output."""
        t1 = np.ma.array([[0.3, 0.4]], dtype=np.float32, mask=[[True, False]])
        t2 = np.ma.array([[0.1, 0.2]], dtype=np.float32)
        result = compute_ndvi_change(t1, t2)
        assert result.mask[0, 0] == True
        assert result.mask[0, 1] == False

    def test_masked_in_t2_propagated(self):
        """Pixel masked in t2 must be masked in output."""
        t1 = np.ma.array([[0.3, 0.4]], dtype=np.float32)
        t2 = np.ma.array([[0.1, 0.2]], dtype=np.float32, mask=[[False, True]])
        result = compute_ndvi_change(t1, t2)
        assert result.mask[0, 1] == True

    def test_masked_in_both_propagated(self):
        """Pixel masked in both images must be masked in output."""
        t1 = np.ma.array([[0.3]], dtype=np.float32, mask=[[True]])
        t2 = np.ma.array([[0.1]], dtype=np.float32, mask=[[True]])
        result = compute_ndvi_change(t1, t2)
        assert result.mask[0, 0] == True

    def test_returns_masked_array(self):
        t1 = np.ma.array([[0.4]], dtype=np.float32)
        t2 = np.ma.array([[0.2]], dtype=np.float32)
        result = compute_ndvi_change(t1, t2)
        assert isinstance(result, np.ma.MaskedArray)

    def test_vegetation_loss_is_negative(self):
        """Vegetation decrease: t1 > t2 → negative ΔNDVI."""
        t1 = np.ma.array([[0.80]], dtype=np.float32)  # healthy veg
        t2 = np.ma.array([[0.20]], dtype=np.float32)  # post-disturbance
        result = compute_ndvi_change(t1, t2)
        assert float(result[0, 0]) < 0.0


class TestDetectDisturbance:
    def test_below_threshold_flagged(self):
        """ΔNDVI ≤ threshold → disturbance_mask True."""
        change = np.ma.array([[-0.20, -0.10, 0.05]], dtype=np.float32)
        result = detect_disturbance(change, threshold=-0.15)
        # -0.20 ≤ -0.15 → flagged
        assert result["disturbance_mask"][0, 0] == True
        # -0.10 > -0.15 → not flagged
        assert result["disturbance_mask"][0, 1] == False
        # 0.05 > -0.15 → not flagged
        assert result["disturbance_mask"][0, 2] == False

    def test_flagged_count_matches(self):
        """flagged_pixel_count should equal number of True pixels in mask."""
        change = np.ma.array([-0.20, -0.30, 0.0, 0.10], dtype=np.float32)
        result = detect_disturbance(change, threshold=-0.15)
        assert result["flagged_pixel_count"] == 2
        assert int(np.sum(result["disturbance_mask"])) == 2

    def test_no_disturbance_when_all_above_threshold(self):
        """Stable scene → zero flagged pixels."""
        change = np.ma.array([0.0, 0.1, 0.05], dtype=np.float32)
        result = detect_disturbance(change, threshold=-0.15)
        assert result["flagged_pixel_count"] == 0
        assert result["flagged_fraction"] == 0.0

    def test_masked_pixels_not_flagged(self):
        """Masked pixels must not appear as disturbances."""
        change = np.ma.array([-0.30, -0.20], dtype=np.float32, mask=[True, False])
        result = detect_disturbance(change, threshold=-0.15)
        # Only the unmasked -0.20 should be flagged
        assert result["flagged_pixel_count"] == 1
        assert result["total_valid_pixels"] == 1

    def test_output_dict_keys(self):
        """Result dict must have all expected keys."""
        change = np.ma.array([0.0], dtype=np.float32)
        result = detect_disturbance(change)
        assert "disturbance_mask" in result
        assert "ndvi_change" in result
        assert "threshold_used" in result
        assert "flagged_pixel_count" in result
        assert "total_valid_pixels" in result
        assert "flagged_fraction" in result

    def test_threshold_recorded_in_output(self):
        """The threshold used must be returned in the result."""
        change = np.ma.array([0.0], dtype=np.float32)
        result = detect_disturbance(change, threshold=-0.25)
        assert result["threshold_used"] == -0.25

    def test_all_masked_gives_none_fraction(self):
        """All masked → no valid pixels → flagged_fraction should be None."""
        change = np.ma.array([-0.5], dtype=np.float32, mask=[True])
        result = detect_disturbance(change, threshold=-0.15)
        assert result["flagged_fraction"] is None
        assert result["total_valid_pixels"] == 0
