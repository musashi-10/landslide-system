"""
satellite/tests/test_pipeline.py
End-to-end tests for the satellite pipeline using in-memory arrays.
No real satellite data or internet access required.
"""

import numpy as np
import pytest

from satellite.pipeline import extract_features_from_arrays
from satellite.config import SatelliteConfig
from satellite.schemas.satellite_schema import SatelliteFeatureRecord
from satellite.tests.conftest import ACQUISITION_TIME, SOURCE_CRS


class TestExtractFeaturesFromArrays:
    def test_returns_list_of_records(self, red_band, nir_band, swir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data, "swir": swir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        assert isinstance(records, list)
        assert len(records) > 0
        assert all(isinstance(r, SatelliteFeatureRecord) for r in records)

    def test_all_records_have_location_id(self, red_band, nir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        for rec in records:
            assert rec.location_id.startswith("LOC_LAT")

    def test_ndvi_present_in_output(self, red_band, nir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        for rec in records:
            assert rec.ndvi is not None
            assert -1.0 <= rec.ndvi <= 1.0

    def test_bsi_present_when_swir_provided(self, red_band, nir_band, swir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={
                "red": red_band.data,
                "nir": nir_band.data,
                "swir": swir_band.data,
            },
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        for rec in records:
            assert rec.bare_surface_index is not None

    def test_bsi_absent_when_no_swir(self, red_band, nir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        for rec in records:
            assert rec.bare_surface_index is None

    def test_change_detection_adds_ndvi_change(self, red_band, nir_band, affine_transform):
        # Use a prior NDVI (t1) that is higher — simulates vegetation loss
        ndvi_t1 = np.full((10, 10), 0.7, dtype=np.float32)
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
            ndvi_t1=ndvi_t1,
        )
        for rec in records:
            assert rec.ndvi_change is not None

    def test_no_change_detection_without_t1(self, red_band, nir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
            ndvi_t1=None,
        )
        for rec in records:
            assert rec.ndvi_change is None

    def test_acquisition_time_preserved(self, red_band, nir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        for rec in records:
            assert rec.acquisition_time == ACQUISITION_TIME

    def test_output_crs_is_epsg4326(self, red_band, nir_band, affine_transform):
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        for rec in records:
            assert rec.output_crs == "EPSG:4326"

    def test_processing_version_from_config(self, red_band, nir_band, affine_transform):
        config = SatelliteConfig(processing_version="v2")
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
            config=config,
        )
        for rec in records:
            assert rec.processing_version == "v2"

    def test_missing_red_band_raises(self, nir_band, affine_transform):
        with pytest.raises(ValueError, match="red"):
            extract_features_from_arrays(
                band_arrays={"nir": nir_band.data},
                transform=affine_transform,
                crs=SOURCE_CRS,
                acquisition_time=ACQUISITION_TIME,
            )

    def test_cloud_masked_pixels_excluded(self, red_band, nir_band, affine_transform):
        # Mask all red pixels → no valid output records
        cloud = np.ones((10, 10), dtype=bool)
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
            cloud_masks={"red": cloud},
        )
        assert len(records) == 0

    def test_all_records_pass_schema_validation(self, red_band, nir_band, swir_band, affine_transform):
        """Every output record must be a valid SatelliteFeatureRecord."""
        records = extract_features_from_arrays(
            band_arrays={
                "red": red_band.data,
                "nir": nir_band.data,
                "swir": swir_band.data,
            },
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        # Re-validate through Pydantic model_validate
        for rec in records:
            validated = SatelliteFeatureRecord.model_validate(rec.model_dump())
            assert validated.location_id == rec.location_id

    def test_can_convert_to_dict_for_dataframe(self, red_band, nir_band, affine_transform):
        """Records must be serializable to dicts (for Parquet export)."""
        import pandas as pd  # noqa: PLC0415
        records = extract_features_from_arrays(
            band_arrays={"red": red_band.data, "nir": nir_band.data},
            transform=affine_transform,
            crs=SOURCE_CRS,
            acquisition_time=ACQUISITION_TIME,
        )
        dicts = [r.model_dump() for r in records]
        df = pd.DataFrame(dicts)
        assert "location_id" in df.columns
        assert "ndvi" in df.columns
        assert len(df) == len(records)
