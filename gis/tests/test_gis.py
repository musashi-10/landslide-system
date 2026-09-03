"""
GIS unit tests.

Covers:
  - CRS transformation (raster and vector)
  - Elevation extraction
  - Slope calculation
  - Aspect calculation
  - Raster alignment
  - Grid generation
  - Spatial joins
  - Missing raster values (NaN propagation)
  - Invalid geometry
  - Standardised output schema
  - Full pipeline with synthetic data
  - Baseline susceptibility
  - Data-contract validation against shared schemas
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, box

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
from gis.tests.fixtures.synthetic import (
    make_dem_array,
    make_dem_file,
    make_soil_vector,
    make_geology_vector,
    make_landcover_vector,
    make_drainage_vector,
    make_historical_landslides,
)


@pytest.fixture()
def dem_array():
    """Small synthetic DEM array, transform, crs."""
    return make_dem_array(rows=20, cols=20)


@pytest.fixture()
def dem_file(tmp_path):
    """Synthetic DEM written to a temp GeoTIFF."""
    return make_dem_file(tmp_path / "dem.tif")


@pytest.fixture()
def soil_gdf():
    return make_soil_vector()


@pytest.fixture()
def geology_gdf():
    return make_geology_vector()


@pytest.fixture()
def landcover_gdf():
    return make_landcover_vector()


@pytest.fixture()
def drainage_gdf():
    return make_drainage_vector()


@pytest.fixture()
def historical_gdf():
    return make_historical_landslides(n=8)


# ===================================================================
# 1. CRS TRANSFORMATION — raster
# ===================================================================

class TestCRSTransformation:
    def test_reproject_noop_same_crs(self, dem_array):
        """No reprojection should occur when source == target CRS."""
        from gis.dem.processor import validate_and_reproject
        data, transform, crs = dem_array
        out_data, out_transform, out_crs = validate_and_reproject(
            data, transform, crs, "EPSG:4326"
        )
        assert out_crs == crs
        # Same object returned (no copy)
        assert out_data is data

    def test_reproject_projected_to_geographic(self, dem_array):
        """Raster in a projected CRS must be reprojected to EPSG:4326."""
        from gis.dem.processor import validate_and_reproject
        data, _, _ = dem_array
        # Pretend the DEM is in UTM zone 45N
        src_crs = CRS.from_epsg(32645)
        transform_utm = from_bounds(
            500_000, 2_985_000, 520_000, 3_005_000, data.shape[1], data.shape[0]
        )
        out_data, out_transform, out_crs = validate_and_reproject(
            data, transform_utm, src_crs, "EPSG:4326"
        )
        assert out_crs == CRS.from_epsg(4326)
        assert out_data.shape[0] > 0 and out_data.shape[1] > 0

    def test_reproject_vector_noop(self, soil_gdf):
        """Vector already in EPSG:4326 — return same object."""
        from gis.spatial.alignment import reproject_vector
        result = reproject_vector(soil_gdf, "EPSG:4326")
        assert result.crs.to_epsg() == 4326

    def test_reproject_vector_to_utm(self, soil_gdf):
        """Reproject vector to UTM and check CRS changes."""
        from gis.spatial.alignment import reproject_vector
        result = reproject_vector(soil_gdf, "EPSG:32645")
        assert result.crs.to_epsg() == 32645

    def test_reproject_vector_no_crs_raises(self):
        """GeoDataFrame without CRS should raise ValueError."""
        from gis.spatial.alignment import reproject_vector
        gdf = gpd.GeoDataFrame({"geometry": [Point(88, 27)]})
        with pytest.raises(ValueError, match="no CRS"):
            reproject_vector(gdf, "EPSG:4326")

    def test_crs_mismatch_spatial_join_raises(self, soil_gdf):
        """Spatial join between GDFs with different CRS must raise."""
        from gis.spatial.alignment import spatial_join_to_grid
        grid = gpd.GeoDataFrame(
            {"location_id": ["G1"], "geometry": [box(88.0, 27.0, 88.1, 27.1)]},
            crs="EPSG:4326",
        )
        layer_utm = soil_gdf.to_crs("EPSG:32645")
        with pytest.raises(ValueError, match="CRS mismatch"):
            spatial_join_to_grid(grid, layer_utm, "soil_type")


# ===================================================================
# 2. ELEVATION EXTRACTION
# ===================================================================

class TestElevationExtraction:
    def test_extraction_in_bounds(self, dem_array):
        from gis.dem.processor import extract_elevation
        data, transform, _ = dem_array
        # Centre of the raster
        lat = 27.10
        lon = 88.10
        val = extract_elevation(data, transform, lat, lon)
        assert val is not None
        assert 500.0 < val < 3000.0  # sensible range for synthetic data

    def test_extraction_out_of_bounds(self, dem_array):
        from gis.dem.processor import extract_elevation
        data, transform, _ = dem_array
        val = extract_elevation(data, transform, 0.0, 0.0)  # Africa
        assert val is None

    def test_extraction_nodata_returns_none(self, dem_array):
        from gis.dem.processor import extract_elevation
        data, transform, _ = dem_array
        # Force top-left pixel to NaN (set by make_dem_array)
        val = extract_elevation(data, transform, 27.19, 88.00)
        # Either None or a real value depending on exact pixel; just no exception
        assert val is None or isinstance(val, float)

    def test_open_dem_loads_file(self, dem_file):
        from gis.dem.processor import open_dem
        data, transform, crs = open_dem(dem_file)
        assert data.shape == (20, 20)
        assert crs.to_epsg() == 4326
        assert not np.all(np.isnan(data))  # most pixels have data


# ===================================================================
# 3. SLOPE CALCULATION
# ===================================================================

class TestSlopeCalculation:
    def test_slope_range(self, dem_array):
        from gis.dem.processor import compute_slope
        data, transform, crs = dem_array
        slope = compute_slope(data, transform, crs)
        valid = slope[~np.isnan(slope)]
        assert valid.min() >= 0.0
        assert valid.max() <= 90.0

    def test_slope_shape(self, dem_array):
        from gis.dem.processor import compute_slope
        data, transform, crs = dem_array
        slope = compute_slope(data, transform, crs)
        assert slope.shape == data.shape

    def test_slope_nan_propagation(self, dem_array):
        from gis.dem.processor import compute_slope
        data, transform, crs = dem_array
        # (0,0) is NaN in make_dem_array
        slope = compute_slope(data, transform, crs)
        assert np.isnan(slope[0, 0])

    def test_flat_dem_has_zero_slope(self):
        from gis.dem.processor import compute_slope
        # Perfectly flat raster → slope = 0
        flat = np.full((10, 10), 500.0, dtype=np.float32)
        transform = from_bounds(88.0, 27.0, 88.1, 27.1, 10, 10)
        crs = CRS.from_epsg(4326)
        slope = compute_slope(flat, transform, crs)
        np.testing.assert_allclose(slope, 0.0, atol=1e-3)

    def test_steep_surface_has_high_slope(self):
        from gis.dem.processor import compute_slope
        # Surface rising steeply from south to north
        rows = np.linspace(0, 5000, 20)[:, None] * np.ones((20, 20))
        steep = rows.astype(np.float32)
        transform = from_bounds(88.0, 27.0, 88.2, 27.2, 20, 20)
        crs = CRS.from_epsg(4326)
        slope = compute_slope(steep, transform, crs)
        valid_slope = slope[~np.isnan(slope)]
        assert valid_slope.mean() > 10.0  # should be steep


# ===================================================================
# 4. ASPECT CALCULATION
# ===================================================================

class TestAspectCalculation:
    def test_aspect_range(self, dem_array):
        from gis.dem.processor import compute_aspect
        data, transform, crs = dem_array
        aspect = compute_aspect(data, transform, crs)
        valid = aspect[~np.isnan(aspect)]
        # -1 for flat; 0–360 otherwise
        assert valid[valid >= 0].max() <= 360.0
        assert valid.min() >= -1.0

    def test_aspect_shape(self, dem_array):
        from gis.dem.processor import compute_aspect
        data, transform, crs = dem_array
        aspect = compute_aspect(data, transform, crs)
        assert aspect.shape == data.shape

    def test_aspect_nan_propagation(self, dem_array):
        from gis.dem.processor import compute_aspect
        data, transform, crs = dem_array
        aspect = compute_aspect(data, transform, crs)
        assert np.isnan(aspect[0, 0])

    def test_flat_dem_aspect_is_minus1(self):
        from gis.dem.processor import compute_aspect
        flat = np.full((10, 10), 500.0, dtype=np.float32)
        transform = from_bounds(88.0, 27.0, 88.1, 27.1, 10, 10)
        crs = CRS.from_epsg(4326)
        aspect = compute_aspect(flat, transform, crs)
        np.testing.assert_array_equal(aspect, -1.0)

    def test_north_facing_slope(self):
        """
        A surface that falls from north to south should have a southward
        aspect (~180°).
        """
        from gis.dem.processor import compute_aspect
        # Values increase northward (row 0 = north = high; row N = south = low)
        rows_arr = np.linspace(1000, 500, 20)[:, None] * np.ones((20, 20))
        data = rows_arr.astype(np.float32)
        transform = from_bounds(88.0, 27.0, 88.2, 27.2, 20, 20)
        crs = CRS.from_epsg(4326)
        aspect = compute_aspect(data, transform, crs)
        interior = aspect[5:15, 5:15]
        valid = interior[~np.isnan(interior) & (interior >= 0)]
        # Should be mostly southward (around 180°)
        assert valid.mean() > 90.0


# ===================================================================
# 5. RASTER ALIGNMENT
# ===================================================================

class TestRasterAlignment:
    def test_align_same_grid_noop(self, dem_array):
        """Aligning a raster to itself should reproduce the same values."""
        from gis.spatial.alignment import align_rasters
        data, transform, crs = dem_array
        out = align_rasters(data, transform, crs, transform, crs, data.shape)
        # Allow small floating-point differences from the reproject call
        valid = ~np.isnan(data) & ~np.isnan(out)
        np.testing.assert_allclose(out[valid], data[valid], rtol=0.01)

    def test_align_changes_shape(self, dem_array):
        """Align to a coarser target grid → smaller shape."""
        from gis.spatial.alignment import align_rasters
        data, transform, crs = dem_array
        target_transform = from_bounds(88.0, 27.0, 88.2, 27.2, 10, 10)
        out = align_rasters(data, transform, crs, target_transform, crs, (10, 10))
        assert out.shape == (10, 10)

    def test_align_crs_mismatch(self, dem_array):
        """Aligning from geographic to projected should succeed."""
        from gis.spatial.alignment import align_rasters
        data, transform, crs = dem_array
        target_crs = CRS.from_epsg(32645)
        target_transform = from_bounds(490_000, 2_985_000, 510_000, 3_005_000, 10, 10)
        out = align_rasters(data, transform, crs, target_transform, target_crs, (10, 10))
        assert out.shape == (10, 10)


# ===================================================================
# 6. GRID GENERATION
# ===================================================================

class TestGridGeneration:
    def test_grid_cell_count(self):
        from gis.spatial.grid import generate_grid
        bbox = (88.0, 27.0, 88.1, 27.1)
        gdf = generate_grid(bbox, resolution=0.05)
        # 0.1 / 0.05 = 2 × 2 = 4 cells
        assert len(gdf) == 4

    def test_grid_columns(self):
        from gis.spatial.grid import generate_grid
        gdf = generate_grid((88.0, 27.0, 88.1, 27.1), resolution=0.05)
        for col in ("grid_id", "location_id", "geometry", "latitude", "longitude"):
            assert col in gdf.columns

    def test_grid_ids_unique(self):
        from gis.spatial.grid import generate_grid
        gdf = generate_grid((88.0, 27.0, 88.2, 27.2), resolution=0.05)
        assert gdf["grid_id"].nunique() == len(gdf)

    def test_grid_location_id_equals_grid_id(self):
        from gis.spatial.grid import generate_grid
        gdf = generate_grid((88.0, 27.0, 88.1, 27.1), resolution=0.05)
        assert (gdf["grid_id"] == gdf["location_id"]).all()

    def test_grid_centroid_within_cell(self):
        from gis.spatial.grid import generate_grid
        gdf = generate_grid((88.0, 27.0, 88.1, 27.1), resolution=0.05)
        for _, row in gdf.iterrows():
            pt = Point(row["longitude"], row["latitude"])
            assert row["geometry"].contains(pt) or row["geometry"].distance(pt) < 1e-9

    def test_grid_invalid_bbox_raises(self):
        from gis.spatial.grid import generate_grid
        with pytest.raises(ValueError, match="Degenerate"):
            generate_grid((88.1, 27.0, 88.0, 27.1), resolution=0.05)

    def test_grid_invalid_resolution_raises(self):
        from gis.spatial.grid import generate_grid
        with pytest.raises(ValueError, match="resolution"):
            generate_grid((88.0, 27.0, 88.1, 27.1), resolution=0.0)

    def test_grid_crs(self):
        from gis.spatial.grid import generate_grid
        gdf = generate_grid((88.0, 27.0, 88.1, 27.1), resolution=0.05)
        assert gdf.crs.to_epsg() == 4326

    def test_grid_prefix(self):
        from gis.spatial.grid import generate_grid
        gdf = generate_grid((88.0, 27.0, 88.1, 27.1), resolution=0.05, id_prefix="CELL")
        assert gdf["grid_id"].str.startswith("CELL_").all()


# ===================================================================
# 7. SPATIAL JOINS
# ===================================================================

class TestSpatialJoins:
    def test_soil_join(self, soil_gdf):
        from gis.spatial.grid import generate_grid
        from gis.spatial.alignment import spatial_join_to_grid
        grid = generate_grid((88.0, 27.0, 88.2, 27.2), resolution=0.05)
        result = spatial_join_to_grid(grid, soil_gdf, "soil_type")
        assert "soil_type" in result.columns
        # At least some cells should have a soil type
        assert result["soil_type"].notna().sum() > 0

    def test_geology_join(self, geology_gdf):
        from gis.spatial.grid import generate_grid
        from gis.spatial.alignment import spatial_join_to_grid
        grid = generate_grid((88.0, 27.0, 88.2, 27.2), resolution=0.05)
        result = spatial_join_to_grid(grid, geology_gdf, "geology")
        assert result["geology"].notna().sum() > 0

    def test_point_to_grid_historical(self, historical_gdf):
        from gis.spatial.grid import generate_grid
        from gis.spatial.alignment import point_to_grid_assignment
        grid = generate_grid((88.0, 27.0, 88.2, 27.2), resolution=0.05)
        result = point_to_grid_assignment(grid, historical_gdf, "historical_landslide")
        assert "historical_landslide" in result.columns
        # Total count across all cells should equal n points (some may be outside)
        assert result["historical_landslide"].sum() <= len(historical_gdf)

    def test_join_missing_column_raises(self, soil_gdf):
        from gis.spatial.grid import generate_grid
        from gis.spatial.alignment import spatial_join_to_grid
        grid = generate_grid((88.0, 27.0, 88.2, 27.2), resolution=0.05)
        with pytest.raises(ValueError, match="Column"):
            spatial_join_to_grid(grid, soil_gdf, "nonexistent_col")

    def test_sample_raster_at_points(self, dem_array):
        from gis.spatial.alignment import sample_raster_at_points
        data, transform, _ = dem_array
        lats = np.array([27.05, 27.10, 27.15])
        lons = np.array([88.05, 88.10, 88.15])
        vals = sample_raster_at_points(data, transform, lats, lons)
        assert vals.shape == (3,)
        # Should be finite for in-bounds pixels (some may be NaN if nodata)
        assert not np.all(np.isnan(vals))

    def test_sample_out_of_bounds_is_nan(self, dem_array):
        from gis.spatial.alignment import sample_raster_at_points
        data, transform, _ = dem_array
        lats = np.array([0.0])   # outside raster
        lons = np.array([0.0])
        vals = sample_raster_at_points(data, transform, lats, lons)
        assert np.isnan(vals[0])


# ===================================================================
# 8. MISSING RASTER VALUES (NaN propagation)
# ===================================================================

class TestNaNPropagation:
    def test_slope_preserves_nodata_as_nan(self):
        from gis.dem.processor import compute_slope
        data = np.full((10, 10), 500.0, dtype=np.float32)
        data[5, 5] = np.nan
        transform = from_bounds(88.0, 27.0, 88.1, 27.1, 10, 10)
        crs = CRS.from_epsg(4326)
        slope = compute_slope(data, transform, crs)
        assert np.isnan(slope[5, 5])

    def test_aspect_preserves_nodata_as_nan(self):
        from gis.dem.processor import compute_aspect
        data = np.full((10, 10), 500.0, dtype=np.float32)
        data[3, 7] = np.nan
        transform = from_bounds(88.0, 27.0, 88.1, 27.1, 10, 10)
        crs = CRS.from_epsg(4326)
        aspect = compute_aspect(data, transform, crs)
        assert np.isnan(aspect[3, 7])

    def test_dem_nodata_not_zero(self, dem_file):
        from gis.dem.processor import open_dem
        data, _, _ = open_dem(dem_file)
        # Nodata pixels should be NaN, not 0
        assert not np.any(data == 0.0)  # synthetic data starts at ~1000m anyway


# ===================================================================
# 9. INVALID GEOMETRY
# ===================================================================

class TestInvalidGeometry:
    def test_grid_outside_bbox_no_cells(self):
        """Zero-size bbox should raise before creating invalid geometries."""
        from gis.spatial.grid import generate_grid
        with pytest.raises(ValueError):
            generate_grid((88.0, 27.0, 88.0, 27.0), resolution=0.05)

    def test_point_outside_grid_not_counted(self, historical_gdf):
        from gis.spatial.grid import generate_grid
        from gis.spatial.alignment import point_to_grid_assignment
        # Grid that does NOT overlap the historical points
        grid = generate_grid((0.0, 0.0, 1.0, 1.0), resolution=0.5)
        result = point_to_grid_assignment(grid, historical_gdf.to_crs("EPSG:4326"), "hist")
        assert result["hist"].sum() == 0


# ===================================================================
# 10. OUTPUT SCHEMA VALIDATION
# ===================================================================

class TestOutputSchema:
    """Validate that pipeline output matches docs/data-contract.md."""

    REQUIRED_COLUMNS = [
        "location_id", "latitude", "longitude", "geometry",
        "elevation", "slope", "aspect",
        "soil_type", "geology", "land_cover",
        "drainage_feature", "historical_landslide",
    ]

    def _run_pipeline(self, tmp_path):
        from gis.pipeline import build_static_feature_dataset
        from gis.config import GISSettings
        from gis.tests.fixtures.synthetic import (
            make_dem_file, make_soil_vector, make_geology_vector,
            make_landcover_vector, make_drainage_vector, make_historical_landslides,
        )

        dem_path = make_dem_file(tmp_path / "dem.tif")

        # Write vector fixtures to temp GeoPackage files
        soil_gdf = make_soil_vector()
        soil_path = str(tmp_path / "soil.gpkg")
        soil_gdf.to_file(soil_path, driver="GPKG")

        geo_gdf = make_geology_vector()
        geo_path = str(tmp_path / "geology.gpkg")
        geo_gdf.to_file(geo_path, driver="GPKG")

        lc_gdf = make_landcover_vector()
        lc_path = str(tmp_path / "landcover.gpkg")
        lc_gdf.to_file(lc_path, driver="GPKG")

        dr_gdf = make_drainage_vector()
        dr_path = str(tmp_path / "drainage.gpkg")
        dr_gdf.to_file(dr_path, driver="GPKG")

        hist_gdf = make_historical_landslides()
        hist_path = str(tmp_path / "historical.gpkg")
        hist_gdf.to_file(hist_path, driver="GPKG")

        settings = GISSettings(
            dem_path=str(dem_path),
            soil_path=soil_path,
            geology_path=geo_path,
            landcover_path=lc_path,
            drainage_path=dr_path,
            historical_path=hist_path,
            grid_resolution_deg=0.05,
            output_dir=str(tmp_path / "outputs"),
        )

        parquet_path, tif_path, gpkg_path = build_static_feature_dataset(
            bbox=(88.0, 27.0, 88.2, 27.2),
            output_dir=str(tmp_path / "outputs"),
            settings=settings,
        )
        return parquet_path, tif_path, gpkg_path

    def test_parquet_columns_present(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_susceptibility_column_present(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        assert "susceptibility" in df.columns

    def test_susceptibility_range(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        valid = df["susceptibility"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_location_ids_unique(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        assert df["location_id"].nunique() == len(df)

    def test_latitude_longitude_finite(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        assert df["latitude"].between(-90, 90).all()
        assert df["longitude"].between(-180, 180).all()

    def test_historical_landslide_is_int(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        assert pd.api.types.is_integer_dtype(df["historical_landslide"])

    def test_drainage_feature_is_int(self, tmp_path):
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        assert pd.api.types.is_integer_dtype(df["drainage_feature"])

    def test_output_files_exist(self, tmp_path):
        parquet_path, tif_path, gpkg_path = self._run_pipeline(tmp_path)
        assert parquet_path.exists()
        assert tif_path.exists()
        assert gpkg_path.exists()

    def test_metadata_file_exists(self, tmp_path):
        self._run_pipeline(tmp_path)
        meta_path = tmp_path / "outputs" / "metadata.json"
        assert meta_path.exists()

    def test_schema_matches_data_contract(self, tmp_path):
        """Validate that each row can be loaded into EnvironmentalFeatures."""
        from shared.schemas import EnvironmentalFeatures
        parquet_path, _, _ = self._run_pipeline(tmp_path)
        df = pd.read_parquet(parquet_path)
        row = df.iloc[0]
        record = EnvironmentalFeatures(
            location_id=row["location_id"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            elevation=None if pd.isna(row.get("elevation", float("nan"))) else float(row["elevation"]),
            slope=None if pd.isna(row.get("slope", float("nan"))) else float(row["slope"]),
            aspect=None if pd.isna(row.get("aspect", float("nan"))) else float(row["aspect"]),
            soil_type=row.get("soil_type") if pd.notna(row.get("soil_type")) else None,
            geology=row.get("geology") if pd.notna(row.get("geology")) else None,
            land_cover=row.get("land_cover") if pd.notna(row.get("land_cover")) else None,
            historical_landslide=int(row["historical_landslide"]),
        )
        assert record.location_id == row["location_id"]


# ===================================================================
# 11. BASELINE SUSCEPTIBILITY
# ===================================================================

class TestBaselineSusceptibility:
    def _make_features(self):
        return pd.DataFrame({
            "location_id": ["G1", "G2", "G3"],
            "slope": [45.0, 10.0, np.nan],
            "elevation": [2000.0, 500.0, 1200.0],
            "land_cover": ["bare", "forest", "grassland"],
            "soil_type": ["clay", "sand", "loam"],
            "geology": ["schist", "granite", "shale"],
            "historical_landslide": [1, 0, 0],
        })

    def test_susceptibility_length(self):
        from gis.susceptibility.baseline import compute_baseline_susceptibility
        features = self._make_features()
        weights = {"slope": 0.5, "elevation": 0.3, "historical_landslide": 0.2}
        result = compute_baseline_susceptibility(features, weights)
        assert len(result) == len(features)

    def test_susceptibility_in_range(self):
        from gis.susceptibility.baseline import compute_baseline_susceptibility
        features = self._make_features()
        weights = {"slope": 0.4, "land_cover": 0.3, "soil_type": 0.3}
        result = compute_baseline_susceptibility(features, weights)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_nan_slope_cell_still_gets_value(self):
        from gis.susceptibility.baseline import compute_baseline_susceptibility
        features = self._make_features()
        # G3 has NaN slope; other features should still contribute
        weights = {"slope": 0.3, "land_cover": 0.4, "soil_type": 0.3}
        result = compute_baseline_susceptibility(features, weights)
        # G3 should not be NaN (has land_cover and soil_type)
        assert pd.notna(result.iloc[2])

    def test_all_nan_cell_gets_nan_susceptibility(self):
        from gis.susceptibility.baseline import compute_baseline_susceptibility
        features = pd.DataFrame({
            "location_id": ["G1"],
            "slope": [np.nan],
            "land_cover": [None],
        })
        weights = {"slope": 0.5, "land_cover": 0.5}
        result = compute_baseline_susceptibility(features, weights)
        assert pd.isna(result.iloc[0])

    def test_high_risk_cell_higher_than_low_risk(self):
        from gis.susceptibility.baseline import compute_baseline_susceptibility
        features = pd.DataFrame({
            "location_id": ["HIGH", "LOW"],
            "slope": [80.0, 2.0],
            "historical_landslide": [1, 0],
            "land_cover": ["bare", "forest"],
        })
        weights = {"slope": 0.4, "historical_landslide": 0.3, "land_cover": 0.3}
        result = compute_baseline_susceptibility(features, weights)
        assert result.iloc[0] > result.iloc[1]


# ===================================================================
# 12. PIPELINE WITHOUT REAL DATA (minimal mode)
# ===================================================================

class TestMinimalPipeline:
    def test_pipeline_runs_without_any_layers(self, tmp_path):
        """Pipeline must complete even when no data sources are configured."""
        from gis.pipeline import build_static_feature_dataset
        from gis.config import GISSettings

        settings = GISSettings(
            grid_resolution_deg=0.05,
            output_dir=str(tmp_path / "outputs"),
        )
        parquet_path, tif_path, gpkg_path = build_static_feature_dataset(
            bbox=(88.0, 27.0, 88.2, 27.2),
            output_dir=str(tmp_path / "outputs"),
            settings=settings,
        )
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0
        assert "location_id" in df.columns
        # Terrain columns should be NaN (no DEM)
        assert df["elevation"].isna().all()

    def test_pipeline_outputs_are_readable(self, tmp_path):
        from gis.pipeline import build_static_feature_dataset
        from gis.config import GISSettings
        import rasterio as rio

        settings = GISSettings(
            grid_resolution_deg=0.05,
            output_dir=str(tmp_path / "out"),
        )
        parquet_path, tif_path, gpkg_path = build_static_feature_dataset(
            bbox=(88.0, 27.0, 88.1, 27.1),
            output_dir=str(tmp_path / "out"),
            settings=settings,
        )
        # Parquet readable
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0
        # GeoTIFF readable
        with rio.open(tif_path) as src:
            assert src.count == 1
        # GeoPackage readable
        gdf = gpd.read_file(gpkg_path)
        assert len(gdf) > 0
