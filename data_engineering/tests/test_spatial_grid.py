"""
Unit tests for spatial grid creation and point assignment.
"""

import pytest
import pandas as pd
from data_engineering.spatial.grid import create_spatial_grid, assign_grid_id, GridConfig


@pytest.fixture
def small_config():
    return GridConfig(
        min_lat=27.0,
        max_lat=28.0,
        min_lon=88.0,
        max_lon=89.0,
        resolution_deg=0.5,
    )


class TestGridConfig:
    def test_invalid_lat_order(self):
        with pytest.raises(ValueError, match="min_lat"):
            GridConfig(min_lat=28.0, max_lat=27.0, min_lon=88.0, max_lon=89.0)

    def test_invalid_lon_order(self):
        with pytest.raises(ValueError, match="min_lon"):
            GridConfig(min_lat=27.0, max_lat=28.0, min_lon=89.0, max_lon=88.0)

    def test_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution_deg"):
            GridConfig(min_lat=27.0, max_lat=28.0, min_lon=88.0, max_lon=89.0, resolution_deg=0)


class TestCreateSpatialGrid:
    def test_returns_geodataframe(self, small_config):
        import geopandas as gpd
        gdf = create_spatial_grid(small_config)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_correct_number_of_cells(self, small_config):
        gdf = create_spatial_grid(small_config)
        # 1.0 / 0.5 = 2 rows × 2 cols = 4 cells
        assert len(gdf) == 4

    def test_grid_id_column_present(self, small_config):
        gdf = create_spatial_grid(small_config)
        assert "grid_id" in gdf.columns

    def test_grid_id_format(self, small_config):
        gdf = create_spatial_grid(small_config)
        for gid in gdf["grid_id"]:
            assert gid.startswith("GRID_")

    def test_unique_grid_ids(self, small_config):
        gdf = create_spatial_grid(small_config)
        assert gdf["grid_id"].nunique() == len(gdf)

    def test_crs_is_wgs84(self, small_config):
        gdf = create_spatial_grid(small_config)
        assert gdf.crs.to_epsg() == 4326

    def test_center_lat_in_bounds(self, small_config):
        gdf = create_spatial_grid(small_config)
        assert gdf["center_lat"].between(small_config.min_lat, small_config.max_lat).all()


class TestAssignGridId:
    def test_assigns_grid_id_to_valid_point(self, small_config):
        df = pd.DataFrame({"latitude": [27.3], "longitude": [88.4]})
        result = assign_grid_id(df, small_config)
        assert "grid_id" in result.columns
        assert result.loc[0, "grid_id"] is not None
        assert result.loc[0, "grid_id"].startswith("GRID_")

    def test_none_for_point_outside_grid(self, small_config):
        df = pd.DataFrame({"latitude": [50.0], "longitude": [10.0]})
        result = assign_grid_id(df, small_config)
        assert result.loc[0, "grid_id"] is None

    def test_none_for_missing_coords(self, small_config):
        df = pd.DataFrame({"latitude": [None], "longitude": [88.4]})
        result = assign_grid_id(df, small_config)
        assert result.loc[0, "grid_id"] is None

    def test_configurable_resolution(self):
        """Different resolutions produce different grid IDs."""
        config_coarse = GridConfig(27.0, 28.0, 88.0, 89.0, resolution_deg=1.0)
        config_fine = GridConfig(27.0, 28.0, 88.0, 89.0, resolution_deg=0.1)
        df = pd.DataFrame({"latitude": [27.3], "longitude": [88.4]})
        r1 = assign_grid_id(df, config_coarse)
        r2 = assign_grid_id(df, config_fine)
        # The grid_ids may differ because resolutions differ
        assert r1.loc[0, "grid_id"] != r2.loc[0, "grid_id"]
