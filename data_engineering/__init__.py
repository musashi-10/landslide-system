"""
data_engineering
================
Data ingestion, preprocessing, validation, and spatial indexing pipeline
for the landslide early-warning system.

Quick start
-----------
::

    from data_engineering.pipelines import build_dataset
    from data_engineering.spatial import GridConfig

    dataset, report, provenance = build_dataset(
        source_path="path/to/landslides.csv",
        column_map={"latitude": "lat", "longitude": "lon", "timestamp": "date"},
        grid_config=GridConfig(
            min_lat=26.0, max_lat=30.0,
            min_lon=85.0, max_lon=92.0,
            resolution_deg=0.01,
        ),
    )
"""

__version__ = "1.0.0"
