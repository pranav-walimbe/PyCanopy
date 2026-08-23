"""Pinned Apache SpatialBench q1 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame
from shapely.geometry import Point


def q1(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q1 (GeoPandas): Trips starting within 50km of Sedona city center."""
    trip_df = pd.read_parquet(data_paths["trip"])[["t_tripkey", "t_pickuploc", "t_pickuptime"]]
    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    trip_gdf = gpd.GeoDataFrame(trip_df, geometry="pickup_geom", crs="EPSG:4326")
    trip_gdf["pickup_lon"] = trip_gdf.geometry.x
    trip_gdf["pickup_lat"] = trip_gdf.geometry.y
    center = Point(-111.7610, 34.8697)
    trip_gdf["distance_to_center"] = trip_gdf.geometry.distance(center)
    filtered = trip_gdf[
        trip_gdf["distance_to_center"].notna() & (trip_gdf["distance_to_center"] <= 0.45)
    ]
    return (
        filtered.sort_values(  # type: ignore[no-any-return]
            ["distance_to_center", "t_tripkey"], ascending=[True, True]
        )[
            [
                "t_tripkey",
                "pickup_lon",
                "pickup_lat",
                "t_pickuptime",
                "distance_to_center",
            ]
        ]
        .head(100)
        .reset_index(drop=True)
    )
