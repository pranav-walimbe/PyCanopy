"""Pinned Apache SpatialBench q8 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame


def q8(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q8 (GeoPandas): Count nearby pickups for each building within ~500m."""
    trips_df = pd.read_parquet(data_paths["trip"])
    trips_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trips_df["t_pickuploc"], crs="EPSG:4326")
    pickups_gdf = gpd.GeoDataFrame(trips_df, geometry="pickup_geom", crs="EPSG:4326")

    buildings_df = pd.read_parquet(data_paths["building"])
    buildings_df["boundary_geom"] = gpd.GeoSeries.from_wkb(
        buildings_df["b_boundary"], crs="EPSG:4326"
    )
    buildings_gdf = gpd.GeoDataFrame(buildings_df, geometry="boundary_geom", crs="EPSG:4326")

    threshold = 0.0045  # degrees (~500m)
    result = (
        buildings_gdf.sjoin(pickups_gdf, predicate="dwithin", distance=threshold)
        .groupby(["b_buildingkey", "b_name"], as_index=False)
        .size()
        .rename(columns={"size": "nearby_pickup_count"})
        .sort_values(["nearby_pickup_count", "b_buildingkey"], ascending=[False, True])
        .head(100)  # Return only the top 100 busiest buildings (bounded result set)
        .reset_index(drop=True)
    )
    return result
