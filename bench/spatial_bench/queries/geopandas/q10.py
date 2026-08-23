"""Pinned Apache SpatialBench q10 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame


def q10(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q10 (GeoPandas): Zone stats for trips starting within each zone.

    Produces columns: z_zonekey, pickup_zone (z_name), avg_duration (seconds), avg_distance, num_trips
    Ordered by avg_duration DESC (NULLS last), z_zonekey ASC.
    Zones with zero trips retained (avg_* = NaN, num_trips = 0).
    """
    trip_df = pd.read_parquet(data_paths["trip"])
    zone_df = pd.read_parquet(data_paths["zone"])

    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    pickup_points = gpd.GeoDataFrame(trip_df, geometry="pickup_geom", crs="EPSG:4326")

    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")
    zones_gdf = gpd.GeoDataFrame(zone_df, geometry="zone_geom", crs="EPSG:4326")

    aggregations = {
        "duration_seconds": "mean",
        "t_distance": "mean",
        "t_tripkey": "count",
    }
    result = (
        gpd.sjoin(pickup_points, zones_gdf, how="right", predicate="within")
        .assign(
            duration_seconds=lambda d: (d["t_dropofftime"] - d["t_pickuptime"]).dt.total_seconds()
        )
        .groupby(["z_zonekey", "z_name"], dropna=False)
        .agg(aggregations)
        .rename(
            columns={
                "duration_seconds": "avg_duration",
                "t_distance": "avg_distance",
                "t_tripkey": "num_trips",
            }
        )
        .reset_index()
        .assign(num_trips=lambda d: d["num_trips"].fillna(0).astype(int))
        .sort_values(
            by=["avg_duration", "z_zonekey"],
            ascending=[False, True],
            na_position="last",
        )
        .rename(columns={"z_name": "pickup_zone"})
        .head(100)  # Return only the top 100 zones by average trip duration (bounded result set)
        .reset_index(drop=True)
    )
    return result
