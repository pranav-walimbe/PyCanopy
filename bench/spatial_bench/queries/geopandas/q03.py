"""Pinned Apache SpatialBench q3 for GeoPandas."""

from typing import cast

import geopandas as gpd
import pandas as pd
from pandas import DataFrame
from shapely.geometry import Polygon


def q3(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q3 (GeoPandas): Monthly trip stats for a buffered box around Sedona center.

    Implements: filter trips whose pickup location is within 0.045 degrees (~5km) of the ~26.5x30 km bounding
    box polygon (approximating ST_DWithin(pickup_point, polygon, 0.045)). Then aggregates monthly:
      * total_trips   = COUNT(t_tripkey)
      * avg_distance  = AVG(t_distance) (set NaN if column absent)
      * avg_duration  = AVG(t_dropofftime - t_pickuptime) in seconds
      * avg_fare      = AVG(t_fare) (set NaN if column absent)
    Ordered by pickup_month ASC.
    Returns columns: pickup_month, total_trips, avg_distance, avg_duration, avg_fare
    """
    trip_df = pd.read_parquet(data_paths["trip"])

    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    trips_gdf = gpd.GeoDataFrame(trip_df, geometry="pickup_geom", crs="EPSG:4326")

    base_poly = Polygon(
        [
            (-111.9060, 34.7347),
            (-111.6160, 34.7347),
            (-111.6160, 35.0047),
            (-111.9060, 35.0047),
            (-111.9060, 34.7347),
        ]
    )

    distances = trips_gdf["pickup_geom"].distance(base_poly)
    mask = distances <= 0.045
    filtered = trips_gdf.loc[mask]

    filtered["_duration_seconds"] = (
        filtered["t_dropofftime"] - filtered["t_pickuptime"]
    ).dt.total_seconds()

    filtered["pickup_month"] = filtered["t_pickuptime"].dt.to_period("M").dt.to_timestamp()

    agg = (
        filtered.groupby("pickup_month", as_index=False)
        .agg(
            total_trips=("t_tripkey", "count"),
            avg_distance=("t_distance", "mean"),
            avg_duration=("_duration_seconds", "mean"),
            avg_fare=("t_fare", "mean"),
        )
        .sort_values("pickup_month")
        .reset_index(drop=True)
    )
    return cast(DataFrame, agg)
