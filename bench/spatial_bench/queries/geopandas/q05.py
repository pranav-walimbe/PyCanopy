"""Pinned Apache SpatialBench q5 for GeoPandas."""

import geopandas as gpd
import pandas as pd
from pandas import DataFrame
from shapely.geometry import MultiPoint


def q5(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q5 (GeoPandas): Monthly travel patterns for repeat customers (convex hull of dropoff points)."""
    trip_df = pd.read_parquet(data_paths["trip"])
    cust_df = pd.read_parquet(data_paths["customer"])
    trip_df["dropoff_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_dropoffloc"], crs="EPSG:4326")
    joined = trip_df.merge(
        cust_df[["c_custkey", "c_name"]],
        left_on="t_custkey",
        right_on="c_custkey",
        how="inner",
    )
    joined["pickup_month"] = joined["t_pickuptime"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        joined.groupby(["c_custkey", "c_name", "pickup_month"], as_index=False)
        .agg(
            trip_count=("t_tripkey", "count"),
            dropoff_points=("dropoff_geom", lambda x: list(x)),
        )
        .loc[lambda d: d["trip_count"] > 5]
    )
    grouped["monthly_travel_hull_area"] = gpd.GeoSeries(
        grouped["dropoff_points"].map(MultiPoint), crs="EPSG:4326"
    ).convex_hull.area

    result = (
        grouped.sort_values(
            ["monthly_travel_hull_area", "c_custkey", "pickup_month"],
            ascending=[False, True, True],
        )[["c_custkey", "c_name", "pickup_month", "monthly_travel_hull_area", "trip_count"]]
        .rename(columns={"c_name": "customer_name", "trip_count": "dropoff_count"})
        .head(100)  # Return only the top 100 repeat customer-months (bounded result set)
        .reset_index(drop=True)
    )
    return result
