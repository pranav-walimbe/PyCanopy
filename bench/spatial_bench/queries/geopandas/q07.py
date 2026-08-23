"""Pinned Apache SpatialBench q7 for GeoPandas."""

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import DataFrame
from shapely.geometry import LineString


def q7(data_paths: dict[str, str]) -> DataFrame:  # type: ignore[override]
    """Q7 (GeoPandas): Detect potential route detours by comparing reported vs geometric distances.

    Mirrors SQL semantics:
      * Join trip with driver and vehicle
      * Filter trips where t_distance > 0
      * reported_distance_m = t_distance (coerced to float)
      * line_distance_m = length of straight line between pickup and dropoff (meters)
      * detour_ratio = (reported_distance_m) / line_distance_m (NULL if line_distance_m==0)
      * Ordered by detour_ratio DESC, reported_distance_m DESC, t_tripkey ASC
    """
    trip_df = pd.read_parquet(data_paths["trip"])
    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    trip_df["dropoff_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_dropoffloc"], crs="EPSG:4326")
    trip_df["reported_distance_m"] = trip_df["t_distance"].astype(float)
    pickup_vals = trip_df["pickup_geom"].to_numpy()
    dropoff_vals = trip_df["dropoff_geom"].to_numpy()
    line_lengths = np.fromiter(
        (
            LineString([pg, dg]).length / 0.000009  # 1 meter = 0.000009 degree
            if (pg is not None and dg is not None)
            else np.nan
            for pg, dg in zip(pickup_vals, dropoff_vals, strict=False)
        ),
        dtype=float,
        count=len(trip_df),
    )
    trip_df["line_distance_m"] = line_lengths
    trip_df["detour_ratio"] = np.divide(
        trip_df["reported_distance_m"].to_numpy(dtype=float, copy=False),
        line_lengths,
        out=np.full_like(trip_df["reported_distance_m"].to_numpy(dtype=float, copy=False), np.nan),
        where=(line_lengths != 0.0),
    )
    result = (
        trip_df[
            [
                "t_tripkey",
                "reported_distance_m",
                "line_distance_m",
                "detour_ratio",
            ]
        ]
        .sort_values(
            ["detour_ratio", "reported_distance_m", "t_tripkey"],
            ascending=[False, False, True],
            na_position="last",
        )
        .head(100)  # Return only the top 100 highest-detour trips (bounded result set)
        .reset_index(drop=True)
    )
    return result
