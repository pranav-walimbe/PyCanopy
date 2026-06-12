"""Q8: Count trip pickups within ~500m of each building.

PyCanopy: a polygon within-distance join of building footprints against trip
pickup points, grouped per building. The reference uses GeoPandas dwithin sjoin.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl

from bench.spatial_bench import check
from pycanopy import wkb_points_to_xy

id = "q8"
title = "Trip pickups within ~500m of each building"

THRESHOLD = 0.0045  # degrees (~500m), as in the reference


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = pl.DataFrame({"qx": qx, "qy": qy})

    joined = (
        sf.lazy().polygon_within_distance_join(query_df, "qx", "qy", distance=THRESHOLD).collect()
    )
    return (
        joined.group_by(["b_buildingkey", "b_name"])
        .agg(pl.len().alias("nearby_pickup_count"))
        .sort(["nearby_pickup_count", "b_buildingkey"], descending=[True, False])
    )


def reference(paths) -> pd.DataFrame:
    trips_df = pd.read_parquet(paths["trip"], columns=["t_pickuploc"])
    trips_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trips_df["t_pickuploc"], crs="EPSG:4326")
    pickups = gpd.GeoDataFrame(trips_df, geometry="pickup_geom", crs="EPSG:4326")

    buildings_df = pd.read_parquet(
        paths["building"], columns=["b_buildingkey", "b_name", "b_boundary"]
    )
    buildings_df["boundary_geom"] = gpd.GeoSeries.from_wkb(
        buildings_df["b_boundary"], crs="EPSG:4326"
    )
    buildings = gpd.GeoDataFrame(buildings_df, geometry="boundary_geom", crs="EPSG:4326")

    return (
        buildings.sjoin(pickups, predicate="dwithin", distance=THRESHOLD)
        .groupby(["b_buildingkey", "b_name"], as_index=False)
        .size()
        .rename(columns={"size": "nearby_pickup_count"})
        .sort_values(["nearby_pickup_count", "b_buildingkey"], ascending=[False, True])
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {r["b_buildingkey"]: r["nearby_pickup_count"] for r in pc_df.iter_rows(named=True)}
    ref_map = dict(zip(ref_df["b_buildingkey"], ref_df["nearby_pickup_count"], strict=False))
    return check.grouped(pc_map, ref_map)
