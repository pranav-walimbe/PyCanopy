"""Q4: Zone distribution of the top 1000 trips by tip amount.

PyCanopy: take the top 1000 trips by tip, within-join their pickup points against
zone polygons, and count per zone. The reference uses a GeoPandas within sjoin.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl

from bench.spatial_bench import check
from bench.spatial_bench.data import wkb_points_to_xy

id = "q4"
title = "Zone distribution of the top 1000 trips by tip"

TOP_N = 1000


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_tip", "t_pickuploc"])
    top = trip.sort(["t_tip", "t_tripkey"], descending=[True, False]).head(TOP_N)
    qx, qy = wkb_points_to_xy(top["t_pickuploc"])
    query_df = top.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    sf = tables.polygon_frame(zone, "z_boundary")

    joined = sf.lazy().within_join(query_df, "qx", "qy").collect()
    return (
        joined.group_by(["z_zonekey", "z_name"])
        .agg(pl.len().alias("trip_count"))
        .sort(["trip_count", "z_zonekey"], descending=[True, False])
    )


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(paths["trip"], columns=["t_tripkey", "t_tip", "t_pickuploc"])
    top_trips = trip_df.sort_values(["t_tip", "t_tripkey"], ascending=[False, True]).head(TOP_N)
    top_trips["pickup_geom"] = gpd.GeoSeries.from_wkb(top_trips["t_pickuploc"], crs="EPSG:4326")
    top_gdf = gpd.GeoDataFrame(top_trips, geometry="pickup_geom", crs="EPSG:4326")

    zone_df = pd.read_parquet(paths["zone"], columns=["z_zonekey", "z_name", "z_boundary"])
    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")
    zones_gdf = gpd.GeoDataFrame(zone_df, geometry="zone_geom", crs="EPSG:4326")[
        ["z_zonekey", "z_name", "zone_geom"]
    ]

    return (
        gpd.sjoin(top_gdf, zones_gdf, how="inner", predicate="within")
        .groupby(["z_zonekey", "z_name"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
        .sort_values(["trip_count", "z_zonekey"], ascending=[False, True])
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {r["z_zonekey"]: r["trip_count"] for r in pc_df.iter_rows(named=True)}
    ref_map = dict(zip(ref_df["z_zonekey"], ref_df["trip_count"], strict=False))
    return check.grouped(pc_map, ref_map)
