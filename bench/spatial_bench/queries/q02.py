"""Q2: Count trips whose pickup point falls within the Coconino County zone.

PyCanopy: filter the trip points to those within (distance 0 of) the single named
zone polygon. The reference counts trip pickups intersecting that polygon.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely import wkb

from bench.spatial_bench import check
from bench.spatial_bench.data import wkb_to_polygons

id = "q2"
title = "Count trips starting within Coconino County zone"

ZONE_NAME = "Coconino County"


def pycanopy(tables) -> pl.DataFrame:
    zone = tables.table("zone", ["z_name", "z_boundary"])
    target = zone.filter(pl.col("z_name") == ZONE_NAME).head(1)
    if target.height == 0:
        return pl.DataFrame({"trip_count_in_coconino_county": [0]})
    poly = wkb_to_polygons(target["z_boundary"])[0]

    trip = tables.table("trip", ["t_pickuploc"])
    sf = tables.point_frame(trip, "t_pickuploc")
    inside = sf.points_within_distance_of_polygon(poly, 0.0)
    return pl.DataFrame({"trip_count_in_coconino_county": [len(inside)]})


def reference(paths) -> pd.DataFrame:
    zone_df = pd.read_parquet(paths["zone"], columns=["z_name", "z_boundary"])
    target = zone_df[zone_df["z_name"] == ZONE_NAME].head(1)
    if target.empty:
        return pd.DataFrame({"trip_count_in_coconino_county": [0]})
    poly = wkb.loads(target.iloc[0]["z_boundary"])

    trip_df = pd.read_parquet(paths["trip"], columns=["t_pickuploc"])
    pickups = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    count = int(pickups.intersects(poly).sum())
    return pd.DataFrame({"trip_count_in_coconino_county": [count]})


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc = pc_df["trip_count_in_coconino_county"][0]
    ref = int(ref_df["trip_count_in_coconino_county"].iloc[0])
    # Boundary points are measure-zero, so counts should match exactly.
    return check.scalar(pc, ref, abs_tol=0.0)
