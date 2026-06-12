"""Q6: Zone statistics for trips whose pickup falls in zones intersecting a bbox.

PyCanopy: a polygon range query selects zones intersecting the bounding box, then
a within-join counts trip pickups per candidate zone. The reference filters zones
by intersects, then within-joins.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import Polygon

from bench.spatial_bench import check
from pycanopy import wkb_points_to_xy

id = "q6"
title = "Zone stats for trips intersecting a bounding box"

# Axis-aligned bounding box (min_x, min_y, max_x, max_y), matching the reference polygon.
BBOX = (-112.2110, 34.4197, -111.3110, 35.3197)
BBOX_POLY = Polygon(
    [
        (BBOX[0], BBOX[1]),
        (BBOX[2], BBOX[1]),
        (BBOX[2], BBOX[3]),
        (BBOX[0], BBOX[3]),
        (BBOX[0], BBOX[1]),
    ]
)

_TRIP_COLS = ["t_tripkey", "t_pickuploc", "t_totalamount", "t_pickuptime", "t_dropofftime"]


def pycanopy(tables) -> pl.DataFrame:
    zone = tables.table("zone", ["z_zonekey", "z_name", "z_boundary"])
    zsf = tables.polygon_frame(zone, "z_boundary")
    cand_idx = zsf.engine.range_query(*BBOX)
    if not cand_idx:
        return pl.DataFrame(
            schema={
                "z_zonekey": pl.Int64,
                "z_name": pl.Utf8,
                "total_pickups": pl.UInt32,
                "avg_distance": pl.Float64,
                "avg_duration": pl.Float64,
            }
        )
    cand = zone[pl.Series(np.asarray(cand_idx, dtype=np.uint32))]
    cand_sf = tables.polygon_frame(cand, "z_boundary")

    trip = tables.table("trip", _TRIP_COLS)
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    qdf = trip.select(["t_tripkey", "t_totalamount", "t_pickuptime", "t_dropofftime"]).with_columns(
        pl.Series("qx", qx), pl.Series("qy", qy)
    )

    joined = cand_sf.lazy().within_join(qdf, "qx", "qy").collect()
    joined = joined.with_columns(
        duration_seconds=(pl.col("t_dropofftime") - pl.col("t_pickuptime")).dt.total_seconds()
    )
    return (
        joined.group_by(["z_zonekey", "z_name"])
        .agg(
            total_pickups=pl.len(),
            avg_distance=pl.col("t_totalamount").mean(),
            avg_duration=pl.col("duration_seconds").mean(),
        )
        .sort(["total_pickups", "z_zonekey"], descending=[True, False])
    )


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(paths["trip"], columns=_TRIP_COLS)
    trip_df["pickup_geom"] = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    pickups = gpd.GeoDataFrame(trip_df, geometry="pickup_geom", crs="EPSG:4326")

    zone_df = pd.read_parquet(paths["zone"], columns=["z_zonekey", "z_name", "z_boundary"])
    zone_df["zone_geom"] = gpd.GeoSeries.from_wkb(zone_df["z_boundary"], crs="EPSG:4326")
    zones = gpd.GeoDataFrame(zone_df, geometry="zone_geom", crs="EPSG:4326")[
        ["z_zonekey", "z_name", "zone_geom"]
    ]
    candidates = zones[zones["zone_geom"].notna() & zones["zone_geom"].intersects(BBOX_POLY)]

    return (
        gpd.sjoin(pickups, candidates, how="inner", predicate="within")
        .assign(
            _duration_seconds=lambda d: (d["t_dropofftime"] - d["t_pickuptime"]).dt.total_seconds()
        )
        .groupby(["z_zonekey", "z_name"], as_index=False)
        .agg(
            total_pickups=("t_tripkey", "count"),
            avg_distance=("t_totalamount", "mean"),
            avg_duration=("_duration_seconds", "mean"),
        )
        .sort_values(["total_pickups", "z_zonekey"], ascending=[False, True])
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {r["z_zonekey"]: r["total_pickups"] for r in pc_df.iter_rows(named=True)}
    ref_map = dict(zip(ref_df["z_zonekey"], ref_df["total_pickups"], strict=False))
    return check.grouped(pc_map, ref_map)
