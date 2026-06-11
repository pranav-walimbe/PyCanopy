"""Q1: Trips starting within ~50km (0.45 degrees) of the Sedona city center.

PyCanopy: a within-distance join of all trip pickup points against the single
center point. The reference filters trips by distance to the center point.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely.geometry import Point

from bench.spatial_bench import check

id = "q1"
title = "Trips starting within ~50km of Sedona center"

CENTER = (-111.7610, 34.8697)
RADIUS = 0.45  # degrees (~50km, planar as in the reference)


def pycanopy(tables) -> pl.DataFrame:
    trip = tables.table("trip", ["t_tripkey", "t_pickuploc", "t_pickuptime"])
    sf = tables.point_frame(trip, "t_pickuploc")
    center_df = pl.DataFrame({"cx": [CENTER[0]], "cy": [CENTER[1]]})
    return sf.lazy().within_distance_join(center_df, "cx", "cy", distance=RADIUS).collect()


def reference(paths) -> pd.DataFrame:
    trip_df = pd.read_parquet(paths["trip"], columns=["t_tripkey", "t_pickuploc", "t_pickuptime"])
    pickup = gpd.GeoSeries.from_wkb(trip_df["t_pickuploc"], crs="EPSG:4326")
    dist = pickup.distance(Point(*CENTER))
    return trip_df[dist.notna() & (dist <= RADIUS)]


def validate(pc_df, ref_df) -> tuple[bool, str]:
    return check.rowcount(pc_df, ref_df)
