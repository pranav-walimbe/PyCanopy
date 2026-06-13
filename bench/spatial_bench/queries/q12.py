"""Q12: The 5 nearest buildings to each trip pickup location.

PyCanopy: a point-to-polygon kNN join of trip pickups against building footprints.
The reference is a nested-loop join (GeoPandas has no kNN join); it is only
practical at small scale and times out at SF1 in the published numbers.

The library streams the trip points through the building index in morsels and
concatenates internally, so the join intermediate stays bounded without the query
batching by hand (collect auto-streams a large-probe join).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

from bench.spatial_bench import check
from pycanopy import wkb_points_to_xy

id = "q12"
title = "5 nearest buildings to each trip pickup"

K = 5


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_tripkey", "t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    joined = sf.lazy().polygon_knn_join(query_df, "qx", "qy", k=K).collect()
    return joined.sort(["distance_to_polygon", "b_buildingkey"])


def reference(paths) -> pd.DataFrame:
    trips_df = pd.read_parquet(paths["trip"], columns=["t_tripkey", "t_pickuploc"])
    buildings_df = pd.read_parquet(
        paths["building"], columns=["b_buildingkey", "b_name", "b_boundary"]
    )

    pickups = gpd.GeoSeries.from_wkb(trips_df["t_pickuploc"], crs="EPSG:4326").to_list()
    boundaries = gpd.GeoSeries.from_wkb(buildings_df["b_boundary"], crs="EPSG:4326").to_list()
    building_keys = buildings_df["b_buildingkey"].to_numpy()
    building_names = buildings_df["b_name"].to_numpy()

    rows = []
    for i, pt in enumerate(pickups):
        dists = np.array([pt.distance(geom) for geom in boundaries])
        nearest = np.lexsort((building_keys, dists))[:K]
        for idx in nearest:
            rows.append(
                {
                    "t_tripkey": trips_df.iloc[i]["t_tripkey"],
                    "b_buildingkey": building_keys[idx],
                    "building_name": building_names[idx],
                    "distance_to_building": dists[idx],
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["distance_to_building", "b_buildingkey"], ascending=[True, True])
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    # k nearest per trip: row counts must match; tie-breaking on equal distances may
    # pick different buildings, so we compare counts rather than exact pairings.
    return check.rowcount(pc_df, ref_df)
