"""
Q9: Building conflation via IoU — find overlapping building footprints.
"""

from __future__ import annotations

import polars as pl

id = "q9"
title = "Building overlap detection via IoU"

compare = {
    "keys": ["building_1", "building_2"],
    "values": ["area1", "area2", "overlap_area", "iou"],
    "rel_tol": 1e-4,
}


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")
    pairs = sf.intersects_pairs(key_col="b_buildingkey")
    if pairs.height == 0:
        return pl.DataFrame(
            schema={
                "building_1": pl.Int64,
                "building_2": pl.Int64,
                "area1": pl.Float64,
                "area2": pl.Float64,
                "overlap_area": pl.Float64,
                "iou": pl.Float64,
            }
        )
    return pairs.select(
        pl.col("b_buildingkey_1").alias("building_1"),
        pl.col("b_buildingkey_2").alias("building_2"),
        pl.col("area_left").alias("area1"),
        pl.col("area_right").alias("area2"),
        "overlap_area",
        "iou",
    ).sort(["iou", "building_1", "building_2"], descending=[True, False, False])