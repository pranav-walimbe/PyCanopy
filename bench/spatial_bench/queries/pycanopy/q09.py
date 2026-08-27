"""
Q9: Building conflation via IoU — find overlapping building footprints.
"""

from __future__ import annotations

import polars as pl

from bench.spatial_bench.config import STORAGE_OPTIONS
from pycanopy import SpatialFrame

id = "q9"
title = "Building overlap detection via IoU"


def pycanopy(data_paths: dict[str, str]) -> pl.DataFrame:
    buildings = (
        pl.scan_parquet(data_paths["building"], storage_options=STORAGE_OPTIONS)
        .select(["b_buildingkey", "b_boundary"])
        .collect()
    )
    sf = SpatialFrame.from_wkb_polygons(buildings, "b_boundary")
    pairs = sf.intersects_pairs(key_col="b_buildingkey")
    return (
        pairs.select(
            pl.col("b_buildingkey_1").alias("building_1"),
            pl.col("b_buildingkey_2").alias("building_2"),
            pl.col("area_left").alias("area1"),
            pl.col("area_right").alias("area2"),
            "overlap_area",
            "iou",
        )
        .sort(["iou", "building_1", "building_2"], descending=[True, False, False])
        .head(100)
    )
