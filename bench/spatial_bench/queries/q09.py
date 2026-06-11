"""Q9: Building conflation via IoU — find overlapping building footprints.

PyCanopy: a polygon self-intersection join over building footprints, with overlap
area and IoU computed per pair. The reference uses a GeoPandas intersects self-join.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

from bench.spatial_bench import check

id = "q9"
title = "Building overlap detection via IoU"


def pycanopy(tables) -> pl.DataFrame:
    buildings = tables.table("building", ["b_buildingkey", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")
    pairs = sf.intersects_pairs()

    schema = {"building_1": pl.Int64, "building_2": pl.Int64, "iou": pl.Float64}
    if pairs.height == 0:
        return pl.DataFrame(schema=schema)

    keys = buildings["b_buildingkey"].to_numpy()
    k1 = keys[pairs["left"].to_numpy()]
    k2 = keys[pairs["right"].to_numpy()]
    # Order each pair so building_1 < building_2 by key, matching the reference.
    swap = k1 > k2
    b1 = np.where(swap, k2, k1)
    b2 = np.where(swap, k1, k2)
    return pl.DataFrame(
        {"building_1": b1, "building_2": b2, "iou": pairs["iou"].to_numpy()},
        schema=schema,
    ).sort(["iou", "building_1", "building_2"], descending=[True, False, False])


def reference(paths) -> pd.DataFrame:
    buildings_df = pd.read_parquet(paths["building"], columns=["b_buildingkey", "b_boundary"])
    geoms = gpd.GeoSeries.from_wkb(buildings_df["b_boundary"], crs="EPSG:4326")
    bdf = gpd.GeoDataFrame(
        {"building_key": buildings_df["b_buildingkey"].to_numpy()},
        geometry=geoms.to_numpy(),
        crs="EPSG:4326",
    ).reset_index(drop=True)

    pairs = gpd.sjoin(bdf, bdf, how="inner", predicate="intersects")
    left_key = pairs["building_key_left"].to_numpy()
    right_pos = pairs["index_right"].to_numpy()
    right_key = bdf["building_key"].to_numpy()[right_pos]

    keep = left_key < right_key
    left_geom = gpd.GeoSeries(pairs.geometry.to_numpy()[keep], crs="EPSG:4326")
    right_geom = gpd.GeoSeries(bdf.geometry.to_numpy()[right_pos][keep], crs="EPSG:4326")
    area1 = left_geom.area.to_numpy()
    area2 = right_geom.area.to_numpy()
    overlap = left_geom.intersection(right_geom).area.to_numpy()
    union = area1 + area2 - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union != 0.0)

    return (
        pd.DataFrame({"building_1": left_key[keep], "building_2": right_key[keep], "iou": iou})
        .sort_values(["iou", "building_1", "building_2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def validate(pc_df, ref_df) -> tuple[bool, str]:
    pc_map = {(r["building_1"], r["building_2"]): r["iou"] for r in pc_df.iter_rows(named=True)}
    ref_map = {
        (int(a), int(b)): float(i)
        for a, b, i in zip(ref_df["building_1"], ref_df["building_2"], ref_df["iou"], strict=False)
    }
    return check.grouped(pc_map, ref_map, rel_tol=1e-4)
