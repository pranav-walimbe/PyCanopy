"""Q12: The 5 nearest buildings to each trip pickup location."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl

from pycanopy import wkb_points_to_xy

id = "q12"
title = "5 nearest buildings to each trip pickup"

K = 5

TABLES_NEEDED = {
    "building": ["b_buildingkey", "b_name", "b_boundary"],
    "trip": ["t_tripkey", "t_pickuploc"],
}

# kNN ties can pick different buildings, so the per-trip distances are compared rather
# than building identity (the SedonaDB column is distance_to_building).
compare = {"keys": ["t_tripkey"], "values": [("distance_to_polygon", "distance_to_building")]}


_SCRATCH: Path | None = None


def _scratch_dir() -> Path:
    # One unique temp dir per process so repeated benchmark runs never overwrite each other
    global _SCRATCH
    if _SCRATCH is None:
        base = os.environ.get("PYCANOPY_SCRATCH") or tempfile.gettempdir()
        _SCRATCH = Path(tempfile.mkdtemp(dir=base, prefix="pc_q12_"))
    return _SCRATCH


def pycanopy(tables) -> pl.LazyFrame:
    """Run the kNN join, sort and sink to Parquet, return a lazy scan.

    Args:
        tables: SpatialBench table accessor providing the trip and building tables.

    Returns:
        A LazyFrame scanning the sorted (t_tripkey, b_buildingkey, distance_to_polygon)
        Parquet output, which the harness streams rather than materialising in RAM.
    """
    tables.parallel_fetch(TABLES_NEEDED)
    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")

    trip = tables.table("trip", ["t_tripkey", "t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    out_path = _scratch_dir() / "sorted.parquet"
    (
        sf.lazy()
        .polygon_knn_join(query_df, "qx", "qy", k=K, sorted_output=True)
        .select(["t_tripkey", "b_buildingkey", "distance_to_polygon"])
        .collect()
    ).write_parquet(out_path)
    return pl.scan_parquet(out_path)
