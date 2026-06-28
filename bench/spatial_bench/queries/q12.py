"""Q12: The 5 nearest buildings to each trip pickup location."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import polars as pl
import psutil

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
    t0 = time.perf_counter()
    tables.parallel_fetch(TABLES_NEEDED)
    t1 = time.perf_counter()

    buildings = tables.table("building", ["b_buildingkey", "b_name", "b_boundary"])
    sf = tables.polygon_frame(buildings, "b_boundary")
    t2 = time.perf_counter()

    trip = tables.table("trip", ["t_tripkey", "t_pickuploc"])
    qx, qy = wkb_points_to_xy(trip["t_pickuploc"])
    query_df = trip.select("t_tripkey").with_columns(pl.Series("qx", qx), pl.Series("qy", qy))

    _proc = psutil.Process()
    _vm_pre = psutil.virtual_memory()
    _sw_pre = psutil.swap_memory()
    rss_pre = _proc.memory_info().rss / 1024**2
    avail_pre = _vm_pre.available / 1024**2
    swap_pre = _sw_pre.used / 1024**2

    out_path = _scratch_dir() / "sorted.parquet"
    joined = (
        sf.lazy()
        .polygon_knn_join(query_df, "qx", "qy", k=K, sorted_output=True)
        .select(["t_tripkey", "b_buildingkey", "distance_to_polygon"])
        .collect()
    )
    t3 = time.perf_counter()

    _vm_post = psutil.virtual_memory()
    _sw_post = psutil.swap_memory()
    rss_post = _proc.memory_info().rss / 1024**2
    avail_post = _vm_post.available / 1024**2
    swap_post = _sw_post.used / 1024**2

    joined.write_parquet(out_path)
    t4 = time.perf_counter()

    print(
        f"PYCANOPY_Q12_STAGES="
        f"fetch={t1 - t0:.2f}s,"
        f"build={t2 - t1:.2f}s,"
        f"join={t3 - t2:.2f}s,"
        f"write={t4 - t3:.2f}s",
        flush=True,
    )
    print(
        f"PYCANOPY_Q12_MEM="
        f"rss_pre={rss_pre:.0f}MB,"
        f"avail_pre={avail_pre:.0f}MB,"
        f"swap_pre={swap_pre:.0f}MB,"
        f"rss_post={rss_post:.0f}MB,"
        f"avail_post={avail_post:.0f}MB,"
        f"swap_post={swap_post:.0f}MB,"
        f"n_buildings={len(buildings)},"
        f"n_trips={len(trip)}",
        flush=True,
    )
    return pl.scan_parquet(out_path)
