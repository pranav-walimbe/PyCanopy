"""Q12: The 5 nearest buildings to each trip pickup location.

PyCanopy: a point-to-polygon kNN join of trip pickups against building footprints.
The result is k rows per trip, larger than RAM at SF10, so it is produced out of core.
The streamed join is exposed as a native Polars source (lazy_source), so the join,
the distance sort and the Parquet sink fuse into one Polars streaming pipeline that
spills to disk under a memory budget. Nothing the size of the full join is held in RAM
or written as an intermediate. A terminal select is pushed into the join so only the
output columns are gathered.

The scratch directory must be on real disk. On the benchmark box /tmp is tmpfs (RAM),
so PYCANOPY_SCRATCH (and POLARS_TEMP_DIR for the sort spill) are pointed at the data
volume in bootstrap.sh. The fallback is the system temp dir for local runs.
"""

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


def _scratch_dir() -> Path:
    """Return a disk-backed scratch directory for the out-of-core sink and sort spill."""
    base = os.environ.get("PYCANOPY_SCRATCH") or tempfile.gettempdir()
    out = Path(base) / "pc_q12"
    out.mkdir(parents=True, exist_ok=True)
    return out


def pycanopy(tables) -> pl.LazyFrame:
    """Sink the sorted kNN join out of core, return a lazy scan of the result.

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
        .polygon_knn_join(query_df, "qx", "qy", k=K)
        .select(["t_tripkey", "b_buildingkey", "distance_to_polygon"])
        .lazy_source()
        .sort(["distance_to_polygon", "b_buildingkey"])
        .sink_parquet(out_path)
    )
    return pl.scan_parquet(out_path)
