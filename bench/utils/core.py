"""Benchmark primitives for PyCanopy operation comparisons.

Each operation reports cold (first call on a fresh engine, index build included),
warm (second call, index cached), and one timing per naive competitor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

from bench.utils.generators import MockDataset


def time_one(fn: Callable) -> tuple[float, object]:
    """Run fn exactly once. Returns (elapsed_ms, result)."""
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1_000, result


@dataclass
class CompetitorResult:
    """Timing for one competitor on one operation.

    Args:
        label: Display name, e.g. "GeoPandas" or "DuckDB".
        ms: Wall-clock time in milliseconds, or None if skipped.
        skipped: True when the competitor was omitted (not installed, too slow, etc).
    """

    label: str
    ms: float | None = None
    skipped: bool = False


@dataclass
class OperationResult:
    """Cold, warm, and competitor timings for one operation.

    Args:
        name: Short label, e.g. "range query".
        cold_ms: First call on a fresh engine (index build + query).
        warm_ms: Second call on the same engine (index cached).
        competitors: Zero or more competitor timings, in display order.
        index_bytes: Heap bytes used by built indexes after the warm call.
    """

    name: str
    cold_ms: float
    warm_ms: float
    competitors: list[CompetitorResult] = field(default_factory=list)
    index_bytes: int | None = None


class BenchmarkReport:
    """Collected operation results for one dataset, rendered as a table.

    Args:
        label: Human-readable dataset description.
        n: Dataset size.
    """

    def __init__(self, label: str, n: int) -> None:
        self.label = label
        self.n = n
        self._ops: list[OperationResult] = []

    def add(self, op: OperationResult) -> None:
        """Append an operation result."""
        self._ops.append(op)

    def to_polars(self) -> pl.DataFrame:
        """Return results as a DataFrame, one row per (operation, competitor) pair."""
        rows = []
        for op in self._ops:
            base = {
                "operation": op.name,
                "cold_ms": op.cold_ms,
                "warm_ms": op.warm_ms,
                "index_bytes": op.index_bytes,
            }
            if not op.competitors:
                rows.append({**base, "competitor": None, "competitor_ms": None, "speedup": None})
                continue
            for comp in op.competitors:
                speedup = (
                    comp.ms / op.warm_ms
                    if comp.ms is not None and not comp.skipped and op.warm_ms > 0
                    else None
                )
                rows.append(
                    {
                        **base,
                        "competitor": comp.label,
                        "competitor_ms": comp.ms,
                        "speedup": speedup,
                    }
                )
        return pl.DataFrame(rows)

    def display(self) -> None:
        """Print the report as a table (times in ms, speedup as competitor/warm)."""
        if not self._ops:
            return
        print(f"\n{self.label}  |  N={self.n:,}\n")
        df = self.to_polars().with_columns(
            pl.col("cold_ms").round(3),
            pl.col("warm_ms").round(3),
            pl.col("competitor_ms").round(3),
            pl.col("speedup").round(1),
        )
        with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True):
            print(df)
        print()


def measure(
    name: str,
    ds: MockDataset,
    engine_fn: Callable,
    competitors: list[tuple[str, Callable | None]] | None = None,
) -> OperationResult:
    """Cold, warm, and competitor timings for one Engine operation.

    Args:
        name: Operation label.
        ds: Dataset used to build a fresh engine.
        engine_fn: Callable(engine) that runs the operation.
        competitors: List of (label, callable) pairs. A None callable marks the
            competitor as skipped (library not installed, too slow, etc).

    Returns:
        OperationResult with cold, warm, and per-competitor timings.
    """
    engine = ds.as_engine()
    cold_ms, _ = time_one(lambda: engine_fn(engine))
    warm_ms, _ = time_one(lambda: engine_fn(engine))
    return OperationResult(
        name=name,
        cold_ms=cold_ms,
        warm_ms=warm_ms,
        competitors=_run_competitors(competitors),
        index_bytes=engine.index_bytes,
    )


def measure_sf(
    name: str,
    ds: MockDataset,
    sf_fn: Callable,
    competitors: list[tuple[str, Callable | None]] | None = None,
) -> OperationResult:
    """Cold, warm, and competitor timings for a SpatialFrame lazy operation.

    Args:
        name: Operation label.
        ds: Dataset used to build a fresh SpatialFrame.
        sf_fn: Callable(sf) that runs the lazy operation.
        competitors: List of (label, callable) pairs. A None callable marks the
            competitor as skipped.

    Returns:
        OperationResult with cold, warm, and per-competitor timings.
    """
    make_sf = ds.as_polygon_spatial_frame if ds.geometry_type == "polygons" else ds.as_spatial_frame
    sf = make_sf()
    cold_ms, _ = time_one(lambda: sf_fn(sf))
    warm_ms, _ = time_one(lambda: sf_fn(sf))
    return OperationResult(
        name=name,
        cold_ms=cold_ms,
        warm_ms=warm_ms,
        competitors=_run_competitors(competitors),
        index_bytes=sf.engine.index_bytes,
    )


def _run_competitors(
    competitors: list[tuple[str, Callable | None]] | None,
) -> list[CompetitorResult]:
    """Time each competitor callable, marking None callables as skipped."""
    results = []
    for label, fn in competitors or []:
        if fn is None:
            results.append(CompetitorResult(label=label, skipped=True))
        else:
            ms, _ = time_one(fn)
            results.append(CompetitorResult(label=label, ms=ms))
    return results


# naive competitor callables


def polars_range_naive(df: pl.DataFrame, x_col: str = "x", y_col: str = "y") -> Callable:
    """Bounding-box filter via Polars scalar expression (no spatial index)."""

    def fn(q):
        bx0, by0, bx1, by1 = q
        return df.filter(
            (pl.col(x_col) >= bx0)
            & (pl.col(x_col) <= bx1)
            & (pl.col(y_col) >= by0)
            & (pl.col(y_col) <= by1)
        )

    fn((0.0, 0.0, 0.5, 0.5))  # warm Polars JIT before timing
    return fn


def geopandas_range_naive(gs) -> Callable:
    """Bounding-box filter via gs.intersects — O(N) shapely scan."""

    def fn(q):
        return gs[gs.intersects(shapely_box(*q))]

    return fn


def geopandas_knn_naive(gs, k: int) -> Callable:
    """kNN via gs.distance sort — O(N) shapely scan."""

    def fn(q):
        return gs.distance(Point(q[0], q[1])).nsmallest(k)

    return fn


def geopandas_contains_naive(gs) -> Callable:
    """Point-in-polygon via gs.contains — O(N) shapely scan."""

    def fn(q):
        return gs[gs.contains(Point(q[0], q[1]))]

    return fn


def geopandas_intersects_naive(gs) -> Callable:
    """Polygon range via gs.intersects — O(N) shapely scan."""

    def fn(q):
        return gs[gs.intersects(shapely_box(*q))]

    return fn


def geopandas_knn_join_naive(gs, k: int) -> Callable:
    """Batch kNN join via Python loop + gs.distance scan — O(Q * N)."""

    def fn(query_df):
        results = []
        for i in range(len(query_df)):
            pt = Point(float(query_df["qx"][i]), float(query_df["qy"][i]))
            results.append(gs.distance(pt).nsmallest(k).index.tolist())
        return results

    return fn


def polars_within_distance_batch_naive(
    df: pl.DataFrame,
    query_df: pl.DataFrame,
    distance: float,
    x_col: str = "x",
    y_col: str = "y",
) -> Callable:
    """Batch within-distance via a Polars filter loop over all query rows — O(Q * N)."""
    d2 = distance**2
    qxs = query_df["qx"].to_list()
    qys = query_df["qy"].to_list()

    def fn():
        results = []
        for qx, qy in zip(qxs, qys):
            results.append(df.filter(((pl.col(x_col) - qx) ** 2 + (pl.col(y_col) - qy) ** 2) <= d2))
        return results

    fn()  # warm Polars JIT before timing
    return fn


def geopandas_batch_contains_naive(gs) -> Callable:
    """Batch contains via Python loop + gs.contains scan — O(Q * N)."""

    def fn(query_df):
        results = []
        for i in range(len(query_df)):
            pt = Point(float(query_df["qx"][i]), float(query_df["qy"][i]))
            results.append(gs[gs.contains(pt)].index.tolist())
        return results

    return fn


# query generation


def make_range_queries(
    coords: np.ndarray,
    selectivity: float,
    n: int,
    seed: int = 0,
) -> list[tuple[float, float, float, float]]:
    """Return n random bounding-box queries of roughly the given selectivity.

    Args:
        coords: (N, 2) float64 array of dataset points.
        selectivity: Approximate fraction of points each query should match.
        n: Number of query boxes to generate.
        seed: RNG seed.

    Returns:
        List of (min_x, min_y, max_x, max_y) tuples.
    """
    side = selectivity**0.5
    rng = np.random.default_rng(seed)
    bx0 = rng.uniform(0.0, 1.0 - side, n)
    by0 = rng.uniform(0.0, 1.0 - side, n)
    return list(zip(bx0.tolist(), by0.tolist(), (bx0 + side).tolist(), (by0 + side).tolist()))


def make_knn_queries(n: int, seed: int = 0) -> list[tuple[float, float]]:
    """Return n random (x, y) query points uniformly in [0, 1]^2.

    Args:
        n: Number of query points.
        seed: RNG seed.

    Returns:
        List of (x, y) tuples.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, (n, 2))
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in range(n)]
