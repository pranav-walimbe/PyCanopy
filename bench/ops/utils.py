"""Operation-benchmark machinery: synthetic datasets, timing, report, competitors.

Generates point/polygon datasets, runs each PyCanopy operation cold (fresh engine,
index build included) and warm (index cached) against naive competitor baselines,
and collects the timings into a table.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import shapely
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

from pycanopy import Engine, SpatialFrame

# dataset generation


class MockDataset:
    """A generated spatial dataset of uniformly random point or polygon geometries.

    Args:
        geometry_type: Either "points" or "polygons".
        n: Number of geometries to generate.
        seed: RNG seed for reproducibility.
        bounds: Spatial extent as (min_x, min_y, max_x, max_y).
        polygon_size: Width and height of each polygon as a fraction of the
            bounds span (polygons only).
    """

    def __init__(
        self,
        geometry_type: str,
        n: int,
        seed: int = 42,
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        polygon_size: float = 0.02,
    ):
        if geometry_type not in ("points", "polygons"):
            raise ValueError(f"geometry_type must be 'points' or 'polygons', got {geometry_type!r}")

        self.geometry_type = geometry_type
        self.n = n
        self.seed = seed
        self.bounds = bounds

        if geometry_type == "points":
            self._data: np.ndarray = generate_points(n, seed, bounds)
        else:
            self._data = _generate_polygons(n, seed, bounds, polygon_size)

    def as_engine(self) -> Engine:
        """Return a PyCanopy Engine (eager indexing) loaded with this dataset.

        Returns:
            An eager-indexed Engine over the dataset geometries.
        """
        engine = (
            Engine(self._data)
            if self.geometry_type == "points"
            else Engine.from_polygons(self._data)
        )
        engine.set_index_mode("eager")
        return engine

    def as_shapely_list(self) -> list:
        """Return polygon geometries as a list of shapely objects (polygons only).

        Returns:
            The polygons as a list of shapely objects.
        """
        if self.geometry_type != "polygons":
            raise TypeError("as_shapely_list is only supported for polygon datasets.")
        return self._data.tolist()

    def as_polars_df(self) -> pl.DataFrame:
        """Return point data as a Polars DataFrame with x and y columns (points only).

        Returns:
            A DataFrame with "x" and "y" columns.
        """
        if self.geometry_type != "points":
            raise TypeError("as_polars_df is only supported for point datasets.")
        return pl.DataFrame({"x": self._data[:, 0], "y": self._data[:, 1]})

    def as_spatial_frame(self) -> SpatialFrame:
        """Return point data as a PyCanopy SpatialFrame (points only).

        Returns:
            An eager-indexed point SpatialFrame.
        """
        if self.geometry_type != "points":
            raise TypeError("as_spatial_frame is only supported for point datasets.")
        return SpatialFrame(self.as_polars_df(), "x", "y", index_mode="eager")

    def as_polygon_spatial_frame(self) -> SpatialFrame:
        """Return polygon data as a PyCanopy SpatialFrame (polygons only).

        Returns:
            An eager-indexed polygon SpatialFrame.
        """
        if self.geometry_type != "polygons":
            raise TypeError("as_polygon_spatial_frame is only supported for polygon datasets.")
        df = pl.DataFrame({"geom": self._data.tolist()})
        return SpatialFrame.from_polygons(df, geometry_col="geom", index_mode="eager")

    def as_coords(self) -> np.ndarray:
        """Return the raw (N, 2) coordinate array (points only).

        Returns:
            The (N, 2) float64 coordinate array.
        """
        if self.geometry_type != "points":
            raise TypeError("as_coords is only supported for point datasets.")
        return self._data


def generate_points(
    n: int,
    seed: int = 42,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> np.ndarray:
    """Return an (N, 2) float64 array of uniformly random (x, y) points.

    Args:
        n: Number of points.
        seed: RNG seed.
        bounds: Spatial extent as (min_x, min_y, max_x, max_y).

    Returns:
        Array of shape (N, 2).
    """
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = bounds
    return np.column_stack([rng.uniform(min_x, max_x, n), rng.uniform(min_y, max_y, n)])


def _generate_polygons(
    n: int,
    seed: int,
    bounds: tuple[float, float, float, float],
    polygon_size: float,
) -> np.ndarray:
    # Generate N axis-aligned box polygons of the given size, anchored on uniform points
    min_x, min_y, max_x, max_y = bounds
    pw = polygon_size * (max_x - min_x)
    ph = polygon_size * (max_y - min_y)
    anchors = generate_points(n, seed, (min_x, min_y, max_x - pw, max_y - ph))
    # shapely.box is fully vectorized, no Python loop over N geometries
    return shapely.box(anchors[:, 0], anchors[:, 1], anchors[:, 0] + pw, anchors[:, 1] + ph)


# timing, report, measurement


def time_one(fn: Callable) -> tuple[float, object]:
    """Run fn exactly once and time it.

    Args:
        fn: A zero-argument callable to run.

    Returns:
        An (elapsed_ms, result) tuple.
    """
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
        n: Dataset size this operation ran on.
        cold_ms: First call on a fresh engine (index build + query).
        warm_ms: Second call on the same engine (index cached).
        competitors: Zero or more competitor timings, in display order.
        index_bytes: Heap bytes used by built indexes after the warm call.
    """

    name: str
    n: int
    cold_ms: float
    warm_ms: float
    competitors: list[CompetitorResult] = field(default_factory=list)
    index_bytes: int | None = None


@dataclass
class BenchmarkReport:
    """Collected operation results, printed per op and written to a CSV summary."""

    _ops: list[OperationResult] = field(default_factory=list)

    def add(self, op: OperationResult) -> None:
        """Append an operation result and print its one-line summary.

        Args:
            op: The operation result to record.
        """
        self._ops.append(op)
        comp = op.competitors[0] if op.competitors else None
        speedup = ""
        if comp and comp.ms is not None and not comp.skipped and op.warm_ms > 0:
            speedup = f" ({comp.ms / op.warm_ms:.1f}x speedup)"
        print(
            f"[testcase] pycanopy completed {op.name} in {op.warm_ms / 1000:.2f}s{speedup}",
            flush=True,
        )

    def write_table(self, path) -> None:
        """Write a readable one-row-per-operation summary table to a text file.

        Keeps the pretty box-drawing layout (one row per op: n, cold/warm ms, the
        GeoPandas baseline, and the speedup) rather than a raw CSV.

        Args:
            path: Destination text file path.
        """
        rows = []
        for op in self._ops:
            comp = op.competitors[0] if op.competitors else None
            gp_ms = comp.ms if comp and comp.ms is not None and not comp.skipped else None
            rows.append(
                {
                    "operation": op.name,
                    "n": f"{op.n:,}",
                    "cold ms": round(op.cold_ms, 3),
                    "warm ms": round(op.warm_ms, 3),
                    "geopandas ms": round(gp_ms, 1) if gp_ms is not None else None,
                    "speedup": f"{gp_ms / op.warm_ms:.0f}x" if gp_ms and op.warm_ms > 0 else None,
                }
            )
        with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
            table = str(pl.DataFrame(rows))
        with open(path, "w") as f:
            f.write(table + "\n")


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
        n=ds.n,
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
        n=ds.n,
        cold_ms=cold_ms,
        warm_ms=warm_ms,
        competitors=_run_competitors(competitors),
        index_bytes=sf.engine.index_bytes,
    )


def _run_competitors(
    competitors: list[tuple[str, Callable | None]] | None,
) -> list[CompetitorResult]:
    # Time each competitor callable, marking None callables as skipped
    results = []
    for label, fn in competitors or []:
        if fn is None:
            results.append(CompetitorResult(label=label, skipped=True))
        else:
            ms, _ = time_one(fn)
            results.append(CompetitorResult(label=label, ms=ms))
    return results


# naive GeoPandas competitor callables (no spatial index)


def geopandas_range_naive(gs) -> Callable:
    """Build a naive bounding-box range filter via gs.intersects (O(N) shapely scan).

    Args:
        gs: A GeoSeries of the dataset geometries.

    Returns:
        A function taking a (min_x, min_y, max_x, max_y) box and returning matching rows.
    """

    def fn(q):
        return gs[gs.intersects(shapely_box(*q))]

    return fn


def geopandas_knn_naive(gs, k: int) -> Callable:
    """Build a naive kNN query via a gs.distance sort (O(N) shapely scan).

    Args:
        gs: A GeoSeries of the dataset geometries.
        k: Number of nearest neighbours to return.

    Returns:
        A function taking an (x, y) query point and returning the k nearest rows.
    """

    def fn(q):
        return gs.distance(Point(q[0], q[1])).nsmallest(k)

    return fn


def geopandas_contains_naive(gs) -> Callable:
    """Build a naive point-in-polygon query via gs.contains (O(N) shapely scan).

    Args:
        gs: A GeoSeries of the dataset polygons.

    Returns:
        A function taking an (x, y) query point and returning the containing polygons.
    """

    def fn(q):
        return gs[gs.contains(Point(q[0], q[1]))]

    return fn


def geopandas_intersects_naive(gs) -> Callable:
    """Build a naive polygon range query via gs.intersects (O(N) shapely scan).

    Args:
        gs: A GeoSeries of the dataset polygons.

    Returns:
        A function taking a (min_x, min_y, max_x, max_y) box and returning matching rows.
    """

    def fn(q):
        return gs[gs.intersects(shapely_box(*q))]

    return fn


def geopandas_knn_join_naive(gs, k: int) -> Callable:
    """Build a naive batch kNN join via a Python loop over gs.distance (O(Q * N)).

    Args:
        gs: A GeoSeries of the dataset geometries.
        k: Number of nearest neighbours per query point.

    Returns:
        A function taking the query DataFrame and returning each point's k nearest indices.
    """

    def fn(query_df):
        results = []
        for i in range(len(query_df)):
            pt = Point(float(query_df["qx"][i]), float(query_df["qy"][i]))
            results.append(gs.distance(pt).nsmallest(k).index.tolist())
        return results

    return fn


def geopandas_within_distance_naive(gs, distance: float) -> Callable:
    """Build a naive batch within-distance join via a Python loop over gs.distance (O(Q * N)).

    Args:
        gs: A GeoSeries of point or polygon geometries (gs.distance handles either).
        distance: Distance threshold for a match.

    Returns:
        A function taking the query DataFrame and returning each point's within-distance indices.
    """

    def fn(query_df):
        results = []
        for i in range(len(query_df)):
            d = gs.distance(Point(float(query_df["qx"][i]), float(query_df["qy"][i])))
            results.append(gs[d <= distance].index.tolist())
        return results

    return fn


def geopandas_batch_contains_naive(gs) -> Callable:
    """Build a naive batch contains join via a Python loop over gs.contains (O(Q * N)).

    Args:
        gs: A GeoSeries of the dataset polygons.

    Returns:
        A function taking the query DataFrame and returning each point's containing indices.
    """

    def fn(query_df):
        results = []
        for i in range(len(query_df)):
            pt = Point(float(query_df["qx"][i]), float(query_df["qy"][i]))
            results.append(gs[gs.contains(pt)].index.tolist())
        return results

    return fn


def geopandas_intersects_self_join_naive(gs) -> Callable:
    """Build a naive all-pairs polygon intersects self-join (O(N^2) shapely scan).

    Args:
        gs: A GeoSeries of the dataset polygons.

    Returns:
        A zero-argument function returning the intersecting (i, j) index pairs with i < j.
    """

    def fn():
        pairs = []
        for i in range(len(gs)):
            hits = gs.intersects(gs.iloc[i])
            pairs.extend((i, j) for j in hits[hits].index if j > i)
        return pairs

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
