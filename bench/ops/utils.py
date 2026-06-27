"""Operation-benchmark machinery: synthetic datasets, timing, report, and competitor callables."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import shapely
from scipy.spatial import cKDTree

from pycanopy import SpatialFrame

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
    """Cold and warm timings for one competitor on one operation.

    Args:
        label: Display name, e.g. "GeoPandas (STRtree)".
        cold_ms: Time including index construction, or None if not measured.
        ms: Warm time with index pre-built.
    """

    label: str
    cold_ms: float | None = None
    ms: float | None = None


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
        """Append an operation result.

        Args:
            op: The operation result to record.
        """
        self._ops.append(op)

    def write_table(self, path) -> None:
        """Write a readable one-row-per-operation summary table to a text file.

        Keeps the pretty box-drawing layout (one row per op: n, cold/warm ms, the
        competitor cold/warm baseline, and the speedup) rather than a raw CSV.

        Args:
            path: Destination text file path.
        """
        rows = []
        for op in self._ops:
            comp = op.competitors[0] if op.competitors else None
            comp_cold = comp.cold_ms if comp else None
            comp_warm = comp.ms if comp else None
            rows.append(
                {
                    "operation": op.name,
                    "n": f"{op.n:,}",
                    "cold ms": round(op.cold_ms, 3),
                    "warm ms": round(op.warm_ms, 3),
                    "gp index": comp.label if comp else None,
                    "gp cold ms": round(comp_cold, 3) if comp_cold is not None else None,
                    "gp ms": round(comp_warm, 3) if comp_warm is not None else None,
                    "speedup": (
                        f"{comp_warm / op.warm_ms:.1f}x"
                        if comp_warm and op.warm_ms > 0
                        else None
                    ),
                }
            )
        with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
            table = str(pl.DataFrame(rows))
        with open(path, "w") as f:
            f.write(table + "\n")


def measure_sf(
    name: str,
    ds: MockDataset,
    sf_fn: Callable,
    competitors: list[tuple[str, Callable, Callable]] | None = None,
) -> OperationResult:
    """Cold, warm, and competitor timings for a SpatialFrame lazy operation.

    Args:
        name: Operation label.
        ds: Dataset used to build a fresh SpatialFrame.
        sf_fn: Callable(sf) that runs the lazy operation.
        competitors: List of (label, cold_fn, warm_fn) triples.

    Returns:
        OperationResult with cold, warm, and per-competitor timings.
    """
    make_sf = ds.as_polygon_spatial_frame if ds.geometry_type == "polygons" else ds.as_spatial_frame
    cold_ms, _ = time_one(lambda: sf_fn(make_sf()))
    sf = make_sf()
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
    competitors: list[tuple[str, Callable, Callable]] | None,
) -> list[CompetitorResult]:
    # Each entry is (label, cold_fn, warm_fn); both are zero-argument callables
    results = []
    for label, cold_fn, warm_fn in competitors or []:
        cold_ms, _ = time_one(cold_fn)
        ms, _ = time_one(warm_fn)
        results.append(CompetitorResult(label=label, cold_ms=cold_ms, ms=ms))
    return results


# GeoPandas competitor callables: cKDTree for k-nearest joins, STRtree for range-based joins


def geopandas_knn_join_indexed(gs, query_df, k: int) -> tuple[Callable, Callable]:
    """Return (cold_fn, warm_fn) for a batch point kNN join using a scipy cKDTree.

    GeoPandas STRtree only supports k=1 nearest; cKDTree is used for k > 1.

    Args:
        gs: A GeoSeries of point geometries.
        query_df: Polars DataFrame with qx and qy columns for the probe side.
        k: Number of nearest neighbours per query point.

    Returns:
        A (cold_fn, warm_fn) pair, each returning (distances, indices) arrays.
    """
    coords = np.column_stack([gs.x.values, gs.y.values])
    qcoords = np.column_stack([query_df["qx"].to_numpy(), query_df["qy"].to_numpy()])

    def cold():
        return cKDTree(coords).query(qcoords, k=k, workers=-1)

    tree = cKDTree(coords)

    def warm():
        return tree.query(qcoords, k=k, workers=-1)

    return cold, warm


def geopandas_polygon_knn_join_indexed(gs, query_df, k: int) -> tuple[Callable, Callable]:
    """Return (cold_fn, warm_fn) for a batch polygon kNN join using a scipy cKDTree on centroids.

    GeoPandas STRtree only supports k=1 nearest; cKDTree on polygon centroids is used for
    k > 1. Distances are centroid-based (approximate).

    Args:
        gs: A GeoSeries of polygon geometries.
        query_df: Polars DataFrame with qx and qy columns for the probe side.
        k: Number of nearest neighbours per query point.

    Returns:
        A (cold_fn, warm_fn) pair, each returning (distances, indices) arrays.
    """
    centroids = shapely.get_coordinates(shapely.centroid(gs.values))
    qcoords = np.column_stack([query_df["qx"].to_numpy(), query_df["qy"].to_numpy()])

    def cold():
        return cKDTree(centroids).query(qcoords, k=k, workers=-1)

    tree = cKDTree(centroids)

    def warm():
        return tree.query(qcoords, k=k, workers=-1)

    return cold, warm


def geopandas_within_distance_indexed(gs, query_df, distance: float) -> tuple[Callable, Callable]:
    """Return (cold_fn, warm_fn) for a batch within-distance join using a shapely STRtree.

    Args:
        gs: A GeoSeries of point or polygon geometries.
        query_df: Polars DataFrame with qx and qy columns for the probe side.
        distance: Distance threshold for a match.

    Returns:
        A (cold_fn, warm_fn) pair, each returning (query_idx, tree_idx) arrays.
    """
    pts = shapely.points(query_df["qx"].to_numpy(), query_df["qy"].to_numpy())

    def cold():
        tree = shapely.STRtree(gs.values)
        return tree.query(pts, predicate="dwithin", distance=distance)

    tree = shapely.STRtree(gs.values)

    def warm():
        return tree.query(pts, predicate="dwithin", distance=distance)

    return cold, warm


def geopandas_batch_contains_indexed(gs, query_df) -> tuple[Callable, Callable]:
    """Return (cold_fn, warm_fn) for a batch point-in-polygon join using a shapely STRtree.

    Args:
        gs: A GeoSeries of the dataset polygons.
        query_df: Polars DataFrame with qx and qy columns for the probe side.

    Returns:
        A (cold_fn, warm_fn) pair, each returning (query_idx, tree_idx) arrays.
    """
    pts = shapely.points(query_df["qx"].to_numpy(), query_df["qy"].to_numpy())

    def cold():
        tree = shapely.STRtree(gs.values)
        return tree.query(pts, predicate="within")

    tree = shapely.STRtree(gs.values)

    def warm():
        return tree.query(pts, predicate="within")

    return cold, warm


def geopandas_intersects_self_join_indexed(gs) -> tuple[Callable, Callable]:
    """Return (cold_fn, warm_fn) for a polygon intersects self-join using a shapely STRtree.

    Args:
        gs: A GeoSeries of the dataset polygons.

    Returns:
        A (cold_fn, warm_fn) pair, each returning (i, j) index pairs with i < j.
    """

    def cold():
        tree = shapely.STRtree(gs.values)
        pairs = tree.query(gs.values, predicate="intersects")
        mask = pairs[0] < pairs[1]
        return pairs[:, mask]

    tree = shapely.STRtree(gs.values)

    def warm():
        pairs = tree.query(gs.values, predicate="intersects")
        mask = pairs[0] < pairs[1]
        return pairs[:, mask]

    return cold, warm


