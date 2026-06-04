"""High-level Python engine wrapping the Rust core.

Accepts GeoArrow arrays, geopandas GeoSeries, shapely geometry lists,
numpy arrays, or plain coordinate sequences. All point input is normalized
to a pair of contiguous float64 numpy arrays before crossing the Python/Rust
boundary, which the Rust side receives as zero-copy slices via NumPy's C API.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import shapely

try:
    from pycanopy._core import Engine as _CoreEngine
except ImportError:
    raise ImportError(
        "pycanopy native extension not found. Build it first with: maturin develop"
    ) from None

_SHAPELY_POLYGON_TYPE_ID = 3
_SHAPELY_MULTIPOLYGON_TYPE_ID = 6


def _to_numpy_xy(geometries) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) as contiguous float64 numpy arrays.

    Args:
        geometries: GeoArrow PyArrow array, geopandas GeoSeries, numpy (N, 2) array,
            list of shapely Points, or list of (x, y) tuples.

    Returns:
        Pair of 1-D contiguous float64 arrays.
    """
    if isinstance(geometries, np.ndarray):
        if geometries.ndim != 2 or geometries.shape[1] < 2:
            raise ValueError("numpy array must be shape (N, 2)")
        return (
            np.ascontiguousarray(geometries[:, 0], dtype=np.float64),
            np.ascontiguousarray(geometries[:, 1], dtype=np.float64),
        )

    if isinstance(geometries, (pa.Array, pa.ChunkedArray)):
        return _geoarrow_to_numpy_xy(geometries)

    if type(geometries).__name__ == "GeoSeries":
        return (
            np.ascontiguousarray(geometries.x.values, dtype=np.float64),
            np.ascontiguousarray(geometries.y.values, dtype=np.float64),
        )

    pairs = list(geometries)
    if pairs and hasattr(pairs[0], "x") and hasattr(pairs[0], "y"):
        return (
            np.array([float(g.x) for g in pairs], dtype=np.float64),
            np.array([float(g.y) for g in pairs], dtype=np.float64),
        )

    return (
        np.array([float(p[0]) for p in pairs], dtype=np.float64),
        np.array([float(p[1]) for p in pairs], dtype=np.float64),
    )


def _geoarrow_to_numpy_xy(array: pa.Array | pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """Extract x/y from a GeoArrow array as contiguous float64 numpy arrays.

    Supports struct<x: float64, y: float64> and FixedSizeList<2, float64> encodings.
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    arr_type = array.type

    if pa.types.is_struct(arr_type):
        has_x = arr_type.get_field_index("x") >= 0
        has_y = arr_type.get_field_index("y") >= 0
        x_col = array.field("x") if has_x else array.field(0)
        y_col = array.field("y") if has_y else array.field(1)
        return (
            np.ascontiguousarray(x_col.to_numpy(zero_copy_only=False), dtype=np.float64),
            np.ascontiguousarray(y_col.to_numpy(zero_copy_only=False), dtype=np.float64),
        )

    if pa.types.is_fixed_size_list(arr_type) and arr_type.list_size == 2:
        flat = array.flatten().to_pylist()
        return (
            np.array(flat[0::2], dtype=np.float64),
            np.array(flat[1::2], dtype=np.float64),
        )

    raise ValueError(f"Unsupported GeoArrow encoding: {arr_type}")


def _extract_polygon_rings(
    geometries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, offsets) as contiguous numpy arrays from a collection of shapely Polygons.

    Each polygon's exterior ring contributes a slice xs[offsets[i]:offsets[i+1]].
    Uses vectorized shapely 2.0 ops and returns contiguous arrays for zero-copy
    transfer into Rust via the numpy C API.
    MultiPolygon geometries are rejected with a clear message.
    """
    geoms = np.asarray(geometries)

    type_ids = shapely.get_type_id(geoms)
    for idx, (geom, tid) in enumerate(zip(geoms, type_ids)):
        if tid == _SHAPELY_MULTIPOLYGON_TYPE_ID:
            raise TypeError(
                f"Geometry at index {idx} is a MultiPolygon. "
                "Split into individual polygons first with .explode() (geopandas) "
                "or by iterating over .geoms (shapely)."
            )
        if tid != _SHAPELY_POLYGON_TYPE_ID:
            raise TypeError(
                f"Geometry at index {idx} is not a Polygon (got {type(geom).__name__!r}). "
                "Engine.from_polygons requires a collection of Polygon geometries."
            )

    coords = shapely.get_coordinates(geoms)
    counts = shapely.get_num_coordinates(geoms)
    offsets = np.concatenate([[0], np.cumsum(counts)])

    return (
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(offsets, dtype=np.int64),
    )


class Engine:
    """Geospatial query engine with automatic index selection.

    Args:
        geometries: Any of: GeoArrow PyArrow array, geopandas GeoSeries, numpy (N x 2)
            array, list of shapely Points, or list of (x, y) tuples.
    """

    def __init__(self, geometries=None):
        self._core = None
        if geometries is not None:
            xs, ys = _to_numpy_xy(geometries)
            self._core = _CoreEngine.from_points(xs, ys)

    @classmethod
    def from_polygons(cls, geometries) -> Engine:
        """Construct from a collection of simple polygon geometries.

        Args:
            geometries: A geopandas GeoSeries or list of shapely Polygon objects.
                MultiPolygon geometries must be split before loading (use
                GeoSeries.explode() or iterate over shapely MultiPolygon.geoms).

        Returns:
            Engine ready to answer range and contains queries over polygon data.
        """
        xs, ys, offsets = _extract_polygon_rings(geometries)
        eng = cls.__new__(cls)
        eng._core = _CoreEngine.from_polygon_rings(xs, ys, offsets)
        return eng

    @classmethod
    def from_coords(cls, xs: Sequence[float], ys: Sequence[float]) -> Engine:
        """Construct directly from x and y coordinate sequences."""
        eng = cls.__new__(cls)
        eng._core = _CoreEngine.from_points(
            np.ascontiguousarray(xs, dtype=np.float64),
            np.ascontiguousarray(ys, dtype=np.float64),
        )
        return eng

    def knn(self, x: float, y: float, k: int, approximate: bool = False) -> list[int]:
        """Return indices of the k nearest points to (x, y).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            k: Number of neighbours to return.
            approximate: Skip exact geometric refinement for speed.

        Returns:
            Indices into the original dataset, sorted nearest-first.
        """
        return self._core.knn(x, y, k, approximate)

    def range_query(self, min_x: float, min_y: float, max_x: float, max_y: float) -> list[int]:
        """Return indices of all points inside the bounding box.

        Args:
            min_x: Minimum x coordinate of the bounding box.
            min_y: Minimum y coordinate of the bounding box.
            max_x: Maximum x coordinate of the bounding box.
            max_y: Maximum y coordinate of the bounding box.

        Returns:
            Indices of matching geometries in no guaranteed order.
        """
        return self._core.range_query(min_x, min_y, max_x, max_y)

    def contains(self, x: float, y: float) -> list[int]:
        """Return indices of polygons that contain the point (x, y).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.

        Returns:
            Indices of matching polygons in no guaranteed order.
        """
        return self._core.contains_query(x, y)

    def batch_knn_join(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        k: int,
        approximate: bool = False,
    ) -> np.ndarray:
        """For each query point, return the k nearest neighbours in the dataset.

        Crosses the Python/Rust boundary once; loops in Rust via rayon.

        Args:
            query_xs: Contiguous float64 array of query x coordinates, shape (N,).
            query_ys: Contiguous float64 array of query y coordinates, shape (N,).
            k: Number of neighbours per query point.
            approximate: Skip exact geometric refinement for speed.

        Returns:
            uint64 array of shape (N * k,). Block i holds k result indices for query i.
        """
        return self._core.batch_knn_join(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            k,
            approximate,
        )

    def batch_within(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> np.ndarray:
        """For each query point, return its index if it falls within the bounding box.

        Crosses the Python/Rust boundary once; filters in Rust via rayon.

        Args:
            query_xs: Contiguous float64 array of query x coordinates, shape (N,).
            query_ys: Contiguous float64 array of query y coordinates, shape (N,).
            min_x: Left edge of the bounding box.
            min_y: Bottom edge of the bounding box.
            max_x: Right edge of the bounding box.
            max_y: Top edge of the bounding box.

        Returns:
            uint64 array of indices (into the query arrays) of matching points.
        """
        return self._core.batch_within(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            min_x,
            min_y,
            max_x,
            max_y,
        )

    def batch_contains(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
    ) -> np.ndarray:
        """For each query point, return (query_idx, engine_idx) for every polygon
        in the dataset that contains it.

        Crosses the Python/Rust boundary once; loops in Rust via rayon.
        Engine must be a polygon dataset.

        Args:
            query_xs: Contiguous float64 array of query x coordinates, shape (N,).
            query_ys: Contiguous float64 array of query y coordinates, shape (N,).

        Returns:
            uint64 array of shape (M * 2,) where M is the total number of matches.
            Reshape to (-1, 2) to get [query_idx, engine_idx] pairs.
        """
        return self._core.batch_contains(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
        )

    def append_delta(self, xs, ys) -> None:
        """Append new points to the delta buffer (point datasets only).

        Args:
            xs: x coordinates as a float64 array-like.
            ys: y coordinates as a float64 array-like.
        """
        self._core.append_delta(
            np.ascontiguousarray(xs, dtype=np.float64),
            np.ascontiguousarray(ys, dtype=np.float64),
        )

    def flush(self) -> None:
        """Force the delta buffer to be merged into the main index immediately."""
        self._core.flush()

    @property
    def delta_len(self) -> int:
        """Number of points currently in the delta buffer."""
        return self._core.delta_len()

    @property
    def n(self) -> int:
        """Number of geometries in the dataset."""
        return self._core.n()

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """Bounding extent as (min_x, min_y, max_x, max_y), or None if empty."""
        return self._core.extent()

    def stats(self) -> str:
        """Return a human-readable summary of dataset statistics."""
        return self._core.stats_info()

    def __repr__(self) -> str:
        return self._core.__repr__() if self._core else "Engine(uninitialised)"
