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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, ring_offsets, poly_offsets) from a collection of shapely Polygons.

    Uses a two-level offset encoding (GeoArrow-compatible):
      ring_offsets[r]..ring_offsets[r+1] is ring r's coordinate range in xs/ys.
      poly_offsets[i]..poly_offsets[i+1] is polygon i's ring range in ring_offsets.
    The first ring per polygon is the exterior; remaining rings are interior holes.
    All arrays are contiguous for zero-copy transfer into Rust via the numpy C API.
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

    # Build a flat list of rings: for each polygon, exterior first then holes.
    n_interior = shapely.get_num_interior_rings(geoms)
    exterior_rings = shapely.get_exterior_ring(geoms)

    ring_objects = []
    rings_per_poly = []
    for i in range(len(geoms)):
        ring_objects.append(exterior_rings[i])
        for j in range(int(n_interior[i])):
            ring_objects.append(shapely.get_interior_ring(geoms[i], j))
        rings_per_poly.append(1 + int(n_interior[i]))

    all_rings = np.asarray(ring_objects)
    rings_per_poly_arr = np.array(rings_per_poly, dtype=np.int64)

    coords = shapely.get_coordinates(all_rings)
    ring_coord_counts = shapely.get_num_coordinates(all_rings)

    ring_offsets = np.concatenate([[0], np.cumsum(ring_coord_counts)])
    poly_offsets = np.concatenate([[0], np.cumsum(rings_per_poly_arr)])

    return (
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(ring_offsets, dtype=np.int64),
        np.ascontiguousarray(poly_offsets, dtype=np.int64),
    )


def _extract_single_polygon_rings(polygon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, ring_offsets) for a single shapely Polygon (exterior first, then holes).

    ring_offsets[r]..ring_offsets[r+1] is ring r's coordinate range in xs/ys.
    """
    if shapely.get_type_id(polygon) != _SHAPELY_POLYGON_TYPE_ID:
        raise TypeError(
            f"Expected a single Polygon, got {type(polygon).__name__!r}. "
            "Split MultiPolygons and pass one Polygon."
        )
    rings = [shapely.get_exterior_ring(polygon)]
    for j in range(int(shapely.get_num_interior_rings(polygon))):
        rings.append(shapely.get_interior_ring(polygon, j))
    all_rings = np.asarray(rings)
    coords = shapely.get_coordinates(all_rings)
    ring_coord_counts = shapely.get_num_coordinates(all_rings)
    ring_offsets = np.concatenate([[0], np.cumsum(ring_coord_counts)])
    return (
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(ring_offsets, dtype=np.int64),
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
        """Construct from a collection of polygon geometries. Interior holes are supported.

        Args:
            geometries: A geopandas GeoSeries or list of shapely Polygon objects.
                Polygons with holes are accepted. MultiPolygon geometries must be
                split before loading (use GeoSeries.explode() or iterate over
                shapely MultiPolygon.geoms).

        Returns:
            Engine ready to answer range and contains queries over polygon data.
        """
        xs, ys, ring_offsets, poly_offsets = _extract_polygon_rings(geometries)
        eng = cls.__new__(cls)
        eng._core = _CoreEngine.from_polygon_rings(xs, ys, ring_offsets, poly_offsets)
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

    def build_index(self) -> None:
        """Build the spatial index without issuing any query.

        Forces index construction so the first query pays no build cost. Safe to
        call multiple times — no-op if the index is already built.
        """
        self._core.build_index()

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

    def knn_from_candidates(
        self,
        x: float,
        y: float,
        k: int,
        survivor_indices: np.ndarray,
    ) -> list[int]:
        """Return the k nearest indices from a candidate subset of the dataset.

        Computes squared distances from (x, y) to each survivor directly from the
        coordinate arrays and partial-sorts to find the k nearest. Exact, O(M + k log k).
        Use when M survivors are already known (e.g. after scalar pre-filtering).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            k: Number of neighbours to return.
            survivor_indices: Contiguous uint32 array of M row positions in the
                full dataset.

        Returns:
            Up to k indices into the original dataset, sorted nearest-first.
        """
        return self._core.knn_from_candidates(
            x,
            y,
            k,
            np.ascontiguousarray(survivor_indices, dtype=np.uint32),
        )

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

    def intersect_ranges(self, queries: list[tuple[float, float, float, float]]) -> list[int]:
        """Return the sorted intersection of multiple bounding-box queries.

        More efficient than calling range_query per predicate and intersecting in
        Python: performs sorted merge in Rust, operating on O(K * |H|) elements
        rather than O(K * N) bitmap AND passes.

        Args:
            queries: List of (min_x, min_y, max_x, max_y) bounding-box tuples.

        Returns:
            Sorted list of indices present in all query results.
        """
        return self._core.intersect_ranges(queries)

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

    def batch_within_distance(
        self,
        query_xs,
        query_ys,
        distance: float,
        flipped: bool = False,
    ) -> np.ndarray:
        """For each query point return (query_idx, engine_idx) pairs within distance.

        Args:
            query_xs: x coordinates of query points.
            query_ys: y coordinates of query points.
            distance: Maximum Euclidean distance for a match.
            flipped: Index query side and iterate engine points (faster when
                len(query) << engine.n).

        Returns:
            Flat uint64 array of shape (M * 2,) interleaved [q0, e0, q1, e1, ...].
        """
        return self._core.batch_within_distance(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            distance,
            flipped,
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

    def batch_within_distance_to_polygons(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        """For each query point, return (query_idx, polygon_idx) pairs within distance.

        Engine must be a polygon dataset. Distance is measured to the polygon boundary
        (zero when the point is inside).

        Args:
            query_xs: Contiguous float64 array of query x coordinates.
            query_ys: Contiguous float64 array of query y coordinates.
            distance: Maximum Euclidean point-to-polygon distance for a match.

        Returns:
            uint64 array of shape (M * 2,) interleaved [q0, e0, q1, e1, ...].
        """
        return self._core.batch_within_distance_to_polygons(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            distance,
        )

    def batch_knn_to_polygons(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each query point, return the k nearest polygons by point-to-polygon distance.

        Engine must be a polygon dataset.

        Args:
            query_xs: Contiguous float64 array of query x coordinates.
            query_ys: Contiguous float64 array of query y coordinates.
            k: Number of nearest polygons per query point.

        Returns:
            Pair (engine_indices, distances), each a flat array of shape (N * k,) in
            per-query blocks. Padding slots use 2**64 - 1 and inf when fewer than k exist.
        """
        return self._core.batch_knn_to_polygons(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            k,
        )

    def polygon_intersects_self_join(self) -> np.ndarray:
        """Return all intersecting polygon pairs (i, j) with i < j. Polygon datasets only.

        Returns:
            uint64 array of shape (M * 2,) interleaved [i0, j0, i1, j1, ...].
        """
        return self._core.polygon_intersects_self_join()

    def polygon_areas(self) -> np.ndarray:
        """Return the unsigned area of every polygon in dataset order. Polygon datasets only."""
        return self._core.polygon_areas()

    def polygon_pairs_intersection_area(
        self,
        i_idx: np.ndarray,
        j_idx: np.ndarray,
    ) -> np.ndarray:
        """Return the unsigned intersection area for each (i, j) polygon pair.

        Args:
            i_idx: uint64 array of left polygon indices.
            j_idx: uint64 array of right polygon indices, same length as i_idx.

        Returns:
            float64 array of intersection areas, one per pair.
        """
        return self._core.polygon_pairs_intersection_area(
            np.ascontiguousarray(i_idx, dtype=np.uint64),
            np.ascontiguousarray(j_idx, dtype=np.uint64),
        )

    def points_within_distance_of_polygon(self, polygon, distance: float) -> np.ndarray:
        """Return indices of engine points within `distance` of a single query polygon.

        Engine must be a point dataset.

        Args:
            polygon: A single shapely Polygon (interior holes supported).
            distance: Maximum Euclidean point-to-polygon distance for a match.

        Returns:
            uint64 array of matching point indices.
        """
        poly_xs, poly_ys, ring_offsets = _extract_single_polygon_rings(polygon)
        return self._core.points_within_distance_of_polygon(
            poly_xs, poly_ys, ring_offsets, distance
        )

    @staticmethod
    def convex_hull_area(xs, ys) -> float:
        """Return the area of the convex hull of a standalone point set.

        Args:
            xs: x coordinates as a float64 array-like.
            ys: y coordinates as a float64 array-like.

        Returns:
            Area of the convex hull. Zero for fewer than three points.
        """
        return _CoreEngine.convex_hull_area_of(
            np.ascontiguousarray(xs, dtype=np.float64),
            np.ascontiguousarray(ys, dtype=np.float64),
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
    def index_bytes(self) -> int:
        """Heap bytes allocated by all currently-built spatial indexes.

        Excludes the coordinate arrays (xs/ys), which exist regardless of index
        construction. Returns 0 if no query has been issued yet (indexes are built
        lazily). Use this to measure the marginal memory cost of index construction.
        """
        return self._core.index_bytes()

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
