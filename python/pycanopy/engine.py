"""
High-level Python engine wrapping the Rust core.

Normalizes varied point input to contiguous float64 arrays, with standard 2D LE WKB decoding via a strided numpy view.
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
    # Return (xs, ys) as contiguous float64 arrays from a numpy (N, 2) array, GeoArrow
    # array, GeoSeries, shapely Points, or (x, y) tuples.
    if isinstance(geometries, np.ndarray):
        if geometries.ndim != 2 or geometries.shape[1] < 2:
            raise ValueError("numpy array must be shape (N, 2)")
        return (
            np.ascontiguousarray(geometries[:, 0], dtype=np.float64),
            np.ascontiguousarray(geometries[:, 1], dtype=np.float64),
        )

    if isinstance(geometries, (pa.Array, pa.ChunkedArray)):
        if pa.types.is_binary(geometries.type) or pa.types.is_large_binary(geometries.type):
            return wkb_points_to_xy(geometries)
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
    # Extract x/y from a GeoArrow array as contiguous float64 arrays, supporting
    # struct<x, y> and FixedSizeList<2> encodings.
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


# A standard 2D little-endian WKB point is a fixed 21-byte record: a 1-byte byte
# order flag, a 4-byte geometry type, then the x and y doubles.
_WKB_POINT_NBYTES = 21
_WKB_LITTLE_ENDIAN = 1
_WKB_POINT_TYPE = 1
_WKB_POINT_RECORD = np.dtype([("order", "u1"), ("type", "<u4"), ("x", "<f8"), ("y", "<f8")])


def wkb_points_to_xy(points) -> tuple[np.ndarray, np.ndarray]:
    """Decode a column of WKB point geometries to contiguous float64 x and y arrays.

    Standard 2D little-endian points use a vectorised buffer read. Other variants
    (big-endian, Z/M, nulls) fall back to shapely.

    Args:
        points: A column of WKB point geometries in one of the accepted forms.

    Returns:
        Pair (xs, ys) of contiguous float64 numpy arrays.
    """
    if hasattr(points, "to_arrow"):  # e.g. a polars Series
        points = points.to_arrow()
    if isinstance(points, pa.ChunkedArray):
        points = points.combine_chunks()

    if isinstance(points, pa.Array):
        fast = _wkb_points_fast(points)
        if fast is not None:
            return fast
        points = points.to_numpy(zero_copy_only=False)

    geoms = shapely.from_wkb(np.asarray(points, dtype=object))
    return (
        np.ascontiguousarray(shapely.get_x(geoms), dtype=np.float64),
        np.ascontiguousarray(shapely.get_y(geoms), dtype=np.float64),
    )


def wkb_point_distance(series_a, series_b) -> np.ndarray:
    """Compute the Euclidean distance between two WKB point columns in one parallel pass.

    Args:
        series_a: A column of WKB point geometries (first point set).
        series_b: A column of WKB point geometries (second point set).

    Returns:
        Float64 numpy array of per-row distances.
    """
    xs1, ys1 = wkb_points_to_xy(series_a)
    xs2, ys2 = wkb_points_to_xy(series_b)
    return _CoreEngine.euclidean_distance(xs1, ys1, xs2, ys2)


def _wkb_points_fast(arr: pa.Array) -> tuple[np.ndarray, np.ndarray] | None:
    # Read x/y from a uniformly 21-byte WKB point column via one numpy view, or None for
    # nulls or any non-uniform or non-point layout so the caller can fall back to shapely.
    if not (pa.types.is_binary(arr.type) or pa.types.is_large_binary(arr.type)):
        return None
    if arr.null_count != 0:
        return None
    n = len(arr)
    if n == 0:
        return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

    _validity, offsets_buf, data_buf = arr.buffers()
    if offsets_buf is None or data_buf is None:
        return None

    # Offsets are int32 for binary, int64 for large_binary (what polars emits). A
    # sliced array shares its parent's buffers, so index past the slice's offset.
    offset_dtype = "<i8" if pa.types.is_large_binary(arr.type) else "<i4"
    offsets = np.frombuffer(offsets_buf, dtype=offset_dtype)[arr.offset : arr.offset + n + 1]

    # Fast path needs every value to be exactly _WKB_POINT_NBYTES, tightly packed
    if not np.array_equal(offsets - offsets[0], np.arange(n + 1) * _WKB_POINT_NBYTES):
        return None

    block = np.frombuffer(data_buf, dtype=np.uint8)[offsets[0] : offsets[-1]]
    records = block.view(_WKB_POINT_RECORD)
    if not (
        np.all(records["order"] == _WKB_LITTLE_ENDIAN)
        and np.all(records["type"] == _WKB_POINT_TYPE)
    ):
        return None
    return (
        np.ascontiguousarray(records["x"], dtype=np.float64),
        np.ascontiguousarray(records["y"], dtype=np.float64),
    )


def _wkb_binary_buffers(column) -> tuple[np.ndarray, np.ndarray] | None:
    # Return zero-copy (data, offsets) numpy buffers of a WKB binary column, where data is
    # the concatenated value bytes and offsets the n+1 bounds, or None for null/non-binary.
    if hasattr(column, "to_arrow"):
        column = column.to_arrow()
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if not isinstance(column, pa.Array):
        return None
    if not (pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type)):
        return None
    if column.null_count != 0:
        return None
    n = len(column)
    _validity, offsets_buf, data_buf = column.buffers()
    if offsets_buf is None or data_buf is None:
        return None
    offset_dtype = "<i8" if pa.types.is_large_binary(column.type) else "<i4"
    offsets = np.frombuffer(offsets_buf, dtype=offset_dtype)[column.offset : column.offset + n + 1]
    data = np.frombuffer(data_buf, dtype=np.uint8)
    return data, np.ascontiguousarray(offsets, dtype=np.int64)


def _extract_polygon_rings(
    geometries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    # Return (xs, ys, ring_offsets, poly_offsets, part_poly) from Polygons/MultiPolygons
    geoms = np.asarray(geometries)
    type_ids = shapely.get_type_id(geoms)
    invalid = (type_ids != _SHAPELY_POLYGON_TYPE_ID) & (type_ids != _SHAPELY_MULTIPOLYGON_TYPE_ID)
    if invalid.any():
        idx = int(np.argmax(invalid))
        raise TypeError(
            f"Geometry at index {idx} is not a Polygon or MultiPolygon "
            f"(got {type(geoms[idx]).__name__!r}). Engine.from_polygons requires polygonal input."
        )

    # Flatten MultiPolygons into parts (part_poly maps each part to its source geometry), then
    # the rings of each part, exterior first then holes, with the count of rings per part.
    parts_arr, part_poly = shapely.get_parts(geoms, return_index=True)
    all_rings, ring_part = shapely.get_rings(parts_arr, return_index=True)
    rings_per_part = np.bincount(ring_part, minlength=len(parts_arr))

    coords = shapely.get_coordinates(all_rings)
    ring_coord_counts = shapely.get_num_coordinates(all_rings)

    ring_offsets = np.concatenate([[0], np.cumsum(ring_coord_counts)])
    poly_offsets = np.concatenate([[0], np.cumsum(rings_per_part)])

    # None when no geometry expanded into multiple parts: every part is its own polygon
    part_poly_arr = None
    if len(parts_arr) != len(geoms):
        part_poly_arr = np.ascontiguousarray(part_poly, dtype=np.int64)

    return (
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(ring_offsets, dtype=np.int64),
        np.ascontiguousarray(poly_offsets, dtype=np.int64),
        part_poly_arr,
    )


def _extract_query_polygon_rings(
    polygon,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Return (xs, ys, ring_offsets, poly_offsets) for a query Polygon or MultiPolygon via
    # two-level offsets grouping rings into parts (one part per member).
    tid = shapely.get_type_id(polygon)
    if tid == _SHAPELY_POLYGON_TYPE_ID:
        members = (polygon,)
    elif tid == _SHAPELY_MULTIPOLYGON_TYPE_ID:
        members = shapely.get_parts(polygon)
    else:
        raise TypeError(f"Expected a Polygon or MultiPolygon, got {type(polygon).__name__!r}.")
    rings = []
    rings_per_part = []
    for poly in members:
        rings.append(shapely.get_exterior_ring(poly))
        n_interior = int(shapely.get_num_interior_rings(poly))
        for j in range(n_interior):
            rings.append(shapely.get_interior_ring(poly, j))
        rings_per_part.append(1 + n_interior)
    all_rings = np.asarray(rings)
    coords = shapely.get_coordinates(all_rings)
    ring_coord_counts = shapely.get_num_coordinates(all_rings)
    ring_offsets = np.concatenate([[0], np.cumsum(ring_coord_counts)])
    poly_offsets = np.concatenate([[0], np.cumsum(np.array(rings_per_part, dtype=np.int64))])
    return (
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(ring_offsets, dtype=np.int64),
        np.ascontiguousarray(poly_offsets, dtype=np.int64),
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
            geometries: A geopandas GeoSeries or list of shapely Polygon / MultiPolygon
                objects. Polygons with holes are accepted, and a MultiPolygon is treated
                as one logical polygon spanning all of its parts.

        Returns:
            Engine object over polygon data.
        """
        xs, ys, ring_offsets, poly_offsets, part_poly = _extract_polygon_rings(geometries)
        eng = cls.__new__(cls)
        eng._core = _CoreEngine.from_polygon_rings(xs, ys, ring_offsets, poly_offsets, part_poly)
        return eng

    @classmethod
    def from_wkb_polygons(cls, column) -> Engine:
        """Construct from a WKB Polygon/MultiPolygon column, decoding the bytes in Rust.

        Args:
            column: A polars Binary Series or pyarrow Binary/LargeBinary array of WKB
                Polygon / MultiPolygon geometries.

        Returns:
            Engine object over polygon data.
        """
        eng = cls.__new__(cls)
        buffers = _wkb_binary_buffers(column)
        if buffers is not None:
            try:
                eng._core = _CoreEngine.from_wkb_polygons(*buffers)
                return eng
            except ValueError:
                pass  # unusual WKB variant -> shapely fallback
        raw = column.to_numpy() if hasattr(column, "to_numpy") else np.asarray(column)
        return cls.from_polygons(shapely.from_wkb(raw))

    @classmethod
    def from_wkb_points(cls, points) -> Engine:
        """Construct from a column of WKB point geometries.

        Decoded with a vectorised buffer read for standard 2D LE WKB, falling back to
        shapely otherwise.

        Args:
            points: A polars Binary Series, a pyarrow Binary/LargeBinary array, or a
                numpy object array of WKB byte strings.

        Returns:
            Engine object over coord data.
        """
        xs, ys = wkb_points_to_xy(points)
        return cls.from_coords(xs, ys)

    @classmethod
    def from_coords(cls, xs: Sequence[float], ys: Sequence[float]) -> Engine:
        """Construct directly from x and y coordinate sequences.

        Args:
            xs: Sequence of x coordinates.
            ys: Sequence of y coordinates.

        Returns:
            Engine object over coord data.
        """
        eng = cls.__new__(cls)
        eng._core = _CoreEngine.from_points(
            np.ascontiguousarray(xs, dtype=np.float64),
            np.ascontiguousarray(ys, dtype=np.float64),
        )
        return eng

    def build_index(self) -> None:
        """Build the spatial index ahead of any query (idempotent)."""
        self._core.build_index()

    def set_index_mode(self, mode: str) -> str:
        """Set the index build policy, returning the previous mode.

        Modes: "eager" (build when a kind is selected), "none" (always brute-force scan),
        "auto" (build only when the cost model beats a scan) and retains built indexes.

        Args:
            mode: "auto", "eager", or "none".

        Returns:
            The previous index mode.
        """
        return self._core.set_index_mode(mode)

    def knn(self, x: float, y: float, k: int) -> list[int]:
        """Return indices of the k nearest points to (x, y).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            k: Number of neighbours to return.

        Returns:
            Indices into the original dataset, sorted nearest-first.
        """
        return self._core.knn(x, y, k)

    def knn_from_candidates(
        self,
        x: float,
        y: float,
        k: int,
        survivor_indices: np.ndarray,
    ) -> list[int]:
        """Return the k nearest indices from a candidate subset of the dataset.

        Squared distances to each survivor then a partial sort for the k nearest.
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

    def range_mask(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Return a boolean mask over candidates of which points lie in the bounding box.

        The bbox query and the candidate intersection both run in Rust, so the hit set
        never crosses the boundary as a Python list and no per-call bitmap is built here.

        Args:
            min_x: Minimum x coordinate of the bounding box.
            min_y: Minimum y coordinate of the bounding box.
            max_x: Maximum x coordinate of the bounding box.
            max_y: Maximum y coordinate of the bounding box.
            candidates: Contiguous uint32 array of dataset row positions to test.

        Returns:
            Boolean array aligned to candidates, True where the row matches.
        """
        return self._core.range_mask(
            min_x, min_y, max_x, max_y, np.ascontiguousarray(candidates, dtype=np.uint32)
        )

    def contains_mask(self, x: float, y: float, candidates: np.ndarray) -> np.ndarray:
        """Return a boolean mask over candidates of which polygons contain the point (x, y).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            candidates: Contiguous uint32 array of dataset row positions to test.

        Returns:
            Boolean array aligned to candidates, True where the polygon matches.
        """
        return self._core.contains_mask(x, y, np.ascontiguousarray(candidates, dtype=np.uint32))

    def fused_mask(
        self,
        range_queries: list[tuple[float, float, float, float]],
        contains_points: list[tuple[float, float]],
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Return a boolean mask over candidates for fused range and contains predicates.

        The per-predicate queries and their sorted-merge intersection run in Rust, so the
        intermediate hit lists never cross the boundary.

        Args:
            range_queries: List of (min_x, min_y, max_x, max_y) bounding boxes, AND-ed.
            contains_points: List of (x, y) points, each an AND-ed contains predicate.
            candidates: Contiguous uint32 array of dataset row positions to test.

        Returns:
            Boolean array aligned to candidates, True where every predicate matches.
        """
        return self._core.fused_mask(
            range_queries, contains_points, np.ascontiguousarray(candidates, dtype=np.uint32)
        )

    def knn_mask_from_candidates(
        self, x: float, y: float, k: int, candidates: np.ndarray
    ) -> np.ndarray:
        """Return a boolean mask over candidates marking the k nearest of them to (x, y).

        Args:
            x: X coordinate of the query point.
            y: Y coordinate of the query point.
            k: Number of neighbours to mark.
            candidates: Contiguous uint32 array of dataset row positions to search.

        Returns:
            Boolean array aligned to candidates, True for the k nearest.
        """
        return self._core.knn_mask_from_candidates(
            x, y, k, np.ascontiguousarray(candidates, dtype=np.uint32)
        )

    def intersect_ranges(self, queries: list[tuple[float, float, float, float]]) -> list[int]:
        """Return the sorted intersection of multiple bounding-box queries.

        Sorted merge in Rust over O(K * |H|) elements rather than O(K * N) bitmap ANDs.

        Args:
            queries: List of (min_x, min_y, max_x, max_y) bounding-box tuples.

        Returns:
            Sorted list of indices present in all query results.
        """
        return self._core.intersect_ranges(queries)

    def intersect_hits(self, lists: list[list[int]]) -> list[int]:
        """Return the sorted intersection of several hit lists via a Rust sorted merge.

        Args:
            lists: Hit-index lists from heterogeneous queries (range, contains, etc).

        Returns:
            Sorted list of indices present in every input list.
        """
        return self._core.intersect_hits(lists)

    def batch_knn_join(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        k: int,
        total_q_count: int | None = None,
    ) -> np.ndarray:
        """For each query point, return the k nearest neighbours in the dataset.

        Args:
            query_xs: Contiguous float64 array of query x coordinates, shape (N,).
            query_ys: Contiguous float64 array of query y coordinates, shape (N,).
            k: Number of neighbours per query point.
            total_q_count: True probe count to plan against, if this is one streamed morsel.

        Returns:
            uint64 array of shape (N * k,). Block i holds k result indices for query i.
        """
        return self._core.batch_knn_join(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            k,
            total_q_count,
        )

    def batch_within_distance(
        self,
        query_xs,
        query_ys,
        distance: float,
        flipped: bool = False,
        total_q_count: int | None = None,
    ) -> np.ndarray:
        """For each query point return (query_idx, engine_idx) pairs within distance.

        Args:
            query_xs: x coordinates of query points.
            query_ys: y coordinates of query points.
            distance: Maximum Euclidean distance for a match.
            flipped: Index query side and iterate engine points (faster when
                len(query) >> engine.n).
            total_q_count: True probe count to plan against, if this is one streamed morsel.

        Returns:
            Flat uint64 array of shape (M * 2,) interleaved [q0, e0, q1, e1, ...].
        """
        return self._core.batch_within_distance(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            distance,
            flipped,
            total_q_count,
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
        total_q_count: int | None = None,
    ) -> np.ndarray:
        """For each query point, (query_idx, engine_idx) for every containing Engine polygon.

        Args:
            query_xs: Contiguous float64 array of query x coordinates, shape (N,).
            query_ys: Contiguous float64 array of query y coordinates, shape (N,).
            total_q_count: True probe count to plan against, if this is one streamed morsel.

        Returns:
            uint64 array of shape (M * 2,) where M is the total number of matches.
            Reshape to (-1, 2) to get [query_idx, engine_idx] pairs.
        """
        return self._core.batch_contains(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            total_q_count,
        )

    def batch_within_distance_to_polygons(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        distance: float,
        total_q_count: int | None = None,
    ) -> np.ndarray:
        """For each query point, return (query_idx, polygon_idx) pairs within distance.

        Engine must be a polygon dataset. Distance is measured to the polygon boundary
        (zero when the point is inside).

        Args:
            query_xs: Contiguous float64 array of query x coordinates.
            query_ys: Contiguous float64 array of query y coordinates.
            distance: Maximum Euclidean point-to-polygon distance for a match.
            total_q_count: True probe count to plan against, if this is one streamed morsel.

        Returns:
            uint64 array of shape (M * 2,) interleaved [q0, e0, q1, e1, ...].
        """
        return self._core.batch_within_distance_to_polygons(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            distance,
            total_q_count,
        )

    def batch_knn_to_polygons(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        k: int,
        total_q_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each query point, return the k nearest polygons by point-to-polygon distance.

        Args:
            query_xs: Contiguous float64 array of query x coordinates.
            query_ys: Contiguous float64 array of query y coordinates.
            k: Number of nearest polygons per query point.
            total_q_count: True probe count to plan against, if this is one streamed morsel.

        Returns:
            Pair (engine_indices, distances), each a flat array of shape (N * k,) in
            per-query blocks. Padding slots use 2**64 - 1 and inf when fewer than k exist.
        """
        return self._core.batch_knn_to_polygons(
            np.ascontiguousarray(query_xs, dtype=np.float64),
            np.ascontiguousarray(query_ys, dtype=np.float64),
            k,
            total_q_count,
        )

    def batch_knn_to_polygons_sorted(
        self,
        query_xs: np.ndarray,
        query_ys: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like batch_knn_to_polygons but returns all valid pairs sorted by (distance ASC, target_idx ASC).

        The sort runs inside Rust via rayon, so no Polars streaming sort or EBS spill is needed.
        The full result materialises in RAM before returning.

        Args:
            query_xs: Contiguous float64 array of query x coordinates.
            query_ys: Contiguous float64 array of query y coordinates.
            k: Number of nearest polygons per query point.

        Returns:
            Tuple (query_indices, target_indices, distances) as three flat uint64/uint64/float64
            arrays. No per-query block structure, no padding slots.
        """
        return self._core.batch_knn_to_polygons_sorted(
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
        """Return the unsigned area of every polygon in dataset order (polygon datasets only).

        Returns:
            float64 array of unsigned polygon areas in dataset order.
        """
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

    def radius_query(self, cx: float, cy: float, distance: float) -> np.ndarray:
        """Return indices of engine points within `distance` of the center (cx, cy).

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            distance: Maximum Euclidean distance for a match.

        Returns:
            uint64 array of matching point indices.
        """
        return self._core.radius_query(cx, cy, distance)

    def points_within_distance_of_polygon(self, polygon, distance: float) -> np.ndarray:
        """Return indices of engine points within `distance` of a query polygon.

        Args:
            polygon: A shapely Polygon or MultiPolygon (interior holes supported). A
                point matches when within `distance` of any part.
            distance: Maximum Euclidean point-to-polygon distance for a match.

        Returns:
            uint64 array of matching point indices.
        """
        poly_xs, poly_ys, ring_offsets, poly_offsets = _extract_query_polygon_rings(polygon)
        return self._core.points_within_distance_of_polygon(
            poly_xs, poly_ys, ring_offsets, poly_offsets, distance
        )

    @staticmethod
    def group_convex_hull_areas(xs_series, ys_series) -> np.ndarray:
        """Compute the convex hull area for each group in a pair of Polars list Series.

        Args:
            xs_series: A Polars List(Float64) Series of x coordinates, one list per group.
            ys_series: A Polars List(Float64) Series of y coordinates, one list per group.

        Returns:
            Float64 numpy array of convex hull areas, one per group.
        """
        lengths = xs_series.list.len().to_numpy()
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        xs_flat = xs_series.explode().to_numpy().astype(np.float64, copy=False)
        ys_flat = ys_series.explode().to_numpy().astype(np.float64, copy=False)
        return _CoreEngine.group_convex_hull_areas(xs_flat, ys_flat, offsets)

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
        """Report how many points are currently in the delta buffer.

        Returns:
            The number of points in the delta buffer.
        """
        return self._core.delta_len()

    @property
    def index_bytes(self) -> int:
        """Report the heap bytes of all built indexes, excluding the xs/ys arrays.

        Returns:
            Heap bytes of built indexes, zero before any build.
        """
        return self._core.index_bytes()

    @property
    def n(self) -> int:
        """Report the number of geometries in the dataset.

        Returns:
            The number of geometries in the dataset.
        """
        return self._core.n()

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """Report the bounding extent of the dataset.

        Returns:
            The extent as (min_x, min_y, max_x, max_y), or None if empty.
        """
        return self._core.extent()

    def stats(self) -> str:
        """Return a human-readable summary of dataset statistics.

        Returns:
            A human-readable summary of dataset statistics.
        """
        return self._core.stats_info()

    def __repr__(self) -> str:
        return self._core.__repr__() if self._core else "Engine(uninitialised)"
