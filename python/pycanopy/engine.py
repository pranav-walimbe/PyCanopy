"""High-level Python engine wrapping the Rust core.

Accepts GeoArrow arrays, geopandas GeoSeries, shapely geometry lists,
numpy arrays, or plain coordinate sequences and converts them to the
flat (xs, ys) format that the Rust Engine.from_points constructor expects.

Hard runtime dependencies (numpy, pyarrow) are imported at module level.
Optional dependencies (geopandas, shapely) are imported inside the specific
helper that needs them, giving a clear ImportError if they are absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    import geopandas as gpd


def _extract_points(geometries) -> tuple[list[float], list[float]]:
    """Return (xs, ys) from the most common geometry containers"""
    if isinstance(geometries, (pa.Array, pa.ChunkedArray)):
        return _from_geoarrow(geometries)

    if isinstance(geometries, np.ndarray):
        if geometries.ndim != 2 or geometries.shape[1] < 2:
            raise ValueError("numpy array must be shape (N, 2)")
        return geometries[:, 0].tolist(), geometries[:, 1].tolist()

    # Check for geopandas GeoSeries by type name to avoid a hard import.
    # If geopandas is installed the isinstance check works; if not, the
    # object simply won't match and we fall through to the tuple path.
    type_name = type(geometries).__name__
    if type_name == "GeoSeries":
        return _from_geoseries(geometries)

    # Shapely Point list: any iterable whose elements have .x and .y
    pairs = list(geometries)
    if pairs and hasattr(pairs[0], "x") and hasattr(pairs[0], "y"):
        return [float(g.x) for g in pairs], [float(g.y) for g in pairs]

    # Plain sequence of (x, y) tuples / lists
    return [float(p[0]) for p in pairs], [float(p[1]) for p in pairs]


def _from_geoseries(gs: gpd.GeoSeries) -> tuple[list[float], list[float]]:
    """Extract coordinates from a geopandas GeoSeries of Points"""
    import geopandas  # noqa: F401 - raises ImportError if not installed

    return gs.x.tolist(), gs.y.tolist()


def _from_geoarrow(array: pa.Array | pa.ChunkedArray) -> tuple[list[float], list[float]]:
    """Extract (xs, ys) from a GeoArrow point array.

    Supports two common GeoArrow encodings:
    - Separated struct: struct<x: float64, y: float64>
    - Interleaved: FixedSizeList<2, float64>
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    arr_type = array.type

    if pa.types.is_struct(arr_type):
        has_x = arr_type.get_field_index("x") >= 0
        x_col = array.field("x") if has_x else array.field(0)
        has_y = arr_type.get_field_index("y") >= 0
        y_col = array.field("y") if has_y else array.field(1)
        return x_col.to_pylist(), y_col.to_pylist()

    if pa.types.is_fixed_size_list(arr_type) and arr_type.list_size == 2:
        flat = array.flatten().to_pylist()
        return flat[0::2], flat[1::2]

    # Fallback: treat each element as an (x, y) pair
    pairs = array.to_pylist()
    return [float(p[0]) for p in pairs], [float(p[1]) for p in pairs]


def _load_core():
    """Import the compiled Rust extension, with a clear error if it is missing"""
    try:
        from pycanopy._core import Engine as _Engine

        return _Engine
    except ImportError:
        raise ImportError(
            "pycanopy native extension not found. Build it first with: maturin develop"
        ) from None


class Engine:
    """Geospatial query engine with automatic index selection.

    Args:
        geometries: Any of: GeoArrow PyArrow array, geopandas GeoSeries, numpy (N x 2)
            array, list of shapely Points, or list of (x, y) tuples.
    """

    def __init__(self, geometries=None):
        self._core = None
        if geometries is not None:
            xs, ys = _extract_points(geometries)
            self._core = _load_core().from_points(xs, ys)

    @classmethod
    def from_coords(cls, xs: Sequence[float], ys: Sequence[float]) -> Engine:
        """Construct directly from x and y coordinate sequences"""
        eng = cls.__new__(cls)
        eng._core = _load_core().from_points(list(xs), list(ys))
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

    def stats(self) -> str:
        """Return a human-readable summary of dataset statistics"""
        return self._core.stats_info()

    def __repr__(self) -> str:
        return self._core.__repr__() if self._core else "Engine(uninitialised)"
