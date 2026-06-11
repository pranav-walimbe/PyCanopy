"""Synthetic spatial dataset generation for PyCanopy benchmarks."""

from __future__ import annotations

import numpy as np
import polars as pl
import shapely

from pycanopy import Engine, SpatialFrame


class MockDataset:
    """A generated spatial dataset exposing point or polygon geometries.

    Args:
        geometry_type: Either "points" or "polygons".
        n: Number of geometries to generate.
        distribution: Spatial pattern, "uniform" or "clustered".
        seed: RNG seed for reproducibility.
        bounds: Spatial extent as (min_x, min_y, max_x, max_y).
        polygon_size: Width and height of each polygon as a fraction of the
            bounds span (polygons only).
        n_clusters: Number of Gaussian blobs (clustered distribution only).
        cluster_std: Std dev per cluster as a fraction of bounds span (clustered only).
    """

    def __init__(
        self,
        geometry_type: str,
        n: int,
        distribution: str = "uniform",
        seed: int = 42,
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        polygon_size: float = 0.02,
        n_clusters: int = 10,
        cluster_std: float = 0.05,
    ):
        if geometry_type not in ("points", "polygons"):
            raise ValueError(f"geometry_type must be 'points' or 'polygons', got {geometry_type!r}")

        self.geometry_type = geometry_type
        self.n = n
        self.distribution = distribution
        self.seed = seed
        self.bounds = bounds

        if geometry_type == "points":
            self._data: np.ndarray = generate_points(
                n, distribution, seed, bounds, n_clusters, cluster_std
            )
        else:
            self._data = _generate_polygons(
                n, distribution, seed, bounds, polygon_size, n_clusters, cluster_std
            )

    def as_engine(self) -> Engine:
        """Return a PyCanopy Engine loaded with this dataset."""
        if self.geometry_type == "points":
            return Engine(self._data)
        return Engine.from_polygons(self._data)

    def as_shapely_list(self) -> list:
        """Return polygon geometries as a list of shapely objects (polygons only)."""
        if self.geometry_type != "polygons":
            raise TypeError("as_shapely_list is only supported for polygon datasets.")
        return self._data.tolist()

    def as_polars_df(self) -> pl.DataFrame:
        """Return point data as a Polars DataFrame with x and y columns (points only)."""
        if self.geometry_type != "points":
            raise TypeError("as_polars_df is only supported for point datasets.")
        return pl.DataFrame({"x": self._data[:, 0], "y": self._data[:, 1]})

    def as_spatial_frame(self) -> SpatialFrame:
        """Return point data as a PyCanopy SpatialFrame (points only)."""
        if self.geometry_type != "points":
            raise TypeError("as_spatial_frame is only supported for point datasets.")
        return SpatialFrame(self.as_polars_df(), "x", "y")

    def as_polygon_spatial_frame(self) -> SpatialFrame:
        """Return polygon data as a PyCanopy SpatialFrame (polygons only)."""
        if self.geometry_type != "polygons":
            raise TypeError("as_polygon_spatial_frame is only supported for polygon datasets.")
        df = pl.DataFrame({"geom": self._data.tolist()})
        return SpatialFrame.from_polygons(df, geometry_col="geom")

    def as_coords(self) -> np.ndarray:
        """Return the raw (N, 2) coordinate array (points only)."""
        if self.geometry_type != "points":
            raise TypeError("as_coords is only supported for point datasets.")
        return self._data


def generate_points(
    n: int,
    distribution: str = "uniform",
    seed: int = 42,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    n_clusters: int = 10,
    cluster_std: float = 0.05,
) -> np.ndarray:
    """Return an (N, 2) float64 array of synthetic (x, y) points.

    Args:
        n: Number of points.
        distribution: "uniform" or "clustered".
        seed: RNG seed.
        bounds: Spatial extent as (min_x, min_y, max_x, max_y).
        n_clusters: Number of Gaussian blobs (clustered only).
        cluster_std: Std dev per cluster as a fraction of bounds span (clustered only).

    Returns:
        Array of shape (N, 2).
    """
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return _uniform(rng, n, bounds)
    if distribution == "clustered":
        return _clustered(rng, n, bounds, n_clusters, cluster_std)
    raise ValueError(f"Unknown distribution: {distribution!r}. Choose uniform or clustered.")


def _generate_polygons(
    n: int,
    distribution: str,
    seed: int,
    bounds: tuple[float, float, float, float],
    polygon_size: float,
    n_clusters: int,
    cluster_std: float,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    pw = polygon_size * (max_x - min_x)
    ph = polygon_size * (max_y - min_y)
    anchors = generate_points(
        n, distribution, seed, (min_x, min_y, max_x - pw, max_y - ph), n_clusters, cluster_std
    )
    # shapely.box is fully vectorized, no Python loop over N geometries.
    return shapely.box(anchors[:, 0], anchors[:, 1], anchors[:, 0] + pw, anchors[:, 1] + ph)


def _uniform(
    rng: np.random.Generator,
    n: int,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    return np.column_stack([rng.uniform(min_x, max_x, n), rng.uniform(min_y, max_y, n)])


def _clustered(
    rng: np.random.Generator,
    n: int,
    bounds: tuple[float, float, float, float],
    n_clusters: int,
    cluster_std: float,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    span_x, span_y = max_x - min_x, max_y - min_y

    centers_x = rng.uniform(min_x, max_x, n_clusters)
    centers_y = rng.uniform(min_y, max_y, n_clusters)
    idx = rng.integers(0, n_clusters, n)

    xs = np.clip(centers_x[idx] + rng.normal(0.0, cluster_std * span_x, n), min_x, max_x)
    ys = np.clip(centers_y[idx] + rng.normal(0.0, cluster_std * span_y, n), min_y, max_y)
    return np.column_stack([xs, ys])
