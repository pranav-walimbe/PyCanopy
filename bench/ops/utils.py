"""Timing and data-generation utilities for the calibration benchmark."""

from __future__ import annotations

import time

import numpy as np
import shapely


def generate_points(
    n: int,
    seed: int = 42,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> np.ndarray:
    """Return an (N, 2) float64 array of uniformly random points.

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


def generate_clustered_points(
    n: int,
    seed: int = 42,
    n_clusters: int = 20,
    spread: float = 0.05,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> np.ndarray:
    """Return an (N, 2) float64 array of clustered points.

    Args:
        n: Number of points.
        seed: RNG seed.
        n_clusters: Number of cluster centers.
        spread: Per-cluster Gaussian standard deviation in spatial units.
        bounds: Spatial extent as (min_x, min_y, max_x, max_y).

    Returns:
        Array of shape (N, 2).
    """
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = bounds
    centers = np.column_stack(
        [rng.uniform(min_x, max_x, n_clusters), rng.uniform(min_y, max_y, n_clusters)]
    )
    pts = centers[rng.integers(0, n_clusters, n)] + rng.normal(0, spread, (n, 2))
    pts[:, 0] = pts[:, 0].clip(min_x, max_x)
    pts[:, 1] = pts[:, 1].clip(min_y, max_y)
    return pts


def generate_polygons(
    n: int,
    seed: int = 42,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    polygon_size: float = 0.005,
) -> np.ndarray:
    """Return an array of N axis-aligned box shapely polygons.

    Args:
        n: Number of polygons.
        seed: RNG seed.
        bounds: Spatial extent as (min_x, min_y, max_x, max_y).
        polygon_size: Polygon side length as a fraction of the extent span.

    Returns:
        Array of shapely polygon objects of length N.
    """
    min_x, min_y, max_x, max_y = bounds
    pw = polygon_size * (max_x - min_x)
    ph = polygon_size * (max_y - min_y)
    anchors = generate_points(n, seed, (min_x, min_y, max_x - pw, max_y - ph))
    return shapely.box(anchors[:, 0], anchors[:, 1], anchors[:, 0] + pw, anchors[:, 1] + ph)


def time_ms(fn, runs: int = 3) -> float:
    """Run fn `runs` times and return the median elapsed time in milliseconds.

    Args:
        fn: Zero-argument callable.
        runs: Number of repetitions.

    Returns:
        Median elapsed time in milliseconds.
    """
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1_000)
    return float(np.median(times))
