"""
Resolve a frame's coordinate system and check that geographic coordinates are really lon/lat.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

_MAX_ABS_LON = 180.0
_MAX_ABS_LAT = 90.0


class PyCanopyCoordinateWarning(UserWarning):
    """Warns that a geographic frame's coordinates are not lon/lat degrees."""


def resolve_coordinate_system(
    coordinate_system: Literal["planar", "geographic"] | None,
    xs: np.ndarray,
    ys: np.ndarray,
) -> str:
    """Resolve a frame's coordinate system, warning when geographic coordinates are out of range.

    Unspecified always resolves to "planar" and the coordinates are never read to infer it, so
    a frame follows its declaration alone and never the data it happens to hold.

    Args:
        coordinate_system: Declared "planar" or "geographic", or None to default to "planar".
        xs: The frame's x coordinates.
        ys: The frame's y coordinates.

    Returns:
        Either "planar" or "geographic".
    """
    if coordinate_system is None:
        return "planar"
    if coordinate_system not in ("planar", "geographic"):
        raise ValueError(
            f"coordinate_system must be 'planar' or 'geographic', got {coordinate_system!r}"
        )
    # Only geographic constrains its coordinates, so a planar frame skips the scan entirely
    if coordinate_system == "geographic" and len(xs) > 0 and not _looks_geographic(xs, ys):
        warnings.warn(
            "coordinate_system='geographic' reads x/y as WGS84 lon/lat degrees, but these "
            "coordinates fall outside |x| <= 180 or |y| <= 90, so distances will be wrong. "
            "Longitudes must use the -180..180 convention rather than 0..360, and projected "
            "coordinates need coordinate_system='planar'.",
            PyCanopyCoordinateWarning,
            stacklevel=3,
        )
    return coordinate_system


def _looks_geographic(xs: np.ndarray, ys: np.ndarray) -> bool:
    # Lon/lat degrees are bounded, so anything outside those bounds cannot be WGS84
    return bool(np.all(np.abs(xs) <= _MAX_ABS_LON) and np.all(np.abs(ys) <= _MAX_ABS_LAT))
