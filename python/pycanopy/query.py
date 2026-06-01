"""Typed result wrappers returned by Engine query methods"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnnResult:
    indices: list[int]
    query_point: tuple[float, float]
    k: int


@dataclass
class RangeResult:
    indices: list[int]
    bbox: tuple[float, float, float, float]
