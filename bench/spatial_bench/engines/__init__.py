"""Engine adapters for ordinary SpatialBench runs."""

from __future__ import annotations

from importlib import import_module

from bench.spatial_bench.config import ENGINE_IDS


def load_runner(engine: str):
    """Create one engine's ordinary benchmark runner."""
    if engine not in ENGINE_IDS:
        raise ValueError(f"unsupported engine: {engine}")
    module = import_module(f"bench.spatial_bench.engines.{engine}.runner")
    return module.Runner()
