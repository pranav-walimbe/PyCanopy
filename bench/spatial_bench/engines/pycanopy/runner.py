"""PyCanopy execution adapter for ordinary benchmark runs."""

from __future__ import annotations

from bench.spatial_bench.fetch_utils import SpatialBenchTables
from bench.spatial_bench.queries import pycanopy


class Runner:
    """Execute pinned query functions with PyCanopy."""

    def prepare(self, data_dir: str, index_mode: str) -> None:
        self._tables = SpatialBenchTables(data_dir=data_dir, index_mode=index_mode)

    def execute(self, query_id: str):
        return pycanopy.BY_ID[query_id].pycanopy(self._tables)

    def close(self) -> None:
        self._tables = None
