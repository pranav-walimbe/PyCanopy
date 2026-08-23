"""PyCanopy execution adapter for ordinary benchmark runs."""

from __future__ import annotations

from bench.spatial_bench.config import TABLES
from bench.spatial_bench.queries import pycanopy


class Runner:
    """Execute pinned query functions with PyCanopy."""

    def prepare(self, data_dir: str) -> None:
        root = data_dir.rstrip("/")
        self._paths = {table: f"{root}/{table}/**/*.parquet" for table in TABLES}

    def execute(self, query_id: str):
        return pycanopy.BY_ID[query_id].pycanopy(self._paths)

    def close(self) -> None:
        self._paths = {}
