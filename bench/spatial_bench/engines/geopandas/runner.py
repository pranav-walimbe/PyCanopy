"""GeoPandas execution adapter."""

from __future__ import annotations

from bench.spatial_bench.queries.geopandas import QUERIES


class Runner:
    """Execute pinned query functions with GeoPandas."""

    def prepare(self, data_dir: str, index_mode: str) -> None:
        root = data_dir.rstrip("/")
        self._paths = {
            table: f"{root}/{table}"
            for table in ("building", "customer", "driver", "trip", "vehicle", "zone")
        }

    def execute(self, query_id: str):
        return QUERIES[query_id](self._paths)

    def close(self) -> None:
        self._paths = {}
