"""SedonaDB execution adapter."""

from __future__ import annotations

from bench.spatial_bench.config import TABLES
from bench.spatial_bench.queries.sedonadb import QUERIES


class Runner:
    """Execute pinned queries with SedonaDB."""

    def prepare(self, data_dir: str) -> None:
        import sedonadb  # noqa: PLC0415

        self._connection = sedonadb.connect()
        for table in TABLES:
            # SedonaDB 0.4 matches zero objects for a glob and needs the directory prefix
            path = f"{data_dir.rstrip('/')}/{table}/"
            self._connection.read_parquet(
                path,
                options={"aws.skip_signature": True, "aws.region": "us-west-2"},
            ).to_view(table, overwrite=True)

    def execute(self, query_id: str):
        return self._connection.sql(QUERIES[query_id]).to_pandas()

    def close(self) -> None:
        self._connection = None
