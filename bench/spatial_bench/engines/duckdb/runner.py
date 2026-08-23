"""DuckDB execution adapter."""

from __future__ import annotations

from bench.spatial_bench.queries.duckdb import QUERIES


class Runner:
    """Execute pinned queries with DuckDB's spatial extension."""

    def prepare(self, data_dir: str, index_mode: str) -> None:
        import duckdb  # noqa: PLC0415

        self._connection = duckdb.connect()
        self._connection.execute("INSTALL spatial")
        self._connection.execute("LOAD spatial")
        self._connection.execute("INSTALL httpfs")
        self._connection.execute("LOAD httpfs")
        self._connection.execute("CREATE SECRET pycanopy_bench_s3 (TYPE s3, REGION 'us-west-2')")
        self._connection.execute("SET enable_external_file_cache = false")
        for table in ("building", "customer", "driver", "trip", "vehicle", "zone"):
            path = f"{data_dir.rstrip('/')}/{table}/*.parquet"
            self._connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")

    def execute(self, query_id: str):
        return self._connection.execute(QUERIES[query_id]).fetchall()

    def close(self) -> None:
        if connection := getattr(self, "_connection", None):
            connection.close()
