"""Focused tests for multi-engine SpatialBench orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.spatial_bench import driver_utils, report_utils
from bench.spatial_bench.config import (
    DATASET_VERSION,
    DEFAULT_RUNS,
    INSTANCE_TYPE,
    PUBLIC_DATA_TEMPLATE,
    QUERY_TIMEOUT_SECONDS,
    REGION,
    SUPPORTED_SCALE_FACTORS,
)
from bench.spatial_bench.queries.duckdb import QUERIES as DUCKDB_QUERIES
from bench.spatial_bench.queries.sedonadb import QUERIES as SEDONADB_QUERIES


def _transport(path: Path, engine: str, version: str, seconds: float) -> None:
    path.write_text(
        "\n".join(
            (
                f"engine\t{engine}\t{version}",
                "metadata\tdataset\tv0.1.0",
                f"query\tq1\tok\t{seconds}\t{seconds},{seconds + 1}\t",
            )
        )
        + "\n"
    )


def test_pinned_sql_engines_expose_all_queries():
    expected = [f"q{i}" for i in range(1, 13)]

    assert list(DUCKDB_QUERIES) == expected
    assert list(SEDONADB_QUERIES) == expected


def test_config_matches_spatialbench_single_node_protocol():
    assert SUPPORTED_SCALE_FACTORS == (1, 10)
    assert DEFAULT_RUNS == 3
    assert QUERY_TIMEOUT_SECONDS == 1200
    assert REGION == "us-west-2"
    assert INSTANCE_TYPE == "m7i.2xlarge"
    assert PUBLIC_DATA_TEMPLATE.startswith("s3://")
    # The dataset build has to be identifiable from the path, since the committed answers only
    # match one of them.
    assert DATASET_VERSION in PUBLIC_DATA_TEMPLATE


def test_combine_transports_preserves_requested_engine_order(tmp_path):
    duckdb = tmp_path / "duckdb.tsv"
    pycanopy = tmp_path / "pycanopy.tsv"
    _transport(duckdb, "duckdb", "2.0", 2.0)
    _transport(pycanopy, "pycanopy", "1.0", 1.0)

    results = report_utils.combine_transports([duckdb, pycanopy], ["pycanopy", "duckdb"], 1)

    assert results["engine_order"] == ["pycanopy", "duckdb"]
    assert results["engines"]["pycanopy"]["queries"]["q1"]["seconds"] == 1.0
    assert results["metadata"]["engines"] == "PyCanopy 1.0, DuckDB 2.0"


def test_combined_text_contains_engine_columns_and_raw_samples(tmp_path):
    transport = tmp_path / "duckdb.tsv"
    _transport(transport, "duckdb", "2.0", 2.0)
    results = report_utils.combine_transports([transport], ["duckdb"], 10)
    output = tmp_path / "results.txt"

    report_utils.write_results_txt(results, output)

    text = output.read_text()
    assert "Apache SpatialBench SF10" in text
    assert "DuckDB" in text
    assert "DuckDB q1: 2.00, 3.00" in text


@pytest.mark.parametrize("engine", ["pycanopy", "duckdb", "sedonadb", "geopandas"])
def test_all_engines_use_the_shared_runner(monkeypatch, engine):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(stdout="SPATIALBENCH_TIME=1.25\n", stderr="")

    monkeypatch.setattr(driver_utils.subprocess, "run", run)

    result = driver_utils._spawn(engine, "q1", "s3://data")

    assert result["status"] == "ok"
    assert result["time"] == 1.25
    assert captured["command"][-3:] == [engine, "q1", "s3://data"]


def test_grouped_chart_supports_live_engine_results(tmp_path):
    pytest.importorskip("matplotlib")
    duckdb = tmp_path / "duckdb.tsv"
    pycanopy = tmp_path / "pycanopy.tsv"
    _transport(duckdb, "duckdb", "2.0", 2.0)
    _transport(pycanopy, "pycanopy", "1.0", 1.0)
    results = report_utils.combine_transports([duckdb, pycanopy], ["pycanopy", "duckdb"], 1)
    output = tmp_path / "chart.png"

    report_utils.write_chart(results, output)

    assert output.stat().st_size > 0
