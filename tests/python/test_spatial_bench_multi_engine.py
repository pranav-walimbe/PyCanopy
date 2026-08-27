"""Focused tests for multi-engine SpatialBench orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.spatial_bench import driver_utils, report_utils
from bench.spatial_bench.config import (
    DATASET_VERSION,
    DEFAULT_RUNS,
    INSTANCE_TYPE,
    MAX_RUNTIME_MINUTES_BY_SCALE_FACTOR,
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
    assert MAX_RUNTIME_MINUTES_BY_SCALE_FACTOR == {1: 60, 10: 180}
    assert REGION == "us-west-2"
    assert INSTANCE_TYPE == "m7i.2xlarge"
    assert PUBLIC_DATA_TEMPLATE.startswith("s3://")
    # The dataset build has to be identifiable from the path
    # The committed answers match only one of them
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


def test_shared_runner_classifies_sigkill_as_oom(monkeypatch):
    monkeypatch.setattr(
        driver_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=-9),
    )

    result = driver_utils._spawn("geopandas", "q3", "s3://data")

    assert result == {"status": "oom", "error": "OOM: subprocess killed by SIGKILL"}


def test_timing_suite_stops_after_oom_and_writes_remaining_queries(monkeypatch, tmp_path):
    responses = [
        {"status": "ok", "seconds": 1.0, "run_times": [1.0]},
        {"status": "oom", "error": "OOM: subprocess killed by SIGKILL", "run_times": []},
    ]
    monkeypatch.setattr(driver_utils, "ASSETS_DIR", tmp_path)
    monkeypatch.setattr(driver_utils, "version", lambda package: "1.0")
    monkeypatch.setattr(driver_utils, "_measure", lambda *args: responses.pop(0))

    transport = driver_utils.run_timing_suite("geopandas", ["q1", "q2", "q3"], "s3://data", 10, 3)

    results = report_utils.read_transport(transport)
    continuation = json.loads((tmp_path / "geopandas-continuation.json").read_text())
    assert list(results["queries"]) == ["q1", "q2"]
    assert results["queries"]["q2"]["status"] == "oom"
    assert continuation == {"engine": "geopandas", "query_ids": ["q3"]}


def test_timing_suite_does_not_replace_after_final_query_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(driver_utils, "ASSETS_DIR", tmp_path)
    monkeypatch.setattr(driver_utils, "version", lambda package: "1.0")
    monkeypatch.setattr(
        driver_utils,
        "_measure",
        lambda *args: {
            "status": "oom",
            "error": "OOM: subprocess killed by SIGKILL",
            "run_times": [],
        },
    )

    driver_utils.run_timing_suite("geopandas", ["q12"], "s3://data", 10, 3)

    assert not (tmp_path / "geopandas-continuation.json").exists()


def test_combine_transports_merges_replacement_attempts(tmp_path):
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    report_utils.write_transport(
        first,
        "geopandas",
        "1.0",
        {"run ID": "run-1"},
        {"q1": {"status": "oom", "error": "OOM", "run_times": []}},
    )
    report_utils.write_transport(
        second,
        "geopandas",
        "1.0",
        {"run ID": "run-2"},
        {"q2": {"status": "ok", "seconds": 2.0, "run_times": [2.0]}},
    )

    results = report_utils.combine_transports([first, second], ["geopandas"], 10)

    engine = results["engines"]["geopandas"]
    assert list(engine["queries"]) == ["q1", "q2"]
    assert engine["queries"]["q1"]["status"] == "oom"
    assert engine["metadata"]["run ID"] == "run-1, run-2"


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
