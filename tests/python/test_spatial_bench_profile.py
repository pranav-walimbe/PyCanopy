"""Focused tests for the SpatialBench Engine-metrics profile path."""

import builtins

import pytest

_profiling = pytest.importorskip("bench.spatial_bench.profiler_utils")
_results = pytest.importorskip("bench.spatial_bench.report_utils")

MIB = 1024 * 1024


def _engine(engine_id, calls, rows, elapsed):
    return {
        "engine_id": engine_id,
        "n": 10,
        "construction": {"wkb_decode_ns": 2, "statistics_ns": 3},
        "index_builds": [{"index": "r_tree", "build_count": 1, "elapsed_compute_ns": 5}],
        "operations": [
            {
                "name": "batch_contains",
                "index": "r_tree",
                "calls": calls,
                "elapsed_compute_ns": elapsed,
                "output_rows": rows,
            }
        ],
    }


def test_engine_metrics_aggregate_across_engines_and_streamed_calls():
    aggregate = _profiling._aggregate_engine_metrics(
        [_engine(0, calls=2, rows=7, elapsed=11), _engine(1, calls=3, rows=13, elapsed=17)]
    )

    assert aggregate["construction"] == {"wkb_decode_ns": 4, "statistics_ns": 6}
    assert aggregate["index_builds"] == [
        {"index": "r_tree", "build_count": 2, "elapsed_compute_ns": 10}
    ]
    assert aggregate["operations"] == [
        {
            "name": "batch_contains",
            "index": "r_tree",
            "calls": 5,
            "elapsed_compute_ns": 28,
            "output_rows": 20,
        }
    ]


def _payload():
    return {
        "time": {"total": 1.0, "non_engine": 0.7},
        "mem": {"baseline": 100, "peak": 200},
        "engine": _profiling._aggregate_engine_metrics([_engine(0, 1, 2, 3)]),
    }


def test_profile_suite_writes_artifacts_then_rejects_oracle_mismatch(tmp_path, monkeypatch):
    payload = _payload()
    result = {
        "status": "ok",
        "title": "query",
        "profile": payload,
        "verify": "MISMATCH",
        "verify_detail": "rows differ",
    }
    driver = pytest.importorskip("bench.spatial_bench.driver_utils")
    monkeypatch.setattr(driver, "_profile_query", lambda *args: result)
    monkeypatch.setattr(driver, "ASSETS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="q1"):
        driver.run_profile_suite(["q1"], "s3://example")

    assert (tmp_path / "profile-branch.txt").is_file()
    assert (tmp_path / "profile-branch.json").is_file()


def test_released_build_records_a_mismatch_instead_of_failing_the_run(tmp_path, monkeypatch):
    driver = pytest.importorskip("bench.spatial_bench.driver_utils")
    result = {
        "status": "ok",
        "title": "query",
        "profile": _payload(),
        "verify": "MISMATCH",
        "verify_detail": "rows differ",
    }
    monkeypatch.setattr(driver, "_profile_query", lambda *args: result)
    monkeypatch.setattr(driver, "ASSETS_DIR", tmp_path)

    transport = driver.run_profile_suite(["q1"], "s3://example", "release")

    assert transport == tmp_path / "profile-release.json"
    payload = _results.read_profile_transport(transport)
    assert payload["variant"] == "release"
    assert payload["results"]["q1"]["verify"] == "MISMATCH"
    assert "pycanopy build" in payload["metadata"]


def _transport(variant, wall, peak, build_ns, op_ns):
    return {
        "variant": variant,
        "metadata": {"run ID": f"run-{variant}", "pycanopy build": f"pycanopy {variant}"},
        "results": {
            "q1": {
                "status": "ok",
                "title": "query",
                "verify": "match",
                "verify_detail": "1 row",
                "profile": {
                    "time": {"total": wall, "non_engine": wall / 2},
                    "mem": {"baseline": 100 * MIB, "peak": peak * MIB},
                    "engine": {
                        "construction": {"wkb_decode_ns": 0, "statistics_ns": 0},
                        "index_builds": [
                            {
                                "index": "prepared_polygons",
                                "build_count": 1,
                                "elapsed_compute_ns": build_ns,
                            }
                        ],
                        "operations": [
                            {
                                "name": "batch_contains",
                                "index": "r_tree",
                                "calls": 1,
                                "elapsed_compute_ns": op_ns,
                                "output_rows": 1,
                            }
                        ],
                        "engines": [],
                    },
                },
            }
        },
    }


def test_comparison_report_carries_metadata_deltas_and_stage_breakdown(tmp_path):
    transports = {
        "branch": _transport("branch", wall=4.0, peak=400, build_ns=10**7, op_ns=2 * 10**9),
        "release": _transport("release", wall=5.0, peak=800, build_ns=10**9, op_ns=10**9),
    }
    out = tmp_path / "profile.txt"
    _results.write_profile_comparison(transports, out)
    text = out.read_text()

    assert "Run metadata" in text
    assert "pycanopy branch" in text and "pycanopy release" in text
    assert "branch run ID" in text and "released run ID" in text
    # Wall 4 against 5, peak 400 against 800
    assert "-20.0%" in text
    assert "-50.0%" in text
    # The build sheds 0.99s while the operation gains 1.0s
    assert "build prepared_polygons" in text
    assert "batch_contains (r_tree)" in text
    assert "total engine compute" in text
    assert "PASS / PASS" in text


def test_comparison_report_falls_back_to_one_build(tmp_path):
    transports = {"branch": _transport("branch", wall=4.0, peak=400, build_ns=1, op_ns=1)}
    out = tmp_path / "profile.txt"
    _results.write_profile_comparison(transports, out)
    text = out.read_text()

    assert "Run metadata" in text
    assert "Wall time and peak memory" not in text
    assert "q1  query" in text


def test_comparison_report_flags_a_query_only_one_build_ran(tmp_path):
    transports = {
        "branch": _transport("branch", wall=4.0, peak=400, build_ns=1, op_ns=1),
        "release": _transport("release", wall=5.0, peak=800, build_ns=1, op_ns=1),
    }
    transports["release"]["results"]["q1"] = {
        "status": "error",
        "title": "query",
        "error": "AttributeError: no such method",
    }
    out = tmp_path / "profile.txt"
    _results.write_profile_comparison(transports, out)

    assert "no profile from the released build" in out.read_text()


def test_profile_runs_when_the_installed_wheel_has_no_metrics_hook(monkeypatch):
    runner = pytest.importorskip("bench.spatial_bench.run_query")
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pycanopy.engine" and "_capture_engine_metrics" in (fromlist or ()):
            raise ImportError("cannot import name '_capture_engine_metrics'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with runner._capture_metrics() as capture:
        assert capture is None


def test_payload_marks_a_build_that_reports_no_metrics():
    payload = _profiling.profile_payload(_profiling.StageProfiler(), 1.0, [], metrics=False)

    assert payload["metrics"] is False
    assert payload["time"]["non_engine"] == 1.0
    assert _results._stage_times({"status": "ok", "profile": payload}) == {}


def test_stage_table_drops_the_released_column_when_it_has_no_metrics(tmp_path):
    branch = _transport("branch", wall=4.0, peak=400, build_ns=10**7, op_ns=2 * 10**9)
    release = _transport("release", wall=5.0, peak=800, build_ns=0, op_ns=0)
    release["results"]["q1"]["profile"]["metrics"] = False
    release["results"]["q1"]["profile"]["engine"]["index_builds"] = []
    release["results"]["q1"]["profile"]["engine"]["operations"] = []
    out = tmp_path / "profile.txt"
    _results.write_profile_comparison({"branch": branch, "release": release}, out)
    text = out.read_text()

    # Wall and peak still compare, so the released build stays in the top table
    assert "-20.0%" in text and "-50.0%" in text
    assert "reports no engine metrics" in text
    assert "build prepared_polygons" in text
