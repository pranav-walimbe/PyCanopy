"""Focused tests for the SpatialBench Engine-metrics profile path."""

from types import SimpleNamespace

import pytest

_profile = pytest.importorskip("bench.spatial_bench._profile")


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
    aggregate = _profile._aggregate_engine_metrics(
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


def test_profile_suite_writes_artifacts_then_rejects_oracle_mismatch(tmp_path, monkeypatch):
    payload = {
        "time": {
            "total": 1.0,
            "fetch": 0.1,
            "execute": 0.8,
            "materialize": 0.1,
            "non_engine": 0.7,
        },
        "mem": {
            "baseline": 100,
            "peak": 200,
            "fetch": 120,
            "execute": 200,
            "materialize": 180,
        },
        "engine": _profile._aggregate_engine_metrics([_engine(0, 1, 2, 3)]),
    }
    result = {
        "status": "ok",
        "title": "query",
        "profile": payload,
        "verify": "MISMATCH",
        "verify_detail": "rows differ",
    }
    monkeypatch.setattr(_profile, "_ASSETS_DIR", tmp_path)
    monkeypatch.setattr(_profile, "profile_query", lambda *args: result)

    with pytest.raises(RuntimeError, match="q1"):
        _profile.run_profile_suite([SimpleNamespace(id="q1")], "s3://example")

    assert (tmp_path / "profile.txt").is_file()
    assert not (tmp_path / "profile.json").exists()
