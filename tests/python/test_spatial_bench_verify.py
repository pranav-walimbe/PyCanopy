"""Focused tests for the pinned SpatialBench answer gate."""

from types import SimpleNamespace

import polars as pl
import pytest
import shapely

from bench.spatial_bench import _verify
from bench.spatial_bench.queries import q12
from pycanopy import SpatialFrame


def _write_answer(root, query_id: str, frame: pl.DataFrame) -> None:
    # Write one canonical answer under the layout used by the benchmark
    directory = root / "sf1"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / f"{query_id}.parquet")


def test_verify_uses_answer_types_and_column_positions(tmp_path):
    answer = pl.DataFrame(
        {
            "key": [1],
            "metric": [10.0],
            "avg_duration_seconds": [20.0],
        }
    )
    _write_answer(tmp_path, "q3", answer)
    result = pl.DataFrame(
        {
            "different_key_name": [1],
            "different_metric_name": [10.0 + 5e-6],
            "avg_duration": [20.0005],
        }
    )

    matched, detail = _verify.verify_output(result, "q3", 1, tmp_path)

    assert matched
    assert "ordered rows match" in detail


def test_verify_requires_canonical_row_order(tmp_path):
    _write_answer(tmp_path, "q8", pl.DataFrame({"key": [1, 2], "count": [5, 4]}))
    result = pl.DataFrame({"key": [2, 1], "count": [4, 5]})

    matched, detail = _verify.verify_output(result, "q8", 1, tmp_path)

    assert not matched
    assert "starting at row 0" in detail


def test_verify_accepts_eligible_final_row_boundary_tie(tmp_path):
    answer = pl.DataFrame({"key": list(range(100)), "metric": [float(i) for i in range(100)]})
    _write_answer(tmp_path, "q1", answer)
    result = (
        answer.with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 99).then(1000).otherwise(pl.col("key")).alias("key")
        )
        .drop("index")
    )

    matched, detail = _verify.verify_output(result, "q1", 1, tmp_path)

    assert matched
    assert "boundary tie" in detail


def test_measure_query_rejects_partial_sample_set(monkeypatch):
    utils = pytest.importorskip("bench.spatial_bench.utils")
    responses = [
        {
            "status": "ok",
            "time": 1.0,
            "kv": {},
        },
        {"status": "timeout"},
    ]
    monkeypatch.setattr(utils, "spawn_query", lambda *args: responses.pop(0))

    result = utils.measure_query(SimpleNamespace(id="q1"), "/data", 1, runs=3)

    assert result == {"status": "timeout", "run_times": [1.0]}


def test_measure_query_does_not_require_verification(monkeypatch):
    utils = pytest.importorskip("bench.spatial_bench.utils")
    monkeypatch.setattr(
        utils,
        "spawn_query",
        lambda *args: {
            "status": "ok",
            "time": 1.0,
            "kv": {},
        },
    )

    result = utils.measure_query(SimpleNamespace(id="q1"), "/data", 1, runs=1)

    assert result == {"status": "ok", "pycanopy_seconds": 1.0, "run_times": [1.0]}


def test_pinned_answers_include_csv_and_typed_parquet():
    for scale_factor in (1, 10):
        directory = _verify._ANSWERS_DIR / f"sf{scale_factor}"
        for query_number in range(1, 13):
            stem = f"q{query_number}"
            assert (directory / f"{stem}.csv").is_file()
            answer = pl.read_parquet(directory / f"{stem}.parquet")
            assert answer.width > 0


class _QueryTables:
    def __init__(self, frames: dict[str, pl.DataFrame]) -> None:
        self._frames = frames

    def parallel_fetch(self, needs) -> None:
        pass

    def table(self, name: str, columns: list[str]) -> pl.DataFrame:
        return self._frames[name].select(columns)

    def polygon_frame(self, frame: pl.DataFrame, geometry_col: str) -> SpatialFrame:
        return SpatialFrame.from_wkb_polygons(frame, geometry_col, index_mode="none")


def test_q12_reduces_knn_pairs_to_ranked_trip_averages():
    buildings = pl.DataFrame(
        {
            "b_buildingkey": list(range(5)),
            "b_boundary": [shapely.box(x, 0, x + 0.25, 0.25).wkb for x in range(5)],
        }
    )
    trips = pl.DataFrame(
        {
            "t_tripkey": [1, 2],
            "t_pickuploc": [shapely.Point(0, 0).wkb, shapely.Point(20, 0).wkb],
        }
    )

    result = q12.pycanopy(_QueryTables({"building": buildings, "trip": trips}))

    assert result.columns == ["t_tripkey", "avg_distance_to_5_nearest"]
    assert result["t_tripkey"].to_list() == [2, 1]
    assert result.height == 2
