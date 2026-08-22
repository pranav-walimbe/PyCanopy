"""Focused tests for the pinned SpatialBench answer gate."""

from types import SimpleNamespace

import polars as pl
import pytest
import shapely

from bench.spatial_bench import _verify
from bench.spatial_bench.queries import q05, q12
from pycanopy import SpatialFrame, executor


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


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--scale-factor", "1", "--n", "0"], "--n must be at least 1"),
        (["--scale-factor", "1", "--query", "q99"], "unknown query IDs: q99"),
    ],
)
def test_onbox_rejects_invalid_run_selection(args, message):
    _onbox = pytest.importorskip("bench.spatial_bench._onbox")
    with pytest.raises(SystemExit, match=message):
        _onbox.main(args)


def test_results_txt_records_public_metadata_without_source_path(tmp_path):
    utils = pytest.importorskip("bench.spatial_bench.utils")
    metadata = utils.collect_run_metadata("s3://private-bucket/data", ["q1"], 1, "auto", 3)
    results = {
        "scale_factor": 1,
        "index_mode": "auto",
        "metadata": metadata,
        "queries": {"q1": {"status": "ok", "pycanopy_seconds": 1.0, "run_times": [1.0]}},
    }
    output = tmp_path / "results.txt"

    utils.write_results_txt(results, output)
    text = output.read_text()

    assert "Run metadata" in text
    assert "engines: PyCanopy " in text
    assert "source: custom" in text
    assert "s3://private-bucket" not in text


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

    def parallel_fetch(self, needs) -> dict[str, pl.DataFrame]:
        return {name: self._frames[name].select(columns) for name, columns in needs.items()}

    def table(self, name: str, columns: list[str]) -> pl.DataFrame:
        return self._frames[name].select(columns)

    def polygon_frame(self, frame: pl.DataFrame, geometry_col: str) -> SpatialFrame:
        return SpatialFrame.from_wkb_polygons(frame, geometry_col, index_mode="none")


def test_q5_groups_before_customer_lookup_without_changing_result():
    timestamp = pl.datetime_range(
        pl.datetime(2000, 1, 1),
        pl.datetime(2000, 1, 12),
        interval="1d",
        eager=True,
    )
    trips = pl.DataFrame(
        {
            "t_custkey": [1] * 6 + [2] * 6,
            "t_dropoffloc": [
                shapely.Point(x, y).wkb
                for x, y in [
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (0.5, 0.5),
                    (0.25, 0.25),
                    (0, 0),
                    (2, 0),
                    (2, 2),
                    (0, 2),
                    (1, 1),
                    (0.5, 0.5),
                ]
            ],
            "t_pickuptime": timestamp,
        }
    )
    customers = pl.DataFrame({"c_custkey": [1, 2], "c_name": ["alice", "bob"]})

    result = q05.pycanopy(_QueryTables({"trip": trips, "customer": customers}))

    assert result["c_custkey"].to_list() == [2, 1]
    assert result["customer_name"].to_list() == ["bob", "alice"]
    assert result["dropoff_count"].to_list() == [6, 6]
    assert result["monthly_travel_hull_area"].to_list() == [4.0, 1.0]


def test_q12_reduces_knn_pairs_to_ranked_trip_averages(monkeypatch):
    monkeypatch.setattr(executor, "MORSEL_ROWS", 1)
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
