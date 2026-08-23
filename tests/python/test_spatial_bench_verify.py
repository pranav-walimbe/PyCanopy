"""Focused tests for the pinned SpatialBench answer gate."""

import polars as pl
import pytest
import shapely

from bench.spatial_bench import profiler_utils as _verify
from bench.spatial_bench.queries.pycanopy import q04, q05, q12
from pycanopy import executor


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


def test_measure_rejects_partial_sample_set(monkeypatch):
    driver = pytest.importorskip("bench.spatial_bench.driver_utils")
    responses = [{"status": "ok", "time": 1.0, "values": {}}, {"status": "timeout"}]
    monkeypatch.setattr(driver, "_spawn", lambda *args, **kwargs: responses.pop(0))

    result = driver._measure("pycanopy", "q1", "/data", runs=3)

    assert result == {"status": "timeout", "run_times": [1.0]}


def test_measure_averages_every_completed_run(monkeypatch):
    driver = pytest.importorskip("bench.spatial_bench.driver_utils")
    monkeypatch.setattr(
        driver, "_spawn", lambda *args, **kwargs: {"status": "ok", "time": 1.0, "values": {}}
    )

    result = driver._measure("pycanopy", "q1", "/data", runs=1)

    assert result == {"status": "ok", "seconds": 1.0, "run_times": [1.0]}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--scale-factor", "1", "--n", "0"], "--n must be at least 1"),
        (["--scale-factor", "1", "--query", "q99"], "unknown query IDs: q99"),
        (["--scale-factor", "10", "--profile"], "--profile runs the SF1 workload"),
        (
            ["--scale-factor", "1", "--profile", "--engine", "duckdb"],
            "--profile measures pycanopy",
        ),
    ],
)
def test_suite_rejects_invalid_run_selection(args, message):
    driver = pytest.importorskip("bench.spatial_bench.driver_utils")
    with pytest.raises(SystemExit, match=message):
        driver.main(args)


def test_results_txt_records_public_metadata_without_source_path(tmp_path):
    driver = pytest.importorskip("bench.spatial_bench.driver_utils")
    results = pytest.importorskip("bench.spatial_bench.report_utils")

    metadata = driver.collect_metadata("pycanopy", "s3://private-bucket/data", 1, 3)
    path = tmp_path / "pycanopy-results.tsv"
    results.write_transport(
        path,
        "pycanopy",
        "1.0",
        metadata,
        {"q1": {"status": "ok", "seconds": 1.0, "run_times": [1.0]}},
    )
    combined = results.combine_transports([path], ["pycanopy"], 1)
    output = tmp_path / "results.txt"

    results.write_results_txt(combined, output)
    text = output.read_text()

    assert "Run metadata" in text
    assert "engines: PyCanopy 1.0" in text
    assert "source: custom" in text
    assert "s3://private-bucket" not in text


def test_pinned_answers_include_csv_and_typed_parquet():
    for scale_factor in (1, 10):
        directory = _verify.ANSWERS_DIR / f"sf{scale_factor}"
        for query_number in range(1, 13):
            stem = f"q{query_number}"
            assert (directory / f"{stem}.csv").is_file()
            answer = pl.read_parquet(directory / f"{stem}.parquet")
            assert answer.width > 0


def _data_paths(tmp_path, frames: dict[str, pl.DataFrame]) -> dict[str, str]:
    # Write the frames as parquet and return the table -> glob map the runner hands to a query
    paths = {}
    for name, frame in frames.items():
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(directory / "0.parquet")
        paths[name] = f"{directory}/**/*.parquet"
    return paths


def test_q4_fetches_geometry_after_selecting_top_trips(monkeypatch, tmp_path):
    trips = pl.DataFrame(
        {
            "t_tripkey": list(range(1001)),
            "t_tip": list(range(1001)),
            "t_pickuploc": [b"not selected"] + [shapely.Point(0.5, 0.5).wkb for _ in range(1000)],
        }
    )
    zones = pl.DataFrame(
        {
            "z_zonekey": [1],
            "z_name": ["zone"],
            "z_boundary": [shapely.box(0, 0, 1, 1).wkb],
        }
    )
    paths = _data_paths(tmp_path, {"trip": trips, "zone": zones})
    collected: list[tuple[list[str], int]] = []
    base_collect, base_collect_all = pl.LazyFrame.collect, pl.collect_all

    def record(frame: pl.DataFrame) -> pl.DataFrame:
        collected.append((frame.columns, frame.height))
        return frame

    monkeypatch.setattr(
        pl.LazyFrame, "collect", lambda self, *a, **k: record(base_collect(self, *a, **k))
    )
    monkeypatch.setattr(
        pl,
        "collect_all",
        lambda frames, *a, **k: [record(f) for f in base_collect_all(frames, *a, **k)],
    )

    result = q04.pycanopy(paths)

    assert result["trip_count"].to_list() == [1000]
    # The geometry column must only ever materialize for the top trips, never for all 1001.
    assert [height for columns, height in collected if "t_pickuploc" in columns] == [q04.TOP_N]


def test_q5_groups_before_customer_lookup_without_changing_result(tmp_path):
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

    result = q05.pycanopy(_data_paths(tmp_path, {"trip": trips, "customer": customers}))

    assert result["c_custkey"].to_list() == [2, 1]
    assert result["customer_name"].to_list() == ["bob", "alice"]
    assert result["dropoff_count"].to_list() == [6, 6]
    assert result["monthly_travel_hull_area"].to_list() == [4.0, 1.0]


def test_q12_reduces_knn_pairs_to_ranked_trip_averages(monkeypatch, tmp_path):
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

    result = q12.pycanopy(_data_paths(tmp_path, {"building": buildings, "trip": trips}))

    assert result.columns == ["t_tripkey", "avg_distance_to_5_nearest"]
    assert result["t_tripkey"].to_list() == [2, 1]
    assert result.height == 2
