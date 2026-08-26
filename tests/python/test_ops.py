"""Tests for the ops calibration harness."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    return importlib.import_module("bench.ops.__main__")


def test_fit_factor_fits_through_origin(ops):
    observations = [ops._Observation(10.0, 20.0), ops._Observation(20.0, 40.0)]
    assert ops._fit_factor(observations) == pytest.approx(2.0)


def test_write_profile_writes_only_named_factors(ops, tmp_path):
    factors = {name: float(i + 1) for i, name in enumerate(ops._ALL_FIELDS)}
    output = tmp_path / "default.json"

    ops._write_profile(output, factors)

    assert json.loads(output.read_text()) == {"cost_factors": factors}


def test_dry_run_exercises_balanced_matrix_without_writing_profile(ops, tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "_POINT_SIZES", [1_000, 2_000, 3_000])
    monkeypatch.setattr(ops, "_POLY_SIZES", [600, 700, 800])
    monkeypatch.setattr(ops, "_Q", 4)
    monkeypatch.setattr(ops, "_EXTRA_K", [1, 10])
    monkeypatch.setattr(ops, "_ORIENTATION_RATIOS", [0.5, 4.0])
    monkeypatch.setattr(ops, "_ORIENTATION_DENSITIES", [0.0001])
    monkeypatch.setattr(ops, "_ORIENTATION_DENSE_CASES", [])
    monkeypatch.setattr(ops, "_REPORT_PATH", tmp_path / "ops.txt")
    monkeypatch.setattr(ops, "_PROFILE_PATH", tmp_path / "default.json")

    fits = ops.run(1, 1, 42, dry_run=True)

    assert set(fits) == set(ops._ALL_FIELDS)
    assert all(value > 0 for value in fits.values())
    assert (tmp_path / "ops.txt").exists()
    assert "Point-to-polygon orientation" in (tmp_path / "ops.txt").read_text()
    assert not (tmp_path / "default.json").exists()
