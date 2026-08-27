"""Focused tests for the SpatialBench AWS launcher."""

import pytest

_spatial_bench = pytest.importorskip("bench.spatial_bench.__main__")


@pytest.mark.parametrize(("scale_factor", "minutes"), [(1, 60), (10, 180)])
def test_user_data_sets_scale_factor_runtime(scale_factor, minutes):
    script = _spatial_bench._user_data("ami-test", "run-test", scale_factor, False, 3, "pycanopy")

    assert f'MAX_RUNTIME_MIN="{minutes}"' in script
