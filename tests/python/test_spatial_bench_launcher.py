"""Focused tests for the SpatialBench AWS launcher."""

import json

import pytest

_spatial_bench = pytest.importorskip("bench.spatial_bench.__main__")


def _not_found():
    # Build the EC2 error returned after a self-terminated instance disappears
    return _spatial_bench.ClientError(
        {"Error": {"Code": "InvalidInstanceID.NotFound", "Message": "gone"}},
        "DescribeInstances",
    )


@pytest.mark.parametrize(("scale_factor", "minutes"), [(1, 60), (10, 180)])
def test_user_data_sets_scale_factor_runtime(scale_factor, minutes):
    script = _spatial_bench._user_data("ami-test", "run-test", scale_factor, False, 3, "pycanopy")

    assert f'MAX_RUNTIME_MIN="{minutes}"' in script


def test_missing_instance_is_not_alive():
    class EC2:
        def describe_instances(self, **kwargs):
            raise _not_found()

    assert not _spatial_bench._alive(EC2(), "i-gone")


def test_cleanup_ignores_missing_instances():
    terminated = []

    class EC2:
        def terminate_instances(self, *, InstanceIds):
            instance_id = InstanceIds[0]
            if instance_id == "i-gone":
                raise _not_found()
            terminated.append(instance_id)

    _spatial_bench._terminate_instances(EC2(), ["i-gone", "i-live"])

    assert terminated == ["i-live"]


def test_node_chain_launches_replacement_with_only_remaining_queries(monkeypatch, tmp_path):
    continuation = tmp_path / "attempt1-geopandas-continuation.json"
    continuation.write_text(json.dumps({"engine": "geopandas", "query_ids": ["q3", "q4"]}))
    launches = []
    downloads = [[continuation], []]

    def launch(ec2, ssm, run_id, scale_factor, profile, n, engine, query_ids, variant):
        launches.append((run_id, query_ids))
        return f"i-{len(launches)}"

    monkeypatch.setattr(_spatial_bench, "_launch", launch)
    monkeypatch.setattr(_spatial_bench, "_wait_for_success", lambda *args: True)
    monkeypatch.setattr(_spatial_bench, "_download", lambda *args: downloads.pop(0))
    instance_ids = []

    success, paths = _spatial_bench._run_node_chain(
        object(),
        object(),
        object(),
        "group",
        "geopandas",
        10,
        False,
        3,
        ["q2", "q3", "q4"],
        instance_ids,
    )

    assert success
    assert launches == [
        ("group-geopandas", ["q2", "q3", "q4"]),
        ("group-geopandas-attempt2", ["q3", "q4"]),
    ]
    assert instance_ids == ["i-1", "i-2"]
    assert paths == [continuation]
