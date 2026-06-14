"""Provision an ephemeral EC2 box, run SpatialBench on it, fetch the chart.

The whole flow is one command:

    python -m bench.spatial_bench.aws_run

It reads ``config.yaml``, launches an instance whose user-data (``bootstrap.sh``)
does everything (build, fetch data, measure PyCanopy vs SedonaDB, render the
comparison chart, upload it, self-terminate), polls S3 for the ``_SUCCESS`` marker,
and downloads the chart PNG into ``assets/``. The box is SSH-free and always
terminates: on shutdown, on a watchdog timeout, and via the ``finally`` here. AWS
credentials come from the standard boto3 chain (AWS_PROFILE / ~/.aws / SSO).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from pathlib import Path

import boto3
import yaml

_DIR = Path(__file__).parent
_ASSETS_DIR = _DIR.parent.parent / "assets"
_SSM_AL2023 = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
_PROJECT_TAG = "pycanopy-spatialbench"
_RESULT_PREFIX = "spatialbench-runs"
_POLL_SECONDS = 30


def load_config() -> dict:
    """Read config.yaml and fail fast if a required field is unset."""
    cfg = yaml.safe_load((_DIR / "config.yaml").read_text())
    for key in ("result_bucket", "instance_profile"):
        if not cfg.get(key) or str(cfg[key]).startswith("your-"):
            sys.exit(f"set '{key}' in bench/spatial_bench/config.yaml")
    return cfg


def _user_data(cfg: dict, run_id: str, index_mode: str) -> str:
    """Fill the @@NAME@@ placeholders in bootstrap.sh for this run."""
    script = (_DIR / "bootstrap.sh").read_text()
    # Non-eager modes pass a measure flag and a filename suffix so the chart does
    # not overwrite the eager sf{N}.png.
    suffix = "" if index_mode == "eager" else f"_{index_mode}"
    repl = {
        "RUN_ID": run_id,
        "REGION": cfg["region"],
        "RESULT_BUCKET": cfg["result_bucket"],
        "RESULT_PREFIX": _RESULT_PREFIX,
        "REPO_URL": cfg["repo_url"],
        "REPO_BRANCH": cfg["repo_branch"],
        "DATA_TEMPLATE": cfg["data_template"],
        "SCALE_FACTOR": str(cfg["scale_factor"]),
        "MAX_RUNTIME_MIN": str(cfg["max_runtime_min"]),
        "MEASURE_ARGS": "" if index_mode == "eager" else f"--index-{index_mode}",
        "OUT_SUFFIX": suffix,
    }
    for key, value in repl.items():
        script = script.replace(f"@@{key}@@", value)
    return script


def launch(ec2, ssm, cfg: dict, run_id: str, index_mode: str) -> str:
    """Launch the benchmark instance and return its id."""
    ami = ssm.get_parameter(Name=_SSM_AL2023)["Parameter"]["Value"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=cfg["instance_type"],
        MinCount=1,
        MaxCount=1,
        UserData=_user_data(cfg, run_id, index_mode),
        InstanceInitiatedShutdownBehavior="terminate",
        IamInstanceProfile={"Name": cfg["instance_profile"]},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": cfg["volume_gb"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": _PROJECT_TAG},
                    {"Key": "RunId", "Value": run_id},
                ],
            }
        ],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(
        f"launched {instance_id} ({cfg['instance_type']}, {index_mode}, run {run_id})", flush=True
    )
    return instance_id


def _alive(ec2, instance_id: str) -> bool:
    """Return True while the instance is still pending or running."""
    inst = ec2.describe_instances(InstanceIds=[instance_id])
    state = inst["Reservations"][0]["Instances"][0]["State"]["Name"]
    return state in ("pending", "running")


def _emit_progress(s3, cfg: dict, run_id: str, seen: int) -> int:
    """Print the box's per-testcase update lines published since the last poll.

    The box streams its log to progress.log every few seconds. We surface only the
    measure updates (running / completed / failed / mismatch / wrote chart), dropping
    build, data-copy, and byte-counter noise, and print the ones not shown yet.
    """
    key = f"{_RESULT_PREFIX}/{run_id}/progress.log"
    try:
        text = s3.get_object(Bucket=cfg["result_bucket"], Key=key)["Body"].read()
    except s3.exceptions.ClientError:
        return seen
    updates = ("running ", "completed ", "failed ", "wrote ")
    lines = [
        line.rstrip()
        for line in text.decode("utf-8", "replace").splitlines()
        if line.startswith(updates) or "output mismatch" in line
    ]
    for line in lines[seen:]:
        print(line, flush=True)
    return len(lines)


def wait_for_success(s3, ec2, cfg: dict, run_id: str, instance_id: str) -> bool:
    """Poll S3 for the _SUCCESS marker until it appears or the box dies/times out."""
    key = f"{_RESULT_PREFIX}/{run_id}/_SUCCESS"
    deadline = time.monotonic() + (cfg["max_runtime_min"] + 15) * 60
    seen = 0
    while time.monotonic() < deadline:
        seen = _emit_progress(s3, cfg, run_id, seen)
        try:
            s3.head_object(Bucket=cfg["result_bucket"], Key=key)
            _emit_progress(s3, cfg, run_id, seen)  # flush any trailing lines
            return True
        except s3.exceptions.ClientError:
            pass
        if not _alive(ec2, instance_id):
            return False  # terminated without success: a failed run
        time.sleep(_POLL_SECONDS)
    return False


def download(s3, cfg: dict, run_id: str) -> list[Path]:
    """Download this run's artifacts: the chart PNG into assets/, the log to tmp.

    Returns the downloaded local paths.
    """
    prefix = f"{_RESULT_PREFIX}/{run_id}/"
    objs = s3.list_objects_v2(Bucket=cfg["result_bucket"], Prefix=prefix).get("Contents", [])
    paths: list[Path] = []
    for obj in objs:
        name = obj["Key"].rsplit("/", 1)[-1]
        if name in ("_SUCCESS", "progress.log"):
            continue
        dest = _ASSETS_DIR if name.endswith(".png") else Path(tempfile.gettempdir())
        dest.mkdir(parents=True, exist_ok=True)
        local = dest / name
        s3.download_file(cfg["result_bucket"], obj["Key"], str(local))
        paths.append(local)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SpatialBench on an ephemeral EC2 box.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--index-eager",
        action="store_const",
        const="eager",
        dest="index_mode",
        help="Build an index whenever a kind is selected (default).",
    )
    group.add_argument(
        "--index-auto",
        action="store_const",
        const="auto",
        dest="index_mode",
        help="Cost-based: build an index only when it beats a scan. Writes sf{N}_auto.png.",
    )
    group.add_argument(
        "--index-none",
        action="store_const",
        const="none",
        dest="index_mode",
        help="Brute-force every query. Writes sf{N}_none.png.",
    )
    parser.set_defaults(index_mode="eager")
    args = parser.parse_args(argv)

    cfg = load_config()
    region = cfg["region"]
    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    ssm = boto3.client("ssm", region_name=region)

    run_id = uuid.uuid4().hex[:12]
    instance_id = launch(ec2, ssm, cfg, run_id, args.index_mode)
    try:
        ok = wait_for_success(s3, ec2, cfg, run_id, instance_id)
        paths = download(s3, cfg, run_id)
    finally:
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"terminated {instance_id}", flush=True)

    if not ok or not any(p.suffix == ".png" for p in paths):
        logs = [p for p in paths if p.suffix == ".log"]
        print(f"run failed; inspect {logs[0]}" if logs else "run failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
