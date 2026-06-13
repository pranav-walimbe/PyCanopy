"""Provision an ephemeral EC2 box, run SpatialBench on it, render the report.

The whole flow is one command:

    python -m bench.spatial_bench.aws_run

It reads ``config.yaml``, launches an instance whose user-data (``bootstrap.sh``)
does everything (build, fetch data, measure, upload results, self-terminate),
polls S3 for the ``_SUCCESS`` marker, downloads the result JSON, and renders the
comparison report. The box is SSH-free and always terminates: on shutdown, on a
watchdog timeout, and via the ``finally`` here. AWS credentials come from the
standard boto3 chain (AWS_PROFILE / ~/.aws / SSO).
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import boto3
import yaml

from bench.spatial_bench import report

_DIR = Path(__file__).parent
_SSM_AL2023 = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
_PROJECT_TAG = "pycanopy-spatialbench"
_RESULT_PREFIX = "spatialbench-runs"
_POLL_SECONDS = 30


def log(msg: str) -> None:
    """Print one driver line with a consistent, greppable prefix."""
    print(f"[aws_run] {msg}", flush=True)


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
    # Non-eager modes pass a measure flag and a filename suffix so the result does
    # not overwrite the eager sf{N}.json.
    suffix = "" if index_mode == "eager" else f"-{index_mode}"
    repl = {
        "RUN_ID": run_id,
        "REGION": cfg["region"],
        "RESULT_BUCKET": cfg["result_bucket"],
        "RESULT_PREFIX": _RESULT_PREFIX,
        "REPO_URL": cfg["repo_url"],
        "REPO_BRANCH": cfg["repo_branch"],
        "DATA_TEMPLATE": cfg["data_template"],
        "SCALE_FACTORS": " ".join(str(s) for s in cfg["scale_factors"]),
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
    log(f"launched {instance_id} ({cfg['instance_type']}, AMI {ami})")
    return instance_id


def _alive(ec2, instance_id: str) -> bool:
    """Return True while the instance is still pending or running."""
    inst = ec2.describe_instances(InstanceIds=[instance_id])
    state = inst["Reservations"][0]["Instances"][0]["State"]["Name"]
    return state in ("pending", "running")


def _emit_progress(s3, cfg: dict, run_id: str, seen: int) -> int:
    """Print step-progress lines the box has published since the last poll.

    The box uploads its bootstrap log to progress.log every few seconds. We keep
    only the meaningful lines (bootstrap steps and per-query measure output),
    dropping the s3-sync byte-counter spam, and print the ones not shown yet.

    Args:
        s3: boto3 S3 client.
        cfg: Loaded config (for the result bucket).
        run_id: This run's id (S3 key prefix).
        seen: Count of progress lines already printed.

    Returns:
        Updated count of progress lines printed so far.
    """
    key = f"{_RESULT_PREFIX}/{run_id}/progress.log"
    try:
        text = s3.get_object(Bucket=cfg["result_bucket"], Key=key)["Body"].read()
    except s3.exceptions.ClientError:
        return seen
    lines = [
        line.rstrip()
        for line in text.decode("utf-8", "replace").splitlines()
        if "[bootstrap]" in line or "running q" in line or "status=" in line
    ]
    for line in lines[seen:]:
        log(f"box: {line}")
    return len(lines)


def wait_for_success(s3, ec2, cfg: dict, run_id: str, instance_id: str) -> bool:
    """Poll S3 for the _SUCCESS marker until it appears or the box dies/times out."""
    key = f"{_RESULT_PREFIX}/{run_id}/_SUCCESS"
    deadline = time.monotonic() + (cfg["max_runtime_min"] + 15) * 60
    seen = 0
    while time.monotonic() < deadline:
        prev = seen
        seen = _emit_progress(s3, cfg, run_id, seen)
        try:
            s3.head_object(Bucket=cfg["result_bucket"], Key=key)
            _emit_progress(s3, cfg, run_id, seen)  # flush any trailing lines
            return True
        except s3.exceptions.ClientError:
            pass
        if not _alive(ec2, instance_id):
            return False  # terminated without success: a failed run
        if seen == prev:
            log("waiting for results ...")
        time.sleep(_POLL_SECONDS)
    return False


def download(s3, cfg: dict, run_id: str, dest: Path) -> list[Path]:
    """Download the result JSON(s) and bootstrap log this run produced."""
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{_RESULT_PREFIX}/{run_id}/"
    objs = s3.list_objects_v2(Bucket=cfg["result_bucket"], Prefix=prefix).get("Contents", [])
    jsons: list[Path] = []
    for obj in objs:
        name = obj["Key"].rsplit("/", 1)[-1]
        if name == "_SUCCESS":
            continue
        local = dest / name
        s3.download_file(cfg["result_bucket"], obj["Key"], str(local))
        log(f"downloaded {name}")
        if name.endswith(".json"):
            jsons.append(local)
    return jsons


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
        help="Cost-based: build an index only when it beats a scan. Writes sf{N}-auto.json.",
    )
    group.add_argument(
        "--index-none",
        action="store_const",
        const="none",
        dest="index_mode",
        help="Brute-force every query. Writes sf{N}-none.json.",
    )
    parser.set_defaults(index_mode="eager")
    args = parser.parse_args(argv)

    cfg = load_config()
    region = cfg["region"]
    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    ssm = boto3.client("ssm", region_name=region)

    run_id = uuid.uuid4().hex[:12]
    tag = "" if args.index_mode == "eager" else f" [{args.index_mode}]"
    log(f"run {run_id}{tag}: {cfg['instance_type']} in {region}, scale {cfg['scale_factors']}")
    instance_id = launch(ec2, ssm, cfg, run_id, args.index_mode)
    try:
        ok = wait_for_success(s3, ec2, cfg, run_id, instance_id)
        jsons = download(s3, cfg, run_id, _DIR / "results")
    finally:
        ec2.terminate_instances(InstanceIds=[instance_id])
        log(f"terminated {instance_id}")

    if not ok or not jsons:
        log("run failed; inspect the downloaded bootstrap.log")
        return 1
    for path in sorted(jsons):
        results = report.load_results(path)
        print(report.build_markdown_table(results))
        for chart in report.build_charts(results):
            log(f"wrote chart: {chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
