"""Launch an ephemeral EC2 box that runs SpatialBench, then fetch the comparison chart.

The whole flow is one command:

    python -m bench.spatial_bench --scale-factor 1

Reads config.yaml, launches an m7i.2xlarge whose user-data (bootstrap.sh) builds
PyCanopy and measures it against the published SedonaDB / DuckDB / GeoPandas /
Spatial Polars baseline, then renders the chart and self-terminates while this
process polls S3 and downloads the PNG into assets/.
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
from botocore.exceptions import ClientError

_DIR = Path(__file__).parent
_ASSETS_DIR = _DIR.parent.parent / "assets"
_SSM_AL2023 = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
_PROJECT_TAG = "pycanopy-spatialbench"
_RESULT_PREFIX = "spatialbench-runs"
_POLL_SECONDS = 30


def load_config() -> dict:
    """Read config.yaml and fail fast if a required field is unset.

    Returns:
        The parsed config dict.
    """
    cfg = yaml.safe_load((_DIR / "config.yaml").read_text())
    for key in ("result_bucket", "instance_profile", "instance_type"):
        if not cfg.get(key) or str(cfg[key]).startswith("your-"):
            sys.exit(f"set '{key}' in bench/spatial_bench/config.yaml")
    return cfg


def _user_data(cfg: dict, run_id: str, scale_factor: int, index_mode: str, verify: bool, n: int) -> str:
    # Substitute @@NAME@@ placeholders in bootstrap.sh for this run
    script = (_DIR / "bootstrap.sh").read_text()
    suffix = "" if index_mode == "eager" else f"_{index_mode}"
    bench_flags: list[str] = [f"--n {n}"]
    if index_mode == "eager":
        bench_flags.append("--index-eager")
    if verify:
        bench_flags.append("--verify")
    repl = {
        "RUN_ID": run_id,
        "REGION": cfg["region"],
        "RESULT_BUCKET": cfg["result_bucket"],
        "RESULT_PREFIX": _RESULT_PREFIX,
        "REPO_URL": cfg["repo_url"],
        "REPO_BRANCH": cfg["repo_branch"],
        "SCALE_FACTOR": str(scale_factor),
        "MAX_RUNTIME_MIN": str(cfg["max_runtime_min"]),
        "BENCH_FLAGS": " ".join(bench_flags),
        "OUT_SUFFIX": suffix,
    }
    for key, value in repl.items():
        script = script.replace(f"@@{key}@@", value)
    return script


def _launch(ec2, ssm, cfg: dict, run_id: str, scale_factor: int, index_mode: str, verify: bool, n: int) -> str:
    # Launch the benchmark instance and return its id
    ami = ssm.get_parameter(Name=_SSM_AL2023)["Parameter"]["Value"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=cfg["instance_type"],
        MinCount=1,
        MaxCount=1,
        UserData=_user_data(cfg, run_id, scale_factor, index_mode, verify, n),
        InstanceInitiatedShutdownBehavior="terminate",
        IamInstanceProfile={"Name": cfg["instance_profile"]},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": cfg.get("volume_gb", 32),
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
        f"[ec2] launched {instance_id} ({cfg['instance_type']}, sf{scale_factor}, "
        f"{index_mode}, run {run_id})",
        flush=True,
    )
    return instance_id


def _alive(ec2, instance_id: str) -> bool:
    # True while the instance is pending or running
    try:
        inst = ec2.describe_instances(InstanceIds=[instance_id])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "InvalidInstanceID.NotFound":
            return True
        raise
    state = inst["Reservations"][0]["Instances"][0]["State"]["Name"]
    return state in ("pending", "running")


def _emit_progress(s3, cfg: dict, run_id: str, seen: int) -> int:
    # Print [testcase] and [verification] lines from the streamed progress log since the last poll
    key = f"{_RESULT_PREFIX}/{run_id}/progress.log"
    try:
        text = s3.get_object(Bucket=cfg["result_bucket"], Key=key)["Body"].read()
    except ClientError:
        return seen
    lines = [
        line.rstrip()
        for line in text.decode("utf-8", "replace").splitlines()
        if line.startswith(("[testcase]", "[verification]"))
    ]
    for line in lines[seen:]:
        print(line, flush=True)
    return len(lines)


def _wait_for_success(s3, ec2, cfg: dict, run_id: str, instance_id: str) -> bool:
    # Poll S3 for the _SUCCESS marker until it appears or the box dies or the deadline passes
    key = f"{_RESULT_PREFIX}/{run_id}/_SUCCESS"
    deadline = time.monotonic() + (cfg["max_runtime_min"] + 15) * 60
    seen = 0
    while time.monotonic() < deadline:
        seen = _emit_progress(s3, cfg, run_id, seen)
        try:
            s3.head_object(Bucket=cfg["result_bucket"], Key=key)
            _emit_progress(s3, cfg, run_id, seen)
            return True
        except ClientError:
            pass
        if not _alive(ec2, instance_id):
            return False
        time.sleep(_POLL_SECONDS)
    return False


def _download(s3, cfg: dict, run_id: str) -> list[Path]:
    # Download chart PNGs into assets/ and the log into tmp; skip _SUCCESS and progress markers
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


def _build_parser() -> argparse.ArgumentParser:
    # CLI for the EC2 launcher
    parser = argparse.ArgumentParser(description="Run SpatialBench on an ephemeral EC2 box.")
    parser.add_argument(
        "--scale-factor",
        type=int,
        choices=(1, 10),
        required=True,
        help="Scale factor to benchmark (1 or 10).",
    )
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument(
        "--index-eager",
        action="store_const",
        const="eager",
        dest="index_mode",
        help="Build an index at frame construction time (index build cost is inside the timed window).",
    )
    index_group.add_argument(
        "--index-auto",
        action="store_const",
        const="auto",
        dest="index_mode",
        help="Build the index only when the cost model estimates it beats a full scan (default).",
    )
    index_group.add_argument(
        "--index-none",
        action="store_const",
        const="none",
        dest="index_mode",
        help="Always scan; no index is built.",
    )
    parser.set_defaults(index_mode="auto")
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        metavar="N",
        help="Number of timed runs per query; reported time is the average (default 3).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the SedonaDB output check per query (only valid with --scale-factor 1 --n 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch the SpatialBench run on EC2 and return an exit code.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        The process exit code, 0 on success and 1 on failure.
    """
    args = _build_parser().parse_args(argv)
    if args.verify and not (args.scale_factor == 1 and args.n == 1):
        sys.exit("--verify is only allowed with --scale-factor 1 --n 1")
    cfg = load_config()
    region = cfg["region"]
    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    ssm = boto3.client("ssm", region_name=region)

    run_id = uuid.uuid4().hex[:12]
    instance_id = _launch(ec2, ssm, cfg, run_id, args.scale_factor, args.index_mode, args.verify, args.n)
    try:
        ok = _wait_for_success(s3, ec2, cfg, run_id, instance_id)
        paths = _download(s3, cfg, run_id)
    finally:
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"[ec2] terminated {instance_id}", flush=True)

    if not ok or not any(p.suffix == ".png" for p in paths):
        logs = [p for p in paths if p.suffix == ".log"]
        print(f"[ec2] run failed; inspect {logs[0]}" if logs else "[ec2] run failed", flush=True)
        return 1

    png = next(p for p in paths if p.suffix == ".png")
    print(f"[ec2] chart saved to {png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
