"""Launch an ephemeral EC2 box that runs SpatialBench, then fetch the comparison chart.

The whole flow is one command:

    python -m bench.spatial_bench --scale 1 --index-auto

It reads config.yaml, launches an instance whose user-data (bootstrap.sh) builds
PyCanopy and measures it against the published baseline, then renders the chart and
self-terminates while this process polls S3 and downloads the PNG into assets/.

Benchmarking only runs on the matched instance, never locally, so the numbers stay
comparable to the baseline. The hidden --on-box flags let bootstrap.sh re-enter this
module on the box to run the measurement, and are not part of the user interface.
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

from bench.spatial_bench import queries as query_registry
from bench.spatial_bench.utils import run_suite

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
    for key in ("result_bucket", "instance_profile"):
        if not cfg.get(key) or str(cfg[key]).startswith("your-"):
            sys.exit(f"set '{key}' in bench/spatial_bench/config.yaml")
    return cfg


def _user_data(cfg: dict, run_id: str, scale: int, index_mode: str, no_verify: bool) -> str:
    # Fill the @@NAME@@ placeholders in bootstrap.sh for this run. Non-eager modes pass a
    # measure flag and a filename suffix so the chart does not overwrite the eager one.
    script = (_DIR / "bootstrap.sh").read_text()
    suffix = "" if index_mode == "eager" else f"_{index_mode}"
    measure_args = [] if index_mode == "eager" else [f"--index-{index_mode}"]
    if no_verify:
        measure_args.append("--no-verify")
    repl = {
        "RUN_ID": run_id,
        "REGION": cfg["region"],
        "RESULT_BUCKET": cfg["result_bucket"],
        "RESULT_PREFIX": _RESULT_PREFIX,
        "REPO_URL": cfg["repo_url"],
        "REPO_BRANCH": cfg["repo_branch"],
        "DATA_TEMPLATE": cfg["data_template"],
        "SCALE_FACTOR": str(scale),
        "MAX_RUNTIME_MIN": str(cfg["max_runtime_min"]),
        "MEASURE_ARGS": " ".join(measure_args),
        "OUT_SUFFIX": suffix,
    }
    for key, value in repl.items():
        script = script.replace(f"@@{key}@@", value)
    return script


def launch(ec2, ssm, cfg: dict, run_id: str, scale: int, index_mode: str, no_verify: bool) -> str:
    """Launch the benchmark instance and return its id.

    Args:
        ec2: A boto3 EC2 client.
        ssm: A boto3 SSM client (resolves the latest AL2023 AMI).
        cfg: The parsed config dict.
        run_id: Unique id tagged on the instance and used in the S3 prefix.
        scale: Scale factor (1 or 10) the box measures.
        index_mode: Index build policy passed to the box ("eager" / "auto" / "none").
        no_verify: Skip the SedonaDB oracle on the box when True.

    Returns:
        The launched instance id.
    """
    ami = ssm.get_parameter(Name=_SSM_AL2023)["Parameter"]["Value"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=cfg["instance_type"],
        MinCount=1,
        MaxCount=1,
        UserData=_user_data(cfg, run_id, scale, index_mode, no_verify),
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
        f"[ec2] launched {instance_id} ({cfg['instance_type']}, sf{scale}, "
        f"{index_mode}, run {run_id})",
        flush=True,
    )
    return instance_id


def _alive(ec2, instance_id: str) -> bool:
    # True while the instance is pending or running. describe_instances is eventually
    # consistent right after launch, so a transient NotFound counts as still-pending.
    try:
        inst = ec2.describe_instances(InstanceIds=[instance_id])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "InvalidInstanceID.NotFound":
            return True
        raise
    state = inst["Reservations"][0]["Instances"][0]["State"]["Name"]
    return state in ("pending", "running")


def _emit_progress(s3, cfg: dict, run_id: str, seen: int) -> int:
    # Print the box's [testcase]/[verification] lines published since the last poll,
    # dropping the build, data-copy, and byte-counter noise from the streamed progress.log.
    key = f"{_RESULT_PREFIX}/{run_id}/progress.log"
    try:
        text = s3.get_object(Bucket=cfg["result_bucket"], Key=key)["Body"].read()
    except s3.exceptions.ClientError:
        return seen
    lines = [
        line.rstrip()
        for line in text.decode("utf-8", "replace").splitlines()
        if line.startswith(("[testcase]", "[verification]"))
    ]
    for line in lines[seen:]:
        print(line, flush=True)
    return len(lines)


def wait_for_success(s3, ec2, cfg: dict, run_id: str, instance_id: str) -> bool:
    """Poll S3 for the _SUCCESS marker until it appears or the box dies/times out.

    Args:
        s3: A boto3 S3 client.
        ec2: A boto3 EC2 client.
        cfg: The parsed config dict.
        run_id: This run's unique id.
        instance_id: The instance to watch for early death.

    Returns:
        True if _SUCCESS appeared, False if the box died or the deadline passed.
    """
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

    Args:
        s3: A boto3 S3 client.
        cfg: The parsed config dict.
        run_id: This run's unique id.

    Returns:
        The downloaded local paths.
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


def _build_parser() -> argparse.ArgumentParser:
    # Launch flags are user-facing. The --on-box group is filled by bootstrap.sh on the
    # box to re-enter this module and run the measurement, and is hidden from --help.
    parser = argparse.ArgumentParser(description="Run SpatialBench on an ephemeral EC2 box.")
    parser.add_argument(
        "--scale", type=int, choices=(1, 10), help="Scale factor to benchmark (1 or 10)."
    )
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
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the SedonaDB oracle. Avoids the per-query verification memory load.",
    )
    parser.add_argument("--on-box", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", help=argparse.SUPPRESS)
    parser.add_argument("--scale-factor", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--queries", nargs="*", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # On-box mode: bootstrap.sh re-enters here to measure on the matched instance
    if args.on_box:
        run_suite(
            query_registry.select(args.queries),
            args.data_dir,
            args.scale_factor,
            args.index_mode,
            args.output,
            verify=not args.no_verify,
        )
        return 0

    if args.scale is None:
        parser.error("--scale is required (1 or 10)")

    cfg = load_config()
    region = cfg["region"]
    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    ssm = boto3.client("ssm", region_name=region)

    run_id = uuid.uuid4().hex[:12]
    instance_id = launch(ec2, ssm, cfg, run_id, args.scale, args.index_mode, args.no_verify)
    try:
        ok = wait_for_success(s3, ec2, cfg, run_id, instance_id)
        paths = download(s3, cfg, run_id)
    finally:
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"[ec2] terminated {instance_id}", flush=True)

    if not ok or not any(p.suffix == ".png" for p in paths):
        logs = [p for p in paths if p.suffix == ".log"]
        print(f"[ec2] run failed; inspect {logs[0]}" if logs else "[ec2] run failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
