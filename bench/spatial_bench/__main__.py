"""Run SpatialBench on isolated EC2 nodes and combine their results."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from bench.spatial_bench.config import (
    AMI_PARAMETER,
    ASSETS_DIR,
    DEFAULT_RUNS,
    ENGINE_IDS,
    INSTANCE_PROFILE,
    INSTANCE_TYPE,
    MAX_RUNTIME_MINUTES,
    POLL_SECONDS,
    PROJECT_TAG,
    PUBLIC_DATA_TEMPLATE,
    QUERY_IDS,
    REGION,
    REPOSITORY_BRANCH,
    REPOSITORY_URL,
    RESULT_BUCKET,
    RESULT_KEY_PREFIX,
    SUPPORTED_SCALE_FACTORS,
    VOLUME_GB,
    VOLUME_IOPS,
    VOLUME_THROUGHPUT_MBPS,
)
from bench.spatial_bench.results_utils import (
    combine_transports,
    write_chart,
    write_results_txt,
)

_DIR = Path(__file__).parent


def _user_data(
    ami: str,
    run_id: str,
    scale_factor: int,
    index_mode: str,
    profile: bool,
    n: int,
    engine: str,
    query_ids: list[str] | None = None,
) -> str:
    # Substitute @@NAME@@ placeholders in bootstrap.sh for this run
    script = (_DIR / "bootstrap.sh").read_text()
    if profile:
        bench_flags: list[str] = ["--profile"]
    else:
        bench_flags = [f"--n {n}", f"--index-mode {index_mode}"]
    if query_ids:
        bench_flags.append("--query " + " ".join(query_ids))
    repl = {
        "RUN_ID": run_id,
        "REGION": REGION,
        "AMI_ID": ami,
        "INSTANCE_TYPE": INSTANCE_TYPE,
        "VOLUME_GB": str(VOLUME_GB),
        "VOLUME_IOPS": str(VOLUME_IOPS),
        "VOLUME_THROUGHPUT_MBPS": str(VOLUME_THROUGHPUT_MBPS),
        "RESULT_BUCKET": RESULT_BUCKET,
        "RESULT_PREFIX": RESULT_KEY_PREFIX,
        "REPO_URL": REPOSITORY_URL,
        "REPO_BRANCH": REPOSITORY_BRANCH,
        "SCALE_FACTOR": str(scale_factor),
        "DATA_ROOT": PUBLIC_DATA_TEMPLATE.format(scale_factor=scale_factor),
        "MAX_RUNTIME_MIN": str(MAX_RUNTIME_MINUTES),
        "PROFILE_MODE": "1" if profile else "0",
        "ENGINE": engine,
        "BENCH_FLAGS": " ".join(bench_flags),
    }
    for key, value in repl.items():
        script = script.replace(f"@@{key}@@", value)
    return script


def _launch(
    ec2,
    ssm,
    run_id: str,
    scale_factor: int,
    index_mode: str,
    profile: bool,
    n: int,
    engine: str,
    query_ids: list[str] | None = None,
) -> str:
    # Launch the benchmark instance and return its id
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        UserData=_user_data(ami, run_id, scale_factor, index_mode, profile, n, engine, query_ids),
        InstanceInitiatedShutdownBehavior="terminate",
        IamInstanceProfile={"Name": INSTANCE_PROFILE},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": VOLUME_GB,
                    "VolumeType": "gp3",
                    "Iops": VOLUME_IOPS,
                    "Throughput": VOLUME_THROUGHPUT_MBPS,
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": PROJECT_TAG},
                    {"Key": "RunId", "Value": run_id},
                ],
            }
        ],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(
        f"[ec2] launched {instance_id} ({INSTANCE_TYPE}, sf{scale_factor}, "
        f"{engine}, {index_mode}, run {run_id})",
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


def _emit_progress(s3, run_id: str, seen: int, engine: str) -> int:
    # Print [testcase] and [verification] lines from the streamed progress log since the last poll
    key = f"{RESULT_KEY_PREFIX}/{run_id}/progress.log"
    try:
        text = s3.get_object(Bucket=RESULT_BUCKET, Key=key)["Body"].read()
    except ClientError:
        return seen
    lines = [
        line.rstrip()
        for line in text.decode("utf-8", "replace").splitlines()
        if line.startswith(("[testcase]", "[verification]", "[timing]"))
    ]
    for line in lines[seen:]:
        print(f"[{engine}] {line}", flush=True)
    return len(lines)


def _wait_for_success(s3, ec2, run_id: str, instance_id: str, engine: str = "pycanopy") -> bool:
    # Poll S3 for the _SUCCESS marker until it appears or the box dies or the deadline passes
    key = f"{RESULT_KEY_PREFIX}/{run_id}/_SUCCESS"
    deadline = time.monotonic() + (MAX_RUNTIME_MINUTES + 15) * 60
    seen = 0
    while time.monotonic() < deadline:
        seen = _emit_progress(s3, run_id, seen, engine)
        try:
            s3.head_object(Bucket=RESULT_BUCKET, Key=key)
            _emit_progress(s3, run_id, seen, engine)
            return True
        except ClientError:
            pass
        if not _alive(ec2, instance_id):
            return False
        time.sleep(POLL_SECONDS)
    return False


def _download(s3, run_id: str) -> list[Path]:
    # Download benchmark/profile artifacts into assets/ and the log into tmp, skipping markers
    prefix = f"{RESULT_KEY_PREFIX}/{run_id}/"
    objs = s3.list_objects_v2(Bucket=RESULT_BUCKET, Prefix=prefix).get("Contents", [])
    paths: list[Path] = []
    for obj in objs:
        name = obj["Key"].rsplit("/", 1)[-1]
        if name in ("_SUCCESS", "progress.log"):
            continue
        keep = name.endswith((".png", ".txt"))
        dest = ASSETS_DIR if keep else Path(tempfile.gettempdir())
        dest.mkdir(parents=True, exist_ok=True)
        local = dest / name if keep else dest / f"{run_id}-{name}"
        s3.download_file(RESULT_BUCKET, obj["Key"], str(local))
        paths.append(local)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    # CLI for the EC2 launcher
    parser = argparse.ArgumentParser(description="Run SpatialBench on an ephemeral EC2 box.")
    parser.add_argument(
        "--engine",
        nargs="+",
        choices=ENGINE_IDS,
        help="Engines to run on separate EC2 nodes.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        choices=SUPPORTED_SCALE_FACTORS,
        help="Scale factor to benchmark (1 or 10). Required unless --profile is given.",
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
    parser.add_argument(
        "--n",
        type=int,
        metavar="N",
        help="Number of timed runs per query; reported time is the average (default 3).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="SF1 profiling mode (one run/query, Engine metrics + RSS + verify); takes no other flags.",
    )
    parser.add_argument(
        "--query",
        nargs="+",
        choices=QUERY_IDS,
        metavar="ID",
        help="Run only these query IDs on the box (e.g. --query q12).",
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
    if args.profile:
        if (
            args.engine is not None
            or args.scale_factor is not None
            or args.index_mode is not None
            or args.n is not None
        ):
            sys.exit("--profile takes no other flags (it runs SF1, one run, with verification)")
        scale_factor, index_mode, n = 1, "auto", 1
        engines = ["pycanopy"]
    else:
        if args.scale_factor is None:
            sys.exit("pass --scale-factor {1,10}, or --profile")
        if not args.engine:
            sys.exit("pass one or more engines with --engine")
        if len(set(args.engine)) != len(args.engine):
            sys.exit("--engine values must be unique")
        scale_factor = args.scale_factor
        index_mode = args.index_mode or "auto"
        n = args.n if args.n is not None else DEFAULT_RUNS
        engines = args.engine
        if n < 1:
            sys.exit("--n must be at least 1")

    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)

    instances = {}
    paths: list[Path] = []
    statuses = {}
    try:
        group_id = uuid.uuid4().hex[:12]
        for engine in engines:
            run_id = group_id if args.profile else f"{group_id}-{engine}"
            instances[engine] = (
                run_id,
                _launch(
                    ec2,
                    ssm,
                    run_id,
                    scale_factor,
                    index_mode,
                    args.profile,
                    n,
                    engine,
                    args.query,
                ),
            )
        with ThreadPoolExecutor(max_workers=len(instances)) as executor:
            futures = {
                engine: executor.submit(_wait_for_success, s3, ec2, run_id, instance_id, engine)
                for engine, (run_id, instance_id) in instances.items()
            }
            statuses = {engine: future.result() for engine, future in futures.items()}
        for run_id, _ in instances.values():
            paths.extend(_download(s3, run_id))
    finally:
        instance_ids = [instance_id for _, instance_id in instances.values()]
        if instance_ids:
            ec2.terminate_instances(InstanceIds=instance_ids)
            print(f"[ec2] terminated {', '.join(instance_ids)}", flush=True)

    if args.profile:
        produced = [path for path in paths if path.name == "profile.txt"]
    else:
        transports = [path for path in paths if path.suffix == ".tsv"]
        if transports:
            results = combine_transports(transports, engines, scale_factor, index_mode)
            suffix = "" if "pycanopy" not in engines or index_mode == "eager" else f"_{index_mode}"
            chart_path = ASSETS_DIR / f"spatialbench_sf{scale_factor}{suffix}.png"
            text_path = ASSETS_DIR / f"spatial-bench-sf{scale_factor}{suffix}-results.txt"
            write_chart(results, chart_path)
            write_results_txt(results, text_path)
            produced = [chart_path, text_path]
            print(f"[results] wrote {text_path} and {chart_path}", flush=True)
        else:
            produced = []

    if not all(statuses.values()) or not produced:
        logs = [p for p in paths if p.suffix == ".log"]
        print(f"[ec2] run failed; inspect {logs[0]}" if logs else "[ec2] run failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
