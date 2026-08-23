"""Copy the upstream Hugging Face dataset into the harness data bucket from an EC2 node."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from bench.spatial_bench.config import (
    AMI_PARAMETER,
    DATA_BUCKET,
    DATA_KEY_PREFIX,
    DATASET_VERSION,
    HF_DATASET,
    INSTANCE_PROFILE,
    POLL_SECONDS,
    PROJECT_TAG,
    REGION,
    RESULT_BUCKET,
    RESULT_KEY_PREFIX,
    SUPPORTED_SCALE_FACTORS,
)

_DIR = Path(__file__).parent

# The node stages one scale factor at a time, but SF10 alone is ~6 GB before upload
MIRROR_INSTANCE_TYPE = "m7i.xlarge"
MIRROR_VOLUME_GB = 64
MIRROR_TIMEOUT_MINUTES = 90


def _user_data(run_id: str, scale_factors: list[int]) -> str:
    # Substitute @@NAME@@ placeholders in mirror.sh for this transfer
    script = (_DIR / "mirror.sh").read_text()
    repl = {
        "RUN_ID": run_id,
        "REGION": REGION,
        "RESULT_BUCKET": RESULT_BUCKET,
        "RESULT_PREFIX": RESULT_KEY_PREFIX,
        "HF_DATASET": HF_DATASET,
        "DATASET_VERSION": DATASET_VERSION,
        "DATA_BUCKET": DATA_BUCKET,
        "DATA_PREFIX": DATA_KEY_PREFIX,
        "SCALE_FACTORS": " ".join(str(factor) for factor in scale_factors),
    }
    for key, value in repl.items():
        script = script.replace(f"@@{key}@@", value)
    return script


def _launch(ec2, ssm, run_id: str, scale_factors: list[int]) -> str:
    # Launch the mirroring instance and return its id
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=MIRROR_INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        UserData=_user_data(run_id, scale_factors),
        InstanceInitiatedShutdownBehavior="terminate",
        IamInstanceProfile={"Name": INSTANCE_PROFILE},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": MIRROR_VOLUME_GB,
                    "VolumeType": "gp3",
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
    factors = ", ".join(f"sf{factor}" for factor in scale_factors)
    print(f"[ec2] launched {instance_id} ({MIRROR_INSTANCE_TYPE}, {factors}, run {run_id})")
    return instance_id


def _alive(ec2, instance_id: str) -> bool:
    # True while the instance is pending or running
    try:
        inst = ec2.describe_instances(InstanceIds=[instance_id])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "InvalidInstanceID.NotFound":
            return True
        raise
    return inst["Reservations"][0]["Instances"][0]["State"]["Name"] in ("pending", "running")


def _emit_progress(s3, run_id: str, seen: int) -> int:
    # Print [mirror] lines from the streamed log since the last poll
    key = f"{RESULT_KEY_PREFIX}/{run_id}/progress.log"
    try:
        text = s3.get_object(Bucket=RESULT_BUCKET, Key=key)["Body"].read()
    except ClientError:
        return seen
    lines = [line.rstrip() for line in text.decode("utf-8", "replace").splitlines()]
    for line in lines[seen:]:
        print(line, flush=True)
    return len(lines)


def _wait_for_success(s3, ec2, run_id: str, instance_id: str) -> bool:
    # Poll S3 for the _SUCCESS marker until it appears or the box dies or the deadline passes
    key = f"{RESULT_KEY_PREFIX}/{run_id}/_SUCCESS"
    deadline = time.monotonic() + MIRROR_TIMEOUT_MINUTES * 60
    seen = 0
    while time.monotonic() < deadline:
        seen = _emit_progress(s3, run_id, seen)
        try:
            s3.head_object(Bucket=RESULT_BUCKET, Key=key)
            _emit_progress(s3, run_id, seen)
            return True
        except ClientError:
            pass
        if not _alive(ec2, instance_id):
            _emit_progress(s3, run_id, seen)
            return False
        time.sleep(POLL_SECONDS)
    return False


def _summarize(s3, scale_factors: list[int]) -> None:
    # Print what now sits under each mirrored scale factor
    for factor in scale_factors:
        prefix = f"{DATA_KEY_PREFIX}/{DATASET_VERSION}/sf{factor}/"
        paginator = s3.get_paginator("list_objects_v2")
        objects = [
            obj
            for page in paginator.paginate(Bucket=DATA_BUCKET, Prefix=prefix)
            for obj in page.get("Contents", [])
        ]
        total = sum(obj["Size"] for obj in objects)
        print(f"[data] sf{factor}: {len(objects)} objects, {total / 1e9:.2f} GB")


def main(argv: list[str] | None = None) -> int:
    """Mirror the requested scale factors from Hugging Face into the data bucket.

    Args:
        argv: Command-line arguments, defaulting to sys.argv.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Copy the upstream SpatialBench dataset into the harness data bucket."
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        nargs="+",
        choices=SUPPORTED_SCALE_FACTORS,
        default=list(SUPPORTED_SCALE_FACTORS),
        help="Scale factors to mirror (default: every supported factor).",
    )
    args = parser.parse_args(argv)
    scale_factors = sorted(set(args.scale_factor))

    run_id = uuid.uuid4().hex[:12]
    ec2 = boto3.client("ec2", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    instance_id = _launch(ec2, ssm, run_id, scale_factors)
    try:
        ok = _wait_for_success(s3, ec2, run_id, instance_id)
    finally:
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"[ec2] terminated {instance_id}")

    if not ok:
        print("[ec2] mirror failed; see the progress log above")
        return 1
    _summarize(s3, scale_factors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
