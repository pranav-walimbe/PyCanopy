#!/usr/bin/env bash
# User-data for one unattended node that copies the upstream Hugging Face dataset into the
# harness data bucket. Nothing lands on the launching machine.

set -uo pipefail

# Cloud-init runs this with no HOME set; under `set -u` any $HOME use aborts the run.
export HOME=/root

RUN_ID="@@RUN_ID@@"
REGION="@@REGION@@"
RESULT_BUCKET="@@RESULT_BUCKET@@"
RESULT_PREFIX="@@RESULT_PREFIX@@"
HF_DATASET="@@HF_DATASET@@"
DATASET_VERSION="@@DATASET_VERSION@@"
DATA_BUCKET="@@DATA_BUCKET@@"
DATA_PREFIX="@@DATA_PREFIX@@"
SCALE_FACTORS="@@SCALE_FACTORS@@"

S3_BASE="s3://${RESULT_BUCKET}/${RESULT_PREFIX}/${RUN_ID}"
LOG=/var/log/pycanopy-mirror.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[mirror] $*"; }

export AWS_DEFAULT_REGION="$REGION"

fail() {
  log "FAILED: $*"
  aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" || true
  shutdown -h now
  exit 1
}

# The root volume is small, so stage the download on the instance store or a data volume.
STAGE=/data/hf
mkdir -p "$STAGE" || fail "cannot create $STAGE"

log "installing tooling"
dnf install -y python3-pip >/dev/null 2>&1 || fail "dnf install failed"
pip3 install --quiet huggingface_hub hf_transfer || fail "pip install failed"

export HF_HUB_ENABLE_HF_TRANSFER=1
# The download heredoc is quoted, so it reads these from the environment rather than the shell.
export HF_DATASET DATASET_VERSION STAGE

for SF in $SCALE_FACTORS; do
  log "downloading sf${SF} from ${HF_DATASET}"
  python3 - "$SF" <<'PYTHON' || fail "snapshot_download failed"
import os
import sys

from huggingface_hub import snapshot_download

scale_factor = sys.argv[1]
version = os.environ["DATASET_VERSION"]
snapshot_download(
    repo_id=os.environ["HF_DATASET"],
    repo_type="dataset",
    allow_patterns=[f"{version}/sf{scale_factor}/**"],
    local_dir=os.environ["STAGE"],
    max_workers=8,
)
PYTHON

  SRC="${STAGE}/${DATASET_VERSION}/sf${SF}"
  DEST="s3://${DATA_BUCKET}/${DATA_PREFIX}/${DATASET_VERSION}/sf${SF}"
  log "uploading sf${SF} to ${DEST}"
  aws s3 sync "$SRC" "$DEST" --region "$REGION" --only-show-errors || fail "s3 sync failed"

  # Free the staged copy before the next scale factor so the volume only ever holds one.
  rm -rf "$SRC"
  log "sf${SF} done"
done

log "verifying"
aws s3 ls --recursive "s3://${DATA_BUCKET}/${DATA_PREFIX}/${DATASET_VERSION}/" --region "$REGION"

log "done"
aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" || true
echo ok | aws s3 cp - "${S3_BASE}/_SUCCESS" --region "$REGION"
shutdown -h now
