#!/usr/bin/env bash
# User-data for an unattended PyCanopy SpatialBench run on Amazon Linux 2023.
# Builds PyCanopy, reads data directly from S3, measures the scale factor, uploads the
# chart, then writes _SUCCESS. The EXIT trap always ships the log and shuts the box
# down, and the launch terminates on shutdown. @@NAME@@ placeholders are substituted
# before deployment. No AWS keys are injected (instance role).

set -uo pipefail

# Cloud-init runs this with no HOME set; under `set -u` any $HOME use aborts the
# run (e.g. sourcing the rust env below). Pin it before anything reads it.
export HOME=/root

RUN_ID="@@RUN_ID@@"
REGION="@@REGION@@"
RESULT_BUCKET="@@RESULT_BUCKET@@"
RESULT_PREFIX="@@RESULT_PREFIX@@"
REPO_URL="@@REPO_URL@@"
REPO_BRANCH="@@REPO_BRANCH@@"
SCALE_FACTOR="@@SCALE_FACTOR@@"
MAX_RUNTIME_MIN="@@MAX_RUNTIME_MIN@@"

S3_BASE="s3://${RESULT_BUCKET}/${RESULT_PREFIX}/${RUN_ID}"
LOG=/var/log/pycanopy-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[bootstrap] $*"; }

# Always ship the log and self-terminate, whether we succeed or fail
cleanup() { aws s3 cp "$LOG" "${S3_BASE}/bootstrap.log" --region "$REGION" || true; shutdown -h now; }
trap cleanup EXIT

# Hard cap: terminate even if a step wedges
( sleep $((MAX_RUNTIME_MIN * 60)); log "watchdog timeout"; shutdown -h now ) &

# Publish the log to S3 every 15s so the launcher can show live step progress
( while true; do
    aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" >/dev/null 2>&1 || true
    sleep 15
  done ) &

set -e
log "installing packages"
dnf install -y gcc git >/dev/null

log "installing rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

log "installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Amazon Linux 2023 ships Python 3.9, below the project floor, so pin uv to a managed
# 3.10 for every sync and run. It is the supported floor, so the box exercises it.
export UV_PYTHON=3.10

log "cloning ${REPO_URL} @ ${REPO_BRANCH}"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" /opt/pycanopy
cd /opt/pycanopy

log "building PyCanopy (release)"

# uv provisions Python and installs the bench group from the committed uv.lock.
# --no-install-project skips an editable build during sync since maturin develop does it next.
uv sync --no-install-project --group bench
uv run maturin develop --release

mkdir -p /data/scratch /opt/pycanopy/assets

# /tmp is tmpfs (RAM) on Amazon Linux 2023, so spill out-of-core scratch and Polars sort to the EBS data volume.
export PYCANOPY_SCRATCH=/data/scratch
export POLARS_TEMP_DIR=/data/scratch
export TMPDIR=/data/scratch

# object_store picks up IMDS credentials automatically once the region is set
export AWS_DEFAULT_REGION="$REGION"
log "measuring sf${SCALE_FACTOR}"
uv run python -m bench.spatial_bench._onbox --scale-factor "$SCALE_FACTOR" @@BENCH_FLAGS@@

# Normal mode writes the chart PNG, profile mode writes profile.txt; upload whichever exists.
# Plain [ -f ] && cp would return non-zero when absent and abort under set -e, so use if.
OUT="spatialbench_sf${SCALE_FACTOR}@@OUT_SUFFIX@@.png"
if [ -f "/opt/pycanopy/assets/$OUT" ]; then
  aws s3 cp "/opt/pycanopy/assets/$OUT" "${S3_BASE}/$OUT" --region "$REGION"
fi
if [ -f "/opt/pycanopy/assets/profile.txt" ]; then
  aws s3 cp "/opt/pycanopy/assets/profile.txt" "${S3_BASE}/profile.txt" --region "$REGION"
fi

log "done"
echo ok | aws s3 cp - "${S3_BASE}/_SUCCESS" --region "$REGION"