#!/usr/bin/env bash
# User-data for an unattended PyCanopy SpatialBench run on Amazon Linux 2023.
# Builds PyCanopy, copies the data locally, runs measure.py per scale factor,
# uploads each result JSON, then writes _SUCCESS. The EXIT trap always ships the
# log and shuts the box down; the launch terminates on shutdown. @@NAME@@
# placeholders are filled in by aws_run.py. No AWS keys are injected (instance role).

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
DATA_TEMPLATE="@@DATA_TEMPLATE@@"
SCALE_FACTORS="@@SCALE_FACTORS@@"
MAX_RUNTIME_MIN="@@MAX_RUNTIME_MIN@@"
MEASURE_ARGS="@@MEASURE_ARGS@@"
OUT_SUFFIX="@@OUT_SUFFIX@@"

S3_BASE="s3://${RESULT_BUCKET}/${RESULT_PREFIX}/${RUN_ID}"
LOG=/var/log/pycanopy-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[bootstrap] $*"; }

# Always ship the log and self-terminate, whether we succeed or fail.
cleanup() { aws s3 cp "$LOG" "${S3_BASE}/bootstrap.log" --region "$REGION" || true; shutdown -h now; }
trap cleanup EXIT

# Hard cap: terminate even if a step wedges.
( sleep $((MAX_RUNTIME_MIN * 60)); log "watchdog timeout"; shutdown -h now ) &

# Publish the log to S3 every 15s so aws_run can show live step progress.
( while true; do
    aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" >/dev/null 2>&1 || true
    sleep 15
  done ) &

set -e
log "installing packages"
dnf install -y gcc git python3.11 python3.11-pip python3.11-devel >/dev/null

log "installing rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

log "cloning ${REPO_URL} @ ${REPO_BRANCH}"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" /opt/pycanopy
cd /opt/pycanopy

log "building PyCanopy (release)"
python3.11 -m venv /opt/venv
source /opt/venv/bin/activate
pip install -q --upgrade pip maturin
maturin develop --release
pip install -q --group bench  # benchmark deps, single list in pyproject.toml

mkdir -p /data /opt/pycanopy/bench/spatial_bench/results
for SF in $SCALE_FACTORS; do
  SRC="${DATA_TEMPLATE//\{sf\}/$SF}"
  log "copying data ${SRC} -> /data/sf${SF}"
  aws s3 sync "$SRC" "/data/sf${SF}" --region "$REGION" \
    || aws s3 sync --no-sign-request "$SRC" "/data/sf${SF}" --region "$REGION"
  OUT="/opt/pycanopy/bench/spatial_bench/results/sf${SF}${OUT_SUFFIX}.json"
  log "measuring sf${SF}${OUT_SUFFIX}"
  python -m bench.spatial_bench.measure --data-dir "/data/sf${SF}" --scale-factor "$SF" --output "$OUT" $MEASURE_ARGS
  aws s3 cp "$OUT" "${S3_BASE}/sf${SF}.json" --region "$REGION"
done

log "done"
echo ok | aws s3 cp - "${S3_BASE}/_SUCCESS" --region "$REGION"
