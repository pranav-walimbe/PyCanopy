#!/usr/bin/env bash
# User-data for one unattended SpatialBench engine node on Amazon Linux 2023.
# The node uploads its profile or timing transport before self-termination.

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
DATA_ROOT="@@DATA_ROOT@@"
MAX_RUNTIME_MIN="@@MAX_RUNTIME_MIN@@"
PROFILE_MODE="@@PROFILE_MODE@@"
PROFILE_VARIANT="@@PROFILE_VARIANT@@"
ENGINE="@@ENGINE@@"

export PYCANOPY_BENCH_RUN_ID="$RUN_ID"
export PYCANOPY_BENCH_REGION="$REGION"
export PYCANOPY_BENCH_AMI_ID="@@AMI_ID@@"
export PYCANOPY_BENCH_INSTANCE_TYPE="@@INSTANCE_TYPE@@"
export PYCANOPY_BENCH_VOLUME_TYPE="gp3"
export PYCANOPY_BENCH_VOLUME_GB="@@VOLUME_GB@@"
export PYCANOPY_BENCH_VOLUME_IOPS="@@VOLUME_IOPS@@"
export PYCANOPY_BENCH_VOLUME_THROUGHPUT_MBPS="@@VOLUME_THROUGHPUT_MBPS@@"

S3_BASE="s3://${RESULT_BUCKET}/${RESULT_PREFIX}/${RUN_ID}"
LOG=/var/log/pycanopy-bootstrap.log
ARTIFACTS_UPLOADED=0
exec > >(tee -a "$LOG") 2>&1
log() { echo "[bootstrap] $*"; }

upload_artifacts() {
  if [ "$ARTIFACTS_UPLOADED" = "1" ]; then
    return
  fi
  for artifact in \
    "profile-${PROFILE_VARIANT}.json" \
    "${ENGINE}-continuation.json" \
    "${ENGINE}-results.tsv"; do
    if [ -f "/opt/pycanopy/assets/$artifact" ]; then
      aws s3 cp "/opt/pycanopy/assets/$artifact" "${S3_BASE}/$artifact" --region "$REGION" || return 1
    fi
  done
  ARTIFACTS_UPLOADED=1
}

# Always ship the log and any completed result artifacts, then self-terminate.
cleanup() {
  upload_artifacts || true
  aws s3 cp "$LOG" "${S3_BASE}/bootstrap.log" --region "$REGION" || true
  shutdown -h now
}
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

log "installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Amazon Linux 2023 ships Python 3.9 below the project floor
# Pin uv to a managed 3.10 for every sync and run which is the supported floor
export UV_PYTHON=3.10

log "cloning ${REPO_URL} @ ${REPO_BRANCH}"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" /opt/pycanopy
cd /opt/pycanopy

if [ "$PROFILE_MODE" = "1" ] && [ "$PROFILE_VARIANT" = "branch" ]; then
  log "installing rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"

  log "building PyCanopy (release)"
  uv sync --no-install-project --group bench
  uv run maturin develop --release
  BENCH_PYTHON=(uv run python)
elif [ "$PROFILE_MODE" = "1" ]; then
  # PyCanopy pulls polars, numpy, pyarrow and shapely, all the queries need
  log "installing latest published pycanopy"
  uv venv --python 3.10 /opt/spatialbench-env
  uv pip install --python /opt/spatialbench-env/bin/python pycanopy
  BENCH_PYTHON=(/opt/spatialbench-env/bin/python)
else
  log "installing latest ${ENGINE} release"
  uv venv --python 3.10 /opt/spatialbench-env
  case "$ENGINE" in
    pycanopy)
      uv pip install --python /opt/spatialbench-env/bin/python pycanopy
      ;;
    duckdb)
      uv pip install --python /opt/spatialbench-env/bin/python duckdb pandas pyarrow
      ;;
    sedonadb)
      uv pip install --python /opt/spatialbench-env/bin/python 'sedonadb[geopandas]' pandas pyarrow pyproj
      ;;
    geopandas)
      uv pip install --python /opt/spatialbench-env/bin/python geopandas pandas pyarrow shapely s3fs
      ;;
    *)
      log "unsupported engine: ${ENGINE}"
      exit 2
      ;;
  esac
  BENCH_PYTHON=(/opt/spatialbench-env/bin/python)
fi

mkdir -p /data/scratch /opt/pycanopy/assets

# /tmp is tmpfs on Amazon Linux 2023 so scratch and Polars sort spill to the EBS data volume
export PYCANOPY_SCRATCH=/data/scratch
export POLARS_TEMP_DIR=/data/scratch
export TMPDIR=/data/scratch

# object_store picks up IMDS credentials automatically once the region is set
export AWS_DEFAULT_REGION="$REGION"
log "measuring sf${SCALE_FACTOR}"
rm -f "/opt/pycanopy/assets/profile-${PROFILE_VARIANT}.json" \
  "/opt/pycanopy/assets/${ENGINE}-continuation.json" \
  "/opt/pycanopy/assets/${ENGINE}-results.tsv"
"${BENCH_PYTHON[@]}" -m bench.spatial_bench.driver_utils \
  --engine "$ENGINE" \
  --scale-factor "$SCALE_FACTOR" \
  --data-dir "$DATA_ROOT" \
  --variant "$PROFILE_VARIANT" \
  @@BENCH_FLAGS@@

upload_artifacts

log "done"
aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" || true
echo ok | aws s3 cp - "${S3_BASE}/_SUCCESS" --region "$REGION"
