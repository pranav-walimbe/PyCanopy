"""Configuration for the SpatialBench cloud harness."""

from pathlib import Path

WORKLOAD_REVISION = "b9221a9c4b02b10db20611d79b4019d2b3c4b68e"
DATASET_VERSION = "v0.1.0"
SUPPORTED_SCALE_FACTORS = (1, 10)
DEFAULT_RUNS = 3
QUERY_TIMEOUT_SECONDS = 1200
RUNNER_PREFIX = "SPATIALBENCH"
QUERY_IDS = tuple(f"q{index}" for index in range(1, 13))

PROFILE_VARIANTS = ("branch", "release")
PROFILE_VARIANT_LABELS = {"branch": "branch", "release": "released"}
TABLES = ("building", "customer", "driver", "trip", "vehicle", "zone")

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
ANSWERS_DIR = Path(__file__).resolve().parent / "answers"

MIB = 1024 * 1024
NS_PER_SECOND = 1_000_000_000
RSS_SAMPLE_INTERVAL = 0.02

ENGINES = {
    "pycanopy": {"display_name": "PyCanopy", "package": "pycanopy", "color": "#2C7FB8"},
    "duckdb": {"display_name": "DuckDB", "package": "duckdb", "color": "#8C8C8C"},
    "sedonadb": {"display_name": "SedonaDB", "package": "sedonadb", "color": "#DD8452"},
    "geopandas": {"display_name": "GeoPandas", "package": "geopandas", "color": "#C9BBA8"},
}
ENGINE_IDS = tuple(ENGINES)

STORAGE_OPTIONS = {"skip_signature": "true"}

# Mirror of the upstream Hugging Face dataset, which is what the committed answers were
# generated from. Public read, so queries stay anonymous.
PUBLIC_DATA_ROOT = "s3://pycanopy-bench-data/spatialbench"
PUBLIC_DATA_TEMPLATE = f"{PUBLIC_DATA_ROOT}/{DATASET_VERSION}/sf{{scale_factor}}"

REGION = "us-west-2"
INSTANCE_TYPE = "m7i.2xlarge"
VOLUME_GB = 32
VOLUME_IOPS = 3000
VOLUME_THROUGHPUT_MBPS = 125
MAX_RUNTIME_MINUTES = 60
RESULT_BUCKET = "pycanopy-bench-results"
INSTANCE_PROFILE = "pycanopy-spatialbench"
REPOSITORY_URL = "https://github.com/pranav-walimbe/PyCanopy.git"
REPOSITORY_BRANCH = "main"
AMI_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
PROJECT_TAG = "pycanopy-spatialbench"
RESULT_KEY_PREFIX = "spatialbench-runs"
POLL_SECONDS = 30
