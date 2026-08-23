"""Configuration for the SpatialBench cloud harness."""

from dataclasses import dataclass

WORKLOAD_REVISION = "b9221a9c4b02b10db20611d79b4019d2b3c4b68e"
DATASET_VERSION = "v0.1.0"
SUPPORTED_SCALE_FACTORS = (1, 10)
DEFAULT_RUNS = 3
QUERY_TIMEOUT_SECONDS = 1200

ENGINE_IDS = ("pycanopy", "duckdb", "sedonadb", "geopandas")
DISPLAY_NAMES = {
    "pycanopy": "PyCanopy",
    "duckdb": "DuckDB",
    "sedonadb": "SedonaDB",
    "geopandas": "GeoPandas",
}
PACKAGE_NAMES = {
    "pycanopy": "pycanopy",
    "duckdb": "duckdb",
    "sedonadb": "sedonadb",
    "geopandas": "geopandas",
}

PUBLIC_DATA_ROOT = "s3://wherobots-examples/data/spatialbench"
PUBLIC_DATA_TEMPLATE = f"{PUBLIC_DATA_ROOT}/SpatialBench_sf{{scale_factor}}"


@dataclass(frozen=True)
class CloudConfig:
    """Infrastructure used for comparable cloud runs."""

    region: str = "us-west-2"
    instance_type: str = "m7i.2xlarge"
    volume_gb: int = 32
    volume_iops: int = 3000
    volume_throughput_mbps: int = 125
    max_runtime_minutes: int = 60
    result_bucket: str = "pycanopy-bench-results"
    instance_profile: str = "pycanopy-spatialbench"
    repository_url: str = "https://github.com/pranav-walimbe/PyCanopy.git"
    repository_branch: str = "main"
    ami_parameter: str = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    project_tag: str = "pycanopy-spatialbench"
    result_prefix: str = "spatialbench-runs"
    poll_seconds: int = 30


CLOUD = CloudConfig()
