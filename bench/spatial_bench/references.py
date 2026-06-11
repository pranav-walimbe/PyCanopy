"""Published SpatialBench reference numbers for cross-engine comparison.

These are the official Apache SpatialBench single-node results (seconds per query),
transcribed verbatim from the project documentation so spatial_bench can place a
locally-measured PyCanopy number beside them without re-running the other engines.

Source: SpatialBench v0.1.0 pre-release (commit 9094be8), benchmarked 2025-09-22.
See: https://github.com/apache/sedona-spatialbench  (docs/single-node-benchmarks.md)

IMPORTANT CAVEATS for any comparison drawn from these numbers:
  * Hardware: AWS EC2 m7i.2xlarge (8 vCPU, 32 GB RAM), us-west-2. To compare fairly,
    run PyCanopy on the same instance type/region (see spatial_bench/README.md).
  * Methodology: end-to-end runtime INCLUDING parquet data loading, cold start
    (DuckDB enable_external_file_cache=false), COUNT to force execution, 1200s timeout.
  * ERROR = the engine raised (e.g. OOM); TIMEOUT = exceeded 1200s.
"""

from __future__ import annotations

# Metadata describing the conditions under which REFERENCE_SECONDS was measured.
REFERENCE_META = {
    "source": "Apache SpatialBench v0.1.0 (commit 9094be8)",
    "date": "2025-09-22",
    "hardware": "AWS EC2 m7i.2xlarge (8 vCPU, 32 GB RAM), us-west-2",
    "software": {
        "SedonaDB": "0.1.0",
        "DuckDB": "1.4.0",
        "GeoPandas": "1.1.1",
        "Shapely": "2.1.1",
        "NumPy": "2.3.3",
    },
    "methodology": (
        "End-to-end incl. parquet load, cold start, COUNT to force execution, 1200s timeout."
    ),
    "timeout_seconds": 1200,
}

# Sentinels for non-numeric outcomes in the published tables.
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"

# REFERENCE_SECONDS[scale_factor][query_id][engine] -> seconds | "ERROR" | "TIMEOUT".
REFERENCE_SECONDS: dict[int, dict[str, dict[str, float | str]]] = {
    1: {
        "q1": {"SedonaDB": 0.66, "DuckDB": 0.96, "GeoPandas": 12.78},
        "q2": {"SedonaDB": 8.07, "DuckDB": 9.95, "GeoPandas": 20.74},
        "q3": {"SedonaDB": 0.80, "DuckDB": 1.17, "GeoPandas": 13.59},
        "q4": {"SedonaDB": 8.41, "DuckDB": 9.83, "GeoPandas": 25.24},
        "q5": {"SedonaDB": 5.10, "DuckDB": 1.80, "GeoPandas": 47.08},
        "q6": {"SedonaDB": 8.59, "DuckDB": 9.36, "GeoPandas": 24.43},
        "q7": {"SedonaDB": 1.66, "DuckDB": 1.82, "GeoPandas": 137.00},
        "q8": {"SedonaDB": 1.10, "DuckDB": 1.08, "GeoPandas": 16.08},
        "q9": {"SedonaDB": 0.23, "DuckDB": 50.15, "GeoPandas": 0.28},
        "q10": {"SedonaDB": 18.79, "DuckDB": 207.84, "GeoPandas": 46.13},
        "q11": {"SedonaDB": 32.98, "DuckDB": TIMEOUT, "GeoPandas": 51.01},
        "q12": {"SedonaDB": 14.55, "DuckDB": ERROR, "GeoPandas": TIMEOUT},
    },
    10: {
        "q1": {"SedonaDB": 3.04, "DuckDB": 4.58, "GeoPandas": ERROR},
        "q2": {"SedonaDB": 8.89, "DuckDB": 8.26, "GeoPandas": ERROR},
        "q3": {"SedonaDB": 4.09, "DuckDB": 5.17, "GeoPandas": TIMEOUT},
        "q4": {"SedonaDB": 7.52, "DuckDB": 8.51, "GeoPandas": ERROR},
        "q5": {"SedonaDB": 50.81, "DuckDB": 14.40, "GeoPandas": ERROR},
        "q6": {"SedonaDB": 9.11, "DuckDB": 10.67, "GeoPandas": ERROR},
        "q7": {"SedonaDB": 14.44, "DuckDB": 14.03, "GeoPandas": ERROR},
        "q8": {"SedonaDB": 7.24, "DuckDB": 7.57, "GeoPandas": TIMEOUT},
        "q9": {"SedonaDB": 0.38, "DuckDB": 942.98, "GeoPandas": 0.49},
        "q10": {"SedonaDB": 42.02, "DuckDB": ERROR, "GeoPandas": ERROR},
        "q11": {"SedonaDB": 97.52, "DuckDB": ERROR, "GeoPandas": ERROR},
        "q12": {"SedonaDB": 145.66, "DuckDB": ERROR, "GeoPandas": TIMEOUT},
    },
}

REFERENCE_ENGINES = ("SedonaDB", "DuckDB", "GeoPandas")


def reference_row(scale_factor: int, query_id: str) -> dict[str, float | str]:
    """Return the published per-engine seconds for one query, or an empty dict if absent.

    Args:
        scale_factor: SpatialBench scale factor (1 or 10 have published numbers).
        query_id: Query identifier such as "q1".

    Returns:
        Mapping of engine name to seconds (or ERROR/TIMEOUT). Empty if no reference exists.
    """
    return REFERENCE_SECONDS.get(scale_factor, {}).get(query_id, {})
