# Benchmarks

## Apache SpatialBench

Run on a single `m7i.2xlarge` (8 vCPU, 32 GB), the same hardware used by [Apache SpatialBench](https://github.com/apache/sedona-spatialbench). PyCanopy is measured with `index_mode="auto"`.

PyCanopy is fastest on 11/24 testcases (there is some variance among benchmark runs).

### SF1 (~6M trips)

![PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF1](assets/spatialbench_sf1.png)

*Apache SpatialBench SF1 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated*

| Query | PyCanopy | SedonaDB | DuckDB | GeoPandas |
|:------|:--------:|:--------:|:------:|:---------:|
| q1  | 1.05 | **0.54** | 0.76 | 12.92 |
| q2  | 2.01 | **1.70** | 2.00 | 14.94 |
| q3  | 0.80 | **0.58** | 1.61 | 13.63 |
| q4  | 4.99 | 4.60 | **3.82** | 18.35 |
| q5  | **1.18** | 3.92 | 1.85 | 49.75 |
| q6  | 4.66 | **3.12** | 4.24 | 19.74 |
| q7  | 1.35 | **1.14** | 1.30 | 136.98 |
| q8  | **0.71** | 0.73 | 1.21 | 15.72 |
| q9  | 0.15 | **0.11** | 0.26 | 0.38 |
| q10 | **5.92** | 6.52 | 203.77 | 43.20 |
| q11 | **6.91** | 10.70 | 360.37 | 47.39 |
| q12 | **3.48** | 10.15 | TIMEOUT | TIMEOUT |

### SF10 (~60M trips)

![PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF10](assets/spatialbench_sf10.png)

*Apache SpatialBench SF10 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated*

| Query | PyCanopy | SedonaDB | DuckDB | GeoPandas |
|:------|:--------:|:--------:|:------:|:---------:|
| q1  | 5.33 | **2.29** | 4.78 | TIMEOUT |
| q2  | **4.28** | 5.84 | 5.08 | TIMEOUT |
| q3  | 4.67 | **3.65** | 4.75 | TIMEOUT |
| q4  | 8.52 | **5.88** | 6.14 | OOM |
| q5  | **11.86** | 23.85 | 12.32 | OOM |
| q6  | 10.12 | 11.22 | **8.78** | ERROR |
| q7  | 11.11 | 7.39 | **5.91** | ERROR |
| q8  | 5.71 | **5.29** | 6.85 | ERROR |
| q9  | **0.23** | 0.25 | 0.50 | ERROR |
| q10 | **29.49** | 38.29 | TIMEOUT | ERROR |
| q11 | **32.30** | 61.19 | TIMEOUT | ERROR |
| q12 | **38.59** | 117.50 | TIMEOUT | ERROR |

All times in seconds. **Bold** = fastest on that query. Every engine was measured by the PyCanopy harness against the pinned SpatialBench workload.
