# Benchmarks

## Apache SpatialBench

Run on a single `m7i.2xlarge` (8 vCPU, 32 GB), the same hardware used by [Apache SpatialBench](https://github.com/apache/sedona-spatialbench). PyCanopy is measured with `index_mode="auto"`.

PyCanopy is fastest on 11/24 testcases and lands within 5% of the fastest time on 14/24 testcases (there is some variance among benchmark runs).

### SF1 (~6M trips)

![PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF1](assets/spatialbench_sf1_auto.png)

*Apache SpatialBench SF1 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated*

| Query | PyCanopy | SedonaDB | DuckDB | GeoPandas |
|:------|:--------:|:--------:|:------:|:---------:|
| q1  | 1.39 | **0.66** | 0.96 | 12.78 |
| q2  | **3.74** | 8.07 | 9.95 | 20.74 |
| q3  | 1.23 | **0.80** | 1.17 | 13.59 |
| q4  | **7.44** | 8.41 | 9.83 | 25.24 |
| q5  | **1.71** | 5.10 | 1.80 | 47.08 |
| q6  | **5.51** | 8.59 | 9.36 | 24.43 |
| q7  | 2.15 | **1.66** | 1.82 | 137.00 |
| q8  | **1.04** | 1.10 | 1.08 | 16.08 |
| q9  | **0.23** | 0.23 | 50.15 | 0.28 |
| q10 | **8.65** | 18.79 | 207.84 | 46.13 |
| q11 | **9.90** | 32.98 | TIMEOUT | 51.01 |
| q12 | 14.86 | **14.55** | ERROR | TIMEOUT |

### SF10 (~60M trips)

![PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF10](assets/spatialbench_sf10_auto.png)

*Apache SpatialBench SF10 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated*

| Query | PyCanopy | SedonaDB | DuckDB | GeoPandas |
|:------|:--------:|:--------:|:------:|:---------:|
| q1  | 8.52 | **3.04** | 4.58 | ERROR |
| q2  | 9.39 | 8.89 | **8.26** | ERROR |
| q3  | 6.88 | **4.09** | 5.17 | TIMEOUT |
| q4  | 17.34 | **7.52** | 8.51 | ERROR |
| q5  | 14.60 | 50.81 | **14.40** | ERROR |
| q6  | 11.07 | **9.11** | 10.67 | ERROR |
| q7  | 22.73 | 14.44 | **14.03** | ERROR |
| q8  | 7.30 | **7.24** | 7.57 | TIMEOUT |
| q9  | **0.34** | 0.38 | 942.98 | 0.49 |
| q10 | **27.26** | 42.02 | ERROR | ERROR |
| q11 | **37.21** | 97.52 | ERROR | ERROR |
| q12 | 175.31 | **145.66** | ERROR | TIMEOUT |

All times in seconds. **Bold** = fastest on that query. SedonaDB, DuckDB, and GeoPandas baselines from published SpatialBench results.
