# Benchmarks

## Apache SpatialBench

Run on a single `m7i.2xlarge` (8 vCPU, 32 GB), the same hardware used by [Apache SpatialBench](https://github.com/apache/sedona-spatialbench). PyCanopy is measured with `index_mode="auto"`.

PyCanopy wins a total of 11/24 testcases and lands within 5% of winning 14/24 testcases (there is some variance among benchmark runs).

### SF1 (~6M trips)

![PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF1](assets/spatialbench_sf1_auto.png)

*Apache SpatialBench SF1 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated*

| Query | PyCanopy | SedonaDB | DuckDB | GeoPandas |
|:------|:--------:|:--------:|:------:|:---------:|
| q1  | 1.41 | **0.66** | 0.96 | 12.78 |
| q2  | **3.94** | 8.07 | 9.95 | 20.74 |
| q3  | 1.22 | **0.80** | 1.17 | 13.59 |
| q4  | 10.88 | **8.41** | 9.83 | 25.24 |
| q5  | **1.77** | 5.10 | 1.80 | 47.08 |
| q6  | **5.57** | 8.59 | 9.36 | 24.43 |
| q7  | 2.22 | **1.66** | 1.82 | 137.00 |
| q8  | **1.06** | 1.10 | 1.08 | 16.08 |
| q9  | **0.23** | 0.23 | 50.15 | 0.28 |
| q10 | **11.62** | 18.79 | 207.84 | 46.13 |
| q11 | **12.43** | 32.98 | TIMEOUT | 51.01 |
| q12 | **14.00** | 14.55 | ERROR | TIMEOUT |

### SF10 (~60M trips)

![PyCanopy vs SedonaDB, DuckDB, and GeoPandas on Apache SpatialBench SF10](assets/spatialbench_sf10_auto.png)

*Apache SpatialBench SF10 · lower is better · bars past the cap truncated with their value · TIMEOUT / ERROR annotated*

| Query | PyCanopy | SedonaDB | DuckDB | GeoPandas |
|:------|:--------:|:--------:|:------:|:---------:|
| q1  | 8.59 | **3.04** | 4.58 | ERROR |
| q2  | 8.95 | 8.89 | **8.26** | ERROR |
| q3  | 7.12 | **4.09** | 5.17 | TIMEOUT |
| q4  | 21.34 | **7.52** | 8.51 | ERROR |
| q5  | 15.22 | 50.81 | **14.40** | ERROR |
| q6  | 11.19 | **9.11** | 10.67 | ERROR |
| q7  | 22.73 | 14.44 | **14.03** | ERROR |
| q8  | **7.03** | 7.24 | 7.57 | TIMEOUT |
| q9  | **0.34** | 0.38 | 942.98 | 0.49 |
| q10 | **28.41** | 42.02 | ERROR | ERROR |
| q11 | **37.30** | 97.52 | ERROR | ERROR |
| q12 | 147.67 | **145.66** | ERROR | TIMEOUT |

All times in seconds. **Bold** = fastest on that query. SedonaDB, DuckDB, and GeoPandas baselines from published SpatialBench results.
