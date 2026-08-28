# Native Access Planning and Automatic Indexing

The Python optimizer decides operation order and Polars integration. Once the executor invokes a
spatial kernel, the Rust planner decides how that kernel should search the native geometry dataset.

It answers two questions:

1. Which access structures support this geometry and operation?
2. Is scanning, reusing an existing index, or building a new index cheapest for this workload?

![Native access planning combines the operation, dataset statistics, probe count, and existing indexes before comparing viable paths](../assets/diagrams/automatic-index-planning.png)

## Candidate access paths

| Access path | Typical role |
|:------------|:-------------|
| Parallel scan | Small inputs, broad queries, or index builds that cannot repay their cost |
| Grid | Range and distance workloads over uniform point data |
| KD-tree | Point kNN and non-uniform point range workloads |
| R-tree | Polygon bounds, polygon queries, and polygon kNN candidates |

The table describes candidates, not guarantees. In `auto` mode, a compatible index can still lose
to a scan.

## Planning inputs

The engine collects statistics when it builds the native dataset:

- Row count and extent
- A fixed spatial histogram for non-empty datasets
- Grid-cell variation used to classify point distributions
- Polygon histogram entries based on exterior-ring centroids

The query contributes its operation, geometry, probe count, `k` or distance where applicable, and
an estimated result fraction. Already-built indexes contribute their remaining probe cost without
another build charge.

## Cost comparison

For query count $Q$, dataset size $N$, and estimated selectivity $s$:

![The native planner compares the estimated total cost of scanning, reusing an index, and building a new index](../assets/diagrams/index-cost-comparison.png)

- Scan cost grows with $Q \times N$
- Grid build cost grows approximately with $N$
- KD-tree and R-tree build costs grow approximately with $N\log_2N$
- Tree probes include traversal and expected result work
- Grid probes depend mainly on expected result work
- A built index competes on its remaining probe cost

The model compares native alternatives. It does not predict end-to-end query time, and it excludes
file reads, Polars expressions, row gathering, and result materialization.

## Index modes

| Mode | Native planner behavior |
|:-----|:------------------------|
| `auto` | Compare a scan, compatible built indexes, and the best new index candidate |
| `eager` | Use the rule-selected index candidate without the scan cost gate |
| `none` | Use the parallel scan |

Explicit index modes can also force a supported index kind. The public API reference documents
those values.

## Join direction

Some native batch kernels compare more than one orientation. A point-to-polygon distance join, for
example, can compare probing polygon structures from the point side with building a grid over the
points and probing it from polygon bounds. The planner includes build and probe work for the full
probe count, even when the Python executor later feeds that work in morsels.

## Calibration

PyCanopy fits its bundled factors from release-mode timings of native scans, index builds, and
index probes. Calibration varies geometry type, input size, spatial distribution, selectivity,
probe count, and kNN `k`. The fitted factors guide relative choices rather than predicting wall
time on a specific machine.
