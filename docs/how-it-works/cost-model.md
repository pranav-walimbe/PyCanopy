# Cost Model and Automatic Indexing

Automatic indexing answers two separate questions:

1. Which access structure fits this geometry and query?
2. Is building it cheaper than scanning or reusing an existing index?

![Automatic index planning combines the query workload, dataset statistics, and available indexes before comparing costs](../assets/diagrams/automatic-index-planning.png)

## Access paths

| Access path | Typical role |
|:------------|:-------------|
| Parallel scan | Small inputs, broad queries, or index builds that cannot repay their cost |
| Grid | Range and distance workloads over uniform point data |
| KD-tree | Point kNN and non-uniform point range workloads |
| R-tree | Polygon bounds, polygon queries, and polygon kNN candidates |

These are candidates rather than promises. In `auto` mode, a query can still choose a scan even
when an index type fits its shape.

## Statistics collected once

- Row count gates index work on small datasets
- Extent provides a fallback area-based selectivity estimate
- Grid-cell variation classifies point data as uniform or clustered
- The histogram estimates range-query output more accurately than extent area alone
- Polygon histograms bin each polygon by its exterior-ring centroid

## Cost comparison

For query count $Q$, dataset size $N$, and estimated selectivity $s$:

![The planner compares the estimated total cost of scanning, reusing an index, and building a new index](../assets/diagrams/index-cost-comparison.png)

- Grid build cost grows approximately with $N$
- KD-tree and R-tree build costs grow approximately with $N\log_2N$
- Tree probes include traversal plus expected result work
- Grid probes scale mainly with expected result work
- A built index has zero remaining build cost and competes on probe cost alone

The model predicts relative choices, not user-visible runtime. File reads, Polars expressions, row
gathers, and result materialization sit outside these native index estimates.

## Index modes

| Mode | Planner behavior |
|:-----|:-----------------|
| `auto` | Compare scan, built indexes, and the best new candidate |
| `eager` | Use the rule-selected candidate without the scan cost gate |
| `none` | Always use the parallel scan |

## Calibration

The bundled cost factors are fitted from release-mode timings of native scans, index builds, and
index probes. The calibration workloads vary geometry type, dataset size, spatial distribution,
selectivity, probe count, and kNN `k` so the model captures the main ways index costs scale.

PyCanopy ships those fitted factors as defaults. They guide relative planning choices rather than
predicting end-to-end query time on a particular machine.
