# Architecture Overview

PyCanopy adds spatial planning and native geometry kernels to Polars. Polars remains responsible
for reading tables, evaluating scalar expressions, gathering rows, and producing DataFrames.

![A PyCanopy query moves from API calls through source preparation and optimization before the executor coordinates Polars and the Rust engine](../assets/diagrams/architecture-overview.png)

## Query lifecycle

1. `SpatialLazyFrame` records each operation as a typed node without executing it.
2. A terminal operation prepares a deferred source, if present, and builds the complete native
   geometry engine.
3. `SpatialOptimizer` estimates and reorders the remaining nodes, then selects how spatial results
   will reconnect to Polars rows.
4. `SpatialExecutor` translates the optimized nodes into Polars operations and Rust calls.
5. Each Rust kernel chooses its own scan or index access path from the engine's statistics and
   configured index mode.

The Python optimizer and the Rust access-path planner answer different questions. The Python layer
orders logical operations and chooses a Polars integration path. The Rust layer decides how a
specific spatial kernel should find candidates.

## Division of work

| Layer | Responsibilities |
|:------|:-----------------|
| Polars | Scans, scalar filters, projection, row gathering, tabular joins, and final output |
| PyCanopy Python | Plan construction, deferred-source preparation, operation ordering, projection planning, and execution routing |
| PyCanopy Rust | Geometry storage, statistics, candidate search, exact predicates, distances, and eligible grouped aggregation |

PyCanopy exchanges row indices, masks, distances, and aggregate state across the Python/Rust
boundary. Full attribute rows stay in Polars.

## Materialization boundary

A `SpatialFrame` always owns a complete native geometry dataset when spatial execution starts.
Deferred ingestion can reduce the rows and columns read before that point, while morsel execution
can bound join intermediates afterward. Neither feature partitions the indexed geometry engine.
