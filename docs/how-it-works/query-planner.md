# Query Planner

PyCanopy separates declaration from execution. You chain operations on a `SpatialLazyFrame` in any order; the `SpatialOptimizer` rewrites the plan before `SpatialExecutor` runs it.

```mermaid
flowchart LR
    A[User chain] --> B[SpatialOptimizer] --> C[SpatialExecutor] --> D[pl.DataFrame]
```

## Optimizer passes

The optimizer runs a fixed sequence of passes over the plan:

**1. Selectivity estimation**

Each node is annotated with an estimated selectivity — the fraction of the dataset expected to survive. Scalar Polars filters get an estimate from column statistics; spatial predicates use the cost model's histogram or kNN ratio.

**2. Predicate pushdown**

Scalar `filter()` nodes are sunk to the bottom of the plan so they run first, reducing the row count before any spatial work begins. A scalar filter that eliminates 90% of rows makes every subsequent spatial operation 10x cheaper.

**3. Cost-sort**

Spatial predicates are reordered by ascending estimated output size. Cheaper, more selective predicates run earlier. kNN and join nodes act as barriers — no reordering crosses them.

**4. Filter fusion**

Consecutive `range_query` or `contains` nodes are merged into a single operation. Two overlapping bounding-box queries become one tighter query; two containment checks on the same polygon set are intersected. This avoids building the index twice.

**5. Join side selection**

For join operations, the optimizer selects which side of the join carries the index. It builds on the side that minimises total probe cost given the estimated sizes of both inputs.

**6. Projection pushdown**

A terminal `.select(cols)` is pinned as the last node and its column set is propagated back into any join as `keep_columns`. Both sides of the join are narrowed before the gather step, so the only full-width materialisation is the final output.

## IO vs EXPR path selection

After optimization, the executor picks one of two execution strategies per node:

- **IO path** — used when selectivity is low (few results expected). The index is queried directly and the result is returned as a slice of the DataFrame. No Polars expression pipeline is involved.
- **EXPR path** — used when selectivity is high. The spatial closure runs as a Polars `map_batches` plugin, processing the DataFrame in batches. Scalar filters run first inside the batch, then the spatial query runs on the surviving rows.

The optimizer annotates each node with the chosen path based on the selectivity threshold from the cost model.
