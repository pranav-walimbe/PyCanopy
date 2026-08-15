# Query Planner

PyCanopy separates declaration from execution. You chain filter operations on a `SpatialLazyFrame`, and the `SpatialOptimizer` may reorder compatible operations before execution. kNN and join operations act as barriers, and projection pushdown applies to a terminal `.select()`.

```mermaid
flowchart LR
    A[User chain] --> B[SpatialOptimizer] --> C[SpatialExecutor] --> D[pl.DataFrame]
```

## Optimizer passes

The optimizer runs a fixed sequence of passes over the plan:

**1. Selectivity estimation**

Spatial predicates receive estimated selectivities based on dataset extent or simple output ratios. Scalar Polars filters receive heuristic execution costs derived from their expression structure; PyCanopy does not currently estimate their selectivity from column statistics.

**2. Predicate pushdown**

Scalar `filter()` nodes are moved ahead of compatible spatial predicates. On the EXPR path, this reduces the candidate row indices passed into the spatial masking step.

**3. Cost-sort**

Spatial predicates are reordered by ascending estimated output size. Cheaper, more selective predicates run earlier. kNN and join nodes act as barriers. No reordering crosses them.

**4. Filter fusion**

On datasets with at least 500 rows, consecutive eligible `range_query` and `contains` predicates may be fused into one Rust call. Predicates estimated to retain less than 5% of rows are executed separately. Each fused predicate is still queried separately against the index, then the hit lists are intersected in Rust via a sorted merge.

**5. Join side selection**

For `within_join` and `within_distance_join`, the optimizer may flip the join when the query side contains more than half as many rows as the indexed dataset. Other join types retain their declared orientation.

**6. Projection pushdown**

A terminal `.select(cols)` remains the final operation, and its required columns are propagated into join nodes as `keep_columns`. Each join side is narrowed before gathering rows, avoiding construction of an unnecessary full-width joined frame.

## IO vs EXPR path selection

The optimizer chooses one execution path for the complete plan. Plans containing kNN or join operations use EXPR. For filter-only plans, a sufficiently selective range, contains, or fused spatial predicate selects IO; otherwise the plan uses EXPR:

- **IO path**: used when selectivity is low (few results expected). The index is queried directly and the result is returned as a slice of the DataFrame. No Polars expression pipeline is involved.
- **EXPR path**: used when selectivity is high. The spatial closure runs as a Polars `map_batches` plugin, processing the DataFrame in batches. Scalar filters run first inside the batch, then the spatial query runs on the surviving rows.
