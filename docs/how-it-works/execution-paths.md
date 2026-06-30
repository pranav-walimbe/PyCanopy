# Execution Paths

After optimization, `SpatialExecutor` runs the plan using one of two strategies. The choice is made per-node based on selectivity.

## EXPR path

Used when a large fraction of the dataset is expected to match (high selectivity). The spatial operation is registered as a Polars `map_batches` plugin and runs inside the Polars expression engine.

Execution order within a batch:

1. Scalar Polars filters run first on the full batch, reducing rows cheaply.
2. The spatial closure runs on the surviving rows, querying the global index.
3. Results are assembled and returned as a Polars Series.

The EXPR path keeps the spatial engine inside Polars' processing loop, allowing Polars' own parallelism and memory management to apply around the spatial work.

## IO path

Used when few results are expected (low selectivity — e.g. a tight bounding box or a point-in-polygon query on a sparse dataset). The index is queried directly, and the result row indices are used to slice the DataFrame:

```
index.range_query(bbox) → [i, j, k, ...]  →  df[i, j, k, ...]
```

No Polars expression pipeline is involved. This avoids the overhead of batch processing when the output is a tiny fraction of the input.

## Polars / PyO3 integration

The Rust engine is compiled as a PyO3 extension (`pycanopy._core`). Coordinate arrays are passed as zero-copy numpy views at the Python/Rust boundary — no allocation occurs for the handoff. The index structures themselves (KD-tree, R-tree, grid) are packed immutable Rust structs that live for the lifetime of the `Engine` object.

For the EXPR path, the plugin is registered via Polars' `register_plugin_function` API, which lets the Rust closure participate in Polars' lazy evaluation graph natively.

## Join assembly

Spatial join kernels (`knn_join`, `within_distance_join`, `within_join`, etc.) return raw `(query_idx, target_idx)` index pairs from Rust. `SpatialExecutor._assemble_join` then:

1. Gathers both sides of the join by index.
2. Horizontal-concatenates them into a single DataFrame.
3. Renames any conflicting column names on the right side with a `right_` prefix.

If a `.select()` was pushed down, both sides are narrowed to the keep-set before the gather, so the full-width DataFrame is never materialised.
