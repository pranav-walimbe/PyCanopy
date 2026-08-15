# Execution Paths

After optimization, `SpatialExecutor` runs the plan using one of two strategies. The optimizer chooses one execution path for the complete plan. Plans containing kNN or join operations use EXPR. For filter-only plans, a sufficiently selective range, contains, or fused spatial predicate selects IO; otherwise the plan uses EXPR.

## EXPR path

Used when a large fraction of the dataset is expected to match. The spatial operation runs as a Polars `map_batches` expression. Its callback receives the surviving row indices and invokes the Rust engine to produce a Boolean mask.

Execution order within a batch:

1. Scalar Polars filters run first on the full batch, reducing rows cheaply.
2. The spatial closure runs on the surviving rows, querying the global index.
3. Results are assembled and returned as a Polars Series.

The EXPR path keeps the spatial engine inside Polars' processing loop, allowing Polars' own parallelism and memory management to apply around the spatial work.

## IO path

Used when few results are expected (low selectivity, e.g. a tight bounding box or a point-in-polygon query on a sparse dataset). The index is queried directly, and the result row indices are used to slice the DataFrame:

```
index.range_query(bbox) → [i, j, k, ...]  →  df[i, j, k, ...]
```

No Polars expression pipeline is involved. This avoids the overhead of batch processing when the output is a tiny fraction of the input.

## Polars / PyO3 integration

The Rust engine is compiled as a PyO3 extension (`pycanopy._core`). The Rust extension reads contiguous NumPy coordinate buffers directly at the Python/Rust boundary. When an `Engine` is constructed, the coordinates are copied once into Rust-owned `Arc<[f64]>` buffers. Subsequent index builds share those buffers without copying the coordinate data again. The index structures themselves (KD-tree, R-tree, grid) are packed immutable Rust structs that live for the lifetime of the `Engine` object.

The EXPR path uses Polars' Python `map_batches` API rather than a native Polars expression plugin. The callback crosses into the PyO3 extension for spatial computation, while Polars manages the surrounding lazy expression pipeline.

## Join assembly

Spatial join kernels (`knn_join`, `within_distance_join`, `within_join`, etc.) return raw `(query_idx, target_idx)` index pairs from Rust. `SpatialExecutor._assemble_join` then:

1. Gathers both sides of the join by index.
2. Horizontal-concatenates them into a single DataFrame.
3. Renames any conflicting column names on the right side with a `right_` prefix.

If a `.select()` was pushed down, both sides are narrowed to the keep-set before the gather, so the full-width DataFrame is never materialized.
