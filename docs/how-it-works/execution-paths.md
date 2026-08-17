# Execution Paths

`SpatialExecutor` selects one path for the complete plan. Selective `range_query`,
`contains`, and fused filters can use IO; broader filters, kNN, and joins use EXPR.

## EXPR path

The spatial predicate runs as a Polars `map_batches` expression:

Execution order:

1. Scalar Polars filters run first.
2. The spatial callback receives the surviving original row indices.
3. Rust evaluates the spatial predicate and returns a Boolean mask aligned with those
   candidates.

Polars evaluates the surrounding lazy pipeline; the callback crosses into Rust for the
spatial predicate. This uses Python `map_batches`, not a native Polars expression plugin.

## IO path

When few matches are expected, the engine returns row indices used to slice the DataFrame:

```
index.range_query(bbox) → [i, j, k, ...]  →  df[i, j, k, ...]
```

Spatial selection bypasses `map_batches`; Polars then applies any scalar filters to the
selected slice.

## Polars / PyO3 integration

The PyO3 extension copies contiguous NumPy coordinates into Rust-owned `Arc<[f64]>`
buffers. Brute force and grid share them. The KD-tree also builds a packed, reordered
coordinate buffer; the R-tree builds a packed bounding-box buffer. Each `geo-index` tree
is immutable and stored in one contiguous byte buffer.

## Join assembly

Rust join kernels return `(query_idx, target_idx)` pairs. The executor then:

1. Gathers both sides of the join by index.
2. Horizontal-concatenates them into a single DataFrame.
3. Renames any conflicting column names on the right side with a `right_` prefix.

A pushed-down `.select()` narrows both sides before the gather.
