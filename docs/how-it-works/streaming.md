# Streaming Architecture

PyCanopy bounds memory on large join results through a morsel-based streaming design. All streaming is implemented in Python on top of Polars' native lazy infrastructure.

## Morsel design

The probe side of any join is sliced into fixed-size chunks called morsels:

```
MORSEL_ROWS = 262_144  # 256K rows per morsel
```

Morsels are produced via `iter_slices` — a zero-copy operation that yields views into the probe DataFrame without copying data. Each morsel is joined independently against the full index, and its result is either yielded, accumulated, or written to disk depending on the collection method.

## collect()

`collect()` automatically streams when the probe DataFrame exceeds a threshold. It accumulates morsel results in memory and concatenates them at the end. For very large probes this bounds the transient memory overhead to one morsel at a time during the join phase, while the final result still materialises fully.

The `batch_size` parameter overrides the morsel size if you need finer control.

## collect_batched()

Returns an iterator of result DataFrames, one per morsel. The caller receives results incrementally and never holds the full output in memory:

```python
for batch in sf.lazy().knn_join(query_df, "qx", "qy", k=3).collect_batched():
    process(batch)
```

Useful for pipelines that can process results as they arrive, or for writing to multiple sinks.

## sink_parquet()

Streams the join result directly to a Parquet file. Each morsel is processed and written before the next is read, so peak memory is bounded to one morsel regardless of output size:

```python
sf.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("result.parquet")
```

## lazy_source()

Exposes the join result as a native Polars `LazyFrame` source. This lets you fuse spatial join output with downstream Polars operations — sorts, sinks, further filters — into a single spilling pipeline:

```python
(
    sf.lazy()
    .polygon_knn_join(trips, "lon", "lat", k=5)
    .select(["trip_id", "building_id", "distance_to_polygon"])
    .lazy_source()
    .sort("distance_to_polygon")
    .sink_parquet("nearest_sorted.parquet")
)
```

Polars handles spilling to disk for the sort if the result exceeds memory, so the entire pipeline — join, select, sort, write — never requires the full result to be in RAM at once.

## Aggregate-join streaming

`.group_by(keys).agg(...)` reduces over the morsel stream using associative partial aggregations. Each morsel produces per-group partials (counts, sums, etc.) that are combined across morsels at the end. The full pair frame never materialises — only the per-group accumulators are held in memory, which are bounded by the number of unique groups rather than the number of join pairs.
