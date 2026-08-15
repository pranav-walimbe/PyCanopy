# Streaming Architecture

PyCanopy reduces memory growth for large join results through a morsel-based streaming design. All streaming is implemented in Python on top of Polars' lazy infrastructure.

## Morsel design

For supported joins with a query DataFrame, the probe side is sliced into fixed-size chunks called morsels:

```
MORSEL_ROWS = 262_144  # 256K rows per morsel
```

Morsels are produced via `iter_slices`, a zero-copy operation that yields views into the probe DataFrame without copying data. Each morsel is joined independently against the full index, and its result is either yielded, accumulated, or written to disk depending on the collection method.

## collect()

`collect()` automatically streams when the probe DataFrame exceeds a threshold. It accumulates morsel results in memory and concatenates them at the end. For very large probes this bounds the transient memory overhead to one morsel at a time during the join phase, while the final result still materializes fully.

The `batch_size` parameter overrides the morsel size if you need finer control.

## collect_batched()

Returns an iterator of result DataFrames, one per morsel. The caller receives results incrementally and never holds the full output in memory:

```python
for batch in sf.lazy().knn_join(query_df, "qx", "qy", k=3).collect_batched():
    process(batch)
```

Useful for pipelines that can process results as they arrive, or for writing to multiple sinks.

## sink_parquet()

Streams the join result directly to a Parquet file. Each probe morsel is processed and written before the next one begins, preventing accumulation of the complete join result. Memory use for an individual morsel still depends on join fan-out: a morsel that produces many matches can produce a large result batch.

```python
sf.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("result.parquet")
```

## lazy_source()

`lazy_source()` exposes the spatial result through Polars' Python IO-source interface and returns a `pl.LazyFrame`. This allows downstream Polars operations to consume PyCanopy's morsel stream and enables projection, predicate, and row-limit pushdown into the source:

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

Downstream operations such as sorting are managed by Polars and may stream or spill depending on the Polars version, execution engine, and configuration. Memory use is therefore not guaranteed to remain at one morsel.

## Aggregate-join streaming

`.group_by(keys).agg(...)` reduces over the morsel stream using associative partial aggregations. Each morsel produces per-group partials (counts, sums, etc.) that are combined across morsels at the end. The full pair frame never materializes. Only the per-group accumulators are held in memory, bounded by the number of unique groups rather than the number of join pairs.
