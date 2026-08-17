# Streaming Architecture

Supported joins process the query side in morsels. Python orchestrates batches, Rust runs
spatial kernels, and Polars or PyArrow handles frames and sinks.

## Morsel design

For supported joins with a query DataFrame, the probe side is sliced into fixed-size chunks called morsels:

```
MORSEL_ROWS = 262_144  # 256K rows per morsel
```

Polars `iter_slices` slices the query DataFrame without copying its underlying buffers.
Each morsel joins independently against the complete indexed side.

## collect()

For large supported joins, `collect()` joins one morsel at a time, then concatenates the
results. Completed morsels and the final result remain in memory, and high fan-out can
still make an individual morsel large.

The `batch_size` parameter overrides the morsel size if you need finer control.

## collect_batched()

Returns one result DataFrame per morsel without assembling the complete output:

```python
for batch in sf.lazy().knn_join(query_df, "qx", "qy", k=3).collect_batched():
    process(batch)
```

## sink_parquet()

Writes each result morsel before processing the next, avoiding accumulation of the complete
join result. Per-morsel memory still depends on join fan-out.

```python
sf.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("result.parquet")
```

## lazy_source()

Exposes the morsel stream as a `pl.LazyFrame` through Polars' Python IO-source interface.
Projection, predicate, and row-limit requests are applied to emitted morsels:

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

Polars manages downstream operations, which may stream, spill, or materialize.

## Aggregate-join streaming

Supported aggregations reduce each joined morsel to per-group partials, then combine the
partial frames. The complete pair frame is not materialized; memory depends on morsel
fan-out, group count, and the number of morsels.
