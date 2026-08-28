# Morsel Execution and Aggregation

Large spatial joins divide the query-side DataFrame into fixed-size slices called **morsels**.
Each morsel probes the complete indexed `SpatialFrame`, then post-join work runs before the next
morsel begins.

![Each query morsel probes the same indexed SpatialFrame before its result is filtered, projected, and consumed](../assets/diagrams/morsel-execution.png)

## What morsels bound

- Query coordinates passed to one native join call
- Pair indices and gathered columns produced by one probe slice
- Post-join filters and projections applied before the next slice
- The result held by incremental terminals

Polars `iter_slices` reuses the query DataFrame's underlying buffers. `batch_size` controls the
morsel size, while the native planner still receives the complete probe count when choosing a
kernel or index strategy.

Morsels do **not** partition the indexed side. Every morsel queries the same fully materialized
`SpatialFrame` engine.

## Result terminals

| Terminal | What happens to each result morsel | Complete output retained? |
|:---------|:-----------------------------------|:--------------------------|
| `collect()` | Results are concatenated into one DataFrame | Yes |
| `collect_batched()` | The caller receives each DataFrame immediately | No |
| `sink_parquet()` | Each DataFrame is written to one Parquet file | No |
| `lazy_source()` | Each DataFrame enters a downstream Polars plan | Depends on that plan |

### Consume batches directly

```python
for batch in zones.lazy().within_join(trips, "lon", "lat").collect_batched():
    process(batch)
```

### Write without collecting the full result

```python
zones.lazy().polygon_knn_join(trips, "lon", "lat", k=5).sink_parquet("nearest.parquet")
```

### Continue in Polars

```python
(
    zones.lazy()
    .polygon_knn_join(trips, "lon", "lat", k=5)
    .select("trip_id", "zone_id", "distance_to_polygon")
    .lazy_source()
    .filter(pl.col("distance_to_polygon") < 100)
    .sink_parquet("nearby.parquet")
)
```

- Polars can push requested columns, predicates, and row limits into the Python IO source
- A one-row execution establishes the source schema before the downstream plan starts
- Polars decides whether later operations stream, spill, or materialize

## Grouped aggregation

Grouped spatial results take one of two paths:

![Eligible grouped joins aggregate match state directly in Rust while other plans reduce and combine per-morsel partials](../assets/diagrams/morsel-aggregation.png)

### Fused native path

- Applies to a single point-in-polygon or point-to-polygon-distance join
- Requires grouping keys from the target polygon frame
- Supports count, sum, and mean over eligible query-side numeric columns
- Updates group state in Rust without constructing match-pair rows

### Partial aggregation path

- Handles other supported shapes, including min and max
- Reduces each joined morsel to a small partial DataFrame
- Combines counts, sums, extrema, and mean components after all morsels finish
- Avoids retaining the complete pair frame

## Memory boundaries

!!! important
    Morsel execution bounds one join batch, not every allocation in the query.

- `collect()` retains completed morsels and constructs the final output
- High-fan-out joins can still produce a large result from one morsel
- Partial-aggregate memory grows with group count and morsel count
- `lazy_source()` cannot prevent downstream blocking operations such as a global sort
- `polygon_knn_join(sorted_output=True)` processes the full probe and global sort in Rust
- The indexed `SpatialFrame` remains fully materialized throughout execution
