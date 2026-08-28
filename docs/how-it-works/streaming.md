# Morsel Execution and Aggregation

Spatial join kernels accept a query-side DataFrame and probe a complete `SpatialFrame` engine.
Morsel execution slices that query side so PyCanopy can process one join intermediate at a time.

![The query side is sliced into morsels that probe the same complete SpatialFrame before each result is consumed](../assets/diagrams/morsel-execution.png)

## When joins use morsels

`collect()` uses morsels automatically when the first join's probe side exceeds the configured
batch size. Smaller joins run once through the same join kernel and result-assembly path.

Streaming terminals call the morsel iterator for every join plan; a probe side smaller than one
batch produces one morsel. Non-join plans yield one complete DataFrame.

`batch_size` controls rows per probe slice. Polars `iter_slices()` shares the query DataFrame's
underlying buffers, and the native planner still receives the complete probe count when selecting
a kernel or access path.

Morsels never partition the indexed side. Each one queries the same fully materialized engine.

## What one morsel bounds

- Query coordinates passed to one native batch call
- Match indices and distances returned by that call
- Projected columns gathered for the matched pairs
- Post-join filters applied before the next result is produced

A morsel does not impose a row limit on its result. A high-fan-out join can produce many match rows
from one probe slice.

## Result terminals

| Terminal | Treatment of each morsel | Complete output retained? |
|:---------|:-------------------------|:--------------------------|
| `collect()` | Concatenate results into one DataFrame | Yes |
| `collect_batched()` | Yield each result DataFrame to the caller | No |
| `sink_parquet()` | Write each result DataFrame to one Parquet file | No |
| `lazy_source()` | Feed each result DataFrame into a downstream Polars plan | Depends on that plan |

### Consume batches directly

```python
for batch in zones.lazy().within_join(trips, "lon", "lat").collect_batched():
    process(batch)
```

### Write without collecting the complete result

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

`lazy_source()` first executes a one-row probe to establish its schema. Polars can then push
requested columns, predicates, and limits into the Python IO source. Polars decides whether later
operations stream, spill, or materialize.

## Grouped aggregation

![Eligible grouped joins aggregate in Rust while other plans combine per-morsel Polars partials](../assets/diagrams/morsel-aggregation.png)

### Fused native path

The fused path applies only when the plan contains one supported polygon join and no other body
nodes:

- `within_join()` or `polygon_within_distance_join()`
- Group keys come only from the target polygon frame
- `count`, `sum`, and `mean` are supported
- Numeric values for `sum` and `mean` come only from the query side

The kernel updates group state in Rust and returns compact per-group arrays rather than a match-pair
frame.

### Per-morsel partial path

Other supported plans assemble each joined morsel and reduce it with Polars. PyCanopy retains the
small partial frames, combines counts, sums, extrema, and mean components, then finalizes one
grouped DataFrame. This path also supports `min` and `max`.

## Remaining memory boundaries

!!! important
    Morsel execution bounds one join intermediate, not every allocation in the query.

- `collect()` retains the complete final output
- High-fan-out joins can produce a large single-morsel result
- The partial aggregation path retains one partial DataFrame per morsel until the final reduction
- `lazy_source()` cannot make a downstream global sort non-blocking
- `polygon_knn_join(sorted_output=True)` runs the complete probe and global ordering in Rust
- The indexed `SpatialFrame` remains fully materialized throughout execution
