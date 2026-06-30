# Aggregations

Aggregation specs used with `.group_by(...).agg(...)`. Each spec reduces over a streamed spatial join without materialising the full pair frame.

```python
import pycanopy as pc

result = (
    zones.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .group_by(["zone_id"])
    .agg(
        n=pc.agg.count(),
        total_fare=pc.agg.sum("fare"),
        avg_fare=pc.agg.mean("fare"),
        min_fare=pc.agg.min("fare"),
        max_fare=pc.agg.max("fare"),
    )
)
```

::: pycanopy.agg
