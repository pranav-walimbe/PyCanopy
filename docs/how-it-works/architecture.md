# Architecture Overview

PyCanopy is a spatial query layer around Polars rather than a replacement DataFrame engine.

```mermaid
flowchart LR
    Q["<b>Python API</b><br/>Declare the spatial query"]
    P["<b>PyCanopy planner</b><br/>Dictate execution order and strategy"]
    E["<b>Polars + Rust</b><br/>Run tabular and spatial work"]
    O["<b>Polars</b><br/>Return a DataFrame or write to a sink"]
    Q --> P --> E --> O
```

## How Things Fit Together

- **Polars owns tabular work:** file scans, scalar expressions, column projection, row gathering,
  joins by key, sorting, and final output
- **Rust owns spatial work:** native geometry, dataset statistics, spatial indexes, candidate
  traversal, exact geometry predicates, and parallel kernels
- **Python connects the two:** immutable plan nodes, optimization passes, execution routing, and
  morsel orchestration

## One query through the system

```python
result = (
    zones.lazy()
    .within_join(trips, x_col="lon", y_col="lat")
    .filter(pl.col("fare") > 20)
    .select("trip_id", "zone_id")
    .collect()
)
```

```mermaid
flowchart LR
    PLAN["<b>QUERY</b><br/>join → fare filter → projection"]
    OPT["<b>PLAN</b><br/>preserve join barrier · narrow gathered columns"]
    JOIN["<b>SPATIAL KERNEL</b><br/>return query and target row indices"]
    RESULT["<b>TABULAR RESULT</b><br/>gather columns · filter fare · build DataFrame"]
    PLAN --> OPT --> JOIN --> RESULT
```

The following pages trace each layer in more detail, from source ingestion through result
materialization.
