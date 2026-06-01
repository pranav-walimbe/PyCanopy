# PyCanopy

Geospatial query engine with dynamic index selection. Rust core, Python bindings via PyO3/Maturin.

---

## What This Is

A standalone Python-callable geospatial query engine that automatically selects the best spatial index based on dataset statistics and query type. The novel part is the query planner — the index implementations wrap existing crates.

Not a geometry operations library. Not a distributed engine. Scope: local, single-node, in-memory, pure spatial lookup (no attribute storage).

---

## Key Decisions

**Language:** Rust core + PyO3/Maturin. R-tree and KD-tree use `geo-index` packed immutable implementations. Grid written from scratch. `rstar` not used.

**Memory:** Hard cap at ~10GB. Fail with a clear error above the threshold.

**Wire format:** GeoArrow at the Python→Rust boundary (Arrow C Data Interface), converted to `Vec<Geometry<f64>>` (geo-types) at load time. geo-types used throughout Rust core.

**Query model:** Pure spatial lookup. Returns indices into the caller's dataset. No attribute storage or compound predicates in v1.

---

## Query Planner Flow

```
query arrives
  │
  ├─ size check → error if > 10GB
  │
  ├─ selectivity estimate (query_bbox_area / total_extent_area)
  │     > 0.5 or k/N > 0.1 → bypass index, full scan
  │
  ├─ index selection
  │     N < 500                        → brute force
  │     points + kNN                   → KD-tree
  │     points + uniform + range       → grid
  │     points + clustered + range     → KD-tree
  │     polygons / mixed               → R-tree
  │     N > 1M + uniform               → grid
  │
  ├─ execution strategy
  │     range    → two-phase: MBR candidates → exact check
  │     kNN      → density-based radius estimate → bounded traversal
  │     contains → two-phase
  │
  ├─ parallelism
  │     batch queries or N > 100K → rayon
  │
  └─ result pre-allocation from selectivity estimate
```

---

## Query Types (v1)

```rust
pub enum Query {
    Knn { point: Point, k: usize, approximate: bool },
    Range { bbox: Rect },
    Contains { point: Point },
}
```

---

## Implementation Order

1. `Cargo.toml` + `pyproject.toml`
2. `src/stats/types.rs` — `DatasetStats`, `GeometryKind`
3. `src/stats/collector.rs`
4. `src/index/mod.rs` — `SpatialIndex` trait ✓
5. `src/index/brute.rs`
6. `src/index/rtree.rs` — wraps geo-index packed R-tree
7. `src/index/kdtree.rs` — wraps geo-index packed KD-tree
8. `src/index/grid.rs`
9. `src/planner/cost.rs`
10. `src/planner/calibration.rs`
11. `src/planner/selector.rs`
12. `src/query/types.rs`
13. `src/query/nearest.rs` + `range.rs`
14. `src/lib.rs` — PyO3 bindings
15. `python/` layer

---

## Deferred

- Spatial joins + join order (v2)
- Predicate pushdown (v2)
- Histogram-based selectivity estimation (v2, v1 uses `query_area / total_extent`)
- Learned/ML planner (v2)
- Out-of-core / mmap support (v2)
- **Delta buffer for incremental inserts (v2):** The packed immutable indices (geo-index RTree/KDTree) cannot accept point inserts after construction. v1 is load-once — recreate the Engine when data changes. v2 should add a write buffer: accumulate inserts into a small `delta: Vec<Geometry>`, answer queries against both the main index and a brute-force scan of the delta, and flush (rebuild the full index) when `delta.len()` exceeds a threshold (e.g. 5% of N).

---

## Style Guide

### Period convention (Rust and Python)

- Isolated single-line doc comment or docstring: **no period**
- Single-line within a multi-line block: **period**
- Last line of a multi-line block: **period**

Never use em dashes in comments or docstrings.

### Rust

- `///` rustdoc on all `pub` items
- Single-line: `/// Short description`
- Multi-line: summary line ends with period, body lines end with period
- Formatting enforced by `rustfmt` (`rustfmt.toml` at repo root)
- Linting via `cargo clippy`

### Python

- Google-style docstrings (`Args:` / `Returns:` sections, not NumPy `---` separators)
- Single-line: `"""Short description"""`
- Multi-line: summary line ends with period, `Args`/`Returns` entries end with period
- Hard dependencies imported at module level
- Optional dependencies imported inside the function that needs them (not in `try/except` at module level)
- Formatting and linting enforced by `ruff` (configured in `pyproject.toml`)

## Notes

