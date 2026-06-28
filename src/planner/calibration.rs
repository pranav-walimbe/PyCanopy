/// Empirical cost factors (ns per operation) for the auto-mode cost model.
/// Defaults are rough hand-calibrated constants. They can be tuned against the
/// ops benchmark without changing the planner logic.
#[derive(Debug, Clone)]
pub struct CostFactors {
    /// Brute-force scan cost per (probe, item) pair
    pub scan_ns_per_item: f64,
    /// Index build cost per item (multiplied by log2(n) for tree indexes)
    pub build_ns_per_item: f64,
    /// KD-tree probe cost per unit for kNN queries
    pub kdtree_knn_ns: f64,
    /// KD-tree probe cost per unit for range queries
    pub kdtree_range_ns: f64,
    /// R-tree probe cost per unit for kNN queries
    pub rtree_knn_ns: f64,
    /// R-tree probe cost per unit for range queries
    pub rtree_range_ns: f64,
    /// Grid probe cost per result (range only; kNN always routes to KD-tree or R-tree)
    pub grid_range_ns: f64,
}

impl Default for CostFactors {
    fn default() -> Self {
        CostFactors {
            scan_ns_per_item: 1.0,
            build_ns_per_item: 1.0,
            kdtree_knn_ns: 30.0,
            kdtree_range_ns: 1.0,
            rtree_knn_ns: 930.0,
            rtree_range_ns: 20.0,
            grid_range_ns: 20.0,
        }
    }
}
