/// Empirical cost factors (ns per operation) for the auto-mode cost model.
/// Defaults are rough hand-calibrated constants. They can be tuned against the
/// ops benchmark without changing the planner logic.
#[derive(Debug, Clone)]
pub struct CostFactors {
    /// Brute-force scan cost per (probe, item) pair
    pub scan_ns_per_item: f64,
    /// Index build cost per item (multiplied by log2(n) for tree indexes)
    pub build_ns_per_item: f64,
    /// R-tree probe cost per node visited or result reported
    pub rtree_ns_per_result: f64,
    /// KD-tree probe cost per node visited or result reported
    pub kdtree_ns_per_result: f64,
    /// Grid probe cost per cell visited or result reported
    pub grid_ns_per_result: f64,
}

impl Default for CostFactors {
    fn default() -> Self {
        CostFactors {
            scan_ns_per_item: 5.0,
            build_ns_per_item: 25.0,
            rtree_ns_per_result: 20.0,
            kdtree_ns_per_result: 15.0,
            grid_ns_per_result: 10.0,
        }
    }
}
