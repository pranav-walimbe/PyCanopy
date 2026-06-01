/// Empirical cost factors (ns per operation) reserved for v2 cost-model calibration
#[derive(Debug, Clone)]
pub struct CostFactors {
    pub scan_ns_per_item: f64,
    pub rtree_ns_per_result: f64,
    pub kdtree_ns_per_result: f64,
    pub grid_ns_per_result: f64,
}

impl Default for CostFactors {
    fn default() -> Self {
        CostFactors {
            scan_ns_per_item: 5.0,
            rtree_ns_per_result: 20.0,
            kdtree_ns_per_result: 15.0,
            grid_ns_per_result: 10.0,
        }
    }
}
