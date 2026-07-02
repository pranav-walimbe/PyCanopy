/// Empirical cost factors (ns per operation) for the auto-mode cost model
#[derive(Debug, Clone)]
pub struct CostFactors {
    pub scan_ns_per_item: f64,
    pub build_ns_per_item: f64,
    pub kdtree_knn_ns: f64,
    pub kdtree_range_ns: f64,
    pub rtree_knn_ns: f64,
    pub rtree_range_ns: f64,
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
