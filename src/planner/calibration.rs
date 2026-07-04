/// Empirical cost factors (ns per operation) for the auto-mode cost model
#[derive(Debug, Clone)]
pub struct CostFactors {
    pub knn_scan_ns_per_item: f64,     // brute-force kNN scan per item
    pub bbox_scan_ns_per_item: f64,    // brute-force range/radius box scan per item
    pub grid_build_ns_per_item: f64,   // grid build per point
    pub kdtree_build_ns_per_item: f64, // kd-tree build per point
    pub rtree_build_ns_per_item: f64,  // r-tree build per polygon
    pub kdtree_knn_ns: f64,            // kd-tree kNN probe
    pub kdtree_range_ns: f64,          // kd-tree range probe
    pub rtree_knn_ns: f64,             // r-tree kNN probe
    pub rtree_range_ns: f64,           // r-tree range probe
    pub grid_range_ns: f64,            // grid range probe
}

impl Default for CostFactors {
    fn default() -> Self {
        CostFactors {
            knn_scan_ns_per_item: 3.0,
            bbox_scan_ns_per_item: 0.6,
            grid_build_ns_per_item: 10.0,
            kdtree_build_ns_per_item: 0.5,
            rtree_build_ns_per_item: 60.0,
            kdtree_knn_ns: 25.0,
            kdtree_range_ns: 18.0,
            rtree_knn_ns: 130.0,
            rtree_range_ns: 25.0,
            grid_range_ns: 35.0,
        }
    }
}
