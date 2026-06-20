use crate::index::SpatialIndex;

/// Execute a kNN query against the given index
pub fn query_nearest<I: SpatialIndex>(index: &I, qx: f64, qy: f64, k: usize) -> Vec<usize> {
    index.nearest(qx, qy, k)
}
