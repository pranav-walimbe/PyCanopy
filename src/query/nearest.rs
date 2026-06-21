use crate::index::SpatialIndex;

/// k nearest neighbours in the index, sorted nearest-first
pub fn query_nearest<I: SpatialIndex>(index: &I, qx: f64, qy: f64, k: usize) -> Vec<usize> {
    index.nearest(qx, qy, k)
}
