pub mod brute;
pub mod grid;
pub mod kdtree;
pub mod rtree;

use std::sync::Arc;

/// Common interface for all spatial index backends.
/// Coordinates are passed as Arc<[f64]> so indexes can share Engine's allocation
/// without copying — storing an Arc<[f64]> is an atomic refcount bump, not a memcpy.
pub trait SpatialIndex: Send + Sync {
    fn build(xs: Arc<[f64]>, ys: Arc<[f64]>) -> Self
    where
        Self: Sized;
    /// Indices of the k nearest points to (qx, qy), sorted nearest-first
    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize>;
    /// Indices of all geometries whose bounding box intersects [min_x, max_x] × [min_y, max_y]
    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize>;
}
