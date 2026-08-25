//! Spatial index backends: KD-tree, R-tree, grid, and brute-force implementations.

pub mod brute;
pub mod grid;
pub mod kdtree;
pub mod rtree;

use crate::Shared;

/// Common interface for all spatial index backends.
/// Coordinates arrive as Shared<f64> so an index shares the Engine allocation by refcount.
pub trait SpatialIndex: Send + Sync {
    /// Build an index over the given coordinate arrays
    fn build(xs: Shared<f64>, ys: Shared<f64>) -> Self
    where
        Self: Sized;
    /// Indices of the k nearest entries to (qx, qy), sorted nearest-first.
    ///
    /// Point datasets rank by exact point distance. Polygon datasets rank by the squared
    /// point-to-MBR distance, which only lower-bounds the exact point-to-polygon distance,
    /// so callers that need exact ranking must refine the candidates themselves.
    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize>;
    /// Indices of all geometries whose bounding box intersects [min_x, max_x] × [min_y, max_y]
    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize>;
    /// `nearest` into a caller owned buffer, so a per-query kernel can reuse one allocation
    fn nearest_into(&self, qx: f64, qy: f64, k: usize, out: &mut Vec<usize>) {
        out.clear();
        out.extend(self.nearest(qx, qy, k));
    }
    /// `range` into a caller owned buffer, so a per-query kernel can reuse one allocation
    fn range_into(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64, out: &mut Vec<usize>) {
        out.clear();
        out.extend(self.range(min_x, min_y, max_x, max_y));
    }
}

/// Squared distance from a point to an axis-aligned box `[min_x, min_y, max_x, max_y]`,
/// zero when the point is inside. Lower-bounds the exact distance to any geometry the box covers.
#[inline]
pub(crate) fn point_box_dist2(px: f64, py: f64, b: &[f64; 4]) -> f64 {
    let dx = (b[0] - px).max(0.0).max(px - b[2]);
    let dy = (b[1] - py).max(0.0).max(py - b[3]);
    dx * dx + dy * dy
}
