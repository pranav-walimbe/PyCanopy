pub mod brute;
pub mod grid;
pub mod kdtree;
pub mod rtree;

use geo::{BoundingRect, Geometry, Point, Rect};

/// Common interface implemented by all spatial index backends
pub trait SpatialIndex: Send + Sync {
    fn build(geometries: &[Geometry<f64>]) -> Self
    where
        Self: Sized;
    /// Returns original insertion indices sorted nearest-first
    fn nearest(&self, point: &Point<f64>, k: usize) -> Vec<usize>;
    /// Returns original insertion indices of geometries whose MBR intersects bbox
    fn range(&self, bbox: &Rect<f64>) -> Vec<usize>;
    /// Returns original insertion indices of geometries whose MBR contains the point
    fn contains(&self, point: &Point<f64>) -> Vec<usize>;
}

/// Returns (min_x, min_y, max_x, max_y) for any geometry, or None for empty geometries
pub(crate) fn geom_bbox(geom: &Geometry<f64>) -> Option<(f64, f64, f64, f64)> {
    geom.bounding_rect()
        .map(|r| (r.min().x, r.min().y, r.max().x, r.max().y))
}

/// Centroid approximation for non-point geometries used by KD-tree and grid
pub(crate) fn geom_center(geom: &Geometry<f64>) -> (f64, f64) {
    match geom {
        Geometry::Point(p) => (p.x(), p.y()),
        _ => geom_bbox(geom)
            .map(|(min_x, min_y, max_x, max_y)| ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0))
            .unwrap_or((0.0, 0.0)),
    }
}
