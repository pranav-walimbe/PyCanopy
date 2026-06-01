use geo::Point;

use crate::index::SpatialIndex;

/// Execute a kNN query against the given index
pub fn query_nearest<I: SpatialIndex>(
    index: &I,
    point: &Point<f64>,
    k: usize,
    _approximate: bool,
) -> Vec<usize> {
    // `approximate` is reserved for future use (e.g. skip exact refinement).
    index.nearest(point, k)
}
