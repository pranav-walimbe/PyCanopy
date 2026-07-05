//! Query enum describing the spatial query passed to the planner and executor.

use geo::{Point, Rect};

/// A spatial query passed to the planner and execution layer
#[derive(Debug, Clone)]
pub enum Query {
    Knn { point: Point<f64>, k: usize }, // k-nearest-neighbour query from a single point
    Range { bbox: Rect<f64> },           // bounding-box range query
    Contains { point: Point<f64> },      // point-in-polygon containment query
}
