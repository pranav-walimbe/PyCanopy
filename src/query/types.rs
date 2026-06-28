use geo::{Point, Rect};

/// A spatial query passed to the planner and execution layer
#[derive(Debug, Clone)]
pub enum Query {
    /// k-nearest-neighbour query from a single point
    Knn { point: Point<f64>, k: usize },
    /// Bounding-box range query
    Range { bbox: Rect<f64> },
    /// Point-in-polygon containment query
    Contains { point: Point<f64> },
}
