use geo::{Point, Rect};

/// A spatial query passed to the planner and execution layer
#[derive(Debug, Clone)]
pub enum Query {
    Knn {
        point: Point<f64>,
        k: usize,
        approximate: bool,
    },
    Range {
        bbox: Rect<f64>,
    },
    Contains {
        point: Point<f64>,
    },
}
