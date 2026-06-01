use geo::Rect;

/// Dataset statistics used by the query planner to select a spatial index
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub n: usize,
    pub kind: GeometryKind,
    pub extent: Option<Rect<f64>>,
    pub distribution: Distribution,
    /// N / extent_area
    pub mean_density: f64,
}

/// Dominant geometry type in the dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryKind {
    Point,
    LineString,
    Polygon,
    /// More than one geometry type present
    Mixed,
    /// Dataset is empty
    Empty,
}

/// Spatial distribution of point geometries estimated via grid CV test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    Uniform,
    Clustered,
    /// Not enough data to classify or geometry kind is not Point
    Unknown,
}

impl DatasetStats {
    pub fn extent_area(&self) -> f64 {
        self.extent
            .map(|r| {
                let w = r.max().x - r.min().x;
                let h = r.max().y - r.min().y;
                (w * h).max(0.0)
            })
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{coord, Rect};

    fn rect(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Rect<f64> {
        Rect::new(coord! { x: min_x, y: min_y }, coord! { x: max_x, y: max_y })
    }

    #[test]
    fn extent_area_basic() {
        let stats = DatasetStats {
            n: 10,
            kind: GeometryKind::Point,
            extent: Some(rect(0.0, 0.0, 10.0, 10.0)),
            distribution: Distribution::Uniform,
            mean_density: 0.1,
        };
        assert!((stats.extent_area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn extent_area_none_returns_zero() {
        let stats = DatasetStats {
            n: 0,
            kind: GeometryKind::Empty,
            extent: None,
            distribution: Distribution::Unknown,
            mean_density: 0.0,
        };
        assert_eq!(stats.extent_area(), 0.0);
    }

    #[test]
    fn extent_area_non_square() {
        let stats = DatasetStats {
            n: 5,
            kind: GeometryKind::Point,
            extent: Some(rect(0.0, 0.0, 4.0, 10.0)),
            distribution: Distribution::Uniform,
            mean_density: 0.125,
        };
        assert!((stats.extent_area() - 40.0).abs() < 1e-10);
    }
}
