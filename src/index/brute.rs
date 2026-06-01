use geo::{Geometry, Point, Rect};

use crate::index::{geom_bbox, SpatialIndex};

#[cfg(test)]
pub(crate) fn five_point_grid() -> Vec<Geometry<f64>> {
    // Indices: 0=(0,0) 1=(1,0) 2=(2,0) 3=(0,1) 4=(1,1)
    // Query (1.2, 0.1): distances² → 1:[0.05] 2:[0.65] 4:[0.85] 0:[1.45] 3:[2.25]
    vec![
        Geometry::Point(Point::new(0.0, 0.0)),
        Geometry::Point(Point::new(1.0, 0.0)),
        Geometry::Point(Point::new(2.0, 0.0)),
        Geometry::Point(Point::new(0.0, 1.0)),
        Geometry::Point(Point::new(1.0, 1.0)),
    ]
}

/// Linear scan index, used for small datasets or high-selectivity queries
pub struct BruteForce {
    bboxes: Vec<Option<(f64, f64, f64, f64)>>,
    // Points are common enough that we cache their coords for fast distance.
    point_coords: Vec<Option<(f64, f64)>>,
}

impl SpatialIndex for BruteForce {
    fn build(geometries: &[Geometry<f64>]) -> Self {
        let bboxes = geometries.iter().map(geom_bbox).collect();
        let point_coords = geometries
            .iter()
            .map(|g| match g {
                Geometry::Point(p) => Some((p.x(), p.y())),
                _ => None,
            })
            .collect();
        BruteForce {
            bboxes,
            point_coords,
        }
    }

    fn nearest(&self, point: &Point<f64>, k: usize) -> Vec<usize> {
        let px = point.x();
        let py = point.y();
        let mut dists: Vec<(usize, f64)> = self
            .bboxes
            .iter()
            .enumerate()
            .map(|(i, bbox)| {
                let d = if let Some((x, y)) = self.point_coords[i] {
                    (x - px).powi(2) + (y - py).powi(2)
                } else if let Some((min_x, min_y, max_x, max_y)) = bbox {
                    let cx = (min_x + max_x) / 2.0;
                    let cy = (min_y + max_y) / 2.0;
                    (cx - px).powi(2) + (cy - py).powi(2)
                } else {
                    f64::INFINITY
                };
                (i, d)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn range(&self, bbox: &Rect<f64>) -> Vec<usize> {
        let (qmin_x, qmin_y, qmax_x, qmax_y) =
            (bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y);
        self.bboxes
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                b.and_then(|(min_x, min_y, max_x, max_y)| {
                    if max_x >= qmin_x && min_x <= qmax_x && max_y >= qmin_y && min_y <= qmax_y {
                        Some(i)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    fn contains(&self, point: &Point<f64>) -> Vec<usize> {
        let (px, py) = (point.x(), point.y());
        self.bboxes
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                b.and_then(|(min_x, min_y, max_x, max_y)| {
                    if px >= min_x && px <= max_x && py >= min_y && py <= max_y {
                        Some(i)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn nearest_returns_single_closest() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(1.2, 0.1), 1), vec![1]);
    }

    #[test]
    fn nearest_k_two_returns_correct_pair() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        assert_eq!(sorted(idx.nearest(&Point::new(1.2, 0.1), 2)), vec![1, 2]);
    }

    #[test]
    fn nearest_k_larger_than_n_returns_all() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(0.0, 0.0), 100).len(), 5);
    }

    #[test]
    fn range_returns_correct_points() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 0.0, y: 0.0 },
            geo::coord! { x: 1.5, y: 0.5 },
        );
        assert_eq!(sorted(idx.range(&bbox)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 5.0, y: 5.0 },
            geo::coord! { x: 10.0, y: 10.0 },
        );
        assert!(idx.range(&bbox).is_empty());
    }
}
