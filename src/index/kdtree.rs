use std::f64::consts::PI;

use geo::{Geometry, Point, Rect};
use geo_index::kdtree::{KDTree, KDTreeBuilder, KDTreeIndex};

use crate::index::{geom_center, SpatialIndex};
use crate::stats::types::SpatialHistogram;

/// Packed immutable KD-tree backed by geo-index, optimised for point datasets
pub struct PackedKdTree {
    tree: KDTree<f64>,
    coords: Vec<(f64, f64)>,
    extent_area: f64,
    histogram: Option<SpatialHistogram>,
}

impl SpatialIndex for PackedKdTree {
    fn build(geometries: &[Geometry<f64>]) -> Self {
        let n = geometries.len();
        let mut builder = KDTreeBuilder::<f64>::new(n as u32);
        let mut coords = Vec::with_capacity(n);

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for geom in geometries {
            let (x, y) = geom_center(geom);
            builder.add(x, y);
            coords.push((x, y));
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let extent_area = if min_x.is_finite() {
            ((max_x - min_x) * (max_y - min_y)).max(0.0)
        } else {
            0.0
        };

        PackedKdTree {
            tree: builder.finish(),
            coords,
            extent_area,
            histogram: None,
        }
    }

    fn nearest(&self, point: &Point<f64>, k: usize) -> Vec<usize> {
        let n = self.coords.len();
        if n == 0 {
            return vec![];
        }
        let k = k.min(n);

        // Use local density from histogram when available; fall back to global density.
        let density = self
            .histogram
            .as_ref()
            .and_then(|h| h.local_density(point.x(), point.y()))
            .unwrap_or_else(|| {
                if self.extent_area > 0.0 {
                    n as f64 / self.extent_area
                } else {
                    1.0
                }
            });
        let mut radius = (k as f64 / (PI * density)).sqrt() * 1.5;
        if radius <= 0.0 {
            radius = 1.0;
        }

        loop {
            let hits = self.tree.within(point.x(), point.y(), radius);
            if hits.len() >= k || radius > 1.0e15 {
                let mut with_dist: Vec<(u32, f64)> = hits
                    .iter()
                    .map(|&i| {
                        let (cx, cy) = self.coords[i as usize];
                        let d = (cx - point.x()).powi(2) + (cy - point.y()).powi(2);
                        (i, d)
                    })
                    .collect();
                with_dist.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                return with_dist
                    .into_iter()
                    .take(k)
                    .map(|(i, _)| i as usize)
                    .collect();
            }
            radius *= 2.0;
        }
    }

    fn range(&self, bbox: &Rect<f64>) -> Vec<usize> {
        self.tree
            .range(bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y)
            .iter()
            .map(|&i| i as usize)
            .collect()
    }

    fn contains(&self, point: &Point<f64>) -> Vec<usize> {
        // For point datasets, contains = exact match within float epsilon.
        self.tree
            .within(point.x(), point.y(), f64::EPSILON * 1000.0)
            .iter()
            .map(|&i| i as usize)
            .collect()
    }
}

impl PackedKdTree {
    /// Inject the spatial histogram for local-density radius estimation in kNN
    pub fn set_histogram(&mut self, histogram: Option<SpatialHistogram>) {
        self.histogram = histogram;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::five_point_grid;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn nearest_returns_single_closest() {
        let geoms = five_point_grid();
        let idx = PackedKdTree::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(1.2, 0.1), 1), vec![1]);
    }

    #[test]
    fn nearest_k_two_returns_correct_pair() {
        let geoms = five_point_grid();
        let idx = PackedKdTree::build(&geoms);
        assert_eq!(sorted(idx.nearest(&Point::new(1.2, 0.1), 2)), vec![1, 2]);
    }

    #[test]
    fn nearest_k_larger_than_n_returns_all() {
        let geoms = five_point_grid();
        let idx = PackedKdTree::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(0.0, 0.0), 100).len(), 5);
    }

    #[test]
    fn range_returns_correct_points() {
        let geoms = five_point_grid();
        let idx = PackedKdTree::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 0.0, y: 0.0 },
            geo::coord! { x: 1.5, y: 0.5 },
        );
        assert_eq!(sorted(idx.range(&bbox)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let geoms = five_point_grid();
        let idx = PackedKdTree::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 5.0, y: 5.0 },
            geo::coord! { x: 10.0, y: 10.0 },
        );
        assert!(idx.range(&bbox).is_empty());
    }
}
