//! Packed immutable KD-tree index optimised for point datasets.

use std::f64::consts::PI;
use std::sync::Arc;

use geo_index::kdtree::{KDTree, KDTreeBuilder, KDTreeIndex};

use crate::index::SpatialIndex;
use crate::stats::types::SpatialHistogram;

/// Packed immutable KD-tree backed by geo-index, optimised for point datasets.
///
/// xs/ys are shared Arcs from the Engine, so no coordinate data is copied at
/// build time. geo-index makes one internal sorted copy (unavoidable for tree
/// traversal). The xs/ys Arcs are kept for the kNN distance refinement step.
pub struct PackedKdTree {
    tree: KDTree<f64>,
    xs: Arc<[f64]>,
    ys: Arc<[f64]>,
    extent_area: f64,
    histogram: Option<SpatialHistogram>,
}

impl SpatialIndex for PackedKdTree {
    fn build(xs: Arc<[f64]>, ys: Arc<[f64]>) -> Self {
        let n = xs.len();
        let mut builder = KDTreeBuilder::<f64>::new(n as u32);

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            builder.add(x, y);
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
            xs,
            ys,
            extent_area,
            histogram: None,
        }
    }

    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize> {
        let n = self.xs.len();
        if n == 0 {
            return vec![];
        }
        let k = k.min(n);

        let density = self
            .histogram
            .as_ref()
            .and_then(|h| h.local_density(qx, qy))
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
            let hits = self.tree.within(qx, qy, radius);
            if hits.len() >= k || radius > 1.0e15 {
                let mut with_dist: Vec<(u32, f64)> = hits
                    .iter()
                    .map(|&i| {
                        let dx = self.xs[i as usize] - qx;
                        let dy = self.ys[i as usize] - qy;
                        (i, dx * dx + dy * dy)
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

    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        self.tree
            .range(min_x, min_y, max_x, max_y)
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

    /// Heap bytes allocated by this index, excluding coordinates shared with the Engine.
    ///
    /// Counts the geo-index internal flat buffer plus the histogram clone if present.
    /// xs/ys Arcs are shared from the Engine and are not counted.
    pub fn heap_bytes(&self) -> usize {
        self.tree.metadata().data_buffer_length()
            + self.histogram.as_ref().map_or(0, |h| h.heap_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::five_point_grid;

    fn build(xs: Vec<f64>, ys: Vec<f64>) -> PackedKdTree {
        PackedKdTree::build(xs.into(), ys.into())
    }

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn nearest_returns_single_closest() {
        let (xs, ys) = five_point_grid();
        assert_eq!(build(xs, ys).nearest(1.2, 0.1, 1), vec![1]);
    }

    #[test]
    fn nearest_k_two_returns_correct_pair() {
        let (xs, ys) = five_point_grid();
        assert_eq!(sorted(build(xs, ys).nearest(1.2, 0.1, 2)), vec![1, 2]);
    }

    #[test]
    fn nearest_k_larger_than_n_returns_all() {
        let (xs, ys) = five_point_grid();
        assert_eq!(build(xs, ys).nearest(0.0, 0.0, 100).len(), 5);
    }

    #[test]
    fn range_returns_correct_points() {
        let (xs, ys) = five_point_grid();
        assert_eq!(sorted(build(xs, ys).range(0.0, 0.0, 1.5, 0.5)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let (xs, ys) = five_point_grid();
        assert!(build(xs, ys).range(5.0, 5.0, 10.0, 10.0).is_empty());
    }
}
