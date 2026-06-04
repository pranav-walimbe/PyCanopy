use std::sync::Arc;

use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex};

use crate::index::SpatialIndex;

/// Packed immutable R-tree backed by geo-index with Hilbert sort.
///
/// geo-index stores coordinates internally (one unavoidable copy at build time).
/// The xs/ys Arcs passed to build() are not retained — they are iterated once
/// to feed the builder and then dropped.
/// For polygon datasets use build_polygons, which computes per-polygon MBRs.
pub struct PackedRTree {
    tree: RTree<f64>,
}

impl SpatialIndex for PackedRTree {
    /// Build from point coordinates. Each point becomes a degenerate bbox.
    fn build(xs: Arc<[f64]>, ys: Arc<[f64]>) -> Self {
        let n = xs.len() as u32;
        let mut builder = RTreeBuilder::<f64>::new(n);
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            builder.add(x, y, x, y);
        }
        // xs and ys Arcs drop here — geo-index owns its internal copy.
        PackedRTree {
            tree: builder.finish::<HilbertSort>(),
        }
    }

    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize> {
        self.tree
            .neighbors(qx, qy, Some(k), None)
            .iter()
            .map(|&i| i as usize)
            .collect()
    }

    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        self.tree
            .search(min_x, min_y, max_x, max_y)
            .iter()
            .map(|&i| i as usize)
            .collect()
    }
}

impl PackedRTree {
    /// Build from polygon ring coordinates. Each polygon's MBR is computed from its ring vertices.
    pub fn build_polygons(xs: &[f64], ys: &[f64], ring_offsets: &[i64]) -> Self {
        let n_polys = ring_offsets.len().saturating_sub(1);
        let mut builder = RTreeBuilder::<f64>::new(n_polys as u32);
        for i in 0..n_polys {
            let start = ring_offsets[i] as usize;
            let end = ring_offsets[i + 1] as usize;
            if start >= end {
                builder.add(0.0, 0.0, 0.0, 0.0);
                continue;
            }
            let ring_xs = &xs[start..end];
            let ring_ys = &ys[start..end];
            let min_x = ring_xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let min_y = ring_ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_x = ring_xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let max_y = ring_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            builder.add(min_x, min_y, max_x, max_y);
        }
        PackedRTree {
            tree: builder.finish::<HilbertSort>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::five_point_grid;

    fn build(xs: Vec<f64>, ys: Vec<f64>) -> PackedRTree {
        PackedRTree::build(xs.into(), ys.into())
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
    fn range_returns_correct_points() {
        let (xs, ys) = five_point_grid();
        assert_eq!(sorted(build(xs, ys).range(0.0, 0.0, 1.5, 0.5)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let (xs, ys) = five_point_grid();
        assert!(build(xs, ys).range(5.0, 5.0, 10.0, 10.0).is_empty());
    }

    #[test]
    fn range_single_result() {
        let (xs, ys) = five_point_grid();
        assert_eq!(sorted(build(xs, ys).range(0.5, 0.5, 1.5, 1.5)), vec![4]);
    }
}
