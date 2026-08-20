//! Linear-scan brute-force index for small datasets or high-selectivity queries.

use std::sync::Arc;

use crate::index::{point_box_dist2, SpatialIndex};

/// Linear scan index, used for small datasets or high-selectivity queries.
///
/// Stores per-geometry bounding boxes, which serve both MBR filtering and nearest ranking.
/// For point datasets the bbox is degenerate (min == max == coordinate) and every array is a
/// shared Arc from the Engine, so no data is copied and box distance is exact point distance.
/// For polygon datasets the arrays are derived from ring coordinates.
pub struct BruteForce {
    /// Per-geometry bounding boxes.
    /// For point datasets all four are Arc::clone of the Engine's xs/ys (shared, zero cost).
    /// For polygon datasets these are new allocations derived from ring coords.
    bbox_min_x: Arc<[f64]>,
    bbox_min_y: Arc<[f64]>,
    bbox_max_x: Arc<[f64]>,
    bbox_max_y: Arc<[f64]>,
}

impl SpatialIndex for BruteForce {
    fn build(xs: Arc<[f64]>, ys: Arc<[f64]>) -> Self {
        BruteForce {
            bbox_min_x: Arc::clone(&xs),
            bbox_min_y: Arc::clone(&ys),
            bbox_max_x: xs,
            bbox_max_y: ys,
        }
    }

    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize> {
        // Rank by point-to-MBR distance, which is exact for the degenerate boxes of a point
        // dataset and the lower bound the polygon refinement pass needs.
        let n = self.bbox_min_x.len();
        let k = k.min(n);
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let b = [
                    self.bbox_min_x[i],
                    self.bbox_min_y[i],
                    self.bbox_max_x[i],
                    self.bbox_max_y[i],
                ];
                (i, point_box_dist2(qx, qy, &b))
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        (0..self.bbox_min_x.len())
            .filter(|&i| {
                self.bbox_max_x[i] >= min_x
                    && self.bbox_min_x[i] <= max_x
                    && self.bbox_max_y[i] >= min_y
                    && self.bbox_min_y[i] <= max_y
            })
            .collect()
    }
}

impl BruteForce {
    /// Heap bytes allocated by this index, excluding coordinates shared with the Engine.
    ///
    /// Point datasets alias the Engine's xs/ys in all four bbox arcs, so marginal cost is zero.
    /// Polygon datasets allocate new MBR arrays of 4 * N * 8 bytes.
    pub fn heap_bytes(&self) -> usize {
        if Arc::ptr_eq(&self.bbox_min_x, &self.bbox_max_x) {
            0
        } else {
            self.bbox_min_x.len() * std::mem::size_of::<f64>() * 4
        }
    }

    /// Build from two-level polygon ring arrays. Computes per-polygon MBRs from exterior
    /// rings only. Holes do not expand the MBR.
    pub fn build_polygons(
        xs: &[f64],
        ys: &[f64],
        ring_offsets: &[i64],
        poly_offsets: &[i64],
    ) -> Self {
        let n_polys = poly_offsets.len().saturating_sub(1);
        let mut mn_xs = Vec::with_capacity(n_polys);
        let mut mn_ys = Vec::with_capacity(n_polys);
        let mut mx_xs = Vec::with_capacity(n_polys);
        let mut mx_ys = Vec::with_capacity(n_polys);

        for &ext_ring_i64 in poly_offsets.iter().take(n_polys) {
            // The MBR comes from the exterior ring only
            let ext_ring = ext_ring_i64 as usize;
            let start = ring_offsets[ext_ring] as usize;
            let end = ring_offsets[ext_ring + 1] as usize;
            if start >= end {
                mn_xs.push(0.0);
                mn_ys.push(0.0);
                mx_xs.push(0.0);
                mx_ys.push(0.0);
                continue;
            }
            let ring_xs = &xs[start..end];
            let ring_ys = &ys[start..end];
            let (mn_x, mn_y, mx_x, mx_y) = ring_xs.iter().zip(ring_ys.iter()).fold(
                (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ),
                |(lo_x, lo_y, hi_x, hi_y), (&x, &y)| {
                    (lo_x.min(x), lo_y.min(y), hi_x.max(x), hi_y.max(y))
                },
            );
            mn_xs.push(mn_x);
            mn_ys.push(mn_y);
            mx_xs.push(mx_x);
            mx_ys.push(mx_y);
        }

        BruteForce {
            bbox_min_x: mn_xs.into(),
            bbox_min_y: mn_ys.into(),
            bbox_max_x: mx_xs.into(),
            bbox_max_y: mx_ys.into(),
        }
    }
}

#[cfg(test)]
pub(crate) fn five_point_grid() -> (Vec<f64>, Vec<f64>) {
    // Points: 0=(0,0), 1=(1,0), 2=(2,0), 3=(0,1), 4=(1,1).
    // Query (1.2, 0.1): distances² → 1:[0.05] 2:[0.65] 4:[0.85] 0:[1.45] 3:[2.25].
    (vec![0.0, 1.0, 2.0, 0.0, 1.0], vec![0.0, 0.0, 0.0, 1.0, 1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(xs: Vec<f64>, ys: Vec<f64>) -> BruteForce {
        BruteForce::build(xs.into(), ys.into())
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

    #[test]
    fn polygon_nearest_ranks_by_mbr_not_centroid() {
        // A wide sliver spanning x 0..10 at y 0, and a small square at x 4..5, y 3..4. From
        // (5, 0.5) the sliver's edge is 0.5 away but its centroid is 5 away, which centroid
        // ranking put second. Ranking by MBR is the lower bound polygon refinement relies on.
        let xs = vec![
            0.0, 10.0, 10.0, 0.0, 0.0, // sliver ring
            4.0, 5.0, 5.0, 4.0, 4.0, // square ring
        ];
        let ys = vec![
            -0.1, -0.1, 0.1, 0.1, -0.1, // sliver ring
            3.0, 3.0, 4.0, 4.0, 3.0, // square ring
        ];
        let ring_offsets = vec![0, 5, 10];
        let poly_offsets = vec![0, 1, 2];
        let index = BruteForce::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);
        assert_eq!(index.nearest(5.0, 0.5, 2), vec![0, 1]);
    }
}
