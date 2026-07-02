use std::sync::Arc;

use crate::index::SpatialIndex;

/// Linear scan index, used for small datasets or high-selectivity queries.
///
/// Stores per-geometry bounding boxes for MBR filtering. For point datasets the
/// bbox is degenerate (min == max == coordinate) and xs/ys are shared Arcs from
/// the Engine, so no data is copied. For polygon datasets, bbox arrays are derived
/// from ring coordinates and centroids are stored for nearest queries.
pub struct BruteForce {
    /// Representative point coordinates: actual coords for points, centroids for polygons
    xs: Arc<[f64]>,
    ys: Arc<[f64]>,
    /// Per-geometry bounding boxes for MBR filtering.
    /// For point datasets these are Arc::clone of xs/ys (shared, zero cost).
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
            bbox_max_x: Arc::clone(&xs),
            bbox_max_y: Arc::clone(&ys),
            xs,
            ys,
        }
    }

    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize> {
        let n = self.xs.len();
        let k = k.min(n);
        let mut dists: Vec<(usize, f64)> = self
            .xs
            .iter()
            .zip(self.ys.iter())
            .enumerate()
            .map(|(i, (&x, &y))| {
                let dx = x - qx;
                let dy = y - qy;
                (i, dx * dx + dy * dy)
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
    /// Point datasets share all bbox arcs with the Engine's xs/ys, so marginal cost is zero.
    /// Polygon datasets allocate new centroid and MBR arrays of 6 * N * 8 bytes.
    pub fn heap_bytes(&self) -> usize {
        if Arc::ptr_eq(&self.xs, &self.bbox_min_x) {
            0
        } else {
            self.xs.len() * std::mem::size_of::<f64>() * 6
        }
    }

    /// Build from two-level polygon ring arrays. Computes per-polygon MBRs and centroids
    /// from exterior rings only. Holes do not expand the MBR.
    pub fn build_polygons(
        xs: &[f64],
        ys: &[f64],
        ring_offsets: &[i64],
        poly_offsets: &[i64],
    ) -> Self {
        let n_polys = poly_offsets.len().saturating_sub(1);
        let mut cxs = Vec::with_capacity(n_polys);
        let mut cys = Vec::with_capacity(n_polys);
        let mut mn_xs = Vec::with_capacity(n_polys);
        let mut mn_ys = Vec::with_capacity(n_polys);
        let mut mx_xs = Vec::with_capacity(n_polys);
        let mut mx_ys = Vec::with_capacity(n_polys);

        for &ext_ring_i64 in poly_offsets.iter().take(n_polys) {
            // MBR and centroid come from the exterior ring only
            let ext_ring = ext_ring_i64 as usize;
            let start = ring_offsets[ext_ring] as usize;
            let end = ring_offsets[ext_ring + 1] as usize;
            if start >= end {
                cxs.push(0.0);
                cys.push(0.0);
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
            cxs.push((mn_x + mx_x) / 2.0);
            cys.push((mn_y + mx_y) / 2.0);
            mn_xs.push(mn_x);
            mn_ys.push(mn_y);
            mx_xs.push(mx_x);
            mx_ys.push(mx_y);
        }

        BruteForce {
            xs: cxs.into(),
            ys: cys.into(),
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
}
