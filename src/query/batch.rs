//! Batch spatial operations used by Engine's PyO3-exposed batch methods.
//! Each function crosses the Python/Rust boundary once and loops via rayon.
//! Returns Vec<u64> or Vec<(u64, u64)> to avoid per-element Python int allocation.

use std::sync::Arc;

use geo::Contains;
use rayon::prelude::*;

use crate::index::brute::BruteForce;
use crate::index::kdtree::PackedKdTree;
use crate::index::SpatialIndex;
use crate::query::range::make_polygon;

/// For each query point, find the k nearest neighbours in the index.
/// Returns a flat array of shape (n_queries * k,): block i holds results for query i.
pub fn par_knn<I: SpatialIndex + Sync>(index: &I, qxs: &[f64], qys: &[f64], k: usize) -> Vec<u64> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .flat_map_iter(|(&qx, &qy)| index.nearest(qx, qy, k).into_iter().map(|i| i as u64))
        .collect()
}

/// Like par_knn but merges delta candidates into each query result before taking top k.
/// Used when the Engine has a non-empty delta buffer.
#[allow(clippy::too_many_arguments)]
pub fn par_knn_with_delta<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    k: usize,
    xs: &[f64],
    ys: &[f64],
    delta_xs: &[f64],
    delta_ys: &[f64],
) -> Vec<u64> {
    let n_main = xs.len();
    qxs.par_iter()
        .zip(qys.par_iter())
        .flat_map_iter(|(&qx, &qy)| {
            let mut candidates: Vec<(usize, f64)> = index
                .nearest(qx, qy, k)
                .into_iter()
                .map(|i| {
                    let d = (xs[i] - qx).powi(2) + (ys[i] - qy).powi(2);
                    (i, d)
                })
                .collect();
            for (di, (&dx, &dy)) in delta_xs.iter().zip(delta_ys.iter()).enumerate() {
                let d = (dx - qx).powi(2) + (dy - qy).powi(2);
                candidates.push((n_main + di, d));
            }
            candidates.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
            candidates.into_iter().map(|(i, _)| i as u64)
        })
        .collect()
}

/// For each query point, return (query_idx, engine_idx) for every polygon in the
/// Engine's dataset that contains the point. Used for within joins on polygon datasets.
///
/// Results are in ascending query_idx order; pairs for the same query point are adjacent.
pub fn par_contains(
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
) -> Vec<(u64, u64)> {
    let brute = BruteForce::build_polygons(xs, ys, ring_offsets, poly_offsets);

    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            let qpt = geo::Point::new(qx, qy);
            // MBR pre-filter via BruteForce, then exact PIP
            brute
                .range(qx, qy, qx, qy)
                .into_iter()
                .filter(move |&ei| {
                    make_polygon(xs, ys, ring_offsets, poly_offsets, ei).contains(&qpt)
                })
                .map(move |ei| (qi as u64, ei as u64))
        })
        .collect()
}

/// For each query point, return (query_idx, engine_idx) for every engine point within `distance`.
/// Uses a bbox pre-filter via the spatial index, then an exact Euclidean distance check.
/// Returns a flat array of interleaved pairs [q0, e0, q1, e1, ...].
pub fn par_within_distance<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    distance: f64,
) -> Vec<u64> {
    let d2 = distance * distance;
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            index
                .range(qx - distance, qy - distance, qx + distance, qy + distance)
                .into_iter()
                .filter(move |&ei| {
                    let dx = xs[ei] - qx;
                    let dy = ys[ei] - qy;
                    dx * dx + dy * dy <= d2
                })
                .flat_map(move |ei| [qi as u64, ei as u64])
        })
        .collect()
}

/// Flipped variant of par_within_distance: indexes the query points and iterates
/// engine points. Produces the same (query_idx, engine_idx) pairs as par_within_distance
/// but is cheaper when the number of query points is much smaller than engine points.
pub fn par_within_distance_flipped(
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    distance: f64,
) -> Vec<u64> {
    let d2 = distance * distance;
    // Build a KD-tree on the (smaller) query side.
    let q_index = PackedKdTree::build(Arc::from(qxs.to_vec()), Arc::from(qys.to_vec()));
    xs.par_iter()
        .zip(ys.par_iter())
        .enumerate()
        .flat_map_iter(|(ei, (&sx, &sy))| {
            q_index
                .range(sx - distance, sy - distance, sx + distance, sy + distance)
                .into_iter()
                .filter(move |&qi| {
                    let dx = qxs[qi] - sx;
                    let dy = qys[qi] - sy;
                    dx * dx + dy * dy <= d2
                })
                .flat_map(move |qi| [qi as u64, ei as u64])
        })
        .collect()
}

/// For each query point, return its index if it falls within [min_x, max_x] × [min_y, max_y].
/// Used as a batch bounding-box filter on a column of query coordinates.
pub fn par_bbox_filter(
    qxs: &[f64],
    qys: &[f64],
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> Vec<u64> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .filter_map(|(i, (&x, &y))| {
            if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
                Some(i as u64)
            } else {
                None
            }
        })
        .collect()
}
