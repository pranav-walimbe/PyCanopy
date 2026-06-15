//! Batch spatial operations used by Engine's PyO3-exposed batch methods.
//! Each function crosses the Python/Rust boundary once and loops via rayon.
//! Returns Vec<u64> or Vec<(u64, u64)> to avoid per-element Python int allocation.

use std::sync::Arc;

use rayon::prelude::*;

use crate::index::kdtree::PackedKdTree;
use crate::index::SpatialIndex;
use crate::query::geometry::point_to_polygon_distance;
use crate::query::prepared::PreparedPolygons;
use crate::query::range::pip_raw;

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
                    let dx = xs[i] - qx;
                    let dy = ys[i] - qy;
                    (i, dx * dx + dy * dy)
                })
                .collect();
            for (di, (&ex, &ey)) in delta_xs.iter().zip(delta_ys.iter()).enumerate() {
                let dx = ex - qx;
                let dy = ey - qy;
                candidates.push((n_main + di, dx * dx + dy * dy));
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
/// Returns a flat array of interleaved pairs [q0, e0, q1, e1, ...] matching the
/// layout of par_within_distance and par_within_distance_flipped. The point-in-polygon
/// test uses the prepared edge index when supplied, else the linear `pip_raw` scan.
#[allow(clippy::too_many_arguments)]
pub fn par_contains<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    prepared: Option<&PreparedPolygons>,
) -> Vec<u64> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            // MBR pre-filter via index, then exact PIP
            index
                .range(qx, qy, qx, qy)
                .into_iter()
                .filter(move |&ei| match prepared {
                    Some(p) => p.contains(ei, qx, qy),
                    None => pip_raw(qx, qy, xs, ys, ring_offsets, poly_offsets, ei),
                })
                .flat_map(move |ei| [qi as u64, ei as u64])
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

/// For each query point, return (query_idx, polygon_idx) for every Engine polygon
/// within `distance` of the point. The Engine index is built over polygon MBRs, so
/// a query box dilated by `distance` is a superset of candidate polygons; each
/// candidate is refined with the exact point-to-polygon distance.
///
/// Returns a flat array of interleaved pairs [q0, e0, q1, e1, ...].
#[allow(clippy::too_many_arguments)]
pub fn par_within_distance_to_polygons<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    distance: f64,
) -> Vec<u64> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            index
                .range(qx - distance, qy - distance, qx + distance, qy + distance)
                .into_iter()
                .filter(move |&ei| {
                    point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei)
                        <= distance
                })
                .flat_map(move |ei| [qi as u64, ei as u64])
        })
        .collect()
}

/// For each query point, find the k nearest Engine polygons by exact point-to-polygon
/// distance. The MBR index supplies an over-sampled candidate set (MBR-nearest is only
/// approximate for true polygon distance), which is then refined and ranked exactly.
///
/// Returns (engine_indices, distances), each of length n_queries * k laid out in
/// per-query blocks. Blocks for queries with fewer than k candidates are padded with
/// u64::MAX / f64::INFINITY so the layout stays rectangular.
#[allow(clippy::too_many_arguments)]
pub fn par_knn_to_polygons<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    k: usize,
    n_polys: usize,
) -> (Vec<u64>, Vec<f64>) {
    // Over-sample MBR-nearest candidates: an MBR can be nearer than its polygon, so
    // fetch a multiple of k (capped at the dataset size) before exact refinement.
    let fetch = (k.saturating_mul(4)).clamp(k, n_polys.max(k));
    qxs.par_iter()
        .zip(qys.par_iter())
        .flat_map_iter(|(&qx, &qy)| {
            let mut cands: Vec<(u64, f64)> = index
                .nearest(qx, qy, fetch)
                .into_iter()
                .map(|ei| {
                    let d =
                        point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
                    (ei as u64, d)
                })
                .collect();
            cands.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            cands.truncate(k);
            cands.resize(k, (u64::MAX, f64::INFINITY));
            cands.into_iter()
        })
        .fold(
            || (Vec::new(), Vec::new()),
            |mut acc, (i, d)| {
                acc.0.push(i);
                acc.1.push(d);
                acc
            },
        )
        .reduce(
            || (Vec::new(), Vec::new()),
            |mut a, b| {
                a.0.extend(b.0);
                a.1.extend(b.1);
                a
            },
        )
}

/// Self-join: every unordered pair (i, j) with i < j of Engine polygons whose
/// boundaries intersect. The MBR index supplies candidates; each is refined with an
/// exact polygon-polygon intersection test.
///
/// Returns a flat array of interleaved pairs [i0, j0, i1, j1, ...].
pub fn par_polygon_intersects_join<I: SpatialIndex + Sync>(
    index: &I,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
) -> Vec<u64> {
    use crate::query::geometry::polygons_intersect;
    let n_polys = poly_offsets.len().saturating_sub(1);
    (0..n_polys)
        .into_par_iter()
        .flat_map_iter(|i| {
            let r_start = poly_offsets[i] as usize;
            let ext_start = ring_offsets[r_start] as usize;
            let ext_end = ring_offsets[r_start + 1] as usize;
            let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
            let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for k in ext_start..ext_end {
                min_x = min_x.min(xs[k]);
                min_y = min_y.min(ys[k]);
                max_x = max_x.max(xs[k]);
                max_y = max_y.max(ys[k]);
            }
            index
                .range(min_x, min_y, max_x, max_y)
                .into_iter()
                // Keep ordered pairs i < j to emit each unordered pair once.
                .filter(move |&j| j > i)
                .filter(move |&j| polygons_intersect(xs, ys, ring_offsets, poly_offsets, i, j))
                .flat_map(move |j| [i as u64, j as u64])
        })
        .collect()
}

/// Filter Engine points to those within `distance` of a single query polygon.
/// The query polygon is given as its own flat ring arrays. The point index is queried
/// over the polygon MBR dilated by `distance`, then refined with exact distance.
#[allow(clippy::too_many_arguments)]
pub fn par_points_within_distance_of_polygon<I: SpatialIndex + Sync>(
    index: &I,
    xs: &[f64],
    ys: &[f64],
    poly_xs: &[f64],
    poly_ys: &[f64],
    poly_ring_offsets: &[i64],
    poly_offsets: &[i64],
    distance: f64,
) -> Vec<u64> {
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (&x, &y) in poly_xs.iter().zip(poly_ys.iter()) {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    index
        .range(
            min_x - distance,
            min_y - distance,
            max_x + distance,
            max_y + distance,
        )
        .into_par_iter()
        .filter(|&pi| {
            point_to_polygon_distance(
                xs[pi],
                ys[pi],
                poly_xs,
                poly_ys,
                poly_ring_offsets,
                poly_offsets,
                0,
            ) <= distance
        })
        .map(|pi| pi as u64)
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
