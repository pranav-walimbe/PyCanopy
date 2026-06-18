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

/// For each query point, (query_idx, polygon_idx) for every Engine polygon containing it.
/// Prepared edge index when supplied else `pip_raw`. part_poly dedups MultiPolygon parts.
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
    part_poly: Option<&[u32]>,
) -> Vec<u64> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            // MBR pre-filter via index, then exact PIP, mapping parts to polygons.
            let mut out: Vec<u64> = Vec::new();
            let mut seen: Vec<u32> = Vec::new();
            for ei in index.range(qx, qy, qx, qy) {
                let hit = match prepared {
                    Some(p) => p.contains(ei, qx, qy),
                    None => pip_raw(qx, qy, xs, ys, ring_offsets, poly_offsets, ei),
                };
                if !hit {
                    continue;
                }
                match part_poly {
                    Some(pp) if seen.contains(&pp[ei]) => {}
                    Some(pp) => {
                        seen.push(pp[ei]);
                        out.push(qi as u64);
                        out.push(pp[ei] as u64);
                    }
                    None => {
                        out.push(qi as u64);
                        out.push(ei as u64);
                    }
                }
            }
            out.into_iter()
        })
        .collect()
}

/// For each query point, (query_idx, engine_idx) for every engine point within `distance`.
/// Bbox pre-filter then exact Euclidean check.
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

/// Flipped par_within_distance. Indexes the query side and iterates engine points. Same
/// pairs and cheaper when the query count is much larger than the engine count.
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

/// For each query point, (query_idx, polygon_idx) for every Engine polygon within
/// `distance`. MBR candidates (box dilated by `distance`) refined by exact distance.
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
    part_poly: Option<&[u32]>,
) -> Vec<u64> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            // MBR pre-filter, exact distance, then map parts to polygons (dedup per query).
            let mut out: Vec<u64> = Vec::new();
            let mut seen: Vec<u32> = Vec::new();
            for ei in index.range(qx - distance, qy - distance, qx + distance, qy + distance) {
                if point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei)
                    > distance
                {
                    continue;
                }
                match part_poly {
                    Some(pp) if seen.contains(&pp[ei]) => {}
                    Some(pp) => {
                        seen.push(pp[ei]);
                        out.push(qi as u64);
                        out.push(pp[ei] as u64);
                    }
                    None => {
                        out.push(qi as u64);
                        out.push(ei as u64);
                    }
                }
            }
            out.into_iter()
        })
        .collect()
}

/// For each query point, the k nearest Engine polygons by exact point-to-polygon distance.
/// The MBR index over-samples candidates because MBR-nearest only approximates polygon
/// distance. The candidates are then refined exactly. (indices, distances) in n_queries*k
/// blocks. Short blocks padded with MAX and inf.
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
    n_parts: usize,
    part_poly: Option<&[u32]>,
) -> (Vec<u64>, Vec<f64>) {
    // Over-sample MBR-nearest candidates: an MBR can be nearer than its polygon, so
    // fetch a multiple of k (capped at the part count) before exact refinement.
    let fetch = (k.saturating_mul(4)).clamp(k, n_parts.max(k));
    qxs.par_iter()
        .zip(qys.par_iter())
        .flat_map_iter(|(&qx, &qy)| {
            let mut cands: Vec<(u64, f64)> = index
                .nearest(qx, qy, fetch)
                .into_iter()
                .map(|ei| {
                    let d =
                        point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
                    let id = part_poly.map_or(ei as u64, |pp| pp[ei] as u64);
                    (id, d)
                })
                .collect();
            // Reduce parts of one polygon to its nearest part before ranking by distance.
            if part_poly.is_some() {
                cands.sort_unstable_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                });
                cands.dedup_by_key(|c| c.0);
            }
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

/// Self-join over Engine polygons. Unordered pairs (i, j) with i < j whose boundaries
/// intersect. MBR candidates refined by an exact polygon-polygon test.
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

/// Engine points within `distance` of one query polygon given as its own ring arrays.
/// A MultiPolygon counts when any part qualifies. Point index over the dilated MBR then
/// refined by exact distance.
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
    let n_parts = poly_offsets.len().saturating_sub(1);
    index
        .range(
            min_x - distance,
            min_y - distance,
            max_x + distance,
            max_y + distance,
        )
        .into_par_iter()
        .filter(|&pi| {
            (0..n_parts).any(|qp| {
                point_to_polygon_distance(
                    xs[pi],
                    ys[pi],
                    poly_xs,
                    poly_ys,
                    poly_ring_offsets,
                    poly_offsets,
                    qp,
                ) <= distance
            })
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
