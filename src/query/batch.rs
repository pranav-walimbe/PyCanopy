//! Batch spatial operations used by Engine's PyO3-exposed batch methods.
//! Each function crosses the Python/Rust boundary once and loops via rayon.
//! Returns Vec<u64> or Vec<(u64, u64)> to avoid per-element Python int allocation.

use std::cmp::Ordering;
use std::sync::Arc;

use rayon::prelude::*;
use rdst::{RadixKey, RadixSort};

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
            // MBR pre-filter via index, then exact PIP, mapping parts to polygons
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
    // Single probe: parallelize the Euclidean refinement over candidates across threads.
    // The multi-probe path parallelises over queries instead.
    if qxs.len() == 1 {
        let qx = qxs[0];
        let qy = qys[0];
        return index
            .range(qx - distance, qy - distance, qx + distance, qy + distance)
            .into_par_iter()
            .filter(move |&ei| {
                let dx = xs[ei] - qx;
                let dy = ys[ei] - qy;
                dx * dx + dy * dy <= d2
            })
            .flat_map_iter(|ei| [0u64, ei as u64])
            .collect();
    }
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
    // Build a KD-tree on the (smaller) query side
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
            // MBR pre-filter, exact distance, then map parts to polygons (dedup per query)
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

/// Per-part exterior-ring MBR as `[min_x, min_y, max_x, max_y]`, one entry per part.
/// The squared point-to-MBR distance lower-bounds the exact point-to-polygon distance.
fn part_mbrs(xs: &[f64], ys: &[f64], ring_offsets: &[i64], poly_offsets: &[i64]) -> Vec<[f64; 4]> {
    let n_parts = poly_offsets.len().saturating_sub(1);
    let mut out = Vec::with_capacity(n_parts);
    for &ext_ring_i64 in poly_offsets.iter().take(n_parts) {
        let ext_ring = ext_ring_i64 as usize;
        let start = ring_offsets[ext_ring] as usize;
        let end = ring_offsets[ext_ring + 1] as usize;
        if start >= end {
            out.push([0.0, 0.0, 0.0, 0.0]);
            continue;
        }
        let (mut mnx, mut mny, mut mxx, mut mxy) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for k in start..end {
            mnx = mnx.min(xs[k]);
            mny = mny.min(ys[k]);
            mxx = mxx.max(xs[k]);
            mxy = mxy.max(ys[k]);
        }
        out.push([mnx, mny, mxx, mxy]);
    }
    out
}

#[inline]
fn point_box_dist2(px: f64, py: f64, b: &[f64; 4]) -> f64 {
    // Squared distance from a point to an axis-aligned box, zero when the point is inside
    let dx = (b[0] - px).max(0.0).max(px - b[2]);
    let dy = (b[1] - py).max(0.0).max(py - b[3]);
    dx * dx + dy * dy
}

/// k nearest single-part polygons for one query, refining candidates nearest-MBR-first and
/// pruning the per-edge scan once k exact hits bound the search. Padded with `(u64::MAX, inf)`.
#[allow(clippy::too_many_arguments)]
fn knn_polys_pruned<I: SpatialIndex>(
    index: &I,
    qx: f64,
    qy: f64,
    fetch: usize,
    k: usize,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    bbox: &[[f64; 4]],
) -> Vec<(u64, f64)> {
    // Order candidates by MBR lower bound
    let mut cands: Vec<(usize, f64)> = index
        .nearest(qx, qy, fetch)
        .into_iter()
        .map(|ei| (ei, point_box_dist2(qx, qy, &bbox[ei])))
        .collect();
    cands.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut kept: Vec<(u64, f64)> = Vec::with_capacity(k + 1);
    let mut kth_sq = f64::INFINITY;
    for (ei, lb_sq) in cands {
        if kept.len() == k && lb_sq >= kth_sq {
            break;
        }
        let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
        let pos = kept.partition_point(|c| c.1 <= d);
        kept.insert(pos, (ei as u64, d));
        if kept.len() > k {
            kept.pop();
        }
        if kept.len() == k {
            kth_sq = kept[k - 1].1 * kept[k - 1].1;
        }
    }
    kept.resize(k, (u64::MAX, f64::INFINITY));
    kept
}

/// k nearest multi-part polygons for one query, reducing each polygon's parts to its nearest
/// part before ranking. Exhaustive: with the part mapping the k-th distance is not a safe bound.
#[allow(clippy::too_many_arguments)]
fn knn_polys_multipart<I: SpatialIndex>(
    index: &I,
    qx: f64,
    qy: f64,
    fetch: usize,
    k: usize,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    part_poly: &[u32],
) -> Vec<(u64, f64)> {
    let mut cands: Vec<(u64, f64)> = index
        .nearest(qx, qy, fetch)
        .into_iter()
        .map(|ei| {
            let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
            (part_poly[ei] as u64, d)
        })
        .collect();
    cands.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
    });
    cands.dedup_by_key(|c| c.0);
    cands.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    cands.truncate(k);
    cands.resize(k, (u64::MAX, f64::INFINITY));
    cands
}

/// Interleave two 16-bit coordinates into a 32-bit Morton (Z-order) code
fn morton_encode(xi: u32, yi: u32) -> u32 {
    fn spread(mut v: u32) -> u32 {
        v &= 0xffff;
        v = (v | (v << 8)) & 0x00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333;
        (v | (v << 1)) & 0x5555_5555
    }
    spread(xi) | (spread(yi) << 1)
}

/// Argsort of the query points by Morton (Z-order) code, normalised to 16 bits per axis, so
/// neighbouring probes share R-tree paths. Identity order for small inputs that would not pay.
fn morton_order(qxs: &[f64], qys: &[f64]) -> Vec<u32> {
    let n = qxs.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    if n < 1024 {
        return order;
    }
    let (mut minx, mut miny, mut maxx, mut maxy) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (&x, &y) in qxs.iter().zip(qys.iter()) {
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    let sx = if maxx > minx {
        65535.0 / (maxx - minx)
    } else {
        0.0
    };
    let sy = if maxy > miny {
        65535.0 / (maxy - miny)
    } else {
        0.0
    };
    let keys: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = ((qxs[i] - minx) * sx) as u32;
            let yi = ((qys[i] - miny) * sy) as u32;
            morton_encode(xi, yi)
        })
        .collect();
    order.par_sort_unstable_by_key(|&i| keys[i as usize]);
    order
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
    let n = qxs.len();
    // Over-sample MBR-nearest candidates: an MBR can be nearer than its polygon, so
    // fetch a multiple of k (capped at the part count) before exact refinement.
    let fetch = (k.saturating_mul(4)).clamp(k, n_parts.max(k));
    // Only the single-part path uses the MBR table, so build it only then
    let bbox = match part_poly {
        Some(_) => Vec::new(),
        None => part_mbrs(xs, ys, ring_offsets, poly_offsets),
    };

    // Probe in Morton order so neighbouring queries share R-tree paths and reuse warm cache.
    // rayon's collect preserves order, so block `rank` holds the results for query `order[rank]`.
    let order = morton_order(qxs, qys);
    let ranked: Vec<(u64, f64)> = order
        .par_iter()
        .flat_map_iter(|&qi| {
            let qi = qi as usize;
            let (qx, qy) = (qxs[qi], qys[qi]);
            let cands = match part_poly {
                Some(pp) => knn_polys_multipart(
                    index,
                    qx,
                    qy,
                    fetch,
                    k,
                    xs,
                    ys,
                    ring_offsets,
                    poly_offsets,
                    pp,
                ),
                None => knn_polys_pruned(
                    index,
                    qx,
                    qy,
                    fetch,
                    k,
                    xs,
                    ys,
                    ring_offsets,
                    poly_offsets,
                    &bbox,
                ),
            };
            cands.into_iter()
        })
        .collect();

    // Invert the permutation, then gather each query's k-block into its original position.
    // Output chunks are disjoint, so the scatter parallelises without unsafe.
    let mut inv = vec![0u32; n];
    for (rank, &qi) in order.iter().enumerate() {
        inv[qi as usize] = rank as u32;
    }
    let mut idx = vec![0u64; n * k];
    let mut dist = vec![0f64; n * k];
    idx.par_chunks_mut(k)
        .zip(dist.par_chunks_mut(k))
        .enumerate()
        .for_each(|(qi, (ic, dc))| {
            let base = inv[qi] as usize * k;
            for j in 0..k {
                ic[j] = ranked[base + j].0;
                dc[j] = ranked[base + j].1;
            }
        });
    (idx, dist)
}

// dist_bits = f64::to_bits(): non-negative floats sort identically as u64.
// LSD levels 0-7 = target_idx (secondary key), 8-15 = dist_bits (primary key).
#[derive(Clone, Copy)]
struct KnnTriple {
    dist_bits: u64,
    target_idx: u64,
    query_idx: u64,
}

impl RadixKey for KnnTriple {
    const LEVELS: usize = 16;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        if level < 8 {
            (self.target_idx >> (level * 8)) as u8
        } else {
            (self.dist_bits >> ((level - 8) * 8)) as u8
        }
    }
}

/// Like `par_knn_to_polygons` but returns all valid pairs globally sorted by
/// (distance ASC, target_idx ASC), matching `ORDER BY distance_to_building, b_buildingkey`.
///
/// Returns `(query_indices, target_indices, distances)` as three flat Vecs with no padding.
/// The full result is built in RAM and sorted with rayon before returning.
#[allow(clippy::too_many_arguments)]
pub fn par_knn_to_polygons_sorted<I: SpatialIndex + Sync>(
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
) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
    let fetch = (k.saturating_mul(4)).clamp(k, n_parts.max(k));
    let bbox = match part_poly {
        Some(_) => Vec::new(),
        None => part_mbrs(xs, ys, ring_offsets, poly_offsets),
    };
    let order = morton_order(qxs, qys);

    let mut triples: Vec<KnnTriple> = order
        .par_iter()
        .flat_map_iter(|&qi| {
            let qi_usize = qi as usize;
            let (qx, qy) = (qxs[qi_usize], qys[qi_usize]);
            let cands = match part_poly {
                Some(pp) => knn_polys_multipart(
                    index,
                    qx,
                    qy,
                    fetch,
                    k,
                    xs,
                    ys,
                    ring_offsets,
                    poly_offsets,
                    pp,
                ),
                None => knn_polys_pruned(
                    index,
                    qx,
                    qy,
                    fetch,
                    k,
                    xs,
                    ys,
                    ring_offsets,
                    poly_offsets,
                    &bbox,
                ),
            };
            cands
                .into_iter()
                .filter(|(t_idx, _)| *t_idx != u64::MAX)
                .map(move |(t_idx, dist)| KnnTriple {
                    dist_bits: dist.to_bits(),
                    target_idx: t_idx,
                    query_idx: qi as u64,
                })
        })
        .collect();

    triples.radix_sort_unstable();

    let n = triples.len();
    let mut q_idx = Vec::with_capacity(n);
    let mut t_idx = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n);
    for triple in triples {
        q_idx.push(triple.query_idx);
        t_idx.push(triple.target_idx);
        dists.push(f64::from_bits(triple.dist_bits));
    }
    (q_idx, t_idx, dists)
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
                // Keep ordered pairs i < j to emit each unordered pair once
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::rtree::PackedRTree;

    // A grid of g*g unit squares spaced 2 apart, as flat single-part polygon ring arrays
    fn grid_squares(g: usize) -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
        let n = g * g;
        let mut xs = Vec::with_capacity(n * 5);
        let mut ys = Vec::with_capacity(n * 5);
        let mut ring_offsets = Vec::with_capacity(n + 1);
        let mut poly_offsets = Vec::with_capacity(n + 1);
        for p in 0..n {
            let (cx, cy) = ((p % g) as f64 * 2.0, (p / g) as f64 * 2.0);
            ring_offsets.push((p * 5) as i64);
            poly_offsets.push(p as i64);
            for &(dx, dy) in &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)] {
                xs.push(cx + dx);
                ys.push(cy + dy);
            }
        }
        ring_offsets.push((n * 5) as i64);
        poly_offsets.push(n as i64);
        (xs, ys, ring_offsets, poly_offsets)
    }

    #[test]
    fn morton_reordered_knn_preserves_per_query_blocks() {
        // Enough queries (>=1024) to exercise the Morton reorder and gather paths
        let g = 12;
        let (xs, ys, ring_offsets, poly_offsets) = grid_squares(g);
        let n_polys = poly_offsets.len() - 1;
        let index = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);

        let n = 1500;
        let span = (g as f64) * 2.0;
        let mut state = 0x2545f4914f6cdd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let qxs: Vec<f64> = (0..n).map(|_| next() * span).collect();
        let qys: Vec<f64> = (0..n).map(|_| next() * span).collect();

        let k = 3;
        let (idx, dist) = par_knn_to_polygons(
            &index,
            &qxs,
            &qys,
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            k,
            n_polys,
            None,
        );
        assert_eq!(idx.len(), n * k);

        for q in 0..n {
            // Independent brute-force k nearest polygon distances for this query
            let mut all: Vec<f64> = (0..n_polys)
                .map(|p| {
                    point_to_polygon_distance(
                        qxs[q],
                        qys[q],
                        &xs,
                        &ys,
                        &ring_offsets,
                        &poly_offsets,
                        p,
                    )
                })
                .collect();
            all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mut got: Vec<f64> = dist[q * k..q * k + k].to_vec();
            got.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            for j in 0..k {
                assert!(
                    (got[j] - all[j]).abs() < 1e-9,
                    "query {q} neighbour {j}: kernel {} vs brute {}",
                    got[j],
                    all[j]
                );
            }
        }
    }
}
