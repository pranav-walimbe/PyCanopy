//! Batch spatial operations exposed by Engine's PyO3 methods, each crossing the Python/Rust boundary once.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use rayon::prelude::*;
use rdst::{RadixKey, RadixSort};

// Spatial tile grid dimension
const TILE_GRID: usize = 16;

/// Per-tile kNN results
type TileResults = Vec<Vec<(u32, Vec<(u64, f64)>)>>;

use crate::index::kdtree::PackedKdTree;
use crate::index::{point_box_dist2, SpatialIndex};
use crate::query::geodesy::{conservative_degree_box, haversine_distance_m, DistanceMetric};
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
                    Some(p) => p.contains(ei, qx, qy, xs, ys),
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

/// Bare engine point indices within `distance` of a single center. Box candidates from the
/// index refined in parallel by the metric's exact distance.
pub fn par_radius<I: SpatialIndex + Sync>(
    index: &I,
    xs: &[f64],
    ys: &[f64],
    cx: f64,
    cy: f64,
    distance: f64,
    metric: DistanceMetric,
) -> Vec<u64> {
    // Dispatch once per call, never per candidate, so the planar path compiles as it always did
    match metric {
        DistanceMetric::Planar => {
            let d2 = distance * distance;
            index
                .range(cx - distance, cy - distance, cx + distance, cy + distance)
                .into_par_iter()
                .filter(move |&ei| {
                    let dx = xs[ei] - cx;
                    let dy = ys[ei] - cy;
                    dx * dx + dy * dy <= d2
                })
                .map(|ei| ei as u64)
                .collect()
        }
        DistanceMetric::Haversine => {
            let (min_x, min_y, max_x, max_y) = conservative_degree_box(cx, cy, distance);
            // Hoisted out of the candidate loop, where the center's latitude never changes
            let cos_lat = cy.to_radians().cos();
            index
                .range(min_x, min_y, max_x, max_y)
                .into_par_iter()
                .filter(move |&ei| {
                    haversine_distance_m(cx, cy, cos_lat, xs[ei], ys[ei]) <= distance
                })
                .map(|ei| ei as u64)
                .collect()
        }
    }
}

/// For each query point, (query_idx, engine_idx) for every engine point within `distance`
pub fn par_within_distance<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    distance: f64,
    metric: DistanceMetric,
) -> Vec<u64> {
    // Dispatch once per call, never per candidate, so the planar path compiles as it always did
    match metric {
        DistanceMetric::Planar => within_distance_planar(index, qxs, qys, xs, ys, distance),
        DistanceMetric::Haversine => within_distance_haversine(index, qxs, qys, xs, ys, distance),
    }
}

/// par_within_distance over a flat plane, bbox pre-filter then exact Euclidean check
fn within_distance_planar<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    distance: f64,
) -> Vec<u64> {
    let d2 = distance * distance;
    // Single probe parallelises the refinement over candidates, multi-probe over queries
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

/// par_within_distance over lon/lat degrees, widened-box pre-filter then exact haversine check
fn within_distance_haversine<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    distance: f64,
) -> Vec<u64> {
    // Single probe parallelises the refinement over candidates, multi-probe over queries
    if qxs.len() == 1 {
        let qx = qxs[0];
        let qy = qys[0];
        let (min_x, min_y, max_x, max_y) = conservative_degree_box(qx, qy, distance);
        let cos_lat = qy.to_radians().cos();
        return index
            .range(min_x, min_y, max_x, max_y)
            .into_par_iter()
            .filter(move |&ei| haversine_distance_m(qx, qy, cos_lat, xs[ei], ys[ei]) <= distance)
            .flat_map_iter(|ei| [0u64, ei as u64])
            .collect();
    }
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            let (min_x, min_y, max_x, max_y) = conservative_degree_box(qx, qy, distance);
            // Hoisted out of the candidate loop, where the query's latitude never changes
            let cos_lat = qy.to_radians().cos();
            index
                .range(min_x, min_y, max_x, max_y)
                .into_iter()
                .filter(move |&ei| {
                    haversine_distance_m(qx, qy, cos_lat, xs[ei], ys[ei]) <= distance
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
    metric: DistanceMetric,
) -> Vec<u64> {
    // Build a KD-tree on the (smaller) query side
    let q_index = PackedKdTree::build(Arc::from(qxs.to_vec()), Arc::from(qys.to_vec()));
    // Dispatch once per call, never per candidate, so the planar path compiles as it always did
    match metric {
        DistanceMetric::Planar => {
            let d2 = distance * distance;
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
        DistanceMetric::Haversine => xs
            .par_iter()
            .zip(ys.par_iter())
            .enumerate()
            .flat_map_iter(|(ei, (&sx, &sy))| {
                let (min_x, min_y, max_x, max_y) = conservative_degree_box(sx, sy, distance);
                // Hoisted out of the candidate loop, where the engine point's latitude is fixed
                let cos_lat = sy.to_radians().cos();
                q_index
                    .range(min_x, min_y, max_x, max_y)
                    .into_iter()
                    .filter(move |&qi| {
                        haversine_distance_m(sx, sy, cos_lat, qxs[qi], qys[qi]) <= distance
                    })
                    .flat_map(move |qi| [qi as u64, ei as u64])
            })
            .collect(),
    }
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

/// Logical polygon owning a part, the identity when the dataset has no multi-part mapping
#[inline]
fn logical_poly(part_poly: Option<&[u32]>, ei: usize) -> u64 {
    match part_poly {
        Some(pp) => pp[ei] as u64,
        None => ei as u64,
    }
}

/// Insert one polygon into a distance-ascending top-k list, holding the nearest part per polygon.
///
/// Candidates arrive in lower-bound order rather than exact order, so a later part of a polygon
/// already held can still be the nearer one and has to displace the entry that is there.
fn insert_nearest(kept: &mut Vec<(u64, f64)>, pid: u64, d: f64, k: usize) {
    match kept.iter().position(|c| c.0 == pid) {
        Some(pos) if kept[pos].1 <= d => return,
        Some(pos) => {
            kept.remove(pos);
        }
        None if kept.len() == k && d >= kept[k - 1].1 => return,
        None => {}
    }
    let pos = kept.partition_point(|c| c.1 <= d);
    kept.insert(pos, (pid, d));
    if kept.len() > k {
        kept.pop();
    }
}

/// The k nearest polygons to one query by exact point-to-polygon distance, in two phases.
///
/// A seed pass refines the MBR-nearest parts, which bounds the true kth distance from above.
/// That bound then defines a box holding every part that can still qualify, since a part whose
/// MBR is further than the bound cannot have a nearer boundary. The sweep over that box refines
/// nearest-MBR-first and stops once the next lower bound cannot beat the kth exact distance held,
/// so the result is exact for holes, concavity, and MultiPolygons alike. Fixed oversampling
/// cannot make that guarantee: any cutoff can drop a polygon whose MBR ranks poorly.
///
/// The sweep re-refines the seed's own parts, which costs k distances and saves tracking which
/// parts were already seen. Padded with `(u64::MAX, inf)` when fewer than k polygons exist.
#[allow(clippy::too_many_arguments)]
fn knn_polys_exact<I: SpatialIndex>(
    index: &I,
    qx: f64,
    qy: f64,
    k: usize,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    bbox: &[[f64; 4]],
    part_poly: Option<&[u32]>,
    n_parts: usize,
) -> Vec<(u64, f64)> {
    let mut kept: Vec<(u64, f64)> = Vec::with_capacity(k + 1);
    // Seed pass. One fetch of k parts covers k polygons unless parts share a logical polygon,
    // so grow the fetch until it does or until every part has been seen.
    let mut fetch = k;
    loop {
        kept.clear();
        for ei in index.nearest(qx, qy, fetch) {
            let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
            insert_nearest(&mut kept, logical_poly(part_poly, ei), d, k);
        }
        if kept.len() == k || fetch >= n_parts {
            break;
        }
        fetch = fetch.saturating_mul(2).min(n_parts);
    }
    if kept.len() < k {
        // The seed swept every part, so the dataset holds fewer than k polygons
        kept.resize(k, (u64::MAX, f64::INFINITY));
        return kept;
    }

    // Sweep pass over every part the seed bound admits, nearest MBR first. The seed's k stay in
    // `kept` as the working bound, so the sweep can only tighten the answer, never lose it to a
    // rounding edge where a convex polygon's exact distance and its MBR bound differ by an ulp.
    let radius = kept[k - 1].1;
    let mut kth_sq = radius * radius;
    let mut cands: Vec<(usize, f64)> = index
        .range(qx - radius, qy - radius, qx + radius, qy + radius)
        .into_iter()
        .map(|ei| (ei, point_box_dist2(qx, qy, &bbox[ei])))
        .collect();
    cands.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    for (ei, lb_sq) in cands {
        // Candidates arrive nearest-MBR-first, so once a lower bound cannot beat the kth exact
        // distance held, nothing after it can either.
        if lb_sq >= kth_sq {
            break;
        }
        let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
        insert_nearest(&mut kept, logical_poly(part_poly, ei), d, k);
        kth_sq = kept[k - 1].1 * kept[k - 1].1;
    }
    kept
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
/// The MBR index only orders candidates by a lower bound, so `knn_polys_exact` refines them
/// and keeps widening until the bound proves no polygon was missed. (indices, distances) in
/// n_queries*k blocks. Short blocks padded with MAX and inf.
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
    let bbox = part_mbrs(xs, ys, ring_offsets, poly_offsets);

    let order = morton_order(qxs, qys);
    let tiles = build_query_tiles(qxs, qys, &order, TILE_GRID);

    // Parallelise over tiles so each tile's polygon vertices stay warm in L3 across its queries
    let tile_results: TileResults = tiles
        .par_iter()
        .map(|cell| {
            cell.iter()
                .map(|&qi| {
                    let (qx, qy) = (qxs[qi as usize], qys[qi as usize]);
                    let cands = knn_polys_exact(
                        index,
                        qx,
                        qy,
                        k,
                        xs,
                        ys,
                        ring_offsets,
                        poly_offsets,
                        &bbox,
                        part_poly,
                        n_parts,
                    );
                    (qi, cands)
                })
                .collect()
        })
        .collect();

    let mut idx = vec![0u64; n * k];
    let mut dist = vec![0f64; n * k];
    for tile in tile_results {
        for (qi, cands) in tile {
            let base = qi as usize * k;
            for (j, (ti, d)) in cands.into_iter().enumerate() {
                idx[base + j] = ti;
                dist[base + j] = d;
            }
        }
    }
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

// Partition queries into a grid_n×grid_n spatial grid, nearby queries share polygon vertex cache lines
fn build_query_tiles(qxs: &[f64], qys: &[f64], order: &[u32], grid_n: usize) -> Vec<Vec<u32>> {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (&x, &y) in qxs.iter().zip(qys.iter()) {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let gn = grid_n as f64;
    let sx = if max_x > min_x {
        gn / (max_x - min_x)
    } else {
        0.0
    };
    let sy = if max_y > min_y {
        gn / (max_y - min_y)
    } else {
        0.0
    };
    let mut tiles: Vec<Vec<u32>> = vec![vec![]; grid_n * grid_n];
    for &qi in order {
        let cx = (((qxs[qi as usize] - min_x) * sx) as usize).min(grid_n - 1);
        let cy = (((qys[qi as usize] - min_y) * sy) as usize).min(grid_n - 1);
        tiles[cy * grid_n + cx].push(qi);
    }
    tiles
}

fn kway_merge(tiles: Vec<Vec<KnnTriple>>) -> Vec<KnnTriple> {
    let total: usize = tiles.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut heap: BinaryHeap<Reverse<(u64, u64, usize, usize)>> = BinaryHeap::new();
    for (ti, v) in tiles.iter().enumerate() {
        if !v.is_empty() {
            heap.push(Reverse((v[0].dist_bits, v[0].target_idx, ti, 0)));
        }
    }
    while let Some(Reverse((_, _, ti, ei))) = heap.pop() {
        out.push(tiles[ti][ei]);
        let next = ei + 1;
        if next < tiles[ti].len() {
            let t = &tiles[ti][next];
            heap.push(Reverse((t.dist_bits, t.target_idx, ti, next)));
        }
    }
    out
}

/// Like `par_knn_to_polygons` but returns all valid pairs globally sorted by
/// (distance ASC, target_idx ASC), matching `ORDER BY distance_to_building, b_buildingkey`.
///
/// Returns `(query_indices, target_indices, distances)` as three flat Vecs with no padding.
/// Queries are partitioned into spatial tiles (Morton order within each tile) so polygon vertex
/// data for a tile stays in L3 cache across its queries, then per-tile sorted runs are k-way merged.
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
    let bbox = part_mbrs(xs, ys, ring_offsets, poly_offsets);
    let order = morton_order(qxs, qys);
    let tiles = build_query_tiles(qxs, qys, &order, TILE_GRID);

    let sorted_tiles: Vec<Vec<KnnTriple>> = tiles
        .par_iter()
        .map(|cell| {
            let mut triples: Vec<KnnTriple> = cell
                .iter()
                .flat_map(|&qi| {
                    let (qx, qy) = (qxs[qi as usize], qys[qi as usize]);
                    let cands = knn_polys_exact(
                        index,
                        qx,
                        qy,
                        k,
                        xs,
                        ys,
                        ring_offsets,
                        poly_offsets,
                        &bbox,
                        part_poly,
                        n_parts,
                    );
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
            triples
        })
        .collect();

    let triples = kway_merge(sorted_tiles);
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
    use crate::index::brute::BruteForce;
    use crate::index::rtree::PackedRTree;
    use crate::query::geodesy::haversine_distance_m;

    // A lon/lat point cloud spanning the equator, mid latitudes, the poles, and the antimeridian
    fn lonlat_cloud() -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for band in [-89.5, -85.0, -45.0, -0.5, 0.0, 0.5, 45.0, 85.0, 89.5] {
            for step in 0..36 {
                xs.push(-180.0 + step as f64 * 10.0);
                ys.push(band);
            }
            // Straddle the antimeridian, where neighbours sit either side of the +/-180 seam
            for lon in [-179.99, -179.9, 179.9, 179.99] {
                xs.push(lon);
                ys.push(band);
            }
        }
        (xs, ys)
    }

    // Every engine point within `distance` meters of the query, checked pair by pair
    fn brute_haversine(xs: &[f64], ys: &[f64], qx: f64, qy: f64, distance: f64) -> Vec<u64> {
        let cos_lat = qy.to_radians().cos();
        (0..xs.len())
            .filter(|&i| haversine_distance_m(qx, qy, cos_lat, xs[i], ys[i]) <= distance)
            .map(|i| i as u64)
            .collect()
    }

    #[test]
    fn haversine_radius_matches_brute_force_across_latitude_bands() {
        let (xs, ys) = lonlat_cloud();
        let index = PackedRTree::build(Arc::from(xs.clone()), Arc::from(ys.clone()));
        for &(cx, cy) in &[
            (0.0, 0.0),
            (-73.9, 40.7),
            (10.0, 85.0),
            (10.0, 89.5),
            (179.95, 0.0),
            (-179.95, -45.0),
            (0.0, -89.5),
        ] {
            for &distance in &[500.0, 50_000.0, 400_000.0] {
                let mut got = par_radius(
                    &index,
                    &xs,
                    &ys,
                    cx,
                    cy,
                    distance,
                    DistanceMetric::Haversine,
                );
                got.sort_unstable();
                let want = brute_haversine(&xs, &ys, cx, cy, distance);
                assert_eq!(got, want, "radius {distance} m around ({cx}, {cy})");
            }
        }
    }

    #[test]
    fn haversine_within_distance_join_matches_brute_force() {
        let (xs, ys) = lonlat_cloud();
        let index = PackedRTree::build(Arc::from(xs.clone()), Arc::from(ys.clone()));
        let qxs = vec![0.0, -73.9, 10.0, 179.95, -179.95, 0.0];
        let qys = vec![0.0, 40.7, 85.0, 0.0, -45.0, 89.5];
        let distance = 200_000.0;
        let flat = par_within_distance(
            &index,
            &qxs,
            &qys,
            &xs,
            &ys,
            distance,
            DistanceMetric::Haversine,
        );
        for qi in 0..qxs.len() {
            let mut got: Vec<u64> = flat
                .chunks_exact(2)
                .filter(|p| p[0] == qi as u64)
                .map(|p| p[1])
                .collect();
            got.sort_unstable();
            let want = brute_haversine(&xs, &ys, qxs[qi], qys[qi], distance);
            assert_eq!(got, want, "query {qi} at ({}, {})", qxs[qi], qys[qi]);
        }
    }

    #[test]
    fn haversine_flipped_join_matches_the_unflipped_pairs() {
        let (xs, ys) = lonlat_cloud();
        let index = PackedRTree::build(Arc::from(xs.clone()), Arc::from(ys.clone()));
        let qxs = vec![0.0, -73.9, 10.0, 179.95];
        let qys = vec![0.0, 40.7, 85.0, 0.0];
        let distance = 200_000.0;
        let mut want = par_within_distance(
            &index,
            &qxs,
            &qys,
            &xs,
            &ys,
            distance,
            DistanceMetric::Haversine,
        );
        let mut got =
            par_within_distance_flipped(&qxs, &qys, &xs, &ys, distance, DistanceMetric::Haversine);
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn single_probe_haversine_matches_the_multi_probe_path() {
        // The one-query fast path parallelises differently, so it needs its own check
        let (xs, ys) = lonlat_cloud();
        let index = PackedRTree::build(Arc::from(xs.clone()), Arc::from(ys.clone()));
        let (qx, qy, distance) = (-73.9, 40.7, 300_000.0);
        let single = par_within_distance(
            &index,
            &[qx],
            &[qy],
            &xs,
            &ys,
            distance,
            DistanceMetric::Haversine,
        );
        let mut got: Vec<u64> = single.chunks_exact(2).map(|p| p[1]).collect();
        got.sort_unstable();
        assert_eq!(got, brute_haversine(&xs, &ys, qx, qy, distance));
    }

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

    // Flatten parts, each a list of closed rings with the exterior first, into the two-level
    // ring arrays the polygon kernels take.
    fn ring_arrays(parts: &[Vec<Vec<(f64, f64)>>]) -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut ring_offsets = Vec::new();
        let mut poly_offsets = Vec::new();
        for part in parts {
            poly_offsets.push(ring_offsets.len() as i64);
            for ring in part {
                ring_offsets.push(xs.len() as i64);
                for &(x, y) in ring {
                    xs.push(x);
                    ys.push(y);
                }
            }
        }
        poly_offsets.push(ring_offsets.len() as i64);
        ring_offsets.push(xs.len() as i64);
        (xs, ys, ring_offsets, poly_offsets)
    }

    // Closed square ring of half-width r centred on (cx, cy)
    fn square_ring(cx: f64, cy: f64, r: f64) -> Vec<(f64, f64)> {
        vec![
            (cx - r, cy - r),
            (cx + r, cy - r),
            (cx + r, cy + r),
            (cx - r, cy + r),
            (cx - r, cy - r),
        ]
    }

    // A square annulus: its MBR covers the centre while its boundary stays `hole` away from it,
    // which is what makes MBR rank and exact distance disagree.
    fn donut(cx: f64, cy: f64, outer: f64, hole: f64) -> Vec<Vec<(f64, f64)>> {
        vec![square_ring(cx, cy, outer), square_ring(cx, cy, hole)]
    }

    fn xorshift(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // Exact k nearest polygon distances for one query, by scanning every part
    #[allow(clippy::too_many_arguments)]
    fn brute_polys(
        qx: f64,
        qy: f64,
        xs: &[f64],
        ys: &[f64],
        ring_offsets: &[i64],
        poly_offsets: &[i64],
        part_poly: Option<&[u32]>,
        k: usize,
    ) -> Vec<f64> {
        let n_parts = poly_offsets.len() - 1;
        let mut best: Vec<(u64, f64)> = Vec::new();
        for p in 0..n_parts {
            let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, p);
            let pid = logical_poly(part_poly, p);
            match best.iter().position(|c| c.0 == pid) {
                Some(pos) => best[pos].1 = best[pos].1.min(d),
                None => best.push((pid, d)),
            }
        }
        best.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        best.into_iter().take(k).map(|c| c.1).collect()
    }

    #[test]
    fn donut_mbrs_do_not_hide_the_nearest_polygon() {
        // Four annuli whose MBRs all contain the query, so MBR rank puts them first, and one
        // small square that is genuinely nearest. Fixed 4*k oversampling returned an annulus.
        let mut parts: Vec<Vec<Vec<(f64, f64)>>> = (0..4)
            .map(|i| donut(i as f64 * 0.1, 0.0, 100.0, 50.0))
            .collect();
        parts.push(vec![vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
        ]]);
        let (xs, ys, ring_offsets, poly_offsets) = ring_arrays(&parts);
        let n_parts = poly_offsets.len() - 1;

        let rtree = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);
        let (idx, dist) = par_knn_to_polygons(
            &rtree,
            &[0.0],
            &[0.0],
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            1,
            n_parts,
            None,
        );
        assert_eq!(idx, vec![4]);
        assert!((dist[0] - 1.0).abs() < 1e-12, "rtree gave {}", dist[0]);

        let brute = BruteForce::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);
        let (idx, dist) = par_knn_to_polygons(
            &brute,
            &[0.0],
            &[0.0],
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            1,
            n_parts,
            None,
        );
        assert_eq!(idx, vec![4]);
        assert!((dist[0] - 1.0).abs() < 1e-12, "brute gave {}", dist[0]);
    }

    #[test]
    fn random_annuli_knn_matches_brute_force() {
        // Overlapping annuli of varied size: MBR order and exact order disagree constantly
        let mut next = xorshift(0x9e3779b97f4a7c15);
        let parts: Vec<Vec<Vec<(f64, f64)>>> = (0..120)
            .map(|_| {
                let outer = 5.0 + next() * 45.0;
                let hole = outer * (0.2 + next() * 0.7);
                donut(next() * 100.0, next() * 100.0, outer, hole)
            })
            .collect();
        let (xs, ys, ring_offsets, poly_offsets) = ring_arrays(&parts);
        let n_parts = poly_offsets.len() - 1;
        let index = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);

        let n = 300;
        let qxs: Vec<f64> = (0..n).map(|_| next() * 100.0).collect();
        let qys: Vec<f64> = (0..n).map(|_| next() * 100.0).collect();
        let k = 4;
        let (_, dist) = par_knn_to_polygons(
            &index,
            &qxs,
            &qys,
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            k,
            n_parts,
            None,
        );

        for q in 0..n {
            let want = brute_polys(
                qxs[q],
                qys[q],
                &xs,
                &ys,
                &ring_offsets,
                &poly_offsets,
                None,
                k,
            );
            for j in 0..k {
                assert!(
                    (dist[q * k + j] - want[j]).abs() < 1e-9,
                    "query {q} neighbour {j}: kernel {} vs brute {}",
                    dist[q * k + j],
                    want[j]
                );
            }
        }
    }

    #[test]
    fn multipolygon_parts_collapse_to_one_neighbour() {
        // Polygon 0 has a far part and a near one, polygons 1 and 2 sit between them. k counts
        // distinct polygons, so polygon 0 must appear once, at its nearer part's distance.
        let parts = vec![
            vec![square_ring(40.0, 0.0, 1.0)],
            vec![square_ring(5.0, 0.0, 1.0)],
            vec![square_ring(20.0, 0.0, 1.0)],
            vec![square_ring(10.0, 0.0, 1.0)],
        ];
        let (xs, ys, ring_offsets, poly_offsets) = ring_arrays(&parts);
        let part_poly = [0u32, 0, 1, 2];
        let n_parts = poly_offsets.len() - 1;
        let index = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);

        let (idx, dist) = par_knn_to_polygons(
            &index,
            &[0.0],
            &[0.0],
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            3,
            n_parts,
            Some(&part_poly),
        );
        assert_eq!(idx, vec![0, 2, 1]);
        for (got, want) in dist.iter().zip([4.0, 9.0, 19.0]) {
            assert!((got - want).abs() < 1e-12, "got {dist:?}");
        }
    }

    #[test]
    fn fewer_polygons_than_k_pads_the_block() {
        let parts = vec![vec![square_ring(5.0, 0.0, 1.0)]];
        let (xs, ys, ring_offsets, poly_offsets) = ring_arrays(&parts);
        let index = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);
        let (idx, dist) = par_knn_to_polygons(
            &index,
            &[0.0],
            &[0.0],
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            3,
            1,
            None,
        );
        assert_eq!(idx, vec![0, u64::MAX, u64::MAX]);
        assert!((dist[0] - 4.0).abs() < 1e-12);
        assert!(dist[1].is_infinite() && dist[2].is_infinite());
    }

    #[test]
    fn sorted_variant_agrees_with_the_blocked_one() {
        let mut next = xorshift(0xdeadbeefcafef00d);
        let parts: Vec<Vec<Vec<(f64, f64)>>> = (0..60)
            .map(|_| {
                let outer = 4.0 + next() * 20.0;
                donut(next() * 80.0, next() * 80.0, outer, outer * 0.6)
            })
            .collect();
        let (xs, ys, ring_offsets, poly_offsets) = ring_arrays(&parts);
        let n_parts = poly_offsets.len() - 1;
        let index = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);

        let n = 50;
        let qxs: Vec<f64> = (0..n).map(|_| next() * 80.0).collect();
        let qys: Vec<f64> = (0..n).map(|_| next() * 80.0).collect();
        let k = 3;
        let (blocked_idx, blocked_dist) = par_knn_to_polygons(
            &index,
            &qxs,
            &qys,
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            k,
            n_parts,
            None,
        );
        let (q_idx, t_idx, dists) = par_knn_to_polygons_sorted(
            &index,
            &qxs,
            &qys,
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            k,
            n_parts,
            None,
        );

        assert_eq!(q_idx.len(), n * k);
        // The sorted path emits globally ordered pairs, so regroup by query to compare
        let mut grouped: Vec<Vec<(u64, f64)>> = vec![Vec::new(); n];
        for i in 0..q_idx.len() {
            grouped[q_idx[i] as usize].push((t_idx[i], dists[i]));
        }
        // Only the sorted path promises a tie-break, so normalise both sides to (dist, idx)
        let by_dist_then_idx = |v: &mut Vec<(u64, f64)>| {
            v.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            })
        };
        for (q, got) in grouped.iter_mut().enumerate() {
            let mut want: Vec<(u64, f64)> = (0..k)
                .map(|j| (blocked_idx[q * k + j], blocked_dist[q * k + j]))
                .collect();
            by_dist_then_idx(got);
            by_dist_then_idx(&mut want);
            assert_eq!(got, &want, "query {q}");
        }
    }
}
