//! Batch spatial operations exposed by Engine's PyO3 methods, each crossing the Python/Rust boundary once.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use rayon::prelude::*;
use rdst::{RadixKey, RadixSort};

// Spatial tile grid dimension
const TILE_GRID: usize = 16;

// Bound the aggregate states replicated across rayon workers. One state is always
// required, even when a very large target makes it exceed this soft budget.
const AGG_WORKER_STATE_BUDGET: usize = 64 * 1024 * 1024;

/// Per-tile kNN results
type TileResults = Vec<(Vec<u32>, Vec<(u32, f64)>)>;

use crate::index::kdtree::PackedKdTree;
use crate::index::{point_box_dist2, SpatialIndex};
use crate::query::geodesy::{conservative_degree_box, haversine_distance_m, DistanceMetric};
use crate::query::geometry::point_to_polygon_distance;
use crate::query::prepared::PreparedPolygons;
use crate::query::range::pip_raw;

/// For each query point, find the k nearest neighbours in the index as native join indices.
/// Returns a flat array of shape (n_queries * k,): block i holds results for query i.
pub fn par_knn_join<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    k: usize,
) -> Vec<u32> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .flat_map_iter(|(&qx, &qy)| index.nearest(qx, qy, k).into_iter().map(|i| i as u32))
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
) -> Vec<u32> {
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
            candidates.into_iter().map(|(i, _)| i as u32)
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
) -> Vec<u32> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            // MBR pre-filter via index, then exact PIP, mapping parts to polygons
            let mut out: Vec<u32> = Vec::new();
            let mut seen: Vec<u32> = Vec::new();
            for ei in index.range(qx, qy, qx, qy) {
                let hit = match prepared {
                    Some(p) => p.contains(ei, qx, qy, xs, ys, ring_offsets, poly_offsets),
                    None => pip_raw(qx, qy, xs, ys, ring_offsets, poly_offsets, ei),
                };
                if !hit {
                    continue;
                }
                match part_poly {
                    Some(pp) if seen.contains(&pp[ei]) => {}
                    Some(pp) => {
                        seen.push(pp[ei]);
                        out.push(qi as u32);
                        out.push(pp[ei]);
                    }
                    None => {
                        out.push(qi as u32);
                        out.push(ei as u32);
                    }
                }
            }
            out.into_iter()
        })
        .collect()
}

/// Mergeable per-target-row state for fused count/sum/mean spatial join aggregation
pub struct GroupedAgg {
    counts: Vec<u64>,
    sums: Vec<f64>,
    valid_counts: Vec<u64>,
    n_groups: usize,
}

impl GroupedAgg {
    fn new(n_groups: usize, n_values: usize) -> Self {
        Self {
            counts: vec![0; n_groups],
            sums: vec![0.0; n_groups * n_values],
            valid_counts: vec![0; n_groups * n_values],
            n_groups,
        }
    }

    #[inline]
    fn add(&mut self, group: u32, query: usize, values: &[&[f64]], validities: &[&[u8]]) {
        let group = group as usize;
        self.counts[group] += 1;
        for column in 0..values.len() {
            if validities[column][query] != 0 {
                let offset = column * self.n_groups + group;
                self.sums[offset] += values[column][query];
                self.valid_counts[offset] += 1;
            }
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.counts.iter_mut().zip(other.counts) {
            *left += right;
        }
        for (left, right) in self.sums.iter_mut().zip(other.sums) {
            *left += right;
        }
        for (left, right) in self.valid_counts.iter_mut().zip(other.valid_counts) {
            *left += right;
        }
        self
    }

    /// Drop target rows with no matches and return column-major aggregate value state
    pub fn compact(self) -> (Vec<u32>, Vec<u64>, Vec<f64>, Vec<u64>) {
        let groups: Vec<usize> = self
            .counts
            .iter()
            .enumerate()
            .filter_map(|(group, &count)| (count != 0).then_some(group))
            .collect();
        let mut group_indices = Vec::with_capacity(groups.len());
        let mut counts = Vec::with_capacity(groups.len());
        for &group in &groups {
            group_indices.push(group as u32);
            counts.push(self.counts[group]);
        }

        let n_values = self.sums.len().checked_div(self.n_groups).unwrap_or(0);
        let mut sums = Vec::with_capacity(groups.len() * n_values);
        let mut valid_counts = Vec::with_capacity(groups.len() * n_values);
        for column in 0..n_values {
            let base = column * self.n_groups;
            for &group in &groups {
                sums.push(self.sums[base + group]);
                valid_counts.push(self.valid_counts[base + group]);
            }
        }
        (group_indices, counts, sums, valid_counts)
    }
}

fn aggregate_chunks<F>(n_queries: usize, n_groups: usize, values: &[&[f64]], visit: F) -> GroupedAgg
where
    F: Fn(usize, &mut GroupedAgg) + Sync,
{
    if n_queries == 0 {
        return GroupedAgg::new(n_groups, values.len());
    }
    let bytes_per_group = size_of::<u64>().saturating_add(
        values
            .len()
            .saturating_mul(size_of::<f64>() + size_of::<u64>()),
    );
    let state_bytes = n_groups.saturating_mul(bytes_per_group).max(1);
    let workers_by_memory = (AGG_WORKER_STATE_BUDGET / state_bytes).max(1);
    let chunks = rayon::current_num_threads()
        .min(n_queries)
        .min(workers_by_memory);
    let chunk_size = n_queries.div_ceil(chunks);
    (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut state = GroupedAgg::new(n_groups, values.len());
            let end = ((chunk + 1) * chunk_size).min(n_queries);
            for query in chunk * chunk_size..end {
                visit(query, &mut state);
            }
            state
        })
        .reduce(
            || GroupedAgg::new(n_groups, values.len()),
            GroupedAgg::merge,
        )
}

/// Fused point-in-polygon join aggregation, retaining state per logical Engine polygon
#[allow(clippy::too_many_arguments)]
pub fn par_contains_aggregate<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    prepared: Option<&PreparedPolygons>,
    part_poly: Option<&[u32]>,
    n_groups: usize,
    values: &[&[f64]],
    validities: &[&[u8]],
) -> GroupedAgg {
    aggregate_chunks(qxs.len(), n_groups, values, |query, state| {
        let (qx, qy) = (qxs[query], qys[query]);
        let mut seen = Vec::new();
        for part in index.range(qx, qy, qx, qy) {
            let hit = match prepared {
                Some(p) => p.contains(part, qx, qy, xs, ys, ring_offsets, poly_offsets),
                None => pip_raw(qx, qy, xs, ys, ring_offsets, poly_offsets, part),
            };
            if !hit {
                continue;
            }
            let group = part_poly.map_or(part as u32, |mapping| mapping[part]);
            if !seen.contains(&group) {
                seen.push(group);
                state.add(group, query, values, validities);
            }
        }
    })
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
) -> Vec<u32> {
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
) -> Vec<u32> {
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
            .flat_map_iter(|ei| [0u32, ei as u32])
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
                .flat_map(move |ei| [qi as u32, ei as u32])
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
) -> Vec<u32> {
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
            .flat_map_iter(|ei| [0u32, ei as u32])
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
                .flat_map(move |ei| [qi as u32, ei as u32])
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
) -> Vec<u32> {
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
                        .flat_map(move |qi| [qi as u32, ei as u32])
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
                    .flat_map(move |qi| [qi as u32, ei as u32])
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
) -> Vec<u32> {
    qxs.par_iter()
        .zip(qys.par_iter())
        .enumerate()
        .flat_map_iter(|(qi, (&qx, &qy))| {
            // MBR pre-filter, exact distance, then map parts to polygons (dedup per query)
            let mut out: Vec<u32> = Vec::new();
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
                        out.push(qi as u32);
                        out.push(pp[ei]);
                    }
                    None => {
                        out.push(qi as u32);
                        out.push(ei as u32);
                    }
                }
            }
            out.into_iter()
        })
        .collect()
}

/// Fused point-to-polygon distance join aggregation per logical Engine polygon
#[allow(clippy::too_many_arguments)]
pub fn par_within_distance_to_polygons_aggregate<I: SpatialIndex + Sync>(
    index: &I,
    qxs: &[f64],
    qys: &[f64],
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    distance: f64,
    part_poly: Option<&[u32]>,
    n_groups: usize,
    values: &[&[f64]],
    validities: &[&[u8]],
) -> GroupedAgg {
    aggregate_chunks(qxs.len(), n_groups, values, |query, state| {
        let (qx, qy) = (qxs[query], qys[query]);
        let mut seen = Vec::new();
        for part in index.range(qx - distance, qy - distance, qx + distance, qy + distance) {
            if point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, part)
                > distance
            {
                continue;
            }
            let group = part_poly.map_or(part as u32, |mapping| mapping[part]);
            if !seen.contains(&group) {
                seen.push(group);
                state.add(group, query, values, validities);
            }
        }
    })
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
fn logical_poly(part_poly: Option<&[u32]>, ei: usize) -> u32 {
    match part_poly {
        Some(pp) => pp[ei],
        None => ei as u32,
    }
}

/// Insert one polygon into a distance-ascending top-k list, holding the nearest part per polygon.
///
/// Candidates arrive in lower-bound order rather than exact order, so a later part of a polygon
/// already held can still be the nearer one and has to displace the entry that is there.
fn insert_nearest(kept: &mut Vec<(u32, f64)>, pid: u32, d: f64, k: usize) {
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

/// Extra parts the seed fetches beyond k, so its MBR bound more often retires the query without
/// a sweep. Measured against SpatialBench q12: wider than this costs more refinement than the
/// sweeps it avoids.
const SEED_MARGIN: usize = 1;

/// Per worker buffers for `knn_polys_exact`, reused across every query in a tile.
///
/// The kernel runs once per query over millions of queries, so allocating its working set each
/// time dominates the real work and puts every rayon worker on the allocator at once.
#[derive(Default)]
struct KnnScratch {
    seeds: Vec<usize>,        // parts returned by the index fetch, in MBR order
    sweep: Vec<usize>,        // parts the sweep's range query returned
    cands: Vec<(usize, f64)>, // sweep candidates paired with their squared MBR lower bound
    kept: Vec<(u32, f64)>,    // the running top-k, and the result once the kernel returns
}

impl KnnScratch {
    fn with_capacity(k: usize) -> Self {
        Self {
            seeds: Vec::with_capacity(k * 2),
            sweep: Vec::with_capacity(k * 4),
            cands: Vec::with_capacity(k * 4),
            kept: Vec::with_capacity(k + 1),
        }
    }
}

/// The k nearest polygons to one query by exact point-to-polygon distance.
///
/// MBR distance only lower-bounds exact distance, so a seed pass refines the MBR-nearest parts
/// and takes the bound of the furthest it saw: once that reaches the kth exact distance nothing
/// unseen can qualify. Otherwise a sweep refines the radius that distance defines, nearest-MBR
/// first, which keeps holes, concavity, and MultiPolygons exact.
///
/// Leaves the answer in `scratch.kept`, padded with `(u32::MAX, inf)` when fewer than k exist.
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
    scratch: &mut KnnScratch,
) {
    let kept = &mut scratch.kept;
    // Seed pass. Parts can share a logical polygon, so grow the fetch until k distinct polygons
    // are held or every part has been seen.
    let mut fetch = (k + SEED_MARGIN).min(n_parts.max(1));
    loop {
        kept.clear();
        index.nearest_into(qx, qy, fetch, &mut scratch.seeds);
        for &ei in scratch.seeds.iter() {
            let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
            insert_nearest(kept, logical_poly(part_poly, ei), d, k);
        }
        if kept.len() == k || fetch >= n_parts {
            break;
        }
        fetch = fetch.saturating_mul(2).min(n_parts);
    }
    if kept.len() < k {
        // The seed swept every part, so the dataset holds fewer than k polygons
        kept.resize(k, (u32::MAX, f64::INFINITY));
        return;
    }

    let kth = kept[k - 1].1;
    let mut kth_sq = kth * kth;
    let seen_all = scratch.seeds.len() >= n_parts;
    let bound_sq = match scratch.seeds.last() {
        Some(&ei) => point_box_dist2(qx, qy, &bbox[ei]),
        None => f64::INFINITY,
    };
    if seen_all || bound_sq >= kth_sq {
        return;
    }

    // Sweep pass over every part the seed's radius admits, nearest MBR first. The seed's k stay
    // in `kept` as the working bound, so the sweep can only tighten the answer, never lose it to
    // a rounding edge where a polygon's exact distance and its MBR bound differ by an ulp.
    index.range_into(qx - kth, qy - kth, qx + kth, qy + kth, &mut scratch.sweep);
    let seeds = &scratch.seeds;
    let cands = &mut scratch.cands;
    cands.clear();
    // The seed already refined its own parts, so re-measuring them would repeat k exact
    // distances per query. Seeds are the MBR-nearest handful, so a scan beats a set.
    cands.extend(
        scratch
            .sweep
            .iter()
            .filter(|ei| !seeds.contains(ei))
            .map(|&ei| (ei, point_box_dist2(qx, qy, &bbox[ei]))),
    );
    cands.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    for &(ei, lb_sq) in cands.iter() {
        // Candidates arrive nearest-MBR-first, so once a lower bound cannot beat the kth exact
        // distance held, nothing after it can either.
        if lb_sq >= kth_sq {
            break;
        }
        let d = point_to_polygon_distance(qx, qy, xs, ys, ring_offsets, poly_offsets, ei);
        insert_nearest(kept, logical_poly(part_poly, ei), d, k);
        kth_sq = kept[k - 1].1 * kept[k - 1].1;
    }
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
) -> (Vec<u32>, Vec<f64>) {
    let n = qxs.len();
    let bbox = part_mbrs(xs, ys, ring_offsets, poly_offsets);

    let order = morton_order(qxs, qys);
    let tiles = build_query_tiles(qxs, qys, &order, TILE_GRID);

    // Parallelise over tiles so each tile's polygon vertices stay warm in L3 across its queries.
    // Each tile writes its answers into three flat buffers rather than one Vec per query, so the
    // kernel allocates per tile instead of per query.
    let tile_results: TileResults = tiles
        .par_iter()
        .map(|cell| {
            let mut scratch = KnnScratch::with_capacity(k);
            let mut qis: Vec<u32> = Vec::with_capacity(cell.len());
            let mut flat: Vec<(u32, f64)> = Vec::with_capacity(cell.len() * k);
            for &qi in cell.iter() {
                let (qx, qy) = (qxs[qi as usize], qys[qi as usize]);
                knn_polys_exact(
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
                    &mut scratch,
                );
                qis.push(qi);
                flat.extend_from_slice(&scratch.kept);
            }
            (qis, flat)
        })
        .collect();

    let mut idx = vec![0u32; n * k];
    let mut dist = vec![0f64; n * k];
    for (qis, flat) in tile_results {
        for (slot, qi) in qis.into_iter().enumerate() {
            let base = qi as usize * k;
            for (j, &(ti, d)) in flat[slot * k..slot * k + k].iter().enumerate() {
                idx[base + j] = ti;
                dist[base + j] = d;
            }
        }
    }
    (idx, dist)
}

// dist_bits = f64::to_bits(): non-negative floats sort identically as u64.
// LSD levels 0-3 = target_idx (secondary key), 4-11 = dist_bits (primary key).
#[derive(Clone, Copy)]
struct KnnTriple {
    dist_bits: u64,
    target_idx: u32,
    query_idx: u32,
}

impl RadixKey for KnnTriple {
    const LEVELS: usize = 12;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        if level < 4 {
            (self.target_idx >> (level * 8)) as u8
        } else {
            (self.dist_bits >> ((level - 4) * 8)) as u8
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
    let mut heap: BinaryHeap<Reverse<(u64, u32, usize, usize)>> = BinaryHeap::new();
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
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let bbox = part_mbrs(xs, ys, ring_offsets, poly_offsets);
    let order = morton_order(qxs, qys);
    let tiles = build_query_tiles(qxs, qys, &order, TILE_GRID);

    let sorted_tiles: Vec<Vec<KnnTriple>> = tiles
        .par_iter()
        .map(|cell| {
            let mut scratch = KnnScratch::with_capacity(k);
            let mut triples: Vec<KnnTriple> = Vec::with_capacity(cell.len() * k);
            for &qi in cell.iter() {
                let (qx, qy) = (qxs[qi as usize], qys[qi as usize]);
                knn_polys_exact(
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
                    &mut scratch,
                );
                triples.extend(scratch.kept.iter().filter(|(t, _)| *t != u32::MAX).map(
                    |&(t_idx, dist)| KnnTriple {
                        dist_bits: dist.to_bits(),
                        target_idx: t_idx,
                        query_idx: qi,
                    },
                ));
            }
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
) -> Vec<u32> {
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
                .flat_map(move |j| [i as u32, j as u32])
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
            let mut got: Vec<u32> = flat
                .as_chunks::<2>()
                .0
                .iter()
                .filter(|p| p[0] == qi as u32)
                .map(|p| p[1])
                .collect();
            got.sort_unstable();
            let want: Vec<u32> = brute_haversine(&xs, &ys, qxs[qi], qys[qi], distance)
                .into_iter()
                .map(|i| i as u32)
                .collect();
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
        let mut got: Vec<u32> = single.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
        got.sort_unstable();
        let want: Vec<u32> = brute_haversine(&xs, &ys, qx, qy, distance)
            .into_iter()
            .map(|i| i as u32)
            .collect();
        assert_eq!(got, want);
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
        let mut best: Vec<(u32, f64)> = Vec::new();
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
        assert_eq!(idx, vec![0, u32::MAX, u32::MAX]);
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
        let mut grouped: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for i in 0..q_idx.len() {
            grouped[q_idx[i] as usize].push((t_idx[i], dists[i]));
        }
        // Only the sorted path promises a tie-break, so normalise both sides to (dist, idx)
        let by_dist_then_idx = |v: &mut Vec<(u32, f64)>| {
            v.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            })
        };
        for (q, got) in grouped.iter_mut().enumerate() {
            let mut want: Vec<(u32, f64)> = (0..k)
                .map(|j| (blocked_idx[q * k + j], blocked_dist[q * k + j]))
                .collect();
            by_dist_then_idx(got);
            by_dist_then_idx(&mut want);
            assert_eq!(got, &want, "query {q}");
        }
    }
}
