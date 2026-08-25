//! Packed immutable R-tree index with Hilbert sort for point and polygon datasets.

use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;

use rayon::prelude::*;

use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex};

use crate::index::{point_box_dist2, SpatialIndex};

/// One entry in the best-first traversal queue, ordered by squared distance to the node's box.
///
/// The id packs the node or item index with a tag in its low bit, odd for a leaf item and even
/// for an interior node, matching how geo-index encodes the same queue.
#[derive(Clone, Copy, PartialEq)]
struct NeighborNode {
    id: usize, // (index << 1), plus 1 when this is a leaf item
    dist: f64, // squared point-to-box distance, a lower bound for anything under the node
}

impl Eq for NeighborNode {}

impl PartialOrd for NeighborNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

thread_local! {
    // Reused across every query on this worker
    // geo-index's own `neighbors` and `search` allocate a fresh queue and vector per call
    // That allocation dominates a kernel invoked once per query
    static NEIGHBOR_QUEUE: RefCell<BinaryHeap<Reverse<NeighborNode>>> =
        const { RefCell::new(BinaryHeap::new()) };
    static SEARCH_QUEUE: RefCell<VecDeque<usize>> = const { RefCell::new(VecDeque::new()) };
}

/// End of the level holding `value`, mirroring geo-index's internal level lookup
fn level_end(value: usize, level_bounds: &[usize]) -> usize {
    let mut lo = 0;
    let mut hi = level_bounds.len() - 1;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if level_bounds[mid] > value {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    level_bounds[lo]
}

// Exterior ring bounds in one pass with a zero box for an empty ring
fn part_bounds(xs: &[f64], ys: &[f64], ring_offsets: &[i64], ext_ring: usize) -> [f64; 4] {
    let start = ring_offsets[ext_ring] as usize;
    let end = ring_offsets[ext_ring + 1] as usize;
    if start >= end {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (&x, &y) in xs[start..end].iter().zip(ys[start..end].iter()) {
        bounds[0] = bounds[0].min(x);
        bounds[1] = bounds[1].min(y);
        bounds[2] = bounds[2].max(x);
        bounds[3] = bounds[3].max(y);
    }
    bounds
}

/// Packed immutable R-tree backed by geo-index with Hilbert sort.
///
/// geo-index stores coordinates internally (one unavoidable copy at build time).
/// The xs/ys Arcs passed to build() are not retained, they are iterated once
/// to feed the builder and then dropped.
/// For polygon datasets use build_polygons which computes per-part MBRs.
pub struct PackedRTree {
    tree: RTree<f64>,
}

impl SpatialIndex for PackedRTree {
    fn build(xs: Arc<[f64]>, ys: Arc<[f64]>) -> Self {
        let n = xs.len() as u32;
        let mut builder = RTreeBuilder::<f64>::new(n);
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            builder.add(x, y, x, y);
        }
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

    fn nearest_into(&self, qx: f64, qy: f64, k: usize, out: &mut Vec<usize>) {
        out.clear();
        if k == 0 {
            return;
        }
        let boxes = self.tree.boxes();
        let indices = self.tree.indices();
        let level_bounds = self.tree.level_bounds();
        let leaf_limit = self.tree.num_items() as usize * 4;
        let stride = self.tree.node_size() as usize * 4;
        NEIGHBOR_QUEUE.with(|cell| {
            let mut queue = cell.borrow_mut();
            queue.clear();
            let mut outer = boxes.len().checked_sub(4);
            'outer: while let Some(node_index) = outer {
                let end = (node_index + stride).min(level_end(node_index, level_bounds));
                for pos in (node_index..end).step_by(4) {
                    let index = indices.get(pos >> 2);
                    let bbox = [boxes[pos], boxes[pos + 1], boxes[pos + 2], boxes[pos + 3]];
                    let dist = point_box_dist2(qx, qy, &bbox);
                    let leaf = node_index < leaf_limit;
                    queue.push(Reverse(NeighborNode {
                        id: (index << 1) + usize::from(leaf),
                        dist,
                    }));
                }
                while queue.peek().is_some_and(|entry| entry.0.id & 1 != 0) {
                    let item = queue.pop().expect("peek proved the queue is non-empty");
                    out.push(item.0.id >> 1);
                    if out.len() == k {
                        break 'outer;
                    }
                }
                outer = queue.pop().map(|item| item.0.id >> 1);
            }
        });
    }

    fn range_into(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64, out: &mut Vec<usize>) {
        out.clear();
        let boxes = self.tree.boxes();
        if boxes.is_empty() {
            return;
        }
        let indices = self.tree.indices();
        let level_bounds = self.tree.level_bounds();
        let leaf_limit = self.tree.num_items() as usize * 4;
        let stride = self.tree.node_size() as usize * 4;
        SEARCH_QUEUE.with(|cell| {
            let mut queue = cell.borrow_mut();
            queue.clear();
            let mut outer = boxes.len().checked_sub(4);
            while let Some(node_index) = outer {
                let end = (node_index + stride).min(level_end(node_index, level_bounds));
                for pos in (node_index..end).step_by(4) {
                    if max_x < boxes[pos]
                        || max_y < boxes[pos + 1]
                        || min_x > boxes[pos + 2]
                        || min_y > boxes[pos + 3]
                    {
                        continue;
                    }
                    let index = indices.get(pos >> 2);
                    if node_index >= leaf_limit {
                        queue.push_back(index);
                    } else {
                        out.push(index);
                    }
                }
                outer = queue.pop_front();
            }
        });
    }
}

impl PackedRTree {
    /// Heap bytes allocated by this index (the geo-index internal flat buffer).
    ///
    /// The coordinate Arcs passed to build() are not retained, they're dropped
    /// after the builder consumes them, so there is nothing shared to exclude.
    pub fn heap_bytes(&self) -> usize {
        self.tree.metadata().data_buffer_length()
    }

    /// Build from two-level polygon ring arrays with one MBR per part from its exterior ring.
    pub fn build_polygons(
        xs: &[f64],
        ys: &[f64],
        ring_offsets: &[i64],
        poly_offsets: &[i64],
    ) -> Self {
        let n_polys = poly_offsets.len().saturating_sub(1);
        // Bounds compute across the rayon pool then feed the builder in order
        let bounds: Vec<[f64; 4]> = poly_offsets[..n_polys]
            .par_iter()
            .map(|&ext_ring| part_bounds(xs, ys, ring_offsets, ext_ring as usize))
            .collect();
        let mut builder = RTreeBuilder::<f64>::new(n_polys as u32);
        for bound in bounds {
            builder.add(bound[0], bound[1], bound[2], bound[3]);
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

    // Deterministic pseudo-random points reproduce a failure without a rand dependency
    fn scattered(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| (next() * 100.0, next() * 100.0)).unzip()
    }

    #[test]
    fn nearest_into_matches_geo_index_neighbors() {
        // nearest_into reimplements the traversal over geo-index's buffers to reuse its queue,
        // so it has to agree with the crate's own result exactly, order included.
        let (xs, ys) = scattered(500);
        let tree = build(xs.clone(), ys.clone());
        let mut got = Vec::new();
        for i in 0..40 {
            let (qx, qy) = (i as f64 * 2.5, (40 - i) as f64 * 2.5);
            for k in [1usize, 2, 5, 17] {
                tree.nearest_into(qx, qy, k, &mut got);
                assert_eq!(got, tree.nearest(qx, qy, k), "k={k} at ({qx}, {qy})");
            }
        }
    }

    #[test]
    fn range_into_matches_geo_index_search() {
        let (xs, ys) = scattered(500);
        let tree = build(xs.clone(), ys.clone());
        let mut got = Vec::new();
        for i in 0..20 {
            let lo = i as f64 * 5.0;
            for span in [0.0, 1.0, 13.0, 200.0] {
                tree.range_into(lo, lo, lo + span, lo + span, &mut got);
                assert_eq!(
                    sorted(got.clone()),
                    sorted(tree.range(lo, lo, lo + span, lo + span)),
                    "span={span} at {lo}"
                );
            }
        }
    }

    #[test]
    fn nearest_into_asks_for_more_than_the_tree_holds() {
        let (xs, ys) = five_point_grid();
        let tree = build(xs, ys);
        let mut got = Vec::new();
        tree.nearest_into(0.0, 0.0, 50, &mut got);
        assert_eq!(sorted(got), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn nearest_into_handles_exact_distance_ties() {
        // Four coincident points tie on every candidate and only the set is well defined
        let tree = build(vec![1.0, 1.0, 1.0, 1.0, 9.0], vec![1.0, 1.0, 1.0, 1.0, 9.0]);
        let mut got = Vec::new();
        tree.nearest_into(1.0, 1.0, 3, &mut got);
        assert_eq!(got.len(), 3);
        got.sort_unstable();
        got.dedup();
        assert_eq!(got.len(), 3, "a tie must not return the same item twice");
        assert!(
            got.iter().all(|&i| i < 4),
            "the far point must not tie the near ones"
        );
    }

    // Deterministic polygon parts with varied ring sizes plus one empty ring
    fn ring_fixture(parts: usize) -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut xs, mut ys, mut ring_offsets, mut poly_offsets) =
            (Vec::new(), Vec::new(), vec![0i64], vec![0i64]);
        for part in 0..parts {
            let n = if part == 3 { 0 } else { 1 + (part % 17) };
            for _ in 0..n {
                xs.push(next() * 360.0 - 180.0);
                ys.push(next() * 180.0 - 90.0);
            }
            ring_offsets.push(xs.len() as i64);
            poly_offsets.push(ring_offsets.len() as i64 - 1);
        }
        (xs, ys, ring_offsets, poly_offsets)
    }

    // Four separate folds per ring as build_polygons did before
    fn reference_bounds(xs: &[f64], ys: &[f64], ring_offsets: &[i64], ext_ring: usize) -> [f64; 4] {
        let start = ring_offsets[ext_ring] as usize;
        let end = ring_offsets[ext_ring + 1] as usize;
        if start >= end {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let (rx, ry) = (&xs[start..end], &ys[start..end]);
        [
            rx.iter().cloned().fold(f64::INFINITY, f64::min),
            ry.iter().cloned().fold(f64::INFINITY, f64::min),
            rx.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ry.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ]
    }

    #[test]
    fn part_bounds_matches_four_separate_folds() {
        // Fusing the passes and spreading them over rayon must not move a single bound
        let (xs, ys, ring_offsets, poly_offsets) = ring_fixture(500);
        for &ext_ring in &poly_offsets[..poly_offsets.len() - 1] {
            let ext_ring = ext_ring as usize;
            assert_eq!(
                part_bounds(&xs, &ys, &ring_offsets, ext_ring),
                reference_bounds(&xs, &ys, &ring_offsets, ext_ring),
                "part bounds differ at ring {ext_ring}"
            );
        }
    }

    #[test]
    fn build_polygons_answers_ranges_over_an_empty_ring() {
        let (xs, ys, ring_offsets, poly_offsets) = ring_fixture(500);
        let tree = PackedRTree::build_polygons(&xs, &ys, &ring_offsets, &poly_offsets);
        for i in 0..40 {
            let (qx, qy) = (i as f64 * 9.0 - 180.0, i as f64 * 4.5 - 90.0);
            let hits = tree.range(qx, qy, qx + 20.0, qy + 20.0);
            let mut expected: Vec<usize> = (0..poly_offsets.len() - 1)
                .filter(|&part| {
                    let b = reference_bounds(&xs, &ys, &ring_offsets, poly_offsets[part] as usize);
                    b[0] <= qx + 20.0 && b[2] >= qx && b[1] <= qy + 20.0 && b[3] >= qy
                })
                .collect();
            expected.sort_unstable();
            assert_eq!(sorted(hits), expected, "range differs at ({qx}, {qy})");
        }
    }

    #[test]
    fn nearest_into_clears_the_buffer_it_is_handed() {
        let (xs, ys) = five_point_grid();
        let tree = build(xs, ys);
        let mut got = vec![99, 98, 97];
        tree.nearest_into(1.2, 0.1, 1, &mut got);
        assert_eq!(got, vec![1]);
    }
}
