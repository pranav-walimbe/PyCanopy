// Cross-index consistency: BruteForce is the oracle. All other index types must
// return the same result *set* (order may differ) for the same dataset and query.

use std::collections::HashSet;
use std::sync::Arc;

use pycanopy::index::{
    brute::BruteForce, grid::UniformGrid, kdtree::PackedKdTree, rtree::PackedRTree, SpatialIndex,
};
use pycanopy::query::range::{query_contains_polygons, query_range_polygons};

fn five_point_grid() -> (Arc<[f64]>, Arc<[f64]>) {
    (
        Arc::from([0.0f64, 1.0, 2.0, 0.0, 1.0].as_slice()),
        Arc::from([0.0f64, 0.0, 0.0, 1.0, 1.0].as_slice()),
    )
}

// Five non-overlapping unit squares (no holes).
//   3=(0,2)-(1,3)   4=(2,2)-(3,3)
//   0=(0,0)-(1,1)   1=(2,0)-(3,1)   2=(4,0)-(5,1)
//
// Returns (xs, ys, ring_offsets, poly_offsets).
// For simple polygons poly_offsets is [0,1,...,n_polys].
fn five_polygon_grid() -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
    let sq = |min_x: f64, min_y: f64| -> Vec<(f64, f64)> {
        vec![
            (min_x, min_y),
            (min_x + 1.0, min_y),
            (min_x + 1.0, min_y + 1.0),
            (min_x, min_y + 1.0),
            (min_x, min_y),
        ]
    };
    let polys = [
        sq(0.0, 0.0),
        sq(2.0, 0.0),
        sq(4.0, 0.0),
        sq(0.0, 2.0),
        sq(2.0, 2.0),
    ];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ring_offsets: Vec<i64> = vec![0];
    for poly in &polys {
        for &(x, y) in poly {
            xs.push(x);
            ys.push(y);
        }
        ring_offsets.push(xs.len() as i64);
    }
    let poly_offsets: Vec<i64> = (0..=polys.len() as i64).collect();
    (xs, ys, ring_offsets, poly_offsets)
}

// One polygon with a square hole: outer (0,0)-(4,4), hole (1,1)-(3,3).
// Points inside the hole must NOT be contained.
fn polygon_with_hole() -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
    let outer = [
        (0.0f64, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (0.0, 4.0),
        (0.0, 0.0),
    ];
    let hole = [
        (1.0f64, 1.0),
        (3.0, 1.0),
        (3.0, 3.0),
        (1.0, 3.0),
        (1.0, 1.0),
    ];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &(x, y) in outer.iter().chain(hole.iter()) {
        xs.push(x);
        ys.push(y);
    }
    // ring 0 = outer (coords 0..5), ring 1 = hole (coords 5..10)
    let ring_offsets = vec![0i64, 5, 10];
    // polygon 0 uses rings 0..2
    let poly_offsets = vec![0i64, 2];
    (xs, ys, ring_offsets, poly_offsets)
}

fn as_set(v: Vec<usize>) -> HashSet<usize> {
    v.into_iter().collect()
}

// nearest

#[test]
fn nearest_k1_all_implementations_agree() {
    let (xs, ys) = five_point_grid();
    let oracle = as_set(BruteForce::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 1));

    assert_eq!(
        as_set(PackedRTree::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 1)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 1)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 1)),
        oracle,
        "grid"
    );
}

#[test]
fn nearest_k3_all_implementations_agree() {
    let (xs, ys) = five_point_grid();
    let oracle = as_set(BruteForce::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 3));

    assert_eq!(
        as_set(PackedRTree::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 3)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 3)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(1.2, 0.1, 3)),
        oracle,
        "grid"
    );
}

// range

#[test]
fn range_non_empty_all_implementations_agree() {
    let (xs, ys) = five_point_grid();
    let oracle =
        as_set(BruteForce::build(Arc::clone(&xs), Arc::clone(&ys)).range(0.0, 0.0, 1.5, 0.5));

    assert_eq!(
        as_set(PackedRTree::build(Arc::clone(&xs), Arc::clone(&ys)).range(0.0, 0.0, 1.5, 0.5)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(Arc::clone(&xs), Arc::clone(&ys)).range(0.0, 0.0, 1.5, 0.5)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(Arc::clone(&xs), Arc::clone(&ys)).range(0.0, 0.0, 1.5, 0.5)),
        oracle,
        "grid"
    );
}

#[test]
fn range_empty_all_implementations_agree() {
    let (xs, ys) = five_point_grid();

    assert!(
        BruteForce::build(Arc::clone(&xs), Arc::clone(&ys))
            .range(5.0, 5.0, 10.0, 10.0)
            .is_empty(),
        "brute"
    );
    assert!(
        PackedRTree::build(Arc::clone(&xs), Arc::clone(&ys))
            .range(5.0, 5.0, 10.0, 10.0)
            .is_empty(),
        "rtree"
    );
    assert!(
        PackedKdTree::build(Arc::clone(&xs), Arc::clone(&ys))
            .range(5.0, 5.0, 10.0, 10.0)
            .is_empty(),
        "kdtree"
    );
    assert!(
        UniformGrid::build(Arc::clone(&xs), Arc::clone(&ys))
            .range(5.0, 5.0, 10.0, 10.0)
            .is_empty(),
        "grid"
    );
}

// larger synthetic dataset

#[test]
fn nearest_k5_on_larger_dataset_all_agree() {
    let xs: Arc<[f64]> = (0..100).map(|i| (i % 10) as f64).collect::<Vec<_>>().into();
    let ys: Arc<[f64]> = (0..100).map(|i| (i / 10) as f64).collect::<Vec<_>>().into();
    let oracle = as_set(BruteForce::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(4.6, 3.2, 5));

    assert_eq!(
        as_set(PackedRTree::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(4.6, 3.2, 5)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(4.6, 3.2, 5)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(Arc::clone(&xs), Arc::clone(&ys)).nearest(4.6, 3.2, 5)),
        oracle,
        "grid"
    );
}

#[test]
fn range_on_larger_dataset_all_agree() {
    let xs: Arc<[f64]> = (0..100).map(|i| (i % 10) as f64).collect::<Vec<_>>().into();
    let ys: Arc<[f64]> = (0..100).map(|i| (i / 10) as f64).collect::<Vec<_>>().into();
    let oracle =
        as_set(BruteForce::build(Arc::clone(&xs), Arc::clone(&ys)).range(2.0, 2.0, 5.0, 5.0));

    assert_eq!(
        as_set(PackedRTree::build(Arc::clone(&xs), Arc::clone(&ys)).range(2.0, 2.0, 5.0, 5.0)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(Arc::clone(&xs), Arc::clone(&ys)).range(2.0, 2.0, 5.0, 5.0)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(Arc::clone(&xs), Arc::clone(&ys)).range(2.0, 2.0, 5.0, 5.0)),
        oracle,
        "grid"
    );
}

// polygon queries: BruteForce is the oracle, RTree must agree (two-phase: MBR + exact check)
// Each test builds brute + rtree once and runs all related assertions against them.

#[test]
fn polygon_range_brute_and_rtree_agree() {
    let (xs, ys, ring_off, poly_off) = five_polygon_grid();
    let brute = BruteForce::build_polygons(&xs, &ys, &ring_off, &poly_off);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &ring_off, &poly_off);

    // bbox covers squares 0, 1, 3, 4 — misses square 2 at (4,0)-(5,1)
    let b = as_set(query_range_polygons(
        &brute, &xs, &ys, &ring_off, &poly_off, 0.0, 0.0, 3.0, 3.0,
    ));
    let r = as_set(query_range_polygons(
        &rtree, &xs, &ys, &ring_off, &poly_off, 0.0, 0.0, 3.0, 3.0,
    ));
    assert_eq!(b, [0, 1, 3, 4].into_iter().collect());
    assert_eq!(r, b, "rtree agrees");

    // bbox beyond all polygons returns empty
    let b_empty = query_range_polygons(
        &brute, &xs, &ys, &ring_off, &poly_off, 10.0, 10.0, 20.0, 20.0,
    );
    let r_empty = query_range_polygons(
        &rtree, &xs, &ys, &ring_off, &poly_off, 10.0, 10.0, 20.0, 20.0,
    );
    assert!(b_empty.is_empty());
    assert!(r_empty.is_empty());
}

#[test]
fn polygon_contains_brute_and_rtree_agree() {
    let (xs, ys, ring_off, poly_off) = five_polygon_grid();
    let brute = BruteForce::build_polygons(&xs, &ys, &ring_off, &poly_off);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &ring_off, &poly_off);

    // interior point of square 0
    let b = query_contains_polygons(&brute, &xs, &ys, &ring_off, &poly_off, 0.5, 0.5);
    let r = query_contains_polygons(&rtree, &xs, &ys, &ring_off, &poly_off, 0.5, 0.5);
    assert_eq!(b, vec![0]);
    assert_eq!(r, b, "rtree agrees");

    // gap between squares 0 and 1 — no polygon
    let b_gap = query_contains_polygons(&brute, &xs, &ys, &ring_off, &poly_off, 1.5, 0.5);
    let r_gap = query_contains_polygons(&rtree, &xs, &ys, &ring_off, &poly_off, 1.5, 0.5);
    assert!(b_gap.is_empty());
    assert!(r_gap.is_empty());
}

// polygon holes

#[test]
fn polygon_with_hole_contains_and_range() {
    let (xs, ys, ring_off, poly_off) = polygon_with_hole();
    let brute = BruteForce::build_polygons(&xs, &ys, &ring_off, &poly_off);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &ring_off, &poly_off);

    // (0.5, 0.5) is inside the outer ring but outside the hole — must be contained
    let b_in = query_contains_polygons(&brute, &xs, &ys, &ring_off, &poly_off, 0.5, 0.5);
    let r_in = query_contains_polygons(&rtree, &xs, &ys, &ring_off, &poly_off, 0.5, 0.5);
    assert_eq!(b_in, vec![0], "point outside hole is contained");
    assert_eq!(r_in, b_in, "rtree agrees");

    // (2.0, 2.0) is inside the hole — must NOT be contained
    let b_hole = query_contains_polygons(&brute, &xs, &ys, &ring_off, &poly_off, 2.0, 2.0);
    let r_hole = query_contains_polygons(&rtree, &xs, &ys, &ring_off, &poly_off, 2.0, 2.0);
    assert!(b_hole.is_empty(), "point in hole is not contained");
    assert!(r_hole.is_empty(), "rtree agrees");

    // range query overlapping the MBR still returns the polygon
    let b_rng = as_set(query_range_polygons(
        &brute, &xs, &ys, &ring_off, &poly_off, 0.0, 0.0, 2.0, 2.0,
    ));
    let r_rng = as_set(query_range_polygons(
        &rtree, &xs, &ys, &ring_off, &poly_off, 0.0, 0.0, 2.0, 2.0,
    ));
    assert_eq!(b_rng, [0].into_iter().collect());
    assert_eq!(r_rng, b_rng, "rtree agrees");
}
