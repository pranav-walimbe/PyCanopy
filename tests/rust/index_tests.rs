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

// Five non-overlapping unit squares:
//   3=(0,2)-(1,3)   4=(2,2)-(3,3)
//   0=(0,0)-(1,1)   1=(2,0)-(3,1)   2=(4,0)-(5,1)
fn five_polygon_grid() -> (Vec<f64>, Vec<f64>, Vec<i64>) {
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
    let mut offsets: Vec<i64> = vec![0];
    for poly in &polys {
        for &(x, y) in poly {
            xs.push(x);
            ys.push(y);
        }
        offsets.push(xs.len() as i64);
    }
    (xs, ys, offsets)
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

#[test]
fn polygon_range_brute_and_rtree_agree() {
    let (xs, ys, offsets) = five_polygon_grid();
    let brute = BruteForce::build_polygons(&xs, &ys, &offsets);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &offsets);

    // bbox covers squares 0, 1, 3, 4 — misses square 2 at (4,0)-(5,1)
    let b = as_set(query_range_polygons(
        &brute, &xs, &ys, &offsets, 0.0, 0.0, 3.0, 3.0,
    ));
    let r = as_set(query_range_polygons(
        &rtree, &xs, &ys, &offsets, 0.0, 0.0, 3.0, 3.0,
    ));
    assert_eq!(r, b);
    assert_eq!(b, [0, 1, 3, 4].into_iter().collect());
}

#[test]
fn polygon_range_empty_brute_and_rtree_agree() {
    let (xs, ys, offsets) = five_polygon_grid();
    let brute = BruteForce::build_polygons(&xs, &ys, &offsets);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &offsets);

    let b = query_range_polygons(&brute, &xs, &ys, &offsets, 10.0, 10.0, 20.0, 20.0);
    let r = query_range_polygons(&rtree, &xs, &ys, &offsets, 10.0, 10.0, 20.0, 20.0);
    assert!(b.is_empty());
    assert!(r.is_empty());
}

#[test]
fn polygon_contains_interior_point_brute_and_rtree_agree() {
    let (xs, ys, offsets) = five_polygon_grid();
    let brute = BruteForce::build_polygons(&xs, &ys, &offsets);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &offsets);

    let b = query_contains_polygons(&brute, &xs, &ys, &offsets, 0.5, 0.5);
    let r = query_contains_polygons(&rtree, &xs, &ys, &offsets, 0.5, 0.5);
    assert_eq!(r, b);
    assert_eq!(b, vec![0]);
}

#[test]
fn polygon_contains_gap_point_returns_empty() {
    let (xs, ys, offsets) = five_polygon_grid();
    let brute = BruteForce::build_polygons(&xs, &ys, &offsets);
    let rtree = PackedRTree::build_polygons(&xs, &ys, &offsets);

    // gap between squares 0 and 1
    let b = query_contains_polygons(&brute, &xs, &ys, &offsets, 1.5, 0.5);
    let r = query_contains_polygons(&rtree, &xs, &ys, &offsets, 1.5, 0.5);
    assert!(b.is_empty());
    assert!(r.is_empty());
}
