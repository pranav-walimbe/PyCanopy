// Cross-index consistency: BruteForce is the oracle. All other index types must
// return the same result *set* (order may differ) for the same dataset and query.

use std::collections::HashSet;

use geo::{coord, Geometry, Point, Rect};
use pycanopy::index::{
    brute::BruteForce, grid::UniformGrid, kdtree::PackedKdTree, rtree::PackedRTree, SpatialIndex,
};

fn five_point_grid() -> Vec<Geometry<f64>> {
    vec![
        Geometry::Point(Point::new(0.0, 0.0)),
        Geometry::Point(Point::new(1.0, 0.0)),
        Geometry::Point(Point::new(2.0, 0.0)),
        Geometry::Point(Point::new(0.0, 1.0)),
        Geometry::Point(Point::new(1.0, 1.0)),
    ]
}

fn as_set(v: Vec<usize>) -> HashSet<usize> {
    v.into_iter().collect()
}

// nearest

#[test]
fn nearest_k1_all_implementations_agree() {
    let geoms = five_point_grid();
    let q = Point::new(1.2, 0.1);
    let oracle = as_set(BruteForce::build(&geoms).nearest(&q, 1));

    assert_eq!(
        as_set(PackedRTree::build(&geoms).nearest(&q, 1)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(&geoms).nearest(&q, 1)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(&geoms).nearest(&q, 1)),
        oracle,
        "grid"
    );
}

#[test]
fn nearest_k3_all_implementations_agree() {
    let geoms = five_point_grid();
    let q = Point::new(1.2, 0.1);
    let oracle = as_set(BruteForce::build(&geoms).nearest(&q, 3));

    assert_eq!(
        as_set(PackedRTree::build(&geoms).nearest(&q, 3)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(&geoms).nearest(&q, 3)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(&geoms).nearest(&q, 3)),
        oracle,
        "grid"
    );
}

// range

#[test]
fn range_non_empty_all_implementations_agree() {
    let geoms = five_point_grid();
    let bbox = Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 1.5, y: 0.5 });
    let oracle = as_set(BruteForce::build(&geoms).range(&bbox));

    assert_eq!(
        as_set(PackedRTree::build(&geoms).range(&bbox)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(&geoms).range(&bbox)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(&geoms).range(&bbox)),
        oracle,
        "grid"
    );
}

#[test]
fn range_empty_all_implementations_agree() {
    let geoms = five_point_grid();
    let bbox = Rect::new(coord! { x: 5.0, y: 5.0 }, coord! { x: 10.0, y: 10.0 });

    assert!(BruteForce::build(&geoms).range(&bbox).is_empty(), "brute");
    assert!(PackedRTree::build(&geoms).range(&bbox).is_empty(), "rtree");
    assert!(
        PackedKdTree::build(&geoms).range(&bbox).is_empty(),
        "kdtree"
    );
    assert!(UniformGrid::build(&geoms).range(&bbox).is_empty(), "grid");
}

// larger synthetic dataset

#[test]
fn nearest_k5_on_larger_dataset_all_agree() {
    let geoms: Vec<Geometry<f64>> = (0..100)
        .map(|i| {
            let x = (i % 10) as f64;
            let y = (i / 10) as f64;
            Geometry::Point(Point::new(x, y))
        })
        .collect();

    let q = Point::new(4.6, 3.2);
    let oracle = as_set(BruteForce::build(&geoms).nearest(&q, 5));

    assert_eq!(
        as_set(PackedRTree::build(&geoms).nearest(&q, 5)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(&geoms).nearest(&q, 5)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(&geoms).nearest(&q, 5)),
        oracle,
        "grid"
    );
}

#[test]
fn range_on_larger_dataset_all_agree() {
    let geoms: Vec<Geometry<f64>> = (0..100)
        .map(|i| {
            let x = (i % 10) as f64;
            let y = (i / 10) as f64;
            Geometry::Point(Point::new(x, y))
        })
        .collect();

    let bbox = Rect::new(coord! { x: 2.0, y: 2.0 }, coord! { x: 5.0, y: 5.0 });
    let oracle = as_set(BruteForce::build(&geoms).range(&bbox));

    assert_eq!(
        as_set(PackedRTree::build(&geoms).range(&bbox)),
        oracle,
        "rtree"
    );
    assert_eq!(
        as_set(PackedKdTree::build(&geoms).range(&bbox)),
        oracle,
        "kdtree"
    );
    assert_eq!(
        as_set(UniformGrid::build(&geoms).range(&bbox)),
        oracle,
        "grid"
    );
}
