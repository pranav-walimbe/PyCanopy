// Cross-index consistency: BruteForce is the oracle. All other index types must
// return the same result *set* (order may differ) for the same dataset and query.

use std::collections::HashSet;

use geo::{coord, Geometry, LineString, Point, Polygon, Rect};
use pycanopy::index::{
    brute::BruteForce, grid::UniformGrid, kdtree::PackedKdTree, rtree::PackedRTree, SpatialIndex,
};
use pycanopy::query::range::{query_contains, query_range};

fn five_polygon_grid() -> Vec<Geometry<f64>> {
    // Five non-overlapping unit squares:
    //   3=(0,2)-(1,3)   4=(2,2)-(3,3)
    //   0=(0,0)-(1,1)   1=(2,0)-(3,1)   2=(4,0)-(5,1)
    let sq = |min_x: f64, min_y: f64| {
        Geometry::Polygon(Polygon::new(
            LineString::new(vec![
                coord! { x: min_x,       y: min_y       },
                coord! { x: min_x + 1.0, y: min_y       },
                coord! { x: min_x + 1.0, y: min_y + 1.0 },
                coord! { x: min_x,       y: min_y + 1.0 },
                coord! { x: min_x,       y: min_y       },
            ]),
            vec![],
        ))
    };
    vec![
        sq(0.0, 0.0),
        sq(2.0, 0.0),
        sq(4.0, 0.0),
        sq(0.0, 2.0),
        sq(2.0, 2.0),
    ]
}

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

// polygon queries: BruteForce is the oracle, RTree must agree (two-phase: MBR + exact check)

#[test]
fn polygon_range_brute_and_rtree_agree() {
    let geoms = five_polygon_grid();
    // bbox covers squares 0, 1, 3, 4 — misses square 2 at (4,0)-(5,1)
    let bbox = Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 3.0, y: 3.0 });
    let brute = as_set(query_range(&BruteForce::build(&geoms), &bbox, &geoms));
    let rtree = as_set(query_range(&PackedRTree::build(&geoms), &bbox, &geoms));
    assert_eq!(rtree, brute);
    assert_eq!(brute, [0, 1, 3, 4].into_iter().collect());
}

#[test]
fn polygon_range_empty_brute_and_rtree_agree() {
    let geoms = five_polygon_grid();
    let bbox = Rect::new(coord! { x: 10.0, y: 10.0 }, coord! { x: 20.0, y: 20.0 });
    let brute = query_range(&BruteForce::build(&geoms), &bbox, &geoms);
    let rtree = query_range(&PackedRTree::build(&geoms), &bbox, &geoms);
    assert!(brute.is_empty());
    assert!(rtree.is_empty());
}

#[test]
fn polygon_contains_interior_point_brute_and_rtree_agree() {
    let geoms = five_polygon_grid();
    let point = Point::new(0.5, 0.5); // inside square 0
    let brute = query_contains(&BruteForce::build(&geoms), &point, &geoms);
    let rtree = query_contains(&PackedRTree::build(&geoms), &point, &geoms);
    assert_eq!(rtree, brute);
    assert_eq!(brute, vec![0]);
}

#[test]
fn polygon_contains_gap_point_returns_empty() {
    let geoms = five_polygon_grid();
    let point = Point::new(1.5, 0.5); // gap between squares 0 and 1
    let brute = query_contains(&BruteForce::build(&geoms), &point, &geoms);
    let rtree = query_contains(&PackedRTree::build(&geoms), &point, &geoms);
    assert!(brute.is_empty());
    assert!(rtree.is_empty());
}
