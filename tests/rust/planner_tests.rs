use geo::{coord, Point, Rect};
use pycanopy::{
    planner::{
        cost::{selectivity, IndexKind},
        selector::select_index,
    },
    query::types::Query,
    stats::types::{DatasetStats, Distribution, GeometryKind},
};

fn stats(n: usize, kind: GeometryKind, dist: Distribution) -> DatasetStats {
    DatasetStats {
        n,
        kind,
        extent: Some(Rect::new(
            coord! { x: 0.0, y: 0.0 },
            coord! { x: 100.0, y: 100.0 },
        )),
        distribution: dist,
        mean_density: n as f64 / 10_000.0,
        histogram: None,
    }
}

fn small_range_query() -> Query {
    // 1% of 10 000 area = 100 → selectivity = 0.01
    Query::Range {
        bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 10.0, y: 10.0 }),
    }
}

fn full_range_query() -> Query {
    Query::Range {
        bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 100.0, y: 100.0 }),
    }
}

// selectivity

#[test]
fn selectivity_full_extent_is_one() {
    let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
    assert!((selectivity(&s, &full_range_query()) - 1.0).abs() < 1e-10);
}

#[test]
fn selectivity_quarter_extent_is_point_25() {
    let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
    let q = Query::Range {
        bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 50.0, y: 50.0 }),
    };
    assert!((selectivity(&s, &q) - 0.25).abs() < 1e-10);
}

// selector routing

#[test]
fn small_dataset_always_brute_force() {
    let s = stats(100, GeometryKind::Point, Distribution::Uniform);
    assert_eq!(
        select_index(&s, &small_range_query()),
        IndexKind::BruteForce
    );
}

#[test]
fn high_selectivity_bypasses_index() {
    let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
    assert_eq!(select_index(&s, &full_range_query()), IndexKind::BruteForce);
}

#[test]
fn point_knn_routes_to_kdtree() {
    let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
    let q = Query::Knn {
        point: Point::new(50.0, 50.0),
        k: 5,
        approximate: false,
    };
    assert_eq!(select_index(&s, &q), IndexKind::KdTree);
}

#[test]
fn point_uniform_range_routes_to_grid() {
    let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
    assert_eq!(select_index(&s, &small_range_query()), IndexKind::Grid);
}

#[test]
fn point_clustered_range_routes_to_kdtree() {
    let s = stats(1000, GeometryKind::Point, Distribution::Clustered);
    assert_eq!(select_index(&s, &small_range_query()), IndexKind::KdTree);
}

#[test]
fn polygon_range_routes_to_rtree() {
    let s = stats(1000, GeometryKind::Polygon, Distribution::Unknown);
    assert_eq!(select_index(&s, &small_range_query()), IndexKind::RTree);
}

#[test]
fn polygon_contains_routes_to_rtree() {
    let s = stats(1000, GeometryKind::Polygon, Distribution::Unknown);
    let q = Query::Contains {
        point: Point::new(50.0, 50.0),
    };
    assert_eq!(select_index(&s, &q), IndexKind::RTree);
}

#[test]
fn polygon_knn_routes_to_rtree() {
    let s = stats(1000, GeometryKind::Polygon, Distribution::Unknown);
    let q = Query::Knn {
        point: Point::new(50.0, 50.0),
        k: 5,
        approximate: false,
    };
    assert_eq!(select_index(&s, &q), IndexKind::RTree);
}

#[test]
fn knn_with_large_k_fraction_falls_back_to_brute() {
    // k/N = 200/1000 = 0.20 > threshold 0.10
    let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
    let q = Query::Knn {
        point: Point::new(0.0, 0.0),
        k: 200,
        approximate: false,
    };
    assert_eq!(select_index(&s, &q), IndexKind::BruteForce);
}

// end-to-end: collect stats then select index

#[test]
fn pipeline_polygon_range_selects_rtree() {
    use pycanopy::stats::collector::collect_polygons;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut offsets: Vec<i64> = vec![0];
    for i in 0..1000usize {
        let ox = (i % 50) as f64 * 2.0;
        let oy = (i / 50) as f64 * 2.0;
        xs.extend_from_slice(&[ox, ox + 1.0, ox + 1.0, ox, ox]);
        ys.extend_from_slice(&[oy, oy, oy + 1.0, oy + 1.0, oy]);
        offsets.push(xs.len() as i64);
    }

    let stats = collect_polygons(&xs, &ys, &offsets);
    assert_eq!(stats.kind, GeometryKind::Polygon);
    assert_eq!(select_index(&stats, &small_range_query()), IndexKind::RTree);
}

#[test]
fn pipeline_uniform_points_range_selects_grid() {
    use pycanopy::stats::collector::collect_points;

    let xs: Vec<f64> = (0..1000).map(|i| (i % 50) as f64 * 2.0).collect();
    let ys: Vec<f64> = (0..1000).map(|i| (i / 50) as f64 * 2.0).collect();

    let stats = collect_points(&xs, &ys);
    assert_eq!(stats.kind, GeometryKind::Point);
    assert_eq!(stats.distribution, Distribution::Uniform);
    assert_eq!(select_index(&stats, &small_range_query()), IndexKind::Grid);
}
