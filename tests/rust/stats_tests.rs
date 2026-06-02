use geo::{Geometry, Point};
use pycanopy::stats::{
    collector::collect,
    types::{Distribution, GeometryKind},
};

fn pt(x: f64, y: f64) -> Geometry<f64> {
    Geometry::Point(Point::new(x, y))
}

fn point_grid_5x5() -> Vec<Geometry<f64>> {
    (0..5)
        .flat_map(|row| (0..5).map(move |col| pt(col as f64, row as f64)))
        .collect()
}

fn clustered_25() -> Vec<Geometry<f64>> {
    let mut geoms: Vec<_> = (0..24).map(|i| pt(i as f64 * 0.01, 0.0)).collect();
    geoms.push(pt(100.0, 100.0)); // outlier sets a large extent
    geoms
}

#[test]
fn collect_empty_returns_empty_stats() {
    let s = collect(&[]);
    assert_eq!(s.n, 0);
    assert_eq!(s.kind, GeometryKind::Empty);
    assert!(s.extent.is_none());
}

#[test]
fn collect_point_grid_classifies_correctly() {
    let s = collect(&point_grid_5x5());
    assert_eq!(s.n, 25);
    assert_eq!(s.kind, GeometryKind::Point);
}

#[test]
fn collect_point_grid_extent_is_correct() {
    let s = collect(&point_grid_5x5());
    let ext = s.extent.expect("should have extent");
    assert!((ext.min().x - 0.0).abs() < 1e-10);
    assert!((ext.min().y - 0.0).abs() < 1e-10);
    assert!((ext.max().x - 4.0).abs() < 1e-10);
    assert!((ext.max().y - 4.0).abs() < 1e-10);
}

#[test]
fn collect_uniform_grid_detects_uniform_distribution() {
    let s = collect(&point_grid_5x5());
    assert_eq!(s.distribution, Distribution::Uniform);
}

#[test]
fn collect_clustered_points_detects_clustered_distribution() {
    let s = collect(&clustered_25());
    assert_eq!(s.distribution, Distribution::Clustered);
}

#[test]
fn collect_mixed_geometries_classifies_as_mixed() {
    use geo::LineString;
    let geoms = vec![
        pt(0.0, 0.0),
        Geometry::LineString(LineString::from(vec![(1.0, 1.0), (2.0, 2.0)])),
    ];
    let s = collect(&geoms);
    assert_eq!(s.kind, GeometryKind::Mixed);
}

#[test]
fn collect_mean_density_is_n_over_area() {
    let s = collect(&point_grid_5x5()); // 25 pts, extent 4×4 = area 16
    let expected = 25.0 / 16.0;
    assert!((s.mean_density - expected).abs() < 1e-6);
}
