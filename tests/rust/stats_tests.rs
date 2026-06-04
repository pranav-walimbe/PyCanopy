use pycanopy::stats::{
    collector::{collect_points, collect_polygons},
    types::{Distribution, GeometryKind},
};

fn point_grid_5x5() -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..5)
        .flat_map(|_row| (0..5).map(|col| col as f64))
        .collect();
    let ys: Vec<f64> = (0..5)
        .flat_map(|row| (0..5).map(move |_col| row as f64))
        .collect();
    (xs, ys)
}

fn clustered_25() -> (Vec<f64>, Vec<f64>) {
    let mut xs: Vec<f64> = (0..24).map(|i| i as f64 * 0.01).collect();
    let mut ys: Vec<f64> = vec![0.0; 24];
    xs.push(100.0);
    ys.push(100.0);
    (xs, ys)
}

#[test]
fn collect_empty_returns_empty_stats() {
    let s = collect_points(&[], &[]);
    assert_eq!(s.n, 0);
    assert_eq!(s.kind, GeometryKind::Empty);
    assert!(s.extent.is_none());
}

#[test]
fn collect_point_grid_classifies_correctly() {
    let (xs, ys) = point_grid_5x5();
    let s = collect_points(&xs, &ys);
    assert_eq!(s.n, 25);
    assert_eq!(s.kind, GeometryKind::Point);
}

#[test]
fn collect_point_grid_extent_is_correct() {
    let (xs, ys) = point_grid_5x5();
    let s = collect_points(&xs, &ys);
    let ext = s.extent.expect("should have extent");
    assert!((ext.min().x - 0.0).abs() < 1e-10);
    assert!((ext.min().y - 0.0).abs() < 1e-10);
    assert!((ext.max().x - 4.0).abs() < 1e-10);
    assert!((ext.max().y - 4.0).abs() < 1e-10);
}

#[test]
fn collect_uniform_grid_detects_uniform_distribution() {
    let (xs, ys) = point_grid_5x5();
    let s = collect_points(&xs, &ys);
    assert_eq!(s.distribution, Distribution::Uniform);
}

#[test]
fn collect_clustered_points_detects_clustered_distribution() {
    let (xs, ys) = clustered_25();
    let s = collect_points(&xs, &ys);
    assert_eq!(s.distribution, Distribution::Clustered);
}

#[test]
fn collect_mean_density_is_n_over_area() {
    let (xs, ys) = point_grid_5x5(); // 25 pts, extent 4×4 = area 16
    let s = collect_points(&xs, &ys);
    let expected = 25.0 / 16.0;
    assert!((s.mean_density - expected).abs() < 1e-6);
}

#[test]
fn collect_polygons_classifies_correctly() {
    // Two unit squares
    let xs = vec![0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 3.0, 3.0, 2.0, 2.0];
    let ys = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let ring_offsets: Vec<i64> = vec![0, 5, 10];
    // simple polygons: one ring each, poly_offsets = [0, 1, 2]
    let poly_offsets: Vec<i64> = vec![0, 1, 2];
    let s = collect_polygons(&xs, &ys, &ring_offsets, &poly_offsets);
    assert_eq!(s.n, 2);
    assert_eq!(s.kind, GeometryKind::Polygon);
    assert_eq!(s.distribution, Distribution::Unknown);
}
