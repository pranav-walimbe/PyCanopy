use geo::{coord, BoundingRect, Geometry, Rect};

use crate::stats::types::{DatasetStats, Distribution, GeometryKind};

/// Collect statistics from a geometry slice
pub fn collect(geometries: &[Geometry<f64>]) -> DatasetStats {
    let n = geometries.len();
    if n == 0 {
        return DatasetStats {
            n: 0,
            kind: GeometryKind::Empty,
            extent: None,
            distribution: Distribution::Unknown,
            mean_density: 0.0,
        };
    }

    let kind = classify_kind(geometries);
    let extent = compute_extent(geometries);
    let distribution = estimate_distribution(geometries, &extent, kind);
    let mean_density = extent
        .map(|e| {
            let area = (e.max().x - e.min().x) * (e.max().y - e.min().y);
            if area > 0.0 {
                n as f64 / area
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    DatasetStats {
        n,
        kind,
        extent,
        distribution,
        mean_density,
    }
}

fn classify_kind(geometries: &[Geometry<f64>]) -> GeometryKind {
    let mut has_point = false;
    let mut has_line = false;
    let mut has_poly = false;

    for geom in geometries.iter().take(200) {
        match geom {
            Geometry::Point(_) | Geometry::MultiPoint(_) => has_point = true,
            Geometry::Line(_) | Geometry::LineString(_) | Geometry::MultiLineString(_) => {
                has_line = true
            }
            Geometry::Polygon(_) | Geometry::MultiPolygon(_) => has_poly = true,
            _ => {}
        }
    }

    match (has_point, has_line, has_poly) {
        (true, false, false) => GeometryKind::Point,
        (false, true, false) => GeometryKind::LineString,
        (false, false, true) => GeometryKind::Polygon,
        _ => GeometryKind::Mixed,
    }
}

fn compute_extent(geometries: &[Geometry<f64>]) -> Option<Rect<f64>> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for geom in geometries {
        if let Some(bbox) = geom.bounding_rect() {
            min_x = min_x.min(bbox.min().x);
            min_y = min_y.min(bbox.min().y);
            max_x = max_x.max(bbox.max().x);
            max_y = max_y.max(bbox.max().y);
        }
    }

    if min_x.is_finite() {
        Some(Rect::new(
            coord! { x: min_x, y: min_y },
            coord! { x: max_x, y: max_y },
        ))
    } else {
        None
    }
}

// Grid-based coefficient-of-variation test for point datasets.
// CV > 1.5 → clustered, otherwise uniform.
fn estimate_distribution(
    geometries: &[Geometry<f64>],
    extent: &Option<Rect<f64>>,
    kind: GeometryKind,
) -> Distribution {
    if kind != GeometryKind::Point {
        return Distribution::Unknown;
    }
    let ext = match extent {
        Some(e) => e,
        None => return Distribution::Unknown,
    };
    let n = geometries.len();
    if n < 20 {
        return Distribution::Unknown;
    }

    let w = ext.max().x - ext.min().x;
    let h = ext.max().y - ext.min().y;
    if w <= 0.0 || h <= 0.0 {
        return Distribution::Unknown;
    }

    let grid_dim = (n as f64).sqrt().max(4.0) as usize;
    let mut counts = vec![0u32; grid_dim * grid_dim];

    for geom in geometries {
        if let Geometry::Point(p) = geom {
            let cx = ((p.x() - ext.min().x) / w * grid_dim as f64)
                .min(grid_dim as f64 - 1.0)
                .max(0.0) as usize;
            let cy = ((p.y() - ext.min().y) / h * grid_dim as f64)
                .min(grid_dim as f64 - 1.0)
                .max(0.0) as usize;
            counts[cy * grid_dim + cx] += 1;
        }
    }

    let mean = n as f64 / (grid_dim * grid_dim) as f64;
    let variance: f64 = counts
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / (grid_dim * grid_dim) as f64;
    let cv = variance.sqrt() / mean;

    if cv > 1.5 {
        Distribution::Clustered
    } else {
        Distribution::Uniform
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Geometry<f64> {
        use geo::Point;
        Geometry::Point(Point::new(x, y))
    }

    fn point_grid_5x5() -> Vec<Geometry<f64>> {
        (0..5)
            .flat_map(|row| (0..5).map(move |col| pt(col as f64, row as f64)))
            .collect()
    }

    // All 25 points near origin; one outlier at (100, 100) sets a large extent,
    // making almost all grid cells empty → high CV → Clustered.
    fn clustered_25() -> Vec<Geometry<f64>> {
        let mut geoms: Vec<_> = (0..24).map(|i| pt(i as f64 * 0.01, 0.0)).collect();
        geoms.push(pt(100.0, 100.0));
        geoms
    }

    #[test]
    fn collect_empty_dataset() {
        let stats = collect(&[]);
        assert_eq!(stats.n, 0);
        assert_eq!(stats.kind, GeometryKind::Empty);
        assert!(stats.extent.is_none());
    }

    #[test]
    fn classify_pure_points() {
        let geoms = vec![pt(0.0, 0.0), pt(1.0, 1.0)];
        assert_eq!(super::classify_kind(&geoms), GeometryKind::Point);
    }

    #[test]
    fn classify_mixed_kinds() {
        use geo::{Geometry, LineString};
        let geoms = vec![
            pt(0.0, 0.0),
            Geometry::LineString(LineString::from(vec![(1.0, 1.0), (2.0, 2.0)])),
        ];
        assert_eq!(super::classify_kind(&geoms), GeometryKind::Mixed);
    }

    #[test]
    fn extent_computed_correctly() {
        let geoms = vec![pt(1.0, 2.0), pt(3.0, 4.0), pt(-1.0, 0.0)];
        let ext = super::compute_extent(&geoms).expect("should have extent");
        assert!((ext.min().x - (-1.0)).abs() < 1e-10);
        assert!((ext.min().y - 0.0).abs() < 1e-10);
        assert!((ext.max().x - 3.0).abs() < 1e-10);
        assert!((ext.max().y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn uniform_distribution_detected() {
        let geoms = point_grid_5x5();
        let stats = collect(&geoms);
        assert_eq!(stats.distribution, Distribution::Uniform);
    }

    #[test]
    fn clustered_distribution_detected() {
        let geoms = clustered_25();
        let stats = collect(&geoms);
        assert_eq!(stats.distribution, Distribution::Clustered);
    }

    #[test]
    fn mean_density_computed() {
        let geoms = point_grid_5x5(); // 25 pts, extent (0,0)-(4,4) = area 16
        let stats = collect(&geoms);
        let expected = 25.0 / 16.0;
        assert!((stats.mean_density - expected).abs() < 1e-6);
    }
}
