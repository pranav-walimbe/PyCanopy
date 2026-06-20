use geo::{coord, Rect};

use crate::stats::types::{
    DatasetStats, Distribution, GeometryKind, SpatialHistogram, HISTOGRAM_RESOLUTION,
};

/// Collect statistics from a flat point coordinate dataset
pub fn collect_points(xs: &[f64], ys: &[f64]) -> DatasetStats {
    let n = xs.len();
    if n == 0 {
        return DatasetStats {
            n: 0,
            kind: GeometryKind::Empty,
            extent: None,
            distribution: Distribution::Unknown,
            mean_density: 0.0,
            histogram: None,
        };
    }

    let extent = compute_extent(xs, ys);
    let distribution = estimate_distribution(xs, ys, &extent);
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
    let histogram = extent.map(|e| build_histogram(xs, ys, &e));

    DatasetStats {
        n,
        kind: GeometryKind::Point,
        extent,
        distribution,
        mean_density,
        histogram,
    }
}

/// Collect statistics from a two-level polygon coordinate dataset
pub fn collect_polygons(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
) -> DatasetStats {
    let n = poly_offsets.len().saturating_sub(1);
    if n == 0 {
        return DatasetStats {
            n: 0,
            kind: GeometryKind::Empty,
            extent: None,
            distribution: Distribution::Unknown,
            mean_density: 0.0,
            histogram: None,
        };
    }

    let extent = compute_extent(xs, ys);
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
    let histogram =
        extent.map(|e| build_polygon_centroid_histogram(xs, ys, ring_offsets, poly_offsets, &e));

    DatasetStats {
        n,
        kind: GeometryKind::Polygon,
        extent,
        distribution: Distribution::Unknown,
        mean_density,
        histogram,
    }
}

fn compute_extent(xs: &[f64], ys: &[f64]) -> Option<Rect<f64>> {
    let (min_x, min_y, max_x, max_y) = xs.iter().zip(ys.iter()).fold(
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        |(mn_x, mn_y, mx_x, mx_y), (&x, &y)| (mn_x.min(x), mn_y.min(y), mx_x.max(x), mx_y.max(y)),
    );
    if min_x.is_finite() {
        Some(Rect::new(
            coord! { x: min_x, y: min_y },
            coord! { x: max_x, y: max_y },
        ))
    } else {
        None
    }
}

// Grid-based coefficient-of-variation test. CV > 1.5 → Clustered, otherwise Uniform
fn estimate_distribution(xs: &[f64], ys: &[f64], extent: &Option<Rect<f64>>) -> Distribution {
    let n = xs.len();
    if n < 20 {
        return Distribution::Unknown;
    }
    let ext = match extent {
        Some(e) => e,
        None => return Distribution::Unknown,
    };
    let w = ext.max().x - ext.min().x;
    let h = ext.max().y - ext.min().y;
    if w <= 0.0 || h <= 0.0 {
        return Distribution::Unknown;
    }

    let grid_dim = (n as f64).sqrt().max(4.0) as usize;
    let mut counts = vec![0u32; grid_dim * grid_dim];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let cx = ((x - ext.min().x) / w * grid_dim as f64)
            .min(grid_dim as f64 - 1.0)
            .max(0.0) as usize;
        let cy = ((y - ext.min().y) / h * grid_dim as f64)
            .min(grid_dim as f64 - 1.0)
            .max(0.0) as usize;
        counts[cy * grid_dim + cx] += 1;
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

fn build_histogram(xs: &[f64], ys: &[f64], extent: &Rect<f64>) -> SpatialHistogram {
    let w = (extent.max().x - extent.min().x).max(f64::EPSILON);
    let h = (extent.max().y - extent.min().y).max(f64::EPSILON);
    let cell_w = w / HISTOGRAM_RESOLUTION as f64;
    let cell_h = h / HISTOGRAM_RESOLUTION as f64;
    let mut counts = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let col = ((x - extent.min().x) / cell_w)
            .floor()
            .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64) as usize;
        let row = ((y - extent.min().y) / cell_h)
            .floor()
            .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64) as usize;
        counts[row * HISTOGRAM_RESOLUTION + col] += 1;
    }
    SpatialHistogram {
        counts,
        min_x: extent.min().x,
        min_y: extent.min().y,
        cell_w,
        cell_h,
    }
}

fn build_polygon_centroid_histogram(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    extent: &Rect<f64>,
) -> SpatialHistogram {
    let w = (extent.max().x - extent.min().x).max(f64::EPSILON);
    let h = (extent.max().y - extent.min().y).max(f64::EPSILON);
    let cell_w = w / HISTOGRAM_RESOLUTION as f64;
    let cell_h = h / HISTOGRAM_RESOLUTION as f64;
    let mut counts = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
    let n_polys = poly_offsets.len().saturating_sub(1);
    for &ext_ring_i64 in poly_offsets.iter().take(n_polys) {
        // Centroid from exterior ring only
        let ext_ring = ext_ring_i64 as usize;
        let start = ring_offsets[ext_ring] as usize;
        let end = ring_offsets[ext_ring + 1] as usize;
        if start >= end {
            continue;
        }
        let count = (end - start) as f64;
        let cx = xs[start..end].iter().sum::<f64>() / count;
        let cy = ys[start..end].iter().sum::<f64>() / count;
        let col = ((cx - extent.min().x) / cell_w)
            .floor()
            .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64) as usize;
        let row = ((cy - extent.min().y) / cell_h)
            .floor()
            .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64) as usize;
        counts[row * HISTOGRAM_RESOLUTION + col] += 1;
    }
    SpatialHistogram {
        counts,
        min_x: extent.min().x,
        min_y: extent.min().y,
        cell_w,
        cell_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{coord, Rect};

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
    fn collect_empty_dataset() {
        let stats = collect_points(&[], &[]);
        assert_eq!(stats.n, 0);
        assert_eq!(stats.kind, GeometryKind::Empty);
        assert!(stats.extent.is_none());
    }

    #[test]
    fn collect_points_kind() {
        let stats = collect_points(&[0.0, 1.0], &[0.0, 1.0]);
        assert_eq!(stats.kind, GeometryKind::Point);
    }

    #[test]
    fn extent_computed_correctly() {
        let stats = collect_points(&[1.0, 3.0, -1.0], &[2.0, 4.0, 0.0]);
        let ext = stats.extent.expect("should have extent");
        assert!((ext.min().x - (-1.0)).abs() < 1e-10);
        assert!((ext.min().y - 0.0).abs() < 1e-10);
        assert!((ext.max().x - 3.0).abs() < 1e-10);
        assert!((ext.max().y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn uniform_distribution_detected() {
        let (xs, ys) = point_grid_5x5();
        let stats = collect_points(&xs, &ys);
        assert_eq!(stats.distribution, Distribution::Uniform);
    }

    #[test]
    fn clustered_distribution_detected() {
        let (xs, ys) = clustered_25();
        let stats = collect_points(&xs, &ys);
        assert_eq!(stats.distribution, Distribution::Clustered);
    }

    #[test]
    fn mean_density_computed() {
        let (xs, ys) = point_grid_5x5(); // 25 pts, extent (0,0)-(4,4) = area 16
        let stats = collect_points(&xs, &ys);
        let expected = 25.0 / 16.0;
        assert!((stats.mean_density - expected).abs() < 1e-6);
    }

    #[test]
    fn histogram_is_none_for_empty_dataset() {
        let stats = collect_points(&[], &[]);
        assert!(stats.histogram.is_none());
    }

    #[test]
    fn histogram_is_some_for_nonempty_dataset() {
        let (xs, ys) = point_grid_5x5();
        let stats = collect_points(&xs, &ys);
        assert!(stats.histogram.is_some());
    }

    #[test]
    fn histogram_counts_sum_to_n() {
        let (xs, ys) = point_grid_5x5();
        let stats = collect_points(&xs, &ys);
        let hist = stats.histogram.unwrap();
        let total: u32 = hist.counts.iter().sum();
        assert_eq!(total as usize, stats.n);
    }

    #[test]
    fn histogram_skewed_selectivity_beats_area_ratio() {
        let (xs, ys) = clustered_25();
        let stats = collect_points(&xs, &ys);
        let hist = stats.histogram.unwrap();
        let bbox = Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 1.0, y: 1.0 });
        let hist_sel = hist.selectivity(&bbox, stats.n);
        let area_sel = 1.0_f64 / (100.0 * 100.0);
        assert!(
            hist_sel > area_sel * 10.0,
            "histogram sel={hist_sel}, area sel={area_sel}"
        );
    }
}
