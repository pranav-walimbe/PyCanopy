//! Collects dataset statistics from flat point and polygon coordinate arrays.

use geo::{coord, Rect};
use rayon::prelude::*;

use crate::stats::types::{
    DatasetStats, Distribution, GeometryKind, SpatialHistogram, HISTOGRAM_RESOLUTION,
};

/// Per-axis cell count of the counting grid
const COUNT_RESOLUTION: usize = 256;

/// Cells of the counting grid per histogram cell along one axis
const HISTOGRAM_STRIDE: usize = COUNT_RESOLUTION / HISTOGRAM_RESOLUTION;

/// Morisita index above which a dataset counts as clustered
const CLUSTERED_INDEX: f64 = 1.5;

/// Point count below which the distribution is left Unknown
const MIN_POINTS_FOR_DISTRIBUTION: usize = 20;

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

    // One parallel pass fills the counting grid behind both answers
    let counts = extent.map(|e| count_grid(xs, ys, &e));
    let distribution = match (&counts, &extent) {
        (Some(counts), Some(_)) if n >= MIN_POINTS_FOR_DISTRIBUTION => {
            if morisita_index(counts, n) > CLUSTERED_INDEX {
                Distribution::Clustered
            } else {
                Distribution::Uniform
            }
        }
        _ => Distribution::Unknown,
    };
    let histogram = counts
        .as_deref()
        .zip(extent)
        .map(|(counts, e)| fold_to_histogram(counts, &e));

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
    // min/max reduce in parallel: order-independent, so the extent matches a serial fold
    let init = || {
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        )
    };
    let (min_x, min_y, max_x, max_y) = xs
        .par_iter()
        .zip(ys.par_iter())
        .fold(init, |(mn_x, mn_y, mx_x, mx_y), (&x, &y)| {
            (mn_x.min(x), mn_y.min(y), mx_x.max(x), mx_y.max(y))
        })
        .reduce(init, |a, b| {
            (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
        });
    if min_x.is_finite() {
        Some(Rect::new(
            coord! { x: min_x, y: min_y },
            coord! { x: max_x, y: max_y },
        ))
    } else {
        None
    }
}

// Bin every point into a COUNT_RESOLUTION grid
fn count_grid(xs: &[f64], ys: &[f64], extent: &Rect<f64>) -> Vec<u32> {
    let cells = COUNT_RESOLUTION * COUNT_RESOLUTION;
    let w = (extent.max().x - extent.min().x).max(f64::EPSILON);
    let h = (extent.max().y - extent.min().y).max(f64::EPSILON);
    let cell_w = w / COUNT_RESOLUTION as f64;
    let cell_h = h / COUNT_RESOLUTION as f64;
    let (min_x, min_y) = (extent.min().x, extent.min().y);

    // One chunk per thread bounds the number of 256 KiB accumulators
    let chunk = (xs.len() / rayon::current_num_threads().max(1)).max(1 << 14);
    xs.par_chunks(chunk)
        .zip(ys.par_chunks(chunk))
        .map(|(chunk_xs, chunk_ys)| {
            let mut local = vec![0u32; cells];
            for (&x, &y) in chunk_xs.iter().zip(chunk_ys.iter()) {
                let col = ((x - min_x) / cell_w)
                    .floor()
                    .clamp(0.0, (COUNT_RESOLUTION - 1) as f64) as usize;
                let row = ((y - min_y) / cell_h)
                    .floor()
                    .clamp(0.0, (COUNT_RESOLUTION - 1) as f64) as usize;
                local[row * COUNT_RESOLUTION + col] += 1;
            }
            local
        })
        .reduce(
            || vec![0u32; cells],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        )
}

// Q * sum n_i(n_i - 1) / (N(N - 1)) reads 1.0 for a Poisson field at any grid size
fn morisita_index(counts: &[u32], n: usize) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let pairs: f64 = counts
        .iter()
        .map(|&count| {
            let count = count as f64;
            count * (count - 1.0)
        })
        .sum();
    let total = n as f64;
    counts.len() as f64 * pairs / (total * (total - 1.0))
}

// Flooring at 256 then dividing by 8 hits the same cell as flooring at 32
fn fold_to_histogram(counts: &[u32], extent: &Rect<f64>) -> SpatialHistogram {
    let w = (extent.max().x - extent.min().x).max(f64::EPSILON);
    let h = (extent.max().y - extent.min().y).max(f64::EPSILON);
    let mut folded = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
    for row in 0..COUNT_RESOLUTION {
        for col in 0..COUNT_RESOLUTION {
            folded[(row / HISTOGRAM_STRIDE) * HISTOGRAM_RESOLUTION + col / HISTOGRAM_STRIDE] +=
                counts[row * COUNT_RESOLUTION + col];
        }
    }
    SpatialHistogram {
        counts: folded,
        min_x: extent.min().x,
        min_y: extent.min().y,
        cell_w: w / HISTOGRAM_RESOLUTION as f64,
        cell_h: h / HISTOGRAM_RESOLUTION as f64,
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
    let n_polys = poly_offsets.len().saturating_sub(1);
    let cells = HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION;
    // Bin each polygon's exterior-ring centroid in parallel, accumulating into per-thread
    // histograms then summing them. Integer sums are order-independent, so counts are exact.
    let counts = (0..n_polys)
        .into_par_iter()
        .fold(
            || vec![0u32; cells],
            |mut local, p| {
                let ext_ring = poly_offsets[p] as usize;
                let start = ring_offsets[ext_ring] as usize;
                let end = ring_offsets[ext_ring + 1] as usize;
                if start < end {
                    let count = (end - start) as f64;
                    let cx = xs[start..end].iter().sum::<f64>() / count;
                    let cy = ys[start..end].iter().sum::<f64>() / count;
                    let col = ((cx - extent.min().x) / cell_w)
                        .floor()
                        .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64)
                        as usize;
                    let row = ((cy - extent.min().y) / cell_h)
                        .floor()
                        .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64)
                        as usize;
                    local[row * HISTOGRAM_RESOLUTION + col] += 1;
                }
                local
            },
        )
        .reduce(
            || vec![0u32; cells],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );
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

    // Bin straight into HISTOGRAM_RESOLUTION cells
    fn direct_histogram(xs: &[f64], ys: &[f64], extent: &Rect<f64>) -> Vec<u32> {
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
        counts
    }

    #[test]
    fn folded_histogram_matches_direct_binning() {
        // Early exits read a zero cell as proof a region is empty
        let (xs, ys) = scattered(20_000);
        let extent = compute_extent(&xs, &ys).expect("scattered points have an extent");
        let folded = fold_to_histogram(&count_grid(&xs, &ys, &extent), &extent);
        assert_eq!(folded.counts, direct_histogram(&xs, &ys, &extent));
    }

    #[test]
    fn folded_histogram_counts_every_point() {
        let (xs, ys) = scattered(5_000);
        let stats = collect_points(&xs, &ys);
        let total: u32 = stats.histogram.expect("histogram").counts.iter().sum();
        assert_eq!(total as usize, stats.n);
    }

    #[test]
    fn morisita_index_is_one_for_scattered_points() {
        let (xs, ys) = scattered(200_000);
        let extent = compute_extent(&xs, &ys).expect("scattered points have an extent");
        let index = morisita_index(&count_grid(&xs, &ys, &extent), xs.len());
        assert!(
            (index - 1.0).abs() < 0.1,
            "scattered points should sit at the Poisson null, got {index}"
        );
    }

    #[test]
    fn scattered_points_are_not_clustered() {
        let (xs, ys) = scattered(200_000);
        assert_eq!(collect_points(&xs, &ys).distribution, Distribution::Uniform);
    }

    #[test]
    fn distribution_is_unknown_below_the_point_floor() {
        let (xs, ys) = scattered(MIN_POINTS_FOR_DISTRIBUTION - 1);
        assert_eq!(collect_points(&xs, &ys).distribution, Distribution::Unknown);
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
