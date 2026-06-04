use crate::query::types::Query;
use crate::stats::types::DatasetStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Spatial index variant selected by the query planner
pub enum IndexKind {
    BruteForce,
    RTree,
    KdTree,
    Grid,
}

/// Fraction of the dataset expected to match the query (0..=1)
pub fn selectivity(stats: &DatasetStats, query: &Query) -> f64 {
    match query {
        Query::Range { bbox } => {
            if let Some(hist) = &stats.histogram {
                return hist.selectivity(bbox, stats.n);
            }
            let total_area = stats.extent_area();
            if total_area <= 0.0 {
                return 1.0;
            }
            let w = (bbox.max().x - bbox.min().x).abs();
            let h = (bbox.max().y - bbox.min().y).abs();
            (w * h / total_area).min(1.0)
        }
        Query::Knn { k, .. } => (*k as f64 / stats.n.max(1) as f64).min(1.0),
        Query::Contains { .. } => 1.0 / stats.n.max(1) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::types::{
        DatasetStats, Distribution, GeometryKind, SpatialHistogram, HISTOGRAM_RESOLUTION,
    };
    use geo::{coord, Point, Rect};

    fn make_stats(n: usize) -> DatasetStats {
        DatasetStats {
            n,
            kind: GeometryKind::Point,
            extent: Some(Rect::new(
                coord! { x: 0.0, y: 0.0 },
                coord! { x: 100.0, y: 100.0 },
            )),
            distribution: Distribution::Uniform,
            mean_density: n as f64 / 10_000.0,
            histogram: None,
        }
    }

    #[test]
    fn range_covering_full_extent_gives_selectivity_one() {
        let stats = make_stats(1000);
        let q = Query::Range {
            bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 100.0, y: 100.0 }),
        };
        assert!((selectivity(&stats, &q) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn range_covering_quarter_extent() {
        let stats = make_stats(1000);
        let q = Query::Range {
            bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 50.0, y: 50.0 }),
        };
        assert!((selectivity(&stats, &q) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn knn_selectivity_equals_k_over_n() {
        let stats = make_stats(1000);
        let q = Query::Knn {
            point: Point::new(0.0, 0.0),
            k: 10,
            approximate: false,
        };
        assert!((selectivity(&stats, &q) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn contains_selectivity_equals_one_over_n() {
        let stats = make_stats(1000);
        let q = Query::Contains {
            point: Point::new(0.0, 0.0),
        };
        assert!((selectivity(&stats, &q) - 0.001).abs() < 1e-12);
    }

    #[test]
    fn range_selectivity_uses_histogram_when_present() {
        // All 1000 points in a single histogram cell (bottom-left).
        // Area ratio for a 10x10 query over 100x100 extent = 0.01.
        // Histogram for a query covering just that cell should return 1.0.
        let mut counts = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
        counts[0] = 1000;
        let hist = SpatialHistogram {
            counts,
            min_x: 0.0,
            min_y: 0.0,
            cell_w: 100.0 / HISTOGRAM_RESOLUTION as f64,
            cell_h: 100.0 / HISTOGRAM_RESOLUTION as f64,
        };
        let mut stats = make_stats(1000);
        stats.histogram = Some(hist);
        // Query covering exactly the bottom-left cell
        let cell_size = 100.0 / HISTOGRAM_RESOLUTION as f64;
        let q = Query::Range {
            bbox: Rect::new(
                coord! { x: 0.0, y: 0.0 },
                coord! { x: cell_size, y: cell_size },
            ),
        };
        assert!((selectivity(&stats, &q) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn range_selectivity_falls_back_to_area_ratio_without_histogram() {
        let stats = make_stats(1000); // histogram: None
        let q = Query::Range {
            bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 50.0, y: 50.0 }),
        };
        assert!((selectivity(&stats, &q) - 0.25).abs() < 1e-10);
    }
}
