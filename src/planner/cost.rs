use crate::planner::calibration::CostFactors;
use crate::query::types::Query;
use crate::stats::types::DatasetStats;

/// Spatial index variant selected by the query planner
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    BruteForce,
    RTree,
    KdTree,
    Grid,
}

/// How aggressively the planner builds spatial indexes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    None,
    Eager,
    Auto,
}

fn build_cost(kind: IndexKind, n: usize, factors: &CostFactors) -> f64 {
    let n = n as f64;
    match kind {
        IndexKind::BruteForce => 0.0,
        IndexKind::Grid => n * factors.build_ns_per_item,
        IndexKind::KdTree | IndexKind::RTree => n * n.log2().max(1.0) * factors.build_ns_per_item,
    }
}

/// Estimated probe cost for `q_count` queries against an already-built `kind` index, given a precomputed `sel`
pub fn probe_cost(
    kind: IndexKind,
    stats: &DatasetStats,
    query: &Query,
    sel: f64,
    q_count: usize,
    factors: &CostFactors,
) -> f64 {
    let n = stats.n as f64;
    let q = q_count as f64;
    let is_knn = matches!(query, Query::Knn { .. });
    match kind {
        IndexKind::BruteForce => q * n * factors.scan_ns_per_item,
        // Grid is a direct cell lookup with no tree traversal
        IndexKind::Grid => {
            let results = (sel * n).max(1.0);
            q * results * factors.grid_range_ns
        }
        _ => {
            let results = (sel * n).max(1.0);
            let per = match (kind, is_knn) {
                (IndexKind::KdTree, true) => factors.kdtree_knn_ns,
                (IndexKind::KdTree, false) => factors.kdtree_range_ns,
                (IndexKind::RTree, true) => factors.rtree_knn_ns,
                (IndexKind::RTree, false) => factors.rtree_range_ns,
                _ => unreachable!(),
            };
            q * (n.log2().max(1.0) + results) * per
        }
    }
}

/// Total estimated cost of `q_count` probes via `kind`
pub fn total_cost(
    kind: IndexKind,
    stats: &DatasetStats,
    query: &Query,
    sel: f64,
    q_count: usize,
    factors: &CostFactors,
) -> f64 {
    build_cost(kind, stats.n, factors) + probe_cost(kind, stats, query, sel, q_count, factors)
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

    #[test]
    fn grid_probe_cost_proportional_to_results() {
        let stats = make_stats(1000);
        let f = CostFactors::default();
        let q = Query::Range {
            bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 50.0, y: 50.0 }),
        };
        let sel = 0.25_f64;
        let expected = 10.0 * (sel * 1000.0) * f.grid_range_ns;
        assert!((probe_cost(IndexKind::Grid, &stats, &q, sel, 10, &f) - expected).abs() < 1e-6);
    }
}
