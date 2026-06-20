use crate::planner::calibration::CostFactors;
use crate::planner::cost::{selectivity, total_cost, IndexKind, IndexMode};
use crate::query::types::Query;
use crate::stats::types::{DatasetStats, Distribution, GeometryKind};

const FULL_SCAN_SELECTIVITY: f64 = 0.5;
const BRUTE_FORCE_N: usize = 500;
const KNN_FRACTION_THRESHOLD: f64 = 0.1;

/// Choose the best index kind for the given dataset statistics and query
pub fn select_index(stats: &DatasetStats, query: &Query) -> IndexKind {
    if stats.n < BRUTE_FORCE_N {
        return IndexKind::BruteForce;
    }

    let sel = selectivity(stats, query);
    if sel > FULL_SCAN_SELECTIVITY {
        return IndexKind::BruteForce;
    }

    match query {
        Query::Knn { k, .. } => {
            if *k as f64 / stats.n as f64 > KNN_FRACTION_THRESHOLD {
                return IndexKind::BruteForce;
            }
            match stats.kind {
                GeometryKind::Point => IndexKind::KdTree,
                _ => IndexKind::RTree,
            }
        }
        Query::Range { .. } => match stats.kind {
            GeometryKind::Point => {
                if stats.distribution == Distribution::Uniform {
                    IndexKind::Grid
                } else {
                    IndexKind::KdTree
                }
            }
            _ => IndexKind::RTree,
        },
        Query::Contains { .. } => match stats.kind {
            GeometryKind::Point => IndexKind::KdTree,
            _ => IndexKind::RTree,
        },
    }
}

/// Apply the index mode to a candidate kind, returning the kind to actually use.
/// None always scans. Eager keeps the candidate. Auto keeps it only when its
/// estimated cost beats brute force over `q_count` probes. Kernels that require a
/// specific index (e.g. an R-tree) pass that as the candidate, the standard path
/// gets the candidate from `select_index` via `plan_access`.
pub fn plan_access_with_kind(
    stats: &DatasetStats,
    query: &Query,
    q_count: usize,
    mode: IndexMode,
    factors: &CostFactors,
    candidate: IndexKind,
) -> IndexKind {
    if mode == IndexMode::None || candidate == IndexKind::BruteForce {
        return IndexKind::BruteForce;
    }
    if mode == IndexMode::Eager {
        return candidate;
    }
    let indexed = total_cost(candidate, stats, query, q_count, factors);
    let brute = total_cost(IndexKind::BruteForce, stats, query, q_count, factors);
    if indexed < brute {
        candidate
    } else {
        IndexKind::BruteForce
    }
}

/// Candidate index for a point distance probe (brute force / grid / KD-tree)
pub fn point_distance_candidate(stats: &DatasetStats) -> IndexKind {
    if stats.n < BRUTE_FORCE_N {
        IndexKind::BruteForce
    } else if stats.distribution == Distribution::Uniform {
        IndexKind::Grid
    } else {
        IndexKind::KdTree
    }
}

/// Candidate for an R-tree kernel: brute force below the small-dataset threshold
pub fn rtree_candidate(stats: &DatasetStats) -> IndexKind {
    if stats.n < BRUTE_FORCE_N {
        IndexKind::BruteForce
    } else {
        IndexKind::RTree
    }
}

/// Plan the index kind for `query`, honouring the index mode. The candidate kind
/// comes from `select_index`.
pub fn plan_access(
    stats: &DatasetStats,
    query: &Query,
    q_count: usize,
    mode: IndexMode,
    factors: &CostFactors,
) -> IndexKind {
    let candidate = select_index(stats, query);
    plan_access_with_kind(stats, query, q_count, mode, factors, candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::types::{DatasetStats, Distribution, GeometryKind};
    use geo::{coord, Point, Rect};

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

    // A small bbox covering only 1% of the extent → selectivity = 0.01, below threshold
    fn small_bbox() -> Query {
        Query::Range {
            bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 10.0, y: 10.0 }),
        }
    }

    fn full_bbox() -> Query {
        Query::Range {
            bbox: Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 100.0, y: 100.0 }),
        }
    }

    #[test]
    fn small_n_always_brute_force() {
        let s = stats(100, GeometryKind::Point, Distribution::Uniform);
        let q = Query::Knn {
            point: Point::new(0.0, 0.0),
            k: 5,
            approximate: false,
        };
        assert_eq!(select_index(&s, &q), IndexKind::BruteForce);
    }

    #[test]
    fn high_selectivity_bypasses_to_brute_force() {
        let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
        assert_eq!(select_index(&s, &full_bbox()), IndexKind::BruteForce);
    }

    #[test]
    fn points_knn_routes_to_kdtree() {
        let s = stats(1000, GeometryKind::Point, Distribution::Clustered);
        let q = Query::Knn {
            point: Point::new(0.0, 0.0),
            k: 5,
            approximate: false,
        };
        assert_eq!(select_index(&s, &q), IndexKind::KdTree);
    }

    #[test]
    fn points_uniform_range_routes_to_grid() {
        let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
        assert_eq!(select_index(&s, &small_bbox()), IndexKind::Grid);
    }

    #[test]
    fn points_clustered_range_routes_to_kdtree() {
        let s = stats(1000, GeometryKind::Point, Distribution::Clustered);
        assert_eq!(select_index(&s, &small_bbox()), IndexKind::KdTree);
    }

    #[test]
    fn polygon_range_routes_to_rtree() {
        let s = stats(1000, GeometryKind::Polygon, Distribution::Unknown);
        assert_eq!(select_index(&s, &small_bbox()), IndexKind::RTree);
    }

    #[test]
    fn knn_fraction_above_threshold_bypasses_to_brute_force() {
        // k/N = 200/1000 = 0.2 > KNN_FRACTION_THRESHOLD (0.1)
        let s = stats(1000, GeometryKind::Point, Distribution::Uniform);
        let q = Query::Knn {
            point: Point::new(0.0, 0.0),
            k: 200,
            approximate: false,
        };
        assert_eq!(select_index(&s, &q), IndexKind::BruteForce);
    }

    fn big_knn() -> Query {
        Query::Knn {
            point: Point::new(0.0, 0.0),
            k: 5,
            approximate: false,
        }
    }

    #[test]
    fn mode_none_always_brute_force() {
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        assert_eq!(
            plan_access(&s, &big_knn(), 1_000_000, IndexMode::None, &f),
            IndexKind::BruteForce
        );
    }

    #[test]
    fn mode_eager_matches_selector() {
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        // Eager ignores q_count and returns the selector's kind
        assert_eq!(
            plan_access(&s, &big_knn(), 1, IndexMode::Eager, &f),
            select_index(&s, &big_knn())
        );
    }

    #[test]
    fn auto_skips_index_for_single_probe() {
        // One probe against a large dataset: build cost is not amortised
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        assert_eq!(
            plan_access(&s, &big_knn(), 1, IndexMode::Auto, &f),
            IndexKind::BruteForce
        );
    }

    #[test]
    fn auto_builds_index_for_many_probes() {
        // Many probes amortise the build, so the index wins
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        assert_eq!(
            plan_access(&s, &big_knn(), 1_000_000, IndexMode::Auto, &f),
            IndexKind::KdTree
        );
    }
}
