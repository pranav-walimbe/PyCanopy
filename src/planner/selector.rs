//! Selects the cheapest index kind for a dataset and query using the cost model.

use crate::planner::calibration::CostFactors;
use crate::planner::cost::{probe_cost, selectivity, total_cost, IndexKind, IndexMode};
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
    select_index_from_selectivity(stats, query, sel)
}

/// Route to an index kind given an already-computed `sel`, skipping the gates `select_index` already applied
fn select_index_from_selectivity(stats: &DatasetStats, query: &Query, sel: f64) -> IndexKind {
    match query {
        Query::Knn { .. } => {
            if sel > KNN_FRACTION_THRESHOLD {
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

/// Apply the index mode to a candidate kind, returning the kind to actually use
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
    let sel = selectivity(stats, query);
    let indexed = total_cost(candidate, stats, query, sel, q_count, factors);
    let brute = total_cost(IndexKind::BruteForce, stats, query, sel, q_count, factors);
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

/// Pick the cheapest strategy given a set of already-built indexes.
///
/// Compares probe-only cost of each built index (build already paid), full build+probe
/// cost of the optimal new index, and brute-force probe cost. Returns the winner,
/// which may be a kind not yet in `built` when building a new index beats reusing one.
pub fn plan_best_available(
    built: &[IndexKind],
    stats: &DatasetStats,
    query: &Query,
    q_count: usize,
    factors: &CostFactors,
) -> IndexKind {
    // Computed once, only if actually needed, and reused for every call below
    let needs_sel = stats.n >= BRUTE_FORCE_N || built.iter().any(|&k| k != IndexKind::BruteForce);
    let sel = if needs_sel {
        selectivity(stats, query)
    } else {
        0.0
    };

    let brute_cost = probe_cost(IndexKind::BruteForce, stats, query, sel, q_count, factors);
    let mut best_kind = IndexKind::BruteForce;
    let mut best_cost = brute_cost;

    for &k in built {
        let c = probe_cost(k, stats, query, sel, q_count, factors);
        if c < best_cost {
            best_cost = c;
            best_kind = k;
        }
    }

    if stats.n >= BRUTE_FORCE_N && sel <= FULL_SCAN_SELECTIVITY {
        let candidate = select_index_from_selectivity(stats, query, sel);
        if candidate != IndexKind::BruteForce {
            let new_cost = total_cost(candidate, stats, query, sel, q_count, factors);
            if new_cost < best_cost {
                best_kind = candidate;
            }
        }
    }

    best_kind
}

/// Plan the index kind for `query`, honouring the index mode
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
        };
        assert_eq!(select_index(&s, &q), IndexKind::BruteForce);
    }

    fn big_knn() -> Query {
        Query::Knn {
            point: Point::new(0.0, 0.0),
            k: 5,
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
    fn auto_builds_index_for_many_probes() {
        // Many probes amortise the build, so the index wins
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        assert_eq!(
            plan_access(&s, &big_knn(), 1_000_000, IndexMode::Auto, &f),
            IndexKind::KdTree
        );
    }

    #[test]
    fn best_available_empty_falls_through_to_new_vs_brute() {
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        assert_eq!(
            plan_best_available(&[], &s, &big_knn(), 1_000_000, &f),
            IndexKind::KdTree
        );
    }

    #[test]
    fn best_available_reuses_built_index_for_few_probes() {
        // Build cost already paid, so even 1 probe uses the built index
        let s = stats(1_000_000, GeometryKind::Point, Distribution::Clustered);
        let f = CostFactors::default();
        assert_eq!(
            plan_best_available(&[IndexKind::KdTree], &s, &big_knn(), 1, &f),
            IndexKind::KdTree
        );
    }

    #[test]
    fn best_available_selects_optimal_new_index_when_none_built() {
        let s = stats(1_000_000, GeometryKind::Polygon, Distribution::Unknown);
        let f = CostFactors::default();
        assert_eq!(
            plan_best_available(&[], &s, &small_bbox(), 1_000_000, &f),
            IndexKind::RTree
        );
    }
}
