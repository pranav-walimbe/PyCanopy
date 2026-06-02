use crate::planner::cost::{selectivity, IndexKind};
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

    // A small bbox covering only 1% of the extent → selectivity = 0.01, below threshold.
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
}
