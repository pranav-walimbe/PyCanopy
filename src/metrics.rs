//! Low-overhead, always-on metrics for Engine operations.

use crate::planner::cost::IndexKind;

pub(crate) const INDEX_SLOTS: usize = 5;

#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Operation {
    Knn,
    RangeQuery,
    RadiusQuery,
    ContainsQuery,
    BatchKnnJoin,
    BatchWithinDistance,
    BatchContains,
    BatchWithinDistanceToPolygons,
    BatchContainsAggregate,
    BatchWithinDistanceToPolygonsAggregate,
    BatchKnnToPolygons,
    BatchKnnToPolygonsSorted,
    PolygonIntersectsSelfJoin,
    PointsWithinDistanceOfPolygon,
}

impl Operation {
    pub(crate) const ALL: [Self; 14] = [
        Self::Knn,
        Self::RangeQuery,
        Self::RadiusQuery,
        Self::ContainsQuery,
        Self::BatchKnnJoin,
        Self::BatchWithinDistance,
        Self::BatchContains,
        Self::BatchWithinDistanceToPolygons,
        Self::BatchContainsAggregate,
        Self::BatchWithinDistanceToPolygonsAggregate,
        Self::BatchKnnToPolygons,
        Self::BatchKnnToPolygonsSorted,
        Self::PolygonIntersectsSelfJoin,
        Self::PointsWithinDistanceOfPolygon,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Knn => "knn",
            Self::RangeQuery => "range_query",
            Self::RadiusQuery => "radius_query",
            Self::ContainsQuery => "contains_query",
            Self::BatchKnnJoin => "batch_knn_join",
            Self::BatchWithinDistance => "batch_within_distance",
            Self::BatchContains => "batch_contains",
            Self::BatchWithinDistanceToPolygons => "batch_within_distance_to_polygons",
            Self::BatchContainsAggregate => "batch_contains_aggregate",
            Self::BatchWithinDistanceToPolygonsAggregate => {
                "batch_within_distance_to_polygons_aggregate"
            }
            Self::BatchKnnToPolygons => "batch_knn_to_polygons",
            Self::BatchKnnToPolygonsSorted => "batch_knn_to_polygons_sorted",
            Self::PolygonIntersectsSelfJoin => "polygon_intersects_self_join",
            Self::PointsWithinDistanceOfPolygon => "points_within_distance_of_polygon",
        }
    }

    const fn slot(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct OperationMetric {
    pub(crate) calls: u64,
    pub(crate) elapsed_compute_ns: u64,
    pub(crate) output_rows: u64,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct BuildMetric {
    pub(crate) build_count: u64,
    pub(crate) elapsed_compute_ns: u64,
}

pub(crate) struct EngineMetrics {
    operations: [[OperationMetric; INDEX_SLOTS]; Operation::ALL.len()],
    builds: [BuildMetric; 4],
    pub(crate) prepared_build: BuildMetric,
    pub(crate) wkb_decode_ns: u64,
    pub(crate) statistics_ns: u64,
}

impl Default for EngineMetrics {
    fn default() -> Self {
        Self {
            operations: [[OperationMetric::default(); INDEX_SLOTS]; Operation::ALL.len()],
            builds: [BuildMetric::default(); 4],
            prepared_build: BuildMetric::default(),
            wkb_decode_ns: 0,
            statistics_ns: 0,
        }
    }
}

impl EngineMetrics {
    pub(crate) fn with_construction(wkb_decode_ns: u64, statistics_ns: u64) -> Self {
        Self {
            wkb_decode_ns,
            statistics_ns,
            ..Self::default()
        }
    }

    pub(crate) fn record_operation(
        &mut self,
        operation: Operation,
        index: Option<IndexKind>,
        elapsed_compute_ns: u64,
        output_rows: usize,
    ) {
        let metric = &mut self.operations[operation.slot()][index_slot(index)];
        metric.calls = metric.calls.saturating_add(1);
        metric.elapsed_compute_ns = metric.elapsed_compute_ns.saturating_add(elapsed_compute_ns);
        metric.output_rows = metric.output_rows.saturating_add(output_rows as u64);
    }

    pub(crate) fn record_build(&mut self, index: IndexKind, elapsed_compute_ns: u64) {
        let metric = &mut self.builds[index_slot(Some(index))];
        metric.build_count = metric.build_count.saturating_add(1);
        metric.elapsed_compute_ns = metric.elapsed_compute_ns.saturating_add(elapsed_compute_ns);
    }

    pub(crate) fn operation(&self, operation: Operation, index_slot: usize) -> OperationMetric {
        self.operations[operation.slot()][index_slot]
    }

    pub(crate) fn build(&self, index_slot: usize) -> BuildMetric {
        self.builds[index_slot]
    }
}

pub(crate) const fn index_slot(index: Option<IndexKind>) -> usize {
    match index {
        Some(IndexKind::BruteForce) => 0,
        Some(IndexKind::RTree) => 1,
        Some(IndexKind::KdTree) => 2,
        Some(IndexKind::Grid) => 3,
        None => 4,
    }
}

pub(crate) const fn index_name(index_slot: usize) -> &'static str {
    match index_slot {
        0 => "brute_force",
        1 => "r_tree",
        2 => "kd_tree",
        3 => "grid",
        4 => "none",
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_metrics_accumulate_in_fixed_buckets() {
        let mut metrics = EngineMetrics::default();
        metrics.record_operation(Operation::BatchContains, Some(IndexKind::RTree), 10, 3);
        metrics.record_operation(Operation::BatchContains, Some(IndexKind::RTree), 20, 4);

        let metric =
            metrics.operation(Operation::BatchContains, index_slot(Some(IndexKind::RTree)));
        assert_eq!(metric.calls, 2);
        assert_eq!(metric.elapsed_compute_ns, 30);
        assert_eq!(metric.output_rows, 7);
    }

    #[test]
    fn build_metrics_count_only_recorded_builds() {
        let mut metrics = EngineMetrics::default();
        metrics.record_build(IndexKind::KdTree, 42);

        let metric = metrics.build(index_slot(Some(IndexKind::KdTree)));
        assert_eq!(metric.build_count, 1);
        assert_eq!(metric.elapsed_compute_ns, 42);
        assert_eq!(index_name(index_slot(None)), "none");
    }
}
