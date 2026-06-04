use geo::{coord, Contains, Intersects, LineString, Point, Polygon, Rect};

use crate::index::SpatialIndex;

/// Range query against a point dataset. The index performs the exact coordinate check.
pub fn query_range_points<I: SpatialIndex>(
    index: &I,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> Vec<usize> {
    index.range(min_x, min_y, max_x, max_y)
}

/// Range query against a polygon dataset.
/// The index returns MBR candidates; exact intersection is verified per candidate.
#[allow(clippy::too_many_arguments)]
pub fn query_range_polygons<I: SpatialIndex>(
    index: &I,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> Vec<usize> {
    let bbox = Rect::new(coord! { x: min_x, y: min_y }, coord! { x: max_x, y: max_y });
    index
        .range(min_x, min_y, max_x, max_y)
        .into_iter()
        .filter(|&i| make_polygon(xs, ys, ring_offsets, i).intersects(&bbox))
        .collect()
}

/// Point-in-polygon query against a polygon dataset.
/// The index returns MBR candidates; exact containment is verified per candidate.
pub fn query_contains_polygons<I: SpatialIndex>(
    index: &I,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    qx: f64,
    qy: f64,
) -> Vec<usize> {
    let qpt = Point::new(qx, qy);
    index
        .range(qx, qy, qx, qy)
        .into_iter()
        .filter(|&i| make_polygon(xs, ys, ring_offsets, i).contains(&qpt))
        .collect()
}

/// Reconstruct a Polygon from flat ring coordinate arrays for an exact geometric check.
pub fn make_polygon(xs: &[f64], ys: &[f64], ring_offsets: &[i64], i: usize) -> Polygon<f64> {
    let start = ring_offsets[i] as usize;
    let end = ring_offsets[i + 1] as usize;
    let coords: Vec<geo::Coord<f64>> = (start..end)
        .map(|j| coord! { x: xs[j], y: ys[j] })
        .collect();
    Polygon::new(LineString::new(coords), vec![])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::{five_point_grid, BruteForce};
    use crate::index::SpatialIndex;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn query_range_returns_correct_points() {
        let (xs, ys) = five_point_grid();
        let idx = BruteForce::build(xs.into(), ys.into());
        assert_eq!(
            sorted(query_range_points(&idx, 0.0, 0.0, 1.5, 0.5)),
            vec![0, 1]
        );
    }

    #[test]
    fn query_range_empty_returns_empty() {
        let (xs, ys) = five_point_grid();
        let idx = BruteForce::build(xs.into(), ys.into());
        assert!(query_range_points(&idx, 5.0, 5.0, 10.0, 10.0).is_empty());
    }

    #[test]
    fn query_range_single_result() {
        let (xs, ys) = five_point_grid();
        let idx = BruteForce::build(xs.into(), ys.into());
        assert_eq!(query_range_points(&idx, 0.5, 0.5, 1.5, 1.5), vec![4]);
    }
}
