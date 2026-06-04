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
    poly_offsets: &[i64],
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> Vec<usize> {
    let bbox = Rect::new(coord! { x: min_x, y: min_y }, coord! { x: max_x, y: max_y });
    index
        .range(min_x, min_y, max_x, max_y)
        .into_iter()
        .filter(|&i| make_polygon(xs, ys, ring_offsets, poly_offsets, i).intersects(&bbox))
        .collect()
}

/// Point-in-polygon query against a polygon dataset.
/// The index returns MBR candidates; exact containment is verified per candidate.
pub fn query_contains_polygons<I: SpatialIndex>(
    index: &I,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    qx: f64,
    qy: f64,
) -> Vec<usize> {
    let qpt = Point::new(qx, qy);
    index
        .range(qx, qy, qx, qy)
        .into_iter()
        .filter(|&i| make_polygon(xs, ys, ring_offsets, poly_offsets, i).contains(&qpt))
        .collect()
}

/// Reconstruct polygon i from two-level ring arrays, including any interior holes.
///
/// ring_offsets[r]..ring_offsets[r+1] gives ring r's coordinate range in xs/ys.
/// poly_offsets[i]..poly_offsets[i+1] gives polygon i's ring range in ring_offsets.
/// The first ring is the exterior; any remaining rings are interior holes.
pub fn make_polygon(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
) -> Polygon<f64> {
    let r_start = poly_offsets[i] as usize;
    let r_end = poly_offsets[i + 1] as usize;

    let ext_start = ring_offsets[r_start] as usize;
    let ext_end = ring_offsets[r_start + 1] as usize;
    let exterior = LineString::new(
        (ext_start..ext_end)
            .map(|j| coord! { x: xs[j], y: ys[j] })
            .collect(),
    );

    let holes = (r_start + 1..r_end)
        .map(|r| {
            let h_start = ring_offsets[r] as usize;
            let h_end = ring_offsets[r + 1] as usize;
            LineString::new(
                (h_start..h_end)
                    .map(|j| coord! { x: xs[j], y: ys[j] })
                    .collect(),
            )
        })
        .collect();

    Polygon::new(exterior, holes)
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
