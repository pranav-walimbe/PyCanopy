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
    index
        .range(min_x, min_y, max_x, max_y)
        .into_iter()
        .filter(|&i| {
            polygon_intersects_bbox_raw(
                xs,
                ys,
                ring_offsets,
                poly_offsets,
                i,
                min_x,
                min_y,
                max_x,
                max_y,
            )
        })
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
    index
        .range(qx, qy, qx, qy)
        .into_iter()
        .filter(|&i| pip_raw(qx, qy, xs, ys, ring_offsets, poly_offsets, i))
        .collect()
}

/// Ray-casting point-in-ring test. Zero allocation, operates directly on coordinate slices.
fn ring_contains(qx: f64, qy: f64, xs: &[f64], ys: &[f64]) -> bool {
    let n = xs.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        if (ys[i] > qy) != (ys[j] > qy)
            && qx < (xs[j] - xs[i]) * (qy - ys[i]) / (ys[j] - ys[i]) + xs[i]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Point-in-polygon test for polygon i, including hole handling. Zero allocation.
///
/// A point is inside the polygon if it is inside the exterior ring and outside
/// all interior rings (holes). Operates directly on the flat coordinate arrays
/// via ring_offsets and poly_offsets — no heap allocation per call.
pub fn pip_raw(
    qx: f64,
    qy: f64,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
) -> bool {
    let r_start = poly_offsets[i] as usize;
    let r_end = poly_offsets[i + 1] as usize;

    let ext_start = ring_offsets[r_start] as usize;
    let ext_end = ring_offsets[r_start + 1] as usize;
    if !ring_contains(qx, qy, &xs[ext_start..ext_end], &ys[ext_start..ext_end]) {
        return false;
    }

    for r in (r_start + 1)..r_end {
        let h_start = ring_offsets[r] as usize;
        let h_end = ring_offsets[r + 1] as usize;
        if ring_contains(qx, qy, &xs[h_start..h_end], &ys[h_start..h_end]) {
            return false;
        }
    }

    true
}

/// Segment intersection test using cross products. Returns false for parallel segments.
#[allow(clippy::too_many_arguments)]
fn segments_intersect(
    ax1: f64,
    ay1: f64,
    ax2: f64,
    ay2: f64,
    bx1: f64,
    by1: f64,
    bx2: f64,
    by2: f64,
) -> bool {
    let d1x = ax2 - ax1;
    let d1y = ay2 - ay1;
    let d2x = bx2 - bx1;
    let d2y = by2 - by1;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < f64::EPSILON {
        return false;
    }
    let t = ((bx1 - ax1) * d2y - (by1 - ay1) * d2x) / denom;
    let u = ((bx1 - ax1) * d1y - (by1 - ay1) * d1x) / denom;
    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

/// Exact polygon-bbox intersection test. Zero allocation, operates on flat coordinate arrays.
///
/// Three conditions are checked in order:
///   1. Any polygon vertex inside the bbox.
///   2. Any bbox corner inside the polygon exterior ring.
///   3. Any polygon edge crosses any bbox edge.
///
/// Only the exterior ring is tested; holes cannot contribute to intersection.
#[allow(clippy::too_many_arguments)]
pub fn polygon_intersects_bbox_raw(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    let r_start = poly_offsets[i] as usize;
    let ext_start = ring_offsets[r_start] as usize;
    let ext_end = ring_offsets[r_start + 1] as usize;
    let vxs = &xs[ext_start..ext_end];
    let vys = &ys[ext_start..ext_end];
    let n = vxs.len();
    if n == 0 {
        return false;
    }

    for k in 0..n {
        if vxs[k] >= min_x && vxs[k] <= max_x && vys[k] >= min_y && vys[k] <= max_y {
            return true;
        }
    }

    let corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ];
    for (cx, cy) in corners {
        if ring_contains(cx, cy, vxs, vys) {
            return true;
        }
    }

    let bbox_edges = [
        (min_x, min_y, max_x, min_y),
        (max_x, min_y, max_x, max_y),
        (max_x, max_y, min_x, max_y),
        (min_x, max_y, min_x, min_y),
    ];
    let mut j = n - 1;
    for k in 0..n {
        for &(bx1, by1, bx2, by2) in &bbox_edges {
            if segments_intersect(vxs[j], vys[j], vxs[k], vys[k], bx1, by1, bx2, by2) {
                return true;
            }
        }
        j = k;
    }

    false
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
