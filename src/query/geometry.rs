//! Geometry kernels backed by the `geo` crate.
//!
//! These operate on the flat coordinate storage used by the polygon Engine
//! (xs/ys plus ring_offsets/poly_offsets, exterior ring first then holes). A
//! `geo::Polygon` is materialised per geometry on demand. The heavy correctness
//! sensitive operations (boolean intersection, convex hull) are delegated to
//! `geo` rather than hand rolled.

use geo::{Area, BooleanOps, ConvexHull, Intersects};
use geo::{Coord, LineString, MultiPoint, Point, Polygon};

use crate::query::range::pip_raw;

/// Build a `geo::Polygon` for polygon `i` from the flat coordinate arrays.
///
/// The first ring of the polygon is the exterior. Remaining rings are interior holes.
pub fn polygon_from_flat(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
) -> Polygon<f64> {
    let r_start = poly_offsets[i] as usize;
    let r_end = poly_offsets[i + 1] as usize;

    let ring = |r: usize| -> LineString<f64> {
        let start = ring_offsets[r] as usize;
        let end = ring_offsets[r + 1] as usize;
        LineString::new((start..end).map(|k| Coord { x: xs[k], y: ys[k] }).collect())
    };

    let exterior = ring(r_start);
    let interiors = ((r_start + 1)..r_end).map(ring).collect();
    Polygon::new(exterior, interiors)
}

#[inline]
fn point_segment_dist2(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    let t = if len2 <= 0.0 {
        0.0
    } else {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
    };
    let cx = ax + t * dx;
    let cy = ay + t * dy;
    let ex = px - cx;
    let ey = py - cy;
    ex * ex + ey * ey
}

/// Euclidean distance from a point to polygon `i`. Zero when the point is inside.
///
/// Operates directly on the flat coordinate slices with no allocation, mirroring
/// `pip_raw`. The boundary distance is the minimum point-to-edge distance across
/// every ring (exterior and holes), and is zero when the point lies inside the
/// polygon (inside the exterior and outside all holes).
pub fn point_to_polygon_distance(
    px: f64,
    py: f64,
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
) -> f64 {
    if pip_raw(px, py, xs, ys, ring_offsets, poly_offsets, i) {
        return 0.0;
    }
    let r_start = poly_offsets[i] as usize;
    let r_end = poly_offsets[i + 1] as usize;
    let mut best = f64::INFINITY;
    for r in r_start..r_end {
        let start = ring_offsets[r] as usize;
        let end = ring_offsets[r + 1] as usize;
        let n = end - start;
        if n < 2 {
            continue;
        }
        let mut j = end - 1;
        for k in start..end {
            best = best.min(point_segment_dist2(px, py, xs[j], ys[j], xs[k], ys[k]));
            j = k;
        }
    }
    best.sqrt()
}

/// Unsigned area of polygon `i` (exterior minus holes)
pub fn polygon_area(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
) -> f64 {
    polygon_from_flat(xs, ys, ring_offsets, poly_offsets, i).unsigned_area()
}

/// True when polygons `i` and `j` (in the same flat arrays) intersect
pub fn polygons_intersect(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
    j: usize,
) -> bool {
    let a = polygon_from_flat(xs, ys, ring_offsets, poly_offsets, i);
    let b = polygon_from_flat(xs, ys, ring_offsets, poly_offsets, j);
    a.intersects(&b)
}

/// Unsigned area of the intersection of polygons `i` and `j`
pub fn polygon_intersection_area(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    i: usize,
    j: usize,
) -> f64 {
    let a = polygon_from_flat(xs, ys, ring_offsets, poly_offsets, i);
    let b = polygon_from_flat(xs, ys, ring_offsets, poly_offsets, j);
    a.intersection(&b).unsigned_area()
}

/// Area of the convex hull of a set of points
pub fn convex_hull_area(pxs: &[f64], pys: &[f64]) -> f64 {
    if pxs.len() < 3 {
        return 0.0;
    }
    let points: MultiPoint<f64> = pxs
        .iter()
        .zip(pys.iter())
        .map(|(&x, &y)| Point::new(x, y))
        .collect();
    points.convex_hull().unsigned_area()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A unit square at the origin: (0,0)-(1,1), single exterior ring (closed)
    fn unit_square() -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
        let xs = vec![0.0, 1.0, 1.0, 0.0, 0.0];
        let ys = vec![0.0, 0.0, 1.0, 1.0, 0.0];
        let ring_offsets = vec![0, 5];
        let poly_offsets = vec![0, 1];
        (xs, ys, ring_offsets, poly_offsets)
    }

    #[test]
    fn area_of_unit_square_is_one() {
        let (xs, ys, ro, po) = unit_square();
        assert!((polygon_area(&xs, &ys, &ro, &po, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn distance_zero_when_inside() {
        let (xs, ys, ro, po) = unit_square();
        let d = point_to_polygon_distance(0.5, 0.5, &xs, &ys, &ro, &po, 0);
        assert!(d.abs() < 1e-9);
    }

    #[test]
    fn distance_to_left_edge() {
        let (xs, ys, ro, po) = unit_square();
        // Point one unit to the left of the square's left edge
        let d = point_to_polygon_distance(-1.0, 0.5, &xs, &ys, &ro, &po, 0);
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn two_overlapping_squares() {
        // Two overlapping squares with a 1x1 overlap at (1,1)-(2,2)
        let xs = vec![
            0.0, 2.0, 2.0, 0.0, 0.0, // A
            1.0, 3.0, 3.0, 1.0, 1.0, // B
        ];
        let ys = vec![
            0.0, 0.0, 2.0, 2.0, 0.0, // A
            1.0, 1.0, 3.0, 3.0, 1.0, // B
        ];
        let ring_offsets = vec![0, 5, 10];
        let poly_offsets = vec![0, 1, 2];
        assert!(polygons_intersect(
            &xs,
            &ys,
            &ring_offsets,
            &poly_offsets,
            0,
            1
        ));
        let ia = polygon_intersection_area(&xs, &ys, &ring_offsets, &poly_offsets, 0, 1);
        assert!((ia - 1.0).abs() < 1e-9);
    }

    #[test]
    fn convex_hull_area_of_square_corners() {
        // Four corners of a 2x2 square plus an interior point, hull area is 4
        let pxs = vec![0.0, 2.0, 2.0, 0.0, 1.0];
        let pys = vec![0.0, 0.0, 2.0, 2.0, 1.0];
        assert!((convex_hull_area(&pxs, &pys) - 4.0).abs() < 1e-9);
    }
}
