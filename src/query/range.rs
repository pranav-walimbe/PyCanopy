use geo::{Contains, Geometry, Intersects, Point, Rect};

use crate::index::SpatialIndex;

/// Two-phase range query: MBR candidates from the index then exact geometric check
pub fn query_range<I: SpatialIndex>(
    index: &I,
    bbox: &Rect<f64>,
    geometries: &[Geometry<f64>],
) -> Vec<usize> {
    index
        .range(bbox)
        .into_iter()
        .filter(|&i| intersects_rect(&geometries[i], bbox))
        .collect()
}

/// Two-phase contains query: MBR candidates from the index then exact contains check
pub fn query_contains<I: SpatialIndex>(
    index: &I,
    point: &Point<f64>,
    geometries: &[Geometry<f64>],
) -> Vec<usize> {
    index
        .contains(point)
        .into_iter()
        .filter(|&i| contains_point(&geometries[i], point))
        .collect()
}

fn intersects_rect(geom: &Geometry<f64>, bbox: &Rect<f64>) -> bool {
    match geom {
        Geometry::Point(p) => {
            p.x() >= bbox.min().x
                && p.x() <= bbox.max().x
                && p.y() >= bbox.min().y
                && p.y() <= bbox.max().y
        }
        Geometry::Polygon(poly) => poly.intersects(bbox),
        Geometry::MultiPolygon(mpoly) => mpoly.intersects(bbox),
        Geometry::LineString(ls) => ls.intersects(bbox),
        Geometry::MultiLineString(mls) => mls.intersects(bbox),
        _ => {
            // Fallback: MBR intersection (no false negatives given index pre-filtered).
            use geo::BoundingRect;
            geom.bounding_rect()
                .map(|g| {
                    g.max().x >= bbox.min().x
                        && g.min().x <= bbox.max().x
                        && g.max().y >= bbox.min().y
                        && g.min().y <= bbox.max().y
                })
                .unwrap_or(false)
        }
    }
}

fn contains_point(geom: &Geometry<f64>, point: &Point<f64>) -> bool {
    match geom {
        Geometry::Point(p) => {
            (p.x() - point.x()).abs() < f64::EPSILON * 1000.0
                && (p.y() - point.y()).abs() < f64::EPSILON * 1000.0
        }
        Geometry::Polygon(poly) => poly.contains(point),
        Geometry::MultiPolygon(mpoly) => mpoly.contains(point),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::{five_point_grid, BruteForce};
    use crate::index::SpatialIndex;
    use geo::coord;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn query_range_returns_correct_points() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        let bbox = Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 1.5, y: 0.5 });
        assert_eq!(sorted(query_range(&idx, &bbox, &geoms)), vec![0, 1]);
    }

    #[test]
    fn query_range_empty_returns_empty() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        let bbox = Rect::new(coord! { x: 5.0, y: 5.0 }, coord! { x: 10.0, y: 10.0 });
        assert!(query_range(&idx, &bbox, &geoms).is_empty());
    }

    #[test]
    fn query_range_single_result() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        let bbox = Rect::new(coord! { x: 0.5, y: 0.5 }, coord! { x: 1.5, y: 1.5 });
        assert_eq!(query_range(&idx, &bbox, &geoms), vec![4]);
    }

    #[test]
    fn query_contains_matches_exact_point() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        assert_eq!(query_contains(&idx, &Point::new(1.0, 0.0), &geoms), vec![1]);
    }

    #[test]
    fn query_contains_no_match_returns_empty() {
        let geoms = five_point_grid();
        let idx = BruteForce::build(&geoms);
        assert!(query_contains(&idx, &Point::new(0.5, 0.5), &geoms).is_empty());
    }
}
