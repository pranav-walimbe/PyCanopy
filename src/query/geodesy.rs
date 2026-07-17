//! Distance metric selection and exact haversine great-circle math over lon/lat degrees.

const EARTH_MEAN_RADIUS_M: f64 = 6_371_008.8; // IUGG mean radius, the value geo's Haversine uses
const MIN_METERS_PER_DEGREE_LAT: f64 = 110_574.0; // at the equator, a latitude degree's floor
const METERS_PER_DEGREE_LON_EQUATOR: f64 = 111_320.0; // a longitude degree's ceiling
const POLE_GUARD_LAT: f64 = 89.9; // past here a box spans all longitudes, cos(lat) nears zero

/// How the distance between two points is measured
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Planar, // straight-line Euclidean distance in whatever units the coordinates carry
    Haversine, // great-circle distance in meters, over lon/lat degrees
}

impl DistanceMetric {
    /// Parse the Python-facing coordinate system name into the metric it selects
    pub fn from_coordinate_system(name: &str) -> Result<DistanceMetric, String> {
        match name {
            "planar" => Ok(DistanceMetric::Planar),
            "geographic" => Ok(DistanceMetric::Haversine),
            other => Err(format!(
                "coordinate system must be 'planar' or 'geographic', got '{other}'"
            )),
        }
    }

    /// The Python-facing coordinate system name that selects this metric
    pub fn coordinate_system(self) -> &'static str {
        match self {
            DistanceMetric::Planar => "planar",
            DistanceMetric::Haversine => "geographic",
        }
    }
}

/// Exact haversine distance in meters, with the first point's latitude cosine hoisted by the caller
#[inline]
pub fn haversine_distance_m(lon1: f64, lat1: f64, cos_lat1: f64, lon2: f64, lat2: f64) -> f64 {
    // Same math and radius as geo::Haversine.distance, minus the cos(lat1) it redoes per call
    let delta_theta = (lat2 - lat1).to_radians();
    let delta_lambda = (lon2 - lon1).to_radians();
    let a = (delta_theta / 2.0).sin().powi(2)
        + cos_lat1 * lat2.to_radians().cos() * (delta_lambda / 2.0).sin().powi(2);
    // Clamping guards asin against a rounding overshoot past 1.0 on near-antipodal pairs
    EARTH_MEAN_RADIUS_M * 2.0 * a.sqrt().min(1.0).asin()
}

/// Degree-space box enclosing every point within `distance_m` of (cx, cy) on the sphere
pub fn conservative_degree_box(cx: f64, cy: f64, distance_m: f64) -> (f64, f64, f64, f64) {
    let lat_delta = distance_m / MIN_METERS_PER_DEGREE_LAT;
    let min_y = (cy - lat_delta).max(-90.0);
    let max_y = (cy + lat_delta).min(90.0);
    let all_longitudes = (-180.0, min_y, 180.0, max_y);

    // Longitude degrees shrink with cos(lat), so widen at the highest latitude reached, not at cy
    let extreme_lat = min_y.abs().max(max_y.abs());
    if extreme_lat >= POLE_GUARD_LAT {
        return all_longitudes;
    }
    let lon_delta = distance_m / (METERS_PER_DEGREE_LON_EQUATOR * extreme_lat.to_radians().cos());
    // Running past either edge wraps the antimeridian, which no degree interval can express
    if cx - lon_delta < -180.0 || cx + lon_delta > 180.0 {
        return all_longitudes;
    }
    (cx - lon_delta, min_y, cx + lon_delta, max_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Distance, Haversine, Point};

    // The crate's own haversine, the correctness oracle for the hoisted rewrite
    fn oracle(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
        Haversine.distance(Point::new(lon1, lat1), Point::new(lon2, lat2))
    }

    fn hoisted(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
        haversine_distance_m(lon1, lat1, lat1.to_radians().cos(), lon2, lat2)
    }

    #[test]
    fn matches_geo_crate_on_known_pairs() {
        // New York to London, a degree of equator, two points a few km apart, a long diagonal
        let pairs = [
            (-74.006, 40.7128, -0.1278, 51.5074),
            (0.0, 0.0, 1.0, 0.0),
            (-77.036585, 38.897448, -77.009080, 38.889825),
            (-72.1235, 42.3521, 72.1260, 70.612),
        ];
        for (lon1, lat1, lon2, lat2) in pairs {
            let got = hoisted(lon1, lat1, lon2, lat2);
            let want = oracle(lon1, lat1, lon2, lat2);
            assert!(
                (got - want).abs() < 1e-6,
                "({lon1}, {lat1}) to ({lon2}, {lat2}): hoisted {got} vs geo {want}"
            );
        }
    }

    #[test]
    fn matches_geo_crate_across_latitude_bands() {
        // Latitude scales the longitude term by its cosine, so sweep a spread of bands
        for lat in [-85.0, -45.0, -1.0, 0.0, 1.0, 45.0, 85.0] {
            for dlon in [0.001, 0.5, 10.0] {
                let got = hoisted(0.0, lat, dlon, lat + 0.25);
                let want = oracle(0.0, lat, dlon, lat + 0.25);
                assert!(
                    (got - want).abs() < 1e-6,
                    "lat {lat} dlon {dlon}: hoisted {got} vs geo {want}"
                );
            }
        }
    }

    #[test]
    fn antipodal_pair_is_half_the_circumference_not_nan() {
        let d = hoisted(0.0, 0.0, 180.0, 0.0);
        assert!(d.is_finite());
        assert!((d - EARTH_MEAN_RADIUS_M * std::f64::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn distance_is_symmetric_between_the_two_points() {
        let a = hoisted(-74.006, 40.7128, -0.1278, 51.5074);
        let b = hoisted(-0.1278, 51.5074, -74.006, 40.7128);
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn degree_box_widens_longitude_toward_the_poles() {
        // The same radius spans more longitude at higher latitude, while latitude span holds
        let (min_x_eq, min_y_eq, max_x_eq, max_y_eq) = conservative_degree_box(0.0, 0.0, 100_000.0);
        let (min_x_hi, min_y_hi, max_x_hi, max_y_hi) =
            conservative_degree_box(0.0, 60.0, 100_000.0);
        assert!((max_x_hi - min_x_hi) > (max_x_eq - min_x_eq) * 1.9);
        assert!(((max_y_eq - min_y_eq) - (max_y_hi - min_y_hi)).abs() < 1e-9);
    }

    #[test]
    fn degree_box_spans_all_longitudes_near_a_pole() {
        let (min_x, _, max_x, max_y) = conservative_degree_box(10.0, 89.95, 1_000.0);
        assert_eq!((min_x, max_x), (-180.0, 180.0));
        assert!(max_y <= 90.0);
    }

    #[test]
    fn degree_box_spans_all_longitudes_across_the_antimeridian() {
        // A box at 179.9E reaching past 180 wraps, which a degree interval cannot express
        let (min_x, _, max_x, _) = conservative_degree_box(179.9, 0.0, 100_000.0);
        assert_eq!((min_x, max_x), (-180.0, 180.0));
    }

    #[test]
    fn degree_box_encloses_every_point_within_the_radius() {
        // No false negatives: sweep a fine grid and confirm the box holds every true match
        let distance = 150_000.0;
        for &cy in &[0.0, 30.0, 60.0, 80.0] {
            for &cx in &[-120.0, 0.0, 45.0] {
                let (min_x, min_y, max_x, max_y) = conservative_degree_box(cx, cy, distance);
                let mut lat = -90.0;
                while lat <= 90.0 {
                    let mut lon = -180.0;
                    while lon < 180.0 {
                        if hoisted(cx, cy, lon, lat) <= distance {
                            assert!(
                                lon >= min_x && lon <= max_x && lat >= min_y && lat <= max_y,
                                "({lon}, {lat}) is within {distance} m of ({cx}, {cy}) but falls \
                                 outside the box ({min_x}, {min_y}, {max_x}, {max_y})"
                            );
                        }
                        lon += 0.25;
                    }
                    lat += 0.25;
                }
            }
        }
    }
}
