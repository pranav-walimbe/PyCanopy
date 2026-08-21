//! Empirical cost factors used to calibrate the auto-mode cost model.

use serde::Deserialize;
use std::sync::OnceLock;

const DEFAULT_PROFILE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/python/pycanopy/cost_profiles/default.json"
));

static DEFAULT_COST_FACTORS: OnceLock<CostFactors> = OnceLock::new();

/// Empirical cost factors (ns per operation) for the auto-mode cost model
#[derive(Debug, Clone, Deserialize)]
pub struct CostFactors {
    pub knn_scan_ns_per_item: f64,     // brute-force kNN scan per item
    pub bbox_scan_ns_per_item: f64,    // brute-force range/radius box scan per item
    pub grid_build_ns_per_item: f64,   // grid build per point
    pub kdtree_build_ns_per_item: f64, // kd-tree build per point
    pub rtree_build_ns_per_item: f64,  // r-tree build per polygon
    pub kdtree_knn_ns: f64,            // kd-tree kNN probe
    pub kdtree_range_ns: f64,          // kd-tree range probe
    pub rtree_knn_ns: f64,             // r-tree kNN probe
    pub rtree_range_ns: f64,           // r-tree range probe
    pub grid_range_ns: f64,            // grid range probe
}

#[derive(Deserialize)]
struct CostProfile {
    cost_factors: CostFactors,
}

impl CostFactors {
    pub(crate) fn from_values(values: [f64; 10]) -> Result<Self, String> {
        if values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("cost profile values must be finite and greater than zero".to_string());
        }
        Ok(Self {
            knn_scan_ns_per_item: values[0],
            bbox_scan_ns_per_item: values[1],
            grid_build_ns_per_item: values[2],
            kdtree_build_ns_per_item: values[3],
            rtree_build_ns_per_item: values[4],
            kdtree_knn_ns: values[5],
            kdtree_range_ns: values[6],
            rtree_knn_ns: values[7],
            rtree_range_ns: values[8],
            grid_range_ns: values[9],
        })
    }

    fn from_profile_json(json: &str) -> Result<Self, String> {
        let profile: CostProfile =
            serde_json::from_str(json).map_err(|err| format!("invalid cost profile: {err}"))?;
        let factors = profile.cost_factors;
        Self::from_values([
            factors.knn_scan_ns_per_item,
            factors.bbox_scan_ns_per_item,
            factors.grid_build_ns_per_item,
            factors.kdtree_build_ns_per_item,
            factors.rtree_build_ns_per_item,
            factors.kdtree_knn_ns,
            factors.kdtree_range_ns,
            factors.rtree_knn_ns,
            factors.rtree_range_ns,
            factors.grid_range_ns,
        ])
    }
}

impl Default for CostFactors {
    fn default() -> Self {
        DEFAULT_COST_FACTORS
            .get_or_init(|| {
                Self::from_profile_json(DEFAULT_PROFILE_JSON)
                    .expect("bundled cost profile must be valid")
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_profile_supplies_engine_defaults() {
        let factors = CostFactors::default();
        assert!(factors.knn_scan_ns_per_item > 0.0);
        assert!(factors.rtree_knn_ns > 0.0);
        assert!(factors.grid_range_ns > 0.0);
    }

    #[test]
    fn profile_rejects_missing_factors() {
        let json = "{}";
        assert!(CostFactors::from_profile_json(json).is_err());
    }
}
