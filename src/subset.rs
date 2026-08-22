//! Compact native geometry subsets for derived engines.

use std::sync::Arc;

use crate::{query::multipoly::polygon_parts_csr, Engine};

/// Copy selected logical polygons into a compact engine with no inherited indexes.
pub(crate) fn polygons(source: &Engine, indices: &[u32]) -> Result<Engine, String> {
    let ring_offsets = source
        .ring_offsets
        .as_deref()
        .ok_or_else(|| "subset currently requires a polygon dataset".to_string())?;
    let poly_offsets = source
        .poly_offsets
        .as_deref()
        .expect("polygon engines have polygon offsets");

    for &index in indices {
        if index as usize >= source.n_polygons {
            return Err(format!(
                "subset index {index} is out of bounds for {} polygons",
                source.n_polygons
            ));
        }
    }

    let mut selected_part_offsets = Vec::with_capacity(indices.len() + 1);
    let mut selected_parts = Vec::new();
    selected_part_offsets.push(0usize);
    if let Some(part_poly) = source.part_poly.as_deref() {
        let (logical_offsets, logical_parts) = polygon_parts_csr(part_poly, source.n_polygons);
        for &index in indices {
            let start = logical_offsets[index as usize] as usize;
            let end = logical_offsets[index as usize + 1] as usize;
            selected_parts.extend_from_slice(&logical_parts[start..end]);
            selected_part_offsets.push(selected_parts.len());
        }
    } else {
        selected_parts.extend_from_slice(indices);
        selected_part_offsets.extend(1..=indices.len());
    }

    let mut ring_count = 0usize;
    let mut coordinate_count = 0usize;
    for &part in &selected_parts {
        let part = part as usize;
        let first_ring = poly_offsets[part] as usize;
        let end_ring = poly_offsets[part + 1] as usize;
        ring_count += end_ring - first_ring;
        for ring in first_ring..end_ring {
            coordinate_count += ring_offsets[ring + 1] as usize - ring_offsets[ring] as usize;
        }
    }

    let mut xs = Vec::with_capacity(coordinate_count);
    let mut ys = Vec::with_capacity(coordinate_count);
    let mut subset_ring_offsets = Vec::with_capacity(ring_count + 1);
    let mut subset_poly_offsets = Vec::with_capacity(selected_parts.len() + 1);
    subset_ring_offsets.push(0i64);
    subset_poly_offsets.push(0i64);

    for &part in &selected_parts {
        let part = part as usize;
        let first_ring = poly_offsets[part] as usize;
        let end_ring = poly_offsets[part + 1] as usize;
        for ring in first_ring..end_ring {
            let start = ring_offsets[ring] as usize;
            let end = ring_offsets[ring + 1] as usize;
            xs.extend_from_slice(&source.xs[start..end]);
            ys.extend_from_slice(&source.ys[start..end]);
            subset_ring_offsets.push(xs.len() as i64);
        }
        subset_poly_offsets.push((subset_ring_offsets.len() - 1) as i64);
    }

    let subset_part_poly = source.part_poly.as_ref().map(|_| {
        let mut part_poly = Vec::with_capacity(selected_parts.len());
        for (logical, window) in selected_part_offsets.windows(2).enumerate() {
            part_poly.extend(std::iter::repeat_n(logical as u32, window[1] - window[0]));
        }
        Arc::from(part_poly)
    });

    let mut subset = Engine::new_polygons(
        xs.into(),
        ys.into(),
        subset_ring_offsets.into(),
        subset_poly_offsets.into(),
        subset_part_poly,
        indices.len(),
    );
    subset.index_mode = source.index_mode;
    subset.cost_factors = source.cost_factors.clone();
    subset.metric = source.metric;
    Ok(subset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{planner::cost::IndexMode, query::geodesy::DistanceMetric};

    fn engine() -> Engine {
        // Logical polygon 0 is one square. Logical polygon 1 is two disjoint squares.
        Engine::new_polygons(
            vec![
                0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 3.0, 3.0, 2.0, 2.0, 4.0, 5.0, 5.0, 4.0, 4.0,
            ]
            .into(),
            vec![
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            ]
            .into(),
            vec![0, 5, 10, 15].into(),
            vec![0, 1, 2, 3].into(),
            Some(vec![0, 1, 1].into()),
            2,
        )
    }

    #[test]
    fn subset_compacts_and_remaps_multipolygon_parts() {
        let subset = polygons(&engine(), &[1, 0, 1]).unwrap();
        assert_eq!(subset.n_polygons, 3);
        assert_eq!(subset.poly_offsets.as_deref().unwrap(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(subset.part_poly.as_deref().unwrap(), &[0, 0, 1, 2, 2]);
        assert_eq!(subset.xs.len(), 25);
        assert_eq!(
            subset.ring_offsets.as_deref().unwrap(),
            &[0, 5, 10, 15, 20, 25]
        );
    }

    #[test]
    fn subset_supports_empty_selection() {
        let subset = polygons(&engine(), &[]).unwrap();
        assert_eq!(subset.n_polygons, 0);
        assert!(subset.xs.is_empty());
        assert_eq!(subset.ring_offsets.as_deref().unwrap(), &[0]);
        assert_eq!(subset.poly_offsets.as_deref().unwrap(), &[0]);
    }

    #[test]
    fn subset_rejects_invalid_indices() {
        let error = match polygons(&engine(), &[2]) {
            Ok(_) => panic!("out-of-bounds subset unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error, "subset index 2 is out of bounds for 2 polygons");
    }

    #[test]
    fn subset_carries_configuration_without_indexes() {
        let mut source = engine();
        source.index_mode = IndexMode::None;
        source.metric = DistanceMetric::Haversine;
        let subset = polygons(&source, &[0]).unwrap();
        assert_eq!(subset.index_mode, IndexMode::None);
        assert_eq!(subset.metric, DistanceMetric::Haversine);
        assert!(subset.brute.is_none());
        assert!(subset.rtree.is_none());
        assert!(subset.prepared_polys.is_none());
    }
}
