//! Map index-part results to logical polygons for MultiPolygon support.
//!
//! The spatial index is built over parts (single-exterior polygons), so a kernel
//! emits part indices. These helpers fold those parts into the logical polygons
//! recorded by `part_poly`, deduplicating where one polygon owns several parts. All
//! are no-ops conceptually when a dataset has one part per polygon (part_poly is None
//! at the call site, so they are simply not called).

/// Map self-join part pairs to unique logical polygon pairs (i < j).
///
/// Pairs whose two parts belong to the same polygon are dropped, and the remaining
/// logical pairs are sorted and deduplicated so each intersecting polygon pair appears once.
pub fn dedup_self_pairs(pairs: Vec<u64>, part_poly: &[u32]) -> Vec<u64> {
    let mut logical: Vec<(u32, u32)> = pairs
        .chunks_exact(2)
        .filter_map(|c| {
            let (a, b) = (part_poly[c[0] as usize], part_poly[c[1] as usize]);
            (a != b).then_some((a.min(b), a.max(b)))
        })
        .collect();
    logical.sort_unstable();
    logical.dedup();
    let mut out = Vec::with_capacity(logical.len() * 2);
    for (a, b) in logical {
        out.push(a as u64);
        out.push(b as u64);
    }
    out
}

/// Map part indices to logical polygon indices, preserving order and dropping repeats
pub fn dedup_indices(parts: Vec<usize>, part_poly: &[u32]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(parts.len());
    for p in parts {
        let poly = part_poly[p] as usize;
        if !out.contains(&poly) {
            out.push(poly);
        }
    }
    out
}

/// Sum per-part areas into per-polygon areas
pub fn sum_part_areas(part_areas: &[f64], part_poly: &[u32], n_polygons: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_polygons];
    for (p, &a) in part_areas.iter().enumerate() {
        out[part_poly[p] as usize] += a;
    }
    out
}

/// Group parts by polygon as CSR: polygon `g` owns parts `parts[offsets[g]..offsets[g+1]]`
pub fn polygon_parts_csr(part_poly: &[u32], n_polygons: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; n_polygons + 1];
    for &g in part_poly {
        offsets[g as usize + 1] += 1;
    }
    for g in 0..n_polygons {
        offsets[g + 1] += offsets[g];
    }
    let mut parts = vec![0u32; part_poly.len()];
    let mut cursor = offsets.clone();
    for (p, &g) in part_poly.iter().enumerate() {
        parts[cursor[g as usize] as usize] = p as u32;
        cursor[g as usize] += 1;
    }
    (offsets, parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Parts 0,1 -> polygon 0 (a MultiPolygon), part 2 -> polygon 1
    const PART_POLY: [u32; 3] = [0, 0, 1];

    #[test]
    fn self_pairs_drop_same_polygon_and_dedup() {
        // (0,1) same polygon -> dropped, (1,2) and (0,2) both -> logical (0,1)
        let pairs = vec![0, 1, 1, 2, 0, 2];
        assert_eq!(dedup_self_pairs(pairs, &PART_POLY), vec![0, 1]);
    }

    #[test]
    fn areas_sum_over_parts() {
        let areas = sum_part_areas(&[1.5, 2.5, 4.0], &PART_POLY, 2);
        assert_eq!(areas, vec![4.0, 4.0]);
    }

    #[test]
    fn csr_groups_parts() {
        let (offsets, parts) = polygon_parts_csr(&PART_POLY, 2);
        assert_eq!(offsets, vec![0, 2, 3]);
        assert_eq!(parts, vec![0, 1, 2]);
    }
}
