//! Per-polygon Y-slab edge index for sub-linear point-in-polygon.
//!
//! A ray cast at height y only crosses edges spanning y, so bucketing each polygon's
//! edges into horizontal bands lets a containment test scan one band instead of every
//! edge. Even-odd over all rings handles holes, matching `pip_raw` for valid polygons.
//! Each band stores its edges as a contiguous interleaved run so the probe is a single
//! sequential scan.

use rayon::prelude::*;

/// Edges per band: the one tuning knob, trading probe speed against band memory
const EDGES_PER_BAND: usize = 8;
/// Safety cap on bands per polygon, bounding memory for pathological rings
const MAX_BANDS: usize = 1024;
/// Floats per stored edge: [x0, y0, x1, y1]
const STRIDE: usize = 4;

/// Prepared point-in-polygon accelerator over a flat polygon dataset
pub struct PreparedPolygons {
    min_y: Vec<f64>,
    inv_band_h: Vec<f64>,
    /// `band_base[p]..band_base[p + 1]` are polygon p's band slots in `band_ptr`
    band_base: Vec<usize>,
    /// CSR into `band_coords` in edge units (length total_bands + 1)
    band_ptr: Vec<u32>,
    /// Band edges, interleaved [x0, y0, x1, y1] and grouped by band
    band_coords: Vec<f64>,
}

/// One polygon's prepared bands, produced in parallel then concatenated
struct PolyPrep {
    min_y: f64,
    inv_band_h: f64,
    bands: Vec<Vec<f64>>,
}

impl PreparedPolygons {
    /// Build from two-level ring arrays (the Engine's flat polygon layout)
    pub fn build(xs: &[f64], ys: &[f64], ring_offsets: &[i64], poly_offsets: &[i64]) -> Self {
        let n_polys = poly_offsets.len().saturating_sub(1);
        let preps: Vec<PolyPrep> = (0..n_polys)
            .into_par_iter()
            .map(|p| prepare_one(xs, ys, ring_offsets, poly_offsets, p))
            .collect();

        let mut min_y = Vec::with_capacity(n_polys);
        let mut inv_band_h = Vec::with_capacity(n_polys);
        let mut band_base = Vec::with_capacity(n_polys + 1);
        let mut band_ptr: Vec<u32> = vec![0];
        let mut band_coords: Vec<f64> = Vec::new();

        band_base.push(0);
        for prep in &preps {
            min_y.push(prep.min_y);
            inv_band_h.push(prep.inv_band_h);
            for band in &prep.bands {
                band_coords.extend_from_slice(band);
                band_ptr.push((band_coords.len() / STRIDE) as u32);
            }
            band_base.push(band_ptr.len() - 1);
        }

        PreparedPolygons {
            min_y,
            inv_band_h,
            band_base,
            band_ptr,
            band_coords,
        }
    }

    /// True when polygon `p` contains the point. Matches `pip_raw` for valid polygons
    #[inline]
    pub fn contains(&self, p: usize, qx: f64, qy: f64) -> bool {
        let bstart = self.band_base[p];
        let nbands = self.band_base[p + 1] - bstart;
        if nbands == 0 {
            return false;
        }
        let slot = bstart + band_of(qy, self.min_y[p], self.inv_band_h[p], nbands);
        let lo = self.band_ptr[slot] as usize * STRIDE;
        let hi = self.band_ptr[slot + 1] as usize * STRIDE;
        let mut inside = false;
        for e in self.band_coords[lo..hi].chunks_exact(STRIDE) {
            let (x0, y0, x1, y1) = (e[0], e[1], e[2], e[3]);
            if (y0 > qy) != (y1 > qy) && qx < (x1 - x0) * (qy - y0) / (y1 - y0) + x0 {
                inside = !inside;
            }
        }
        inside
    }
}

/// Prepare one polygon: collect its ring edges and bucket them into y-bands
fn prepare_one(
    xs: &[f64],
    ys: &[f64],
    ring_offsets: &[i64],
    poly_offsets: &[i64],
    p: usize,
) -> PolyPrep {
    let r_start = poly_offsets[p] as usize;
    let r_end = poly_offsets[p + 1] as usize;

    let mut edges: Vec<[f64; 4]> = Vec::new();
    let mut pmin_y = f64::INFINITY;
    let mut pmax_y = f64::NEG_INFINITY;
    for r in r_start..r_end {
        let s = ring_offsets[r] as usize;
        let e = ring_offsets[r + 1] as usize;
        if e - s < 2 {
            continue;
        }
        let mut j = e - 1;
        for k in s..e {
            edges.push([xs[k], ys[k], xs[j], ys[j]]);
            pmin_y = pmin_y.min(ys[k]);
            pmax_y = pmax_y.max(ys[k]);
            j = k;
        }
    }

    let nbands = (edges.len() / EDGES_PER_BAND).clamp(1, MAX_BANDS);
    let span = pmax_y - pmin_y;
    let inv = if span > 0.0 {
        nbands as f64 / span
    } else {
        0.0
    };
    let min_y = if pmin_y.is_finite() { pmin_y } else { 0.0 };

    // Append each edge to every band its y-span overlaps
    let mut bands: Vec<Vec<f64>> = vec![Vec::new(); nbands];
    for e in &edges {
        let b_lo = band_of(e[1].min(e[3]), min_y, inv, nbands);
        let b_hi = band_of(e[1].max(e[3]), min_y, inv, nbands);
        for band in bands.iter_mut().take(b_hi + 1).skip(b_lo) {
            band.extend_from_slice(e);
        }
    }

    PolyPrep {
        min_y,
        inv_band_h: inv,
        bands,
    }
}

/// Band index for `y`, clamped so out-of-extent points land in end bands (no straddle)
#[inline]
fn band_of(y: f64, min_y: f64, inv_band_h: f64, nbands: usize) -> usize {
    if inv_band_h == 0.0 {
        return 0;
    }
    (((y - min_y) * inv_band_h) as isize).clamp(0, nbands as isize - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::range::pip_raw;

    fn square_with_hole() -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
        let xs = vec![0.0, 4.0, 4.0, 0.0, 0.0, 1.0, 3.0, 3.0, 1.0, 1.0];
        let ys = vec![0.0, 0.0, 4.0, 4.0, 0.0, 1.0, 1.0, 3.0, 3.0, 1.0];
        (xs, ys, vec![0, 5, 10], vec![0, 2])
    }

    #[test]
    fn prepared_agrees_with_pip_raw_including_hole() {
        let (xs, ys, ring, poly) = square_with_hole();
        let prepared = PreparedPolygons::build(&xs, &ys, &ring, &poly);
        for gx in 0..50 {
            for gy in 0..50 {
                let (qx, qy) = (gx as f64 * 0.1 - 0.5, gy as f64 * 0.1 - 0.5);
                assert_eq!(
                    prepared.contains(0, qx, qy),
                    pip_raw(qx, qy, &xs, &ys, &ring, &poly, 0),
                    "mismatch at ({qx}, {qy})"
                );
            }
        }
    }

    #[test]
    fn prepared_agrees_on_many_vertex_polygon() {
        let n = 200;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for k in 0..n {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            xs.push(a.cos());
            ys.push(a.sin());
        }
        let (ring, poly) = (vec![0, n as i64], vec![0, 1]);
        let prepared = PreparedPolygons::build(&xs, &ys, &ring, &poly);
        for gx in 0..40 {
            for gy in 0..40 {
                let (qx, qy) = (gx as f64 * 0.075 - 1.5, gy as f64 * 0.075 - 1.5);
                assert_eq!(
                    prepared.contains(0, qx, qy),
                    pip_raw(qx, qy, &xs, &ys, &ring, &poly, 0),
                    "mismatch at ({qx}, {qy})"
                );
            }
        }
    }
}
