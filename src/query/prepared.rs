//! Per-polygon Y-slab edge index, built for a part the first time it is probed.

use std::sync::OnceLock;

use crate::query::range::pip_raw;

// Edges per band: the one tuning knob, trading probe speed against band memory
const EDGES_PER_BAND: usize = 8;
// Safety cap on bands per polygon, bounding memory for pathological rings
const MAX_BANDS: usize = 1024;
const VERTS_PER_EDGE: usize = 2; // vertex indices per edge into shared xs/ys

// Below this a probe reads most edges anyway and the band lookup costs more than it skips
const MIN_BANDS_TO_PAY: usize = 4;

/// Prepared point-in-polygon accelerator over a flat polygon dataset.
///
/// A query touches far fewer parts than a dataset holds, so bands are built per part on that
/// part's first probe rather than for everything up front. Probing a part before its bands
/// exist falls back to a raw edge scan, which is also the permanent path for parts too small
/// for banding to pay.
pub struct PreparedPolygons {
    bands: Vec<OnceLock<PartBands>>, // empty until the part's first probe builds it
    bandable: Vec<bool>,             // parts whose band count clears MIN_BANDS_TO_PAY
}

struct PartBands {
    min_y: f64,
    inv_band_h: f64,
    band_ptr: Box<[u32]>,   // CSR into band_edges, length nbands + 1
    band_edges: Box<[u32]>, // per-band edge indices, local to this part
    edge_verts: Box<[u32]>, // each edge as two vertex indices into the shared xs/ys, interleaved
}

impl PreparedPolygons {
    /// Reserve a band slot per part without reading any coordinates.
    ///
    /// Edge counts come from the ring arrays alone, which is enough to decide which parts are
    /// worth banding, so construction stays proportional to the part count.
    pub fn new(ring_offsets: &[i64], poly_offsets: &[i64]) -> Self {
        let n_parts = poly_offsets.len().saturating_sub(1);
        let bandable = (0..n_parts)
            .map(|p| band_count(edge_count(ring_offsets, poly_offsets, p)) >= MIN_BANDS_TO_PAY)
            .collect();
        let bands = (0..n_parts).map(|_| OnceLock::new()).collect();
        PreparedPolygons { bands, bandable }
    }

    /// True when part `p` contains the point. Matches `pip_raw` for valid polygons.
    /// Edge coordinates are read from the shared `xs`/`ys` by index, not stored here.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn contains(
        &self,
        p: usize,
        qx: f64,
        qy: f64,
        xs: &[f64],
        ys: &[f64],
        ring_offsets: &[i64],
        poly_offsets: &[i64],
    ) -> bool {
        if !self.bandable[p] {
            return pip_raw(qx, qy, xs, ys, ring_offsets, poly_offsets, p);
        }
        self.bands[p]
            .get_or_init(|| build_part(ys, ring_offsets, poly_offsets, p))
            .contains(qx, qy, xs, ys)
    }

    /// Count of parts whose bands have been built
    #[cfg(test)]
    fn built_parts(&self) -> usize {
        self.bands
            .iter()
            .filter(|slot| slot.get().is_some())
            .count()
    }
}

impl PartBands {
    #[inline]
    fn contains(&self, qx: f64, qy: f64, xs: &[f64], ys: &[f64]) -> bool {
        let nbands = self.band_ptr.len() - 1;
        let slot = band_of(qy, self.min_y, self.inv_band_h, nbands);
        let lo = self.band_ptr[slot] as usize;
        let hi = self.band_ptr[slot + 1] as usize;
        let mut inside = false;
        for &ei in &self.band_edges[lo..hi] {
            let o = ei as usize * VERTS_PER_EDGE;
            let v0 = self.edge_verts[o] as usize;
            let v1 = self.edge_verts[o + 1] as usize;
            let (x0, y0, x1, y1) = (xs[v0], ys[v0], xs[v1], ys[v1]);
            if (y0 > qy) != (y1 > qy) && qx < (x1 - x0) * (qy - y0) / (y1 - y0) + x0 {
                inside = !inside;
            }
        }
        inside
    }
}

fn edge_count(ring_offsets: &[i64], poly_offsets: &[i64], p: usize) -> usize {
    (poly_offsets[p] as usize..poly_offsets[p + 1] as usize)
        .map(|r| {
            let (s, e) = (ring_offsets[r] as usize, ring_offsets[r + 1] as usize);
            // Rings of under two vertices carry no edges, matching build_part
            if e - s < 2 {
                0
            } else {
                e - s
            }
        })
        .sum()
}

fn band_count(edges: usize) -> usize {
    (edges / EDGES_PER_BAND).clamp(1, MAX_BANDS)
}

fn build_part(ys: &[f64], ring_offsets: &[i64], poly_offsets: &[i64], p: usize) -> PartBands {
    let r_start = poly_offsets[p] as usize;
    let r_end = poly_offsets[p + 1] as usize;

    let mut edge_verts: Vec<u32> = Vec::new();
    let mut pmin_y = f64::INFINITY;
    let mut pmax_y = f64::NEG_INFINITY;
    for r in r_start..r_end {
        let s = ring_offsets[r] as usize;
        let e = ring_offsets[r + 1] as usize;
        if e - s < 2 {
            continue;
        }
        let mut j = e - 1;
        // k is the global vertex index we store rather than just a ys probe
        #[allow(clippy::needless_range_loop)]
        for k in s..e {
            edge_verts.push(k as u32);
            edge_verts.push(j as u32);
            pmin_y = pmin_y.min(ys[k]);
            pmax_y = pmax_y.max(ys[k]);
            j = k;
        }
    }

    let edges = edge_verts.len() / VERTS_PER_EDGE;
    let nbands = band_count(edges);
    let span = pmax_y - pmin_y;
    let inv = if span > 0.0 {
        nbands as f64 / span
    } else {
        0.0
    };
    let min_y = if pmin_y.is_finite() { pmin_y } else { 0.0 };

    // File each edge index into every band its y-span overlaps, storing only vertex indices
    let mut per_band: Vec<Vec<u32>> = vec![Vec::new(); nbands];
    for ei in 0..edges {
        let o = ei * VERTS_PER_EDGE;
        let (ya, yb) = (ys[edge_verts[o] as usize], ys[edge_verts[o + 1] as usize]);
        let b_lo = band_of(ya.min(yb), min_y, inv, nbands);
        let b_hi = band_of(ya.max(yb), min_y, inv, nbands);
        for band in per_band.iter_mut().take(b_hi + 1).skip(b_lo) {
            band.push(ei as u32);
        }
    }

    let mut band_ptr = Vec::with_capacity(nbands + 1);
    let mut band_edges = Vec::new();
    band_ptr.push(0);
    for band in &per_band {
        band_edges.extend_from_slice(band);
        band_ptr.push(band_edges.len() as u32);
    }

    PartBands {
        min_y,
        inv_band_h: inv,
        band_ptr: band_ptr.into_boxed_slice(),
        band_edges: band_edges.into_boxed_slice(),
        edge_verts: edge_verts.into_boxed_slice(),
    }
}

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

    fn ring_polygon(n: usize) -> (Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for k in 0..n {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            xs.push(a.cos());
            ys.push(a.sin());
        }
        (xs, ys, vec![0, n as i64], vec![0, 1])
    }

    #[test]
    fn prepared_agrees_with_pip_raw_including_hole() {
        let (xs, ys, ring, poly) = square_with_hole();
        let prepared = PreparedPolygons::new(&ring, &poly);
        for gx in 0..50 {
            for gy in 0..50 {
                let (qx, qy) = (gx as f64 * 0.1 - 0.5, gy as f64 * 0.1 - 0.5);
                assert_eq!(
                    prepared.contains(0, qx, qy, &xs, &ys, &ring, &poly),
                    pip_raw(qx, qy, &xs, &ys, &ring, &poly, 0),
                    "mismatch at ({qx}, {qy})"
                );
            }
        }
    }

    #[test]
    fn prepared_agrees_on_many_vertex_polygon() {
        let (xs, ys, ring, poly) = ring_polygon(200);
        let prepared = PreparedPolygons::new(&ring, &poly);
        for gx in 0..40 {
            for gy in 0..40 {
                let (qx, qy) = (gx as f64 * 0.075 - 1.5, gy as f64 * 0.075 - 1.5);
                assert_eq!(
                    prepared.contains(0, qx, qy, &xs, &ys, &ring, &poly),
                    pip_raw(qx, qy, &xs, &ys, &ring, &poly, 0),
                    "mismatch at ({qx}, {qy})"
                );
            }
        }
    }

    #[test]
    fn construction_builds_no_bands() {
        let (_, _, ring, poly) = ring_polygon(200);
        assert_eq!(PreparedPolygons::new(&ring, &poly).built_parts(), 0);
    }

    #[test]
    fn only_probed_parts_are_built() {
        // Two ring polygons side by side where a probe of the first never reaches the second
        let (mut xs, mut ys, _, _) = ring_polygon(200);
        let (right_xs, right_ys, _, _) = ring_polygon(200);
        xs.extend(right_xs.iter().map(|x| x + 10.0));
        ys.extend(right_ys.iter().copied());
        let (ring, poly) = (vec![0, 200, 400], vec![0, 1, 2]);

        let prepared = PreparedPolygons::new(&ring, &poly);
        prepared.contains(0, 0.0, 0.0, &xs, &ys, &ring, &poly);

        assert_eq!(prepared.built_parts(), 1);
    }

    #[test]
    fn repeated_probes_build_a_part_once() {
        let (xs, ys, ring, poly) = ring_polygon(200);
        let prepared = PreparedPolygons::new(&ring, &poly);
        for _ in 0..10 {
            assert!(prepared.contains(0, 0.0, 0.0, &xs, &ys, &ring, &poly));
        }
        assert_eq!(prepared.built_parts(), 1);
    }

    #[test]
    fn small_polygons_never_build_bands() {
        // A square yields one band and so stays under MIN_BANDS_TO_PAY
        let xs = vec![0.0, 1.0, 1.0, 0.0];
        let ys = vec![0.0, 0.0, 1.0, 1.0];
        let (ring, poly) = (vec![0, 4], vec![0, 1]);

        let prepared = PreparedPolygons::new(&ring, &poly);
        for gx in 0..20 {
            for gy in 0..20 {
                let (qx, qy) = (gx as f64 * 0.1 - 0.5, gy as f64 * 0.1 - 0.5);
                assert_eq!(
                    prepared.contains(0, qx, qy, &xs, &ys, &ring, &poly),
                    pip_raw(qx, qy, &xs, &ys, &ring, &poly, 0),
                    "mismatch at ({qx}, {qy})"
                );
            }
        }
        assert_eq!(prepared.built_parts(), 0);
    }

    #[test]
    fn banding_starts_at_the_minimum_band_count() {
        let below = MIN_BANDS_TO_PAY * EDGES_PER_BAND - 1;
        let at = MIN_BANDS_TO_PAY * EDGES_PER_BAND;
        let (xs, ys, ring, poly) = ring_polygon(below);
        let prepared = PreparedPolygons::new(&ring, &poly);
        prepared.contains(0, 0.0, 0.0, &xs, &ys, &ring, &poly);
        assert_eq!(prepared.built_parts(), 0);

        let (xs, ys, ring, poly) = ring_polygon(at);
        let prepared = PreparedPolygons::new(&ring, &poly);
        prepared.contains(0, 0.0, 0.0, &xs, &ys, &ring, &poly);
        assert_eq!(prepared.built_parts(), 1);
    }
}
