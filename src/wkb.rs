//! Decodes a WKB Polygon or MultiPolygon column into the engine's flat ring arrays.

use std::mem::MaybeUninit;

use rayon::prelude::*;

const WKB_POLYGON: u32 = 3;
const WKB_MULTIPOLYGON: u32 = 6;

// Below this geometry count the decode stays single chunk
const MIN_DECODE_CHUNK: usize = 256;

/// Flat ring representation parsed from a WKB column (see Engine::from_polygon_rings).
/// part_poly is Some only when some geometry expanded into more than one part.
pub struct ParsedPolygons {
    pub xs: Vec<f64>,                // x coordinates of all ring vertices
    pub ys: Vec<f64>,                // y coordinates of all ring vertices
    pub ring_offsets: Vec<i64>,      // coordinate range per ring
    pub poly_offsets: Vec<i64>,      // ring range per polygon part
    pub part_poly: Option<Vec<u32>>, // logical polygon per part, None if no MultiPolygons
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }

    fn u8(&mut self) -> Result<u8, String> {
        let b = *self.bytes.get(self.pos).ok_or("truncated WKB")?;
        self.pos += 1;
        Ok(b)
    }

    fn u32(&mut self, le: bool) -> Result<u32, String> {
        let s = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or("truncated WKB")?;
        let a: [u8; 4] = s.try_into().unwrap();
        self.pos += 4;
        Ok(if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    }

    fn f64(&mut self, le: bool) -> Result<f64, String> {
        let s = self
            .bytes
            .get(self.pos..self.pos + 8)
            .ok_or("truncated WKB")?;
        let a: [u8; 8] = s.try_into().unwrap();
        self.pos += 8;
        Ok(if le {
            f64::from_le_bytes(a)
        } else {
            f64::from_be_bytes(a)
        })
    }
}

/// Count the coordinates in one polygon body, advancing the reader past the coordinate bytes
fn count_polygon_body(r: &mut Reader, le: bool) -> Result<usize, String> {
    let n_rings = r.u32(le)? as usize;
    let mut coords = 0usize;
    for _ in 0..n_rings {
        let n_pts = r.u32(le)? as usize;
        coords += n_pts;
        let skip = n_pts * 16;
        if r.pos + skip > r.bytes.len() {
            return Err("truncated WKB".into());
        }
        r.pos += skip;
    }
    Ok(coords)
}

/// Count the coordinates in one geometry without decoding them
fn count_geometry(r: &mut Reader) -> Result<usize, String> {
    let le = r.u8()? == 1;
    match r.u32(le)? {
        WKB_POLYGON => count_polygon_body(r, le),
        WKB_MULTIPOLYGON => {
            let n = r.u32(le)? as usize;
            let mut coords = 0usize;
            for _ in 0..n {
                let sub_le = r.u8()? == 1;
                if r.u32(sub_le)? != WKB_POLYGON {
                    return Err("MultiPolygon member is not a Polygon".into());
                }
                coords += count_polygon_body(r, sub_le)?;
            }
            Ok(coords)
        }
        t => Err(format!("unsupported WKB geometry type {t}")),
    }
}

/// Total coordinate count of geometries `[g_start, g_end)`, used to size each chunk's slices
fn count_chunk(
    data: &[u8],
    offsets: &[i64],
    g_start: usize,
    g_end: usize,
) -> Result<usize, String> {
    let mut coords = 0usize;
    for g in g_start..g_end {
        let (start, end) = (offsets[g] as usize, offsets[g + 1] as usize);
        let slice = data.get(start..end).ok_or("WKB offset out of bounds")?;
        coords += count_geometry(&mut Reader::new(slice))?;
    }
    Ok(coords)
}

/// Decode one polygon body, writing coordinates into `xs`/`ys` at `cursor` and recording each
/// ring's chunk-local end position in `ring_offsets`.
fn fill_polygon_body(
    r: &mut Reader,
    le: bool,
    xs: &mut [MaybeUninit<f64>],
    ys: &mut [MaybeUninit<f64>],
    cursor: &mut usize,
    ring_offsets: &mut Vec<i64>,
) -> Result<(), String> {
    let n_rings = r.u32(le)? as usize;
    for _ in 0..n_rings {
        let n_pts = r.u32(le)? as usize;
        for _ in 0..n_pts {
            let x = r.f64(le)?;
            let y = r.f64(le)?;
            xs[*cursor].write(x);
            ys[*cursor].write(y);
            *cursor += 1;
        }
        ring_offsets.push(*cursor as i64);
    }
    Ok(())
}

/// Decode one geometry into the chunk's coordinate slices and chunk-local offset arrays,
/// returning the number of parts it expanded into.
#[allow(clippy::too_many_arguments)]
fn fill_geometry(
    r: &mut Reader,
    geom: usize,
    xs: &mut [MaybeUninit<f64>],
    ys: &mut [MaybeUninit<f64>],
    cursor: &mut usize,
    ring_offsets: &mut Vec<i64>,
    poly_offsets: &mut Vec<i64>,
    part_poly: &mut Vec<u32>,
) -> Result<usize, String> {
    let le = r.u8()? == 1;
    match r.u32(le)? {
        WKB_POLYGON => {
            fill_polygon_body(r, le, xs, ys, cursor, ring_offsets)?;
            poly_offsets.push((ring_offsets.len() - 1) as i64);
            part_poly.push(geom as u32);
            Ok(1)
        }
        WKB_MULTIPOLYGON => {
            let n = r.u32(le)? as usize;
            for _ in 0..n {
                let sub_le = r.u8()? == 1;
                if r.u32(sub_le)? != WKB_POLYGON {
                    return Err("MultiPolygon member is not a Polygon".into());
                }
                fill_polygon_body(r, sub_le, xs, ys, cursor, ring_offsets)?;
                poly_offsets.push((ring_offsets.len() - 1) as i64);
                part_poly.push(geom as u32);
            }
            Ok(n)
        }
        t => Err(format!("unsupported WKB geometry type {t}")),
    }
}

/// One fill task: a geometry range paired with the disjoint `xs`/`ys` slices it writes into
type FillTask<'a> = (
    (usize, usize),
    &'a mut [MaybeUninit<f64>],
    &'a mut [MaybeUninit<f64>],
);

/// Chunk-local offset arrays produced by the fill pass, rebased onto running totals in the merge
struct ChunkSmall {
    ring_offsets: Vec<i64>,
    poly_offsets: Vec<i64>,
    part_poly: Vec<u32>,
    multipart: bool,
}

/// Decode geometries `[g_start, g_end)` straight into the pre-sized `xs`/`ys` slices. The slices
/// must hold exactly this chunk's coordinate count, which is asserted so the caller may set_len.
fn fill_chunk(
    data: &[u8],
    offsets: &[i64],
    g_start: usize,
    g_end: usize,
    xs: &mut [MaybeUninit<f64>],
    ys: &mut [MaybeUninit<f64>],
) -> Result<ChunkSmall, String> {
    let mut cursor = 0usize;
    let mut ring_offsets = vec![0i64];
    let mut poly_offsets = vec![0i64];
    let mut part_poly = Vec::new();
    let mut multipart = false;
    for g in g_start..g_end {
        let (start, end) = (offsets[g] as usize, offsets[g + 1] as usize);
        let slice = data.get(start..end).ok_or("WKB offset out of bounds")?;
        let parts = fill_geometry(
            &mut Reader::new(slice),
            g,
            xs,
            ys,
            &mut cursor,
            &mut ring_offsets,
            &mut poly_offsets,
            &mut part_poly,
        )?;
        multipart |= parts != 1;
    }
    // Every slot must be written, otherwise the caller's set_len would expose uninitialised memory
    if cursor != xs.len() {
        return Err("decode coordinate count mismatch".into());
    }
    Ok(ChunkSmall {
        ring_offsets,
        poly_offsets,
        part_poly,
        multipart,
    })
}

/// Parse a WKB column in `chunk_size`-geometry chunks. A cheap count pass sizes the output, then
/// the chunks decode in parallel into disjoint slices. The result is independent of `chunk_size`.
fn parse_polygons_chunked(
    data: &[u8],
    offsets: &[i64],
    chunk_size: usize,
) -> Result<ParsedPolygons, String> {
    let n = offsets.len().saturating_sub(1);
    if n == 0 {
        return Ok(ParsedPolygons {
            xs: Vec::new(),
            ys: Vec::new(),
            ring_offsets: vec![0],
            poly_offsets: vec![0],
            part_poly: None,
        });
    }
    let chunk_size = chunk_size.max(1);
    let bounds: Vec<(usize, usize)> = (0..n)
        .step_by(chunk_size)
        .map(|s| (s, (s + chunk_size).min(n)))
        .collect();

    // Pass 1: count coordinates per chunk so each chunk's output region is known before filling
    let coord_counts: Vec<usize> = bounds
        .par_iter()
        .map(|&(s, e)| count_chunk(data, offsets, s, e))
        .collect::<Result<_, _>>()?;
    let total_coords: usize = coord_counts.iter().sum();

    let mut xs: Vec<f64> = Vec::with_capacity(total_coords);
    let mut ys: Vec<f64> = Vec::with_capacity(total_coords);

    // Pass 2: carve the uninitialised capacity into one disjoint slice per chunk, then decode the
    // chunks in parallel, each writing its coordinates exactly once into its own slice.
    let smalls: Vec<ChunkSmall> = {
        let mut xs_rest = &mut xs.spare_capacity_mut()[..total_coords];
        let mut ys_rest = &mut ys.spare_capacity_mut()[..total_coords];
        let mut tasks: Vec<FillTask> = Vec::with_capacity(bounds.len());
        for (i, &b) in bounds.iter().enumerate() {
            let (xa, xb) = xs_rest.split_at_mut(coord_counts[i]);
            let (ya, yb) = ys_rest.split_at_mut(coord_counts[i]);
            xs_rest = xb;
            ys_rest = yb;
            tasks.push((b, xa, ya));
        }
        tasks
            .into_par_iter()
            .map(|((s, e), xa, ya)| fill_chunk(data, offsets, s, e, xa, ya))
            .collect::<Result<_, _>>()?
    };

    // SAFETY: every chunk decoded exactly its coordinate count (asserted in fill_chunk) into a
    // SAFETY: disjoint slice partitioning [0, total_coords), so all elements are initialised.
    unsafe {
        xs.set_len(total_coords);
        ys.set_len(total_coords);
    }

    // Merge the small per-chunk offset arrays, rebasing onto the running coord and ring totals
    let total_rings: usize = smalls.iter().map(|c| c.ring_offsets.len() - 1).sum();
    let total_parts: usize = smalls.iter().map(|c| c.poly_offsets.len() - 1).sum();
    let mut ring_offsets = Vec::with_capacity(total_rings + 1);
    ring_offsets.push(0);
    let mut poly_offsets = Vec::with_capacity(total_parts + 1);
    poly_offsets.push(0);
    let mut part_poly = Vec::with_capacity(total_parts);
    let mut multipart = false;
    let mut coord_base = 0i64;
    let mut ring_base = 0i64;
    for (i, c) in smalls.iter().enumerate() {
        ring_offsets.extend(c.ring_offsets[1..].iter().map(|&r| r + coord_base));
        poly_offsets.extend(c.poly_offsets[1..].iter().map(|&p| p + ring_base));
        part_poly.extend_from_slice(&c.part_poly);
        multipart |= c.multipart;
        coord_base += coord_counts[i] as i64;
        ring_base += (c.ring_offsets.len() - 1) as i64;
    }

    Ok(ParsedPolygons {
        xs,
        ys,
        ring_offsets,
        poly_offsets,
        part_poly: multipart.then_some(part_poly),
    })
}

/// Parse a whole WKB column. `data` is the concatenated value buffer, geometry g is
/// `data[offsets[g]..offsets[g+1]]`.
pub fn parse_polygons(data: &[u8], offsets: &[i64]) -> Result<ParsedPolygons, String> {
    let n = offsets.len().saturating_sub(1);
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = n.div_ceil(n_threads).max(MIN_DECODE_CHUNK);
    parse_polygons_chunked(data, offsets, chunk_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Little-endian WKB for a unit square at (cx, cy): one closed 5-point ring
    fn le_square(cx: f64, cy: f64) -> Vec<u8> {
        let mut b = vec![1]; // little-endian
        b.extend_from_slice(&WKB_POLYGON.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // 1 ring
        b.extend_from_slice(&5u32.to_le_bytes()); // 5 points
        for (x, y) in [
            (cx, cy),
            (cx + 1.0, cy),
            (cx + 1.0, cy + 1.0),
            (cx, cy + 1.0),
            (cx, cy),
        ] {
            b.extend_from_slice(&x.to_le_bytes());
            b.extend_from_slice(&y.to_le_bytes());
        }
        b
    }

    fn le_multipolygon(parts: &[Vec<u8>]) -> Vec<u8> {
        let mut b = vec![1];
        b.extend_from_slice(&WKB_MULTIPOLYGON.to_le_bytes());
        b.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for p in parts {
            b.extend_from_slice(p);
        }
        b
    }

    // Little-endian WKB polygon from explicit rings, exterior first then holes
    fn le_polygon_rings(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut b = vec![1];
        b.extend_from_slice(&WKB_POLYGON.to_le_bytes());
        b.extend_from_slice(&(rings.len() as u32).to_le_bytes());
        for ring in rings {
            b.extend_from_slice(&(ring.len() as u32).to_le_bytes());
            for &(x, y) in ring {
                b.extend_from_slice(&x.to_le_bytes());
                b.extend_from_slice(&y.to_le_bytes());
            }
        }
        b
    }

    fn square_with_hole(g: f64) -> Vec<u8> {
        le_polygon_rings(&[
            vec![(g, 0.0), (g + 4.0, 0.0), (g + 4.0, 4.0), (g, 4.0), (g, 0.0)],
            vec![
                (g + 1.0, 1.0),
                (g + 2.0, 1.0),
                (g + 2.0, 2.0),
                (g + 1.0, 1.0),
            ],
        ])
    }

    #[test]
    fn parses_polygon_and_multipolygon() {
        let a = le_square(0.0, 0.0);
        let b = le_multipolygon(&[le_square(10.0, 10.0), le_square(20.0, 20.0)]);
        let mut data = a.clone();
        data.extend_from_slice(&b);
        let offsets = [0, a.len() as i64, (a.len() + b.len()) as i64];

        let p = parse_polygons(&data, &offsets).unwrap();
        assert_eq!(p.poly_offsets, vec![0, 1, 2, 3]);
        assert_eq!(p.ring_offsets, vec![0, 5, 10, 15]);
        assert_eq!(p.part_poly, Some(vec![0, 1, 1]));
        assert_eq!(p.xs.len(), 15);
        assert_eq!(p.xs[10], 20.0);
    }

    #[test]
    fn single_polygons_need_no_part_map() {
        let a = le_square(0.0, 0.0);
        let offsets = [0, a.len() as i64];
        let p = parse_polygons(&a, &offsets).unwrap();
        assert!(p.part_poly.is_none());
    }

    #[test]
    fn chunked_decode_is_invariant_to_chunk_size() {
        // Mix polygons and multipolygons so the merge rebases both offset levels and part_poly
        let mut data = Vec::new();
        let mut offsets = vec![0i64];
        for g in 0..500 {
            let geom = if g % 7 == 0 {
                le_multipolygon(&[le_square(g as f64, 0.0), le_square(g as f64, 10.0)])
            } else if g % 5 == 0 {
                square_with_hole(g as f64)
            } else {
                le_square(g as f64, 0.0)
            };
            data.extend_from_slice(&geom);
            offsets.push(data.len() as i64);
        }
        let n = offsets.len() - 1;
        // chunk_size >= n is a single chunk, the serial reference the parallel paths must match
        let serial = parse_polygons_chunked(&data, &offsets, n).unwrap();
        for chunk_size in [1usize, 2, 13, 64, 256] {
            let p = parse_polygons_chunked(&data, &offsets, chunk_size).unwrap();
            assert_eq!(p.xs, serial.xs, "xs differ at chunk_size {chunk_size}");
            assert_eq!(p.ys, serial.ys, "ys differ at chunk_size {chunk_size}");
            assert_eq!(
                p.ring_offsets, serial.ring_offsets,
                "ring_offsets differ at chunk_size {chunk_size}"
            );
            assert_eq!(
                p.poly_offsets, serial.poly_offsets,
                "poly_offsets differ at chunk_size {chunk_size}"
            );
            assert_eq!(
                p.part_poly, serial.part_poly,
                "part_poly differ at chunk_size {chunk_size}"
            );
        }
    }

    #[test]
    fn empty_column_decodes_to_empty() {
        let p = parse_polygons(&[], &[0]).unwrap();
        assert_eq!(p.xs.len(), 0);
        assert_eq!(p.ring_offsets, vec![0]);
        assert_eq!(p.poly_offsets, vec![0]);
        assert!(p.part_poly.is_none());
    }

    #[test]
    fn decodes_polygon_with_holes() {
        // Exterior ring of 4 points then a hole of 3 points, one logical polygon
        let geom = le_polygon_rings(&[
            vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 0.0)],
            vec![(1.0, 1.0), (2.0, 1.0), (1.0, 1.0)],
        ]);
        let offsets = [0, geom.len() as i64];
        let p = parse_polygons(&geom, &offsets).unwrap();
        assert_eq!(p.ring_offsets, vec![0, 4, 7]);
        assert_eq!(p.poly_offsets, vec![0, 2]);
        assert_eq!(p.xs.len(), 7);
        assert!(p.part_poly.is_none());
    }

    #[test]
    fn truncated_wkb_is_error_not_panic() {
        // Claim a full square but drop trailing coordinate bytes
        let mut geom = le_square(0.0, 0.0);
        geom.truncate(geom.len() - 8);
        let offsets = [0, geom.len() as i64];
        assert!(parse_polygons(&geom, &offsets).is_err());
    }

    #[test]
    fn unsupported_geometry_type_is_error() {
        // Byte order then WKB type 1 (Point), which the polygon decoder rejects
        let mut geom = vec![1u8];
        geom.extend_from_slice(&1u32.to_le_bytes());
        geom.extend_from_slice(&0.0f64.to_le_bytes());
        geom.extend_from_slice(&0.0f64.to_le_bytes());
        let offsets = [0, geom.len() as i64];
        assert!(parse_polygons(&geom, &offsets).is_err());
    }

    #[test]
    fn chunk_error_is_handled_in_parallel_decode() {
        // Many valid geometries plus one truncated, one geometry per chunk so a single chunk
        // fails. The decode must return Err cleanly rather than set_len over uninitialised memory.
        let mut data = Vec::new();
        let mut offsets = vec![0i64];
        for g in 0..300 {
            data.extend_from_slice(&le_square(g as f64, 0.0));
            offsets.push(data.len() as i64);
        }
        let mut bad = le_square(99.0, 0.0);
        bad.truncate(bad.len() - 8);
        data.extend_from_slice(&bad);
        offsets.push(data.len() as i64);
        assert!(parse_polygons_chunked(&data, &offsets, 1).is_err());
    }
}
