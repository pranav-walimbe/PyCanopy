//! Decode a WKB Polygon / MultiPolygon column straight into the engine's flat ring
//! arrays, with no per-geometry object allocation (the polygon analogue of the
//! vectorised WKB point reader). Falls back to shapely on the Python side for any
//! variant this does not recognise (returned here as an Err).

const WKB_POLYGON: u32 = 3;
const WKB_MULTIPOLYGON: u32 = 6;

/// Flat ring representation parsed from a WKB column (see Engine::from_polygon_rings).
/// part_poly is Some only when some geometry expanded into more than one part.
pub struct ParsedPolygons {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub ring_offsets: Vec<i64>,
    pub poly_offsets: Vec<i64>,
    pub part_poly: Option<Vec<u32>>,
}

/// Little cursor over one geometry's bytes with endian-aware reads.
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

/// Read one polygon body (its rings) into the coordinate arrays, recording ring bounds.
fn read_polygon_body(
    r: &mut Reader,
    le: bool,
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
    ring_offsets: &mut Vec<i64>,
) -> Result<(), String> {
    let n_rings = r.u32(le)? as usize;
    for _ in 0..n_rings {
        let n_pts = r.u32(le)? as usize;
        for _ in 0..n_pts {
            xs.push(r.f64(le)?);
            ys.push(r.f64(le)?);
        }
        ring_offsets.push(xs.len() as i64);
    }
    Ok(())
}

/// Append part p as one ring group: poly_offsets tracks the cumulative ring count.
fn push_part(
    poly_offsets: &mut Vec<i64>,
    part_poly: &mut Vec<u32>,
    geom: usize,
    ring_count: usize,
) {
    poly_offsets.push(ring_count as i64);
    part_poly.push(geom as u32);
}

/// Parse one top-level geometry, returning how many parts it contributed.
fn read_geometry(
    r: &mut Reader,
    geom: usize,
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
    ring_offsets: &mut Vec<i64>,
    poly_offsets: &mut Vec<i64>,
    part_poly: &mut Vec<u32>,
) -> Result<usize, String> {
    let le = r.u8()? == 1;
    match r.u32(le)? {
        WKB_POLYGON => {
            read_polygon_body(r, le, xs, ys, ring_offsets)?;
            push_part(poly_offsets, part_poly, geom, ring_offsets.len() - 1);
            Ok(1)
        }
        WKB_MULTIPOLYGON => {
            let n = r.u32(le)? as usize;
            for _ in 0..n {
                let sub_le = r.u8()? == 1;
                if r.u32(sub_le)? != WKB_POLYGON {
                    return Err("MultiPolygon member is not a Polygon".into());
                }
                read_polygon_body(r, sub_le, xs, ys, ring_offsets)?;
                push_part(poly_offsets, part_poly, geom, ring_offsets.len() - 1);
            }
            Ok(n)
        }
        t => Err(format!("unsupported WKB geometry type {t}")),
    }
}

/// Parse a whole WKB column. `data` is the concatenated value buffer; geometry g is
/// `data[offsets[g]..offsets[g+1]]`.
pub fn parse_polygons(data: &[u8], offsets: &[i64]) -> Result<ParsedPolygons, String> {
    let n = offsets.len().saturating_sub(1);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ring_offsets = vec![0i64];
    let mut poly_offsets = vec![0i64];
    let mut part_poly = Vec::with_capacity(n);
    let mut multipart = false;

    for g in 0..n {
        let (start, end) = (offsets[g] as usize, offsets[g + 1] as usize);
        let slice = data.get(start..end).ok_or("WKB offset out of bounds")?;
        let parts = read_geometry(
            &mut Reader::new(slice),
            g,
            &mut xs,
            &mut ys,
            &mut ring_offsets,
            &mut poly_offsets,
            &mut part_poly,
        )?;
        multipart |= parts != 1;
    }

    Ok(ParsedPolygons {
        xs,
        ys,
        ring_offsets,
        poly_offsets,
        part_poly: multipart.then_some(part_poly),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Little-endian WKB for a unit square at (cx, cy): one closed 5-point ring.
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
}
