use geo::{coord, Geometry, Point, Rect};

use crate::index::{geom_center, SpatialIndex};

/// Uniform grid index built from scratch, best for large uniform point datasets
pub struct UniformGrid {
    cells: Vec<Vec<usize>>,
    coords: Vec<(f64, f64)>,
    min_x: f64,
    min_y: f64,
    cell_w: f64,
    cell_h: f64,
    grid_w: usize,
    grid_h: usize,
}

impl UniformGrid {
    fn cell_of(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x - self.min_x) / self.cell_w)
            .floor()
            .clamp(0.0, self.grid_w as f64 - 1.0) as usize;
        let row = ((y - self.min_y) / self.cell_h)
            .floor()
            .clamp(0.0, self.grid_h as f64 - 1.0) as usize;
        (col, row)
    }

    fn bbox_cells(
        &self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> impl Iterator<Item = usize> + '_ {
        let col_lo = ((min_x - self.min_x) / self.cell_w).floor().max(0.0) as usize;
        let row_lo = ((min_y - self.min_y) / self.cell_h).floor().max(0.0) as usize;
        let col_hi = ((max_x - self.min_x) / self.cell_w)
            .ceil()
            .min(self.grid_w as f64 - 1.0) as usize;
        let row_hi = ((max_y - self.min_y) / self.cell_h)
            .ceil()
            .min(self.grid_h as f64 - 1.0) as usize;

        (row_lo..=row_hi).flat_map(move |r| (col_lo..=col_hi).map(move |c| r * self.grid_w + c))
    }
}

impl SpatialIndex for UniformGrid {
    fn build(geometries: &[Geometry<f64>]) -> Self {
        let n = geometries.len();
        let mut coords = Vec::with_capacity(n);

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for geom in geometries {
            let (x, y) = geom_center(geom);
            coords.push((x, y));
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let grid_dim = (n as f64).sqrt().ceil().max(1.0) as usize;
        let grid_w = grid_dim;
        let grid_h = grid_dim;

        let extent_w = max_x - min_x;
        let extent_h = max_y - min_y;
        let cell_w = if extent_w > 0.0 {
            extent_w / grid_w as f64
        } else {
            1.0
        };
        let cell_h = if extent_h > 0.0 {
            extent_h / grid_h as f64
        } else {
            1.0
        };

        let mut cells = vec![Vec::<usize>::new(); grid_w * grid_h];
        for (i, &(x, y)) in coords.iter().enumerate() {
            let col = ((x - min_x) / cell_w)
                .floor()
                .clamp(0.0, grid_w as f64 - 1.0) as usize;
            let row = ((y - min_y) / cell_h)
                .floor()
                .clamp(0.0, grid_h as f64 - 1.0) as usize;
            cells[row * grid_w + col].push(i);
        }

        UniformGrid {
            cells,
            coords,
            min_x,
            min_y,
            cell_w,
            cell_h,
            grid_w,
            grid_h,
        }
    }

    fn nearest(&self, point: &Point<f64>, k: usize) -> Vec<usize> {
        if self.coords.is_empty() {
            return vec![];
        }
        let k = k.min(self.coords.len());
        let px = point.x();
        let py = point.y();

        let (start_col, start_row) = self.cell_of(px, py);
        let sc = start_col as isize;
        let sr = start_row as isize;
        let max_ring = self.grid_w.max(self.grid_h) as isize + 1;
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for ring in 0..=max_ring {
            let r = ring;
            let col_lo = (sc - r).max(0) as usize;
            let col_hi = (sc + r).min(self.grid_w as isize - 1) as usize;
            let row_lo = (sr - r).max(0) as usize;
            let row_hi = (sr + r).min(self.grid_h as isize - 1) as usize;

            // Unclamped bounds — needed to correctly identify ring border cells
            // when the grid edge has restricted the visible range.
            let uc_col_lo = sc - r;
            let uc_col_hi = sc + r;
            let uc_row_lo = sr - r;
            let uc_row_hi = sr + r;

            for row in row_lo..=row_hi {
                for col in col_lo..=col_hi {
                    let on_border = r == 0
                        || col as isize == uc_col_lo
                        || col as isize == uc_col_hi
                        || row as isize == uc_row_lo
                        || row as isize == uc_row_hi;
                    if !on_border {
                        continue;
                    }
                    for &idx in &self.cells[row * self.grid_w + col] {
                        let (x, y) = self.coords[idx];
                        let d = (x - px).powi(2) + (y - py).powi(2);
                        candidates.push((idx, d));
                    }
                }
            }

            if candidates.len() >= k {
                // Stop only when no unvisited cell can contain a closer point
                // than our current k-th best candidate.
                let col_lo_i = (sc - r).max(0);
                let col_hi_i = (sc + r).min(self.grid_w as isize - 1);
                let row_lo_i = (sr - r).max(0);
                let row_hi_i = (sr + r).min(self.grid_h as isize - 1);

                // Entire grid covered — definitely done.
                let full_cover = col_lo_i == 0
                    && col_hi_i == self.grid_w as isize - 1
                    && row_lo_i == 0
                    && row_hi_i == self.grid_h as isize - 1;
                if full_cover {
                    break;
                }

                // Minimum squared distance from (px, py) to any unvisited cell.
                // Unvisited strips exist in any direction not clamped to the grid edge.
                // Each strip is a half-plane, so the nearest unvisited point in each
                // strip has zero distance in the free axis — only the perpendicular
                // distance matters.
                let vx_lo = self.min_x + col_lo_i as f64 * self.cell_w;
                let vx_hi = self.min_x + (col_hi_i + 1) as f64 * self.cell_w;
                let vy_lo = self.min_y + row_lo_i as f64 * self.cell_h;
                let vy_hi = self.min_y + (row_hi_i + 1) as f64 * self.cell_h;

                let mut min_unvisited_sq = f64::INFINITY;
                if col_lo_i > 0 {
                    let d = (px - vx_lo).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }
                if col_hi_i < self.grid_w as isize - 1 {
                    let d = (vx_hi - px).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }
                if row_lo_i > 0 {
                    let d = (py - vy_lo).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }
                if row_hi_i < self.grid_h as isize - 1 {
                    let d = (vy_hi - py).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }

                // k-th best distance² among current candidates.
                let mut dists: Vec<f64> = candidates.iter().map(|c| c.1).collect();
                dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let kth_dist_sq = dists[k - 1];

                if min_unvisited_sq > kth_dist_sq {
                    break;
                }
            }
        }

        candidates
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn range(&self, bbox: &Rect<f64>) -> Vec<usize> {
        if self.grid_w == 0 || self.grid_h == 0 {
            return vec![];
        }
        let (qmin_x, qmin_y, qmax_x, qmax_y) =
            (bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y);

        self.bbox_cells(qmin_x, qmin_y, qmax_x, qmax_y)
            .flat_map(|cell_idx| self.cells[cell_idx].iter().copied())
            .filter(|&i| {
                let (x, y) = self.coords[i];
                x >= qmin_x && x <= qmax_x && y >= qmin_y && y <= qmax_y
            })
            .collect()
    }

    fn contains(&self, point: &Point<f64>) -> Vec<usize> {
        // Reuse range with a zero-area bbox.
        let eps = f64::EPSILON * 1000.0;
        let bbox = Rect::new(
            coord! { x: point.x() - eps, y: point.y() - eps },
            coord! { x: point.x() + eps, y: point.y() + eps },
        );
        self.range(&bbox)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::five_point_grid;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn nearest_returns_single_closest() {
        let geoms = five_point_grid();
        let idx = UniformGrid::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(1.2, 0.1), 1), vec![1]);
    }

    #[test]
    fn nearest_k_two_returns_correct_pair() {
        let geoms = five_point_grid();
        let idx = UniformGrid::build(&geoms);
        assert_eq!(sorted(idx.nearest(&Point::new(1.2, 0.1), 2)), vec![1, 2]);
    }

    #[test]
    fn nearest_k_larger_than_n_returns_all() {
        let geoms = five_point_grid();
        let idx = UniformGrid::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(0.0, 0.0), 100).len(), 5);
    }

    #[test]
    fn range_returns_correct_points() {
        let geoms = five_point_grid();
        let idx = UniformGrid::build(&geoms);
        let bbox = Rect::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 1.5, y: 0.5 });
        assert_eq!(sorted(idx.range(&bbox)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let geoms = five_point_grid();
        let idx = UniformGrid::build(&geoms);
        let bbox = Rect::new(coord! { x: 5.0, y: 5.0 }, coord! { x: 10.0, y: 10.0 });
        assert!(idx.range(&bbox).is_empty());
    }
}
