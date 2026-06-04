use std::sync::Arc;

use rayon::prelude::*;

use crate::index::SpatialIndex;

/// Uniform grid index with CSR (compressed sparse row) cell storage.
/// Best for large datasets with uniform spatial distribution.
///
/// xs/ys are shared Arcs from the Engine — no coordinate data is copied.
/// The CSR arrays (cell_offsets, indices) are new allocations derived from the data.
pub struct UniformGrid {
    /// cell_offsets[i]..cell_offsets[i+1] is the slice of indices in cell i
    cell_offsets: Vec<u32>,
    indices: Vec<u32>,
    xs: Arc<[f64]>,
    ys: Arc<[f64]>,
    min_x: f64,
    min_y: f64,
    cell_w: f64,
    cell_h: f64,
    grid_w: usize,
    grid_h: usize,
}

impl UniformGrid {
    fn bbox_cell_range(
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
    fn build(xs: Arc<[f64]>, ys: Arc<[f64]>) -> Self {
        let n = xs.len();

        let (min_x, min_y, max_x, max_y) = xs.iter().zip(ys.iter()).fold(
            (
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ),
            |(mn_x, mn_y, mx_x, mx_y), (&x, &y)| {
                (mn_x.min(x), mn_y.min(y), mx_x.max(x), mx_y.max(y))
            },
        );

        let grid_dim = (n as f64).sqrt().ceil().max(1.0) as usize;
        let grid_w = grid_dim;
        let grid_h = grid_dim;
        let cell_w = if max_x > min_x {
            (max_x - min_x) / grid_w as f64
        } else {
            1.0
        };
        let cell_h = if max_y > min_y {
            (max_y - min_y) / grid_h as f64
        } else {
            1.0
        };

        // Two-pass CSR build.
        // Pass 1 (parallel): compute cell index for each point.
        let cell_for_point: Vec<u32> = xs
            .par_iter()
            .zip(ys.par_iter())
            .map(|(&x, &y)| {
                let col = ((x - min_x) / cell_w)
                    .floor()
                    .clamp(0.0, grid_w as f64 - 1.0) as u32;
                let row = ((y - min_y) / cell_h)
                    .floor()
                    .clamp(0.0, grid_h as f64 - 1.0) as u32;
                row * grid_w as u32 + col
            })
            .collect();

        let num_cells = grid_w * grid_h;
        let mut counts = vec![0u32; num_cells];
        for &c in &cell_for_point {
            counts[c as usize] += 1;
        }

        // Prefix sum → cell_offsets (length num_cells + 1).
        let mut cell_offsets = Vec::with_capacity(num_cells + 1);
        cell_offsets.push(0u32);
        for &cnt in &counts {
            cell_offsets.push(cell_offsets.last().unwrap() + cnt);
        }

        // Pass 2: scatter point indices into their cell slots.
        let mut indices = vec![0u32; n];
        let mut write_pos = cell_offsets[..num_cells].to_vec();
        for (i, &c) in cell_for_point.iter().enumerate() {
            let pos = write_pos[c as usize] as usize;
            indices[pos] = i as u32;
            write_pos[c as usize] += 1;
        }

        UniformGrid {
            cell_offsets,
            indices,
            xs,
            ys,
            min_x,
            min_y,
            cell_w,
            cell_h,
            grid_w,
            grid_h,
        }
    }

    fn nearest(&self, qx: f64, qy: f64, k: usize) -> Vec<usize> {
        if self.xs.is_empty() {
            return vec![];
        }
        let k = k.min(self.xs.len());

        let start_col = ((qx - self.min_x) / self.cell_w)
            .floor()
            .clamp(0.0, self.grid_w as f64 - 1.0) as isize;
        let start_row = ((qy - self.min_y) / self.cell_h)
            .floor()
            .clamp(0.0, self.grid_h as f64 - 1.0) as isize;
        let max_ring = self.grid_w.max(self.grid_h) as isize + 1;
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for ring in 0..=max_ring {
            let r = ring;
            let col_lo = (start_col - r).max(0) as usize;
            let col_hi = (start_col + r).min(self.grid_w as isize - 1) as usize;
            let row_lo = (start_row - r).max(0) as usize;
            let row_hi = (start_row + r).min(self.grid_h as isize - 1) as usize;
            let uc_col_lo = start_col - r;
            let uc_col_hi = start_col + r;
            let uc_row_lo = start_row - r;
            let uc_row_hi = start_row + r;

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
                    let cell = row * self.grid_w + col;
                    let start = self.cell_offsets[cell] as usize;
                    let end = self.cell_offsets[cell + 1] as usize;
                    for &idx in &self.indices[start..end] {
                        let x = self.xs[idx as usize];
                        let y = self.ys[idx as usize];
                        candidates.push((idx as usize, (x - qx).powi(2) + (y - qy).powi(2)));
                    }
                }
            }

            if candidates.len() >= k {
                let col_lo_i = (start_col - r).max(0);
                let col_hi_i = (start_col + r).min(self.grid_w as isize - 1);
                let row_lo_i = (start_row - r).max(0);
                let row_hi_i = (start_row + r).min(self.grid_h as isize - 1);

                let full_cover = col_lo_i == 0
                    && col_hi_i == self.grid_w as isize - 1
                    && row_lo_i == 0
                    && row_hi_i == self.grid_h as isize - 1;
                if full_cover {
                    break;
                }

                let vx_lo = self.min_x + col_lo_i as f64 * self.cell_w;
                let vx_hi = self.min_x + (col_hi_i + 1) as f64 * self.cell_w;
                let vy_lo = self.min_y + row_lo_i as f64 * self.cell_h;
                let vy_hi = self.min_y + (row_hi_i + 1) as f64 * self.cell_h;

                let mut min_unvisited_sq = f64::INFINITY;
                if col_lo_i > 0 {
                    let d = (qx - vx_lo).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }
                if col_hi_i < self.grid_w as isize - 1 {
                    let d = (vx_hi - qx).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }
                if row_lo_i > 0 {
                    let d = (qy - vy_lo).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }
                if row_hi_i < self.grid_h as isize - 1 {
                    let d = (vy_hi - qy).max(0.0);
                    min_unvisited_sq = min_unvisited_sq.min(d * d);
                }

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

    fn range(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        if self.grid_w == 0 || self.grid_h == 0 {
            return vec![];
        }
        self.bbox_cell_range(min_x, min_y, max_x, max_y)
            .flat_map(|cell_idx| {
                let start = self.cell_offsets[cell_idx] as usize;
                let end = self.cell_offsets[cell_idx + 1] as usize;
                self.indices[start..end].iter().copied()
            })
            .filter(|&idx| {
                let x = self.xs[idx as usize];
                let y = self.ys[idx as usize];
                x >= min_x && x <= max_x && y >= min_y && y <= max_y
            })
            .map(|i| i as usize)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::brute::five_point_grid;

    fn build(xs: Vec<f64>, ys: Vec<f64>) -> UniformGrid {
        UniformGrid::build(xs.into(), ys.into())
    }

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn nearest_returns_single_closest() {
        let (xs, ys) = five_point_grid();
        assert_eq!(build(xs, ys).nearest(1.2, 0.1, 1), vec![1]);
    }

    #[test]
    fn nearest_k_two_returns_correct_pair() {
        let (xs, ys) = five_point_grid();
        assert_eq!(sorted(build(xs, ys).nearest(1.2, 0.1, 2)), vec![1, 2]);
    }

    #[test]
    fn nearest_k_larger_than_n_returns_all() {
        let (xs, ys) = five_point_grid();
        assert_eq!(build(xs, ys).nearest(0.0, 0.0, 100).len(), 5);
    }

    #[test]
    fn range_returns_correct_points() {
        let (xs, ys) = five_point_grid();
        assert_eq!(sorted(build(xs, ys).range(0.0, 0.0, 1.5, 0.5)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let (xs, ys) = five_point_grid();
        assert!(build(xs, ys).range(5.0, 5.0, 10.0, 10.0).is_empty());
    }
}
