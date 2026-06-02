use geo::Rect;

pub const HISTOGRAM_RESOLUTION: usize = 32;

/// Fixed 32x32 spatial histogram of geometry counts built at load time
#[derive(Debug, Clone)]
pub struct SpatialHistogram {
    pub counts: Vec<u32>,
    pub min_x: f64,
    pub min_y: f64,
    pub cell_w: f64,
    pub cell_h: f64,
}

impl SpatialHistogram {
    /// Fraction of N expected to fall within bbox, based on histogram cell counts
    pub fn selectivity(&self, bbox: &Rect<f64>, n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let (col_min, row_min) = self.cell_for(bbox.min().x, bbox.min().y);
        let (col_max, row_max) = self.cell_for(bbox.max().x, bbox.max().y);
        let mut sum = 0u64;
        for row in row_min..=row_max {
            for col in col_min..=col_max {
                sum += self.counts[row * HISTOGRAM_RESOLUTION + col] as u64;
            }
        }
        (sum as f64 / n as f64).min(1.0)
    }

    /// True if any cell overlapping bbox has a non-zero count
    pub(crate) fn has_any_in_bbox(&self, bbox: &Rect<f64>) -> bool {
        let (col_min, row_min) = self.cell_for(bbox.min().x, bbox.min().y);
        let (col_max, row_max) = self.cell_for(bbox.max().x, bbox.max().y);
        for row in row_min..=row_max {
            for col in col_min..=col_max {
                if self.counts[row * HISTOGRAM_RESOLUTION + col] > 0 {
                    return true;
                }
            }
        }
        false
    }

    /// True if the cell containing (x, y) has a non-zero count
    pub(crate) fn has_any_at(&self, x: f64, y: f64) -> bool {
        let (col, row) = self.cell_for(x, y);
        self.counts[row * HISTOGRAM_RESOLUTION + col] > 0
    }

    /// Local point density (count / cell_area) at (x, y), or None if the cell is empty
    pub(crate) fn local_density(&self, x: f64, y: f64) -> Option<f64> {
        let (col, row) = self.cell_for(x, y);
        let count = self.counts[row * HISTOGRAM_RESOLUTION + col];
        if count == 0 {
            return None;
        }
        let cell_area = self.cell_w * self.cell_h;
        if cell_area > 0.0 {
            Some(count as f64 / cell_area)
        } else {
            None
        }
    }

    fn cell_for(&self, x: f64, y: f64) -> (usize, usize) {
        let col = ((x - self.min_x) / self.cell_w)
            .floor()
            .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64) as usize;
        let row = ((y - self.min_y) / self.cell_h)
            .floor()
            .clamp(0.0, (HISTOGRAM_RESOLUTION - 1) as f64) as usize;
        (col, row)
    }
}

/// Dataset statistics used by the query planner to select a spatial index
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub n: usize,
    pub kind: GeometryKind,
    pub extent: Option<Rect<f64>>,
    pub distribution: Distribution,
    /// N / extent_area
    pub mean_density: f64,
    pub histogram: Option<SpatialHistogram>,
}

/// Dominant geometry type in the dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryKind {
    Point,
    LineString,
    Polygon,
    /// More than one geometry type present
    Mixed,
    /// Dataset is empty
    Empty,
}

/// Spatial distribution of point geometries estimated via grid CV test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    Uniform,
    Clustered,
    /// Not enough data to classify or geometry kind is not Point
    Unknown,
}

impl DatasetStats {
    pub fn extent_area(&self) -> f64 {
        self.extent
            .map(|r| {
                let w = r.max().x - r.min().x;
                let h = r.max().y - r.min().y;
                (w * h).max(0.0)
            })
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{coord, Rect};

    fn rect(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Rect<f64> {
        Rect::new(coord! { x: min_x, y: min_y }, coord! { x: max_x, y: max_y })
    }

    #[test]
    fn extent_area_basic() {
        let stats = DatasetStats {
            n: 10,
            kind: GeometryKind::Point,
            extent: Some(rect(0.0, 0.0, 10.0, 10.0)),
            distribution: Distribution::Uniform,
            mean_density: 0.1,
            histogram: None,
        };
        assert!((stats.extent_area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn extent_area_none_returns_zero() {
        let stats = DatasetStats {
            n: 0,
            kind: GeometryKind::Empty,
            extent: None,
            distribution: Distribution::Unknown,
            mean_density: 0.0,
            histogram: None,
        };
        assert_eq!(stats.extent_area(), 0.0);
    }

    fn uniform_histogram(resolution: usize, count_per_cell: u32) -> SpatialHistogram {
        SpatialHistogram {
            counts: vec![count_per_cell; resolution * resolution],
            min_x: 0.0,
            min_y: 0.0,
            cell_w: 1.0 / resolution as f64,
            cell_h: 1.0 / resolution as f64,
        }
    }

    #[test]
    fn histogram_selectivity_full_bbox_returns_one() {
        // 32x32 cells, 1 point each = 1024 total; full bbox should return 1.0
        let hist = uniform_histogram(HISTOGRAM_RESOLUTION, 1);
        let bbox = rect(0.0, 0.0, 1.0, 1.0);
        assert!((hist.selectivity(&bbox, 1024) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn histogram_selectivity_quarter_bbox() {
        // 1024 points uniform; bbox covering bottom-left quarter should return ~0.25.
        // Tolerance is 0.05 because exact cell-boundary alignment causes floor() to
        // include the boundary cell, giving a small over-count (17x17 vs 16x16 cells).
        let hist = uniform_histogram(HISTOGRAM_RESOLUTION, 1);
        let bbox = rect(0.0, 0.0, 0.5, 0.5);
        let sel = hist.selectivity(&bbox, 1024);
        assert!((sel - 0.25).abs() < 0.05);
    }

    #[test]
    fn histogram_selectivity_empty_region_returns_zero() {
        // All counts in bottom-left quarter; top-right query should return 0
        let mut counts = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
        counts[0] = 100; // only cell (col=0, row=0) has data
        let hist = SpatialHistogram {
            counts,
            min_x: 0.0,
            min_y: 0.0,
            cell_w: 1.0 / HISTOGRAM_RESOLUTION as f64,
            cell_h: 1.0 / HISTOGRAM_RESOLUTION as f64,
        };
        let bbox = rect(0.9, 0.9, 1.0, 1.0); // top-right corner, no data
        assert_eq!(hist.selectivity(&bbox, 100), 0.0);
    }

    #[test]
    fn histogram_selectivity_skewed_data_outperforms_area_ratio() {
        // 100 points all in the bottom-left cell; query covers that cell.
        // Area ratio would give 1/1024 ≈ 0.001; histogram should give 1.0.
        let mut counts = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
        counts[0] = 100;
        let hist = SpatialHistogram {
            counts,
            min_x: 0.0,
            min_y: 0.0,
            cell_w: 1.0 / HISTOGRAM_RESOLUTION as f64,
            cell_h: 1.0 / HISTOGRAM_RESOLUTION as f64,
        };
        let cell_size = 1.0 / HISTOGRAM_RESOLUTION as f64;
        let bbox = rect(0.0, 0.0, cell_size, cell_size);
        assert!((hist.selectivity(&bbox, 100) - 1.0).abs() < 1e-10);
    }

    fn sparse_histogram() -> SpatialHistogram {
        let mut counts = vec![0u32; HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION];
        counts[0] = 50; // only bottom-left cell has data
        SpatialHistogram {
            counts,
            min_x: 0.0,
            min_y: 0.0,
            cell_w: 1.0 / HISTOGRAM_RESOLUTION as f64,
            cell_h: 1.0 / HISTOGRAM_RESOLUTION as f64,
        }
    }

    #[test]
    fn has_any_in_bbox_returns_true_over_populated_cell() {
        let hist = sparse_histogram();
        let cell = 1.0 / HISTOGRAM_RESOLUTION as f64;
        assert!(hist.has_any_in_bbox(&rect(0.0, 0.0, cell, cell)));
    }

    #[test]
    fn has_any_in_bbox_returns_false_over_empty_region() {
        let hist = sparse_histogram();
        assert!(!hist.has_any_in_bbox(&rect(0.5, 0.5, 1.0, 1.0)));
    }

    #[test]
    fn has_any_at_returns_true_in_populated_cell() {
        let hist = sparse_histogram();
        assert!(hist.has_any_at(0.0, 0.0));
    }

    #[test]
    fn has_any_at_returns_false_in_empty_cell() {
        let hist = sparse_histogram();
        assert!(!hist.has_any_at(0.9, 0.9));
    }

    #[test]
    fn local_density_returns_none_for_empty_cell() {
        let hist = sparse_histogram();
        assert!(hist.local_density(0.9, 0.9).is_none());
    }

    #[test]
    fn local_density_returns_count_over_cell_area() {
        let hist = sparse_histogram();
        let cell_area = (1.0 / HISTOGRAM_RESOLUTION as f64).powi(2);
        let expected = 50.0 / cell_area;
        let got = hist.local_density(0.0, 0.0).expect("should have density");
        assert!((got - expected).abs() < 1e-6);
    }

    #[test]
    fn extent_area_non_square() {
        let stats = DatasetStats {
            n: 5,
            kind: GeometryKind::Point,
            extent: Some(rect(0.0, 0.0, 4.0, 10.0)),
            distribution: Distribution::Uniform,
            mean_density: 0.125,
            histogram: None,
        };
        assert!((stats.extent_area() - 40.0).abs() < 1e-10);
    }
}
