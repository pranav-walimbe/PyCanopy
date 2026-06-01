use geo::{Geometry, Point, Rect};
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex};

use crate::index::{geom_bbox, SpatialIndex};

/// Packed immutable R-tree backed by geo-index with Hilbert sort
pub struct PackedRTree {
    tree: RTree<f64>,
}

impl SpatialIndex for PackedRTree {
    fn build(geometries: &[Geometry<f64>]) -> Self {
        let n = geometries.len() as u32;
        let mut builder = RTreeBuilder::<f64>::new(n);

        for geom in geometries {
            if let Some((min_x, min_y, max_x, max_y)) = geom_bbox(geom) {
                builder.add(min_x, min_y, max_x, max_y);
            } else {
                // Degenerate box for empty geometries; keeps index aligned.
                builder.add(0.0, 0.0, 0.0, 0.0);
            }
        }

        PackedRTree {
            tree: builder.finish::<HilbertSort>(),
        }
    }

    fn nearest(&self, point: &Point<f64>, k: usize) -> Vec<usize> {
        self.tree
            .neighbors(point.x(), point.y(), Some(k), None)
            .iter()
            .map(|&i| i as usize)
            .collect()
    }

    fn range(&self, bbox: &Rect<f64>) -> Vec<usize> {
        self.tree
            .search(bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y)
            .iter()
            .map(|&i| i as usize)
            .collect()
    }

    fn contains(&self, point: &Point<f64>) -> Vec<usize> {
        // MBR-level contains: returns geometries whose MBR contains the point.
        // Caller performs exact polygon-contains check.
        self.tree
            .search(point.x(), point.y(), point.x(), point.y())
            .iter()
            .map(|&i| i as usize)
            .collect()
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
        let idx = PackedRTree::build(&geoms);
        assert_eq!(idx.nearest(&Point::new(1.2, 0.1), 1), vec![1]);
    }

    #[test]
    fn nearest_k_two_returns_correct_pair() {
        let geoms = five_point_grid();
        let idx = PackedRTree::build(&geoms);
        assert_eq!(sorted(idx.nearest(&Point::new(1.2, 0.1), 2)), vec![1, 2]);
    }

    #[test]
    fn range_returns_correct_points() {
        let geoms = five_point_grid();
        let idx = PackedRTree::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 0.0, y: 0.0 },
            geo::coord! { x: 1.5, y: 0.5 },
        );
        assert_eq!(sorted(idx.range(&bbox)), vec![0, 1]);
    }

    #[test]
    fn range_empty_bbox_returns_empty() {
        let geoms = five_point_grid();
        let idx = PackedRTree::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 5.0, y: 5.0 },
            geo::coord! { x: 10.0, y: 10.0 },
        );
        assert!(idx.range(&bbox).is_empty());
    }

    #[test]
    fn range_single_result() {
        let geoms = five_point_grid();
        let idx = PackedRTree::build(&geoms);
        let bbox = Rect::new(
            geo::coord! { x: 0.5, y: 0.5 },
            geo::coord! { x: 1.5, y: 1.5 },
        );
        assert_eq!(sorted(idx.range(&bbox)), vec![4]);
    }
}
