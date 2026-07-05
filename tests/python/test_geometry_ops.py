"""
Tests for the polygon geometry operations added for SpatialBench coverage.

Covers point-to-polygon distance joins, polygon self-intersection + IoU helpers,
the single-polygon distance filter, and convex hull area.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import shapely
from shapely.geometry import box as shapely_box

from pycanopy import Engine, SpatialFrame, wkb_point_distance


def _poly_engine(boxes) -> Engine:
    return Engine.from_polygons([shapely_box(*b) for b in boxes])


def test_polygon_areas():
    # A 1x1 square and a 2x2 square.
    eng = _poly_engine([(0, 0, 1, 1), (0, 0, 2, 2)])
    areas = eng.polygon_areas()
    assert np.allclose(np.sort(areas), [1.0, 4.0])


def test_within_distance_to_polygons():
    # Square at (0,0)-(1,1). Point (2, 0.5) is distance 1.0 from its right edge.
    eng = _poly_engine([(0, 0, 1, 1)])
    qx = np.array([2.0, 0.5], dtype=np.float64)  # outside (d=1.0), inside (d=0)
    qy = np.array([0.5, 0.5], dtype=np.float64)

    near = eng.batch_within_distance_to_polygons(qx, qy, 1.5).reshape(-1, 2)
    pairs = {(int(q), int(e)) for q, e in near}
    assert (0, 0) in pairs  # outside point matches at d<=1.5
    assert (1, 0) in pairs  # inside point matches

    tight = eng.batch_within_distance_to_polygons(qx, qy, 0.5).reshape(-1, 2)
    tight_pairs = {(int(q), int(e)) for q, e in tight}
    assert (0, 0) not in tight_pairs  # 1.0 > 0.5
    assert (1, 0) in tight_pairs


def test_knn_to_polygons():
    # Three squares spread along x; query near the middle one.
    eng = _poly_engine([(0, 0, 1, 1), (10, 0, 11, 1), (20, 0, 21, 1)])
    qx = np.array([10.5], dtype=np.float64)
    qy = np.array([0.5], dtype=np.float64)
    idx, dist = eng.batch_knn_to_polygons(qx, qy, 2)
    idx = idx.reshape(1, 2)[0]
    dist = dist.reshape(1, 2)[0]
    assert int(idx[0]) == 1  # nearest is the middle square (contains the point)
    assert abs(float(dist[0])) < 1e-9  # inside -> distance 0
    assert int(idx[1]) in (0, 2)  # next nearest is one of the flanking squares


def test_polygon_self_intersection_and_iou():
    # Square A (0,0)-(2,2) and square B (1,1)-(3,3) overlap in a 1x1 region.
    eng = _poly_engine([(0, 0, 2, 2), (1, 1, 3, 3)])
    pairs = eng.polygon_intersects_self_join().reshape(-1, 2)
    assert pairs.shape[0] == 1
    i, j = int(pairs[0][0]), int(pairs[0][1])
    assert (i, j) == (0, 1)

    overlap = eng.polygon_pairs_intersection_area(
        np.array([i], dtype=np.uint64), np.array([j], dtype=np.uint64)
    )
    assert abs(float(overlap[0]) - 1.0) < 1e-9

    areas = eng.polygon_areas()
    union = areas[i] + areas[j] - overlap[0]
    iou = overlap[0] / union
    assert abs(iou - (1.0 / 7.0)) < 1e-9  # 1 / (4 + 4 - 1)


def test_disjoint_polygons_have_no_intersection_pairs():
    eng = _poly_engine([(0, 0, 1, 1), (10, 10, 11, 11)])
    pairs = eng.polygon_intersects_self_join().reshape(-1, 2)
    assert pairs.shape[0] == 0


def test_points_within_distance_of_polygon():
    # Point dataset; query polygon is the unit square at the origin.
    xs = np.array([0.5, 2.0, 5.0], dtype=np.float64)  # inside, d=1.0, far
    ys = np.array([0.5, 0.5, 5.0], dtype=np.float64)
    eng = Engine.from_coords(xs, ys)
    poly = shapely_box(0, 0, 1, 1)

    hit = set(eng.points_within_distance_of_polygon(poly, 1.5).tolist())
    assert hit == {0, 1}  # inside + the d=1.0 point

    tight = set(eng.points_within_distance_of_polygon(poly, 0.5).tolist())
    assert tight == {0}


def test_convex_hull_area():
    # Corners of a 2x2 square plus an interior point: hull area is 4.
    xs = np.array([0.0, 2.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ys = np.array([0.0, 0.0, 2.0, 2.0, 1.0], dtype=np.float64)
    assert abs(Engine.convex_hull_area(xs, ys) - 4.0) < 1e-9


def test_convex_hull_area_degenerate():
    assert Engine.convex_hull_area([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_group_convex_hull_areas():
    # Two groups: a 2x2 square (area 4) and a degenerate group (<3 points, area 0)
    xs = pl.Series([[0.0, 2.0, 2.0, 0.0], [0.0, 1.0]])
    ys = pl.Series([[0.0, 0.0, 2.0, 2.0], [0.0, 1.0]])
    areas = Engine.group_convex_hull_areas(xs, ys)
    assert abs(areas[0] - 4.0) < 1e-9
    assert areas[1] == 0.0


def test_group_convex_hull_areas_matches_scalar():
    # Batch result must match calling Engine.convex_hull_area per group
    rng = np.random.default_rng(42)
    groups_x = [rng.uniform(0, 10, size=rng.integers(3, 20)).tolist() for _ in range(50)]
    groups_y = [rng.uniform(0, 10, size=len(g)).tolist() for g in groups_x]
    xs = pl.Series(groups_x)
    ys = pl.Series(groups_y)
    batch = Engine.group_convex_hull_areas(xs, ys)
    for i, (gx, gy) in enumerate(zip(groups_x, groups_y)):
        expected = Engine.convex_hull_area(np.array(gx), np.array(gy))
        assert abs(batch[i] - expected) < 1e-9, f"group {i}: batch={batch[i]} scalar={expected}"


def test_wkb_point_distance():
    # 3-4-5 right triangle: distance should be exactly 5.0
    pts_a = pl.Series(shapely.to_wkb([shapely.Point(0.0, 0.0), shapely.Point(0.0, 0.0)]))
    pts_b = pl.Series(shapely.to_wkb([shapely.Point(3.0, 4.0), shapely.Point(0.0, 0.0)]))
    dists = wkb_point_distance(pts_a, pts_b)
    assert abs(dists[0] - 5.0) < 1e-9
    assert dists[1] == 0.0


def test_within_distance_to_polygons_rejects_point_engine():
    eng = Engine.from_coords(np.array([0.0]), np.array([0.0]))
    with pytest.raises(Exception):
        eng.batch_within_distance_to_polygons(np.array([0.0]), np.array([0.0]), 1.0)


# Declarative Python API (SpatialFrame / SpatialLazyFrame)


def _poly_frame(boxes):
    polys = [shapely_box(*b) for b in boxes]
    df = pl.DataFrame({"pid": list(range(len(polys)))}).with_columns(
        pl.Series("_geom", polys, dtype=pl.Object)
    )
    return SpatialFrame.from_polygons(df, geometry_col="_geom")


def test_lazy_polygon_within_distance_join():
    sf = _poly_frame([(0, 0, 1, 1), (10, 0, 11, 1)])
    query = pl.DataFrame({"qx": [2.0], "qy": [0.5], "qid": [99]})
    out = sf.lazy().polygon_within_distance_join(query, "qx", "qy", distance=1.5).collect()
    # Point (2, 0.5) is 1.0 from the first square only.
    assert out["pid"].to_list() == [0]
    assert out["qid"].to_list() == [99]


def test_lazy_polygon_knn_join():
    sf = _poly_frame([(0, 0, 1, 1), (10, 0, 11, 1), (20, 0, 21, 1)])
    query = pl.DataFrame({"qx": [10.5], "qy": [0.5]})
    out = sf.lazy().polygon_knn_join(query, "qx", "qy", k=2).collect()
    assert len(out) == 2
    assert "distance_to_polygon" in out.columns
    # Nearest is the containing square (pid 1, distance 0).
    first = out.sort("distance_to_polygon").row(0, named=True)
    assert first["pid"] == 1
    assert abs(first["distance_to_polygon"]) < 1e-9


def test_frame_intersects_pairs_iou():
    sf = _poly_frame([(0, 0, 2, 2), (1, 1, 3, 3)])
    pairs = sf.intersects_pairs()
    assert len(pairs) == 1
    row = pairs.row(0, named=True)
    assert (row["left"], row["right"]) == (0, 1)
    assert abs(row["overlap_area"] - 1.0) < 1e-9
    assert abs(row["iou"] - (1.0 / 7.0)) < 1e-9


def test_frame_intersects_pairs_key_col():
    df = pl.DataFrame(
        {"id": [10, 5], "geom": [shapely.box(0, 0, 2, 2).wkb, shapely.box(1, 1, 3, 3).wkb]}
    )
    sf = SpatialFrame.from_wkb_polygons(df, "geom")
    pairs = sf.intersects_pairs(key_col="id")
    assert len(pairs) == 1
    row = pairs.row(0, named=True)
    assert row["id_1"] == 5 and row["id_2"] == 10
    assert abs(row["iou"] - (1.0 / 7.0)) < 1e-9


def test_frame_polygon_areas_column():
    sf = _poly_frame([(0, 0, 1, 1), (0, 0, 2, 2)])
    out = sf.polygon_areas()
    assert "area" in out.columns
    assert np.allclose(sorted(out["area"].to_list()), [1.0, 4.0])


def test_frame_points_within_distance_of_polygon():
    df = pl.DataFrame({"x": [0.5, 2.0, 5.0], "y": [0.5, 0.5, 5.0], "label": ["a", "b", "c"]})
    sf = SpatialFrame(df, "x", "y")
    hit = sf.points_within_distance_of_polygon(shapely_box(0, 0, 1, 1), 1.5)
    assert set(hit["label"].to_list()) == {"a", "b"}
