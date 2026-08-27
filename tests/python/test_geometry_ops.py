"""
Tests for the polygon geometry operations added for SpatialBench coverage.

Covers point-to-polygon distance joins, polygon self-intersection + IoU helpers,
the single-polygon distance filter, and convex hull area.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import shapely
from shapely.geometry import box as shapely_box

from pycanopy import (
    Engine,
    SpatialFrame,
    distance_to_point,
    point_distance,
    wkb_point_distance,
)


def _haversine_ref(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    # reference great-circle distance in meters for the geographic distance tests
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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
    assert near.dtype == np.uint32
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
    assert idx.dtype == np.uint32
    idx = idx.reshape(1, 2)[0]
    dist = dist.reshape(1, 2)[0]
    assert int(idx[0]) == 1  # nearest is the middle square (contains the point)
    assert abs(float(dist[0])) < 1e-9  # inside -> distance 0
    assert int(idx[1]) in (0, 2)  # next nearest is one of the flanking squares

    q_idx, t_idx, sorted_dist = eng.batch_knn_to_polygons_sorted(qx, qy, 2)
    assert q_idx.dtype == t_idx.dtype == np.uint32
    assert sorted_dist.dtype == np.float64


def test_point_distance_join_indices_are_uint32():
    eng = Engine.from_coords(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    pairs = eng.batch_within_distance(np.array([0.0]), np.array([0.0]), 0.5)
    assert pairs.dtype == np.uint32


def _annulus(cx: float, cy: float, outer: float, hole: float):
    # Square annulus: its MBR covers the centre while its boundary stays `hole` away from it
    ring = [
        (cx - outer, cy - outer),
        (cx + outer, cy - outer),
        (cx + outer, cy + outer),
        (cx - outer, cy + outer),
    ]
    inner = [
        (cx - hole, cy - hole),
        (cx - hole, cy + hole),
        (cx + hole, cy + hole),
        (cx + hole, cy - hole),
    ]
    return shapely.Polygon(ring, [inner])


def test_knn_to_polygons_is_exact_under_covering_mbrs():
    # Four annuli whose MBRs all contain the query outrank the one genuinely nearest square by
    # MBR distance. Oversampling a fixed multiple of k dropped the square and returned 50.0.
    polys = [_annulus(i * 0.1, 0.0, 100.0, 50.0) for i in range(4)]
    polys.append(shapely_box(1, 0, 2, 1))
    eng = Engine.from_polygons(polys)
    idx, dist = eng.batch_knn_to_polygons(
        np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64), 1
    )
    assert int(idx[0]) == 4
    assert abs(float(dist[0]) - 1.0) < 1e-9


def test_knn_to_polygons_matches_brute_force_on_annuli():
    rng = np.random.default_rng(7)
    outer = rng.uniform(5.0, 50.0, 40)
    polys = [
        _annulus(cx, cy, o, o * f)
        for cx, cy, o, f in zip(
            rng.uniform(0, 100, 40), rng.uniform(0, 100, 40), outer, rng.uniform(0.2, 0.9, 40)
        )
    ]
    eng = Engine.from_polygons(polys)
    qx = rng.uniform(0, 100, 50)
    qy = rng.uniform(0, 100, 50)
    k = 3
    _, dist = eng.batch_knn_to_polygons(qx, qy, k)
    dist = dist.reshape(-1, k)

    points = shapely.points(qx, qy)
    for q in range(len(qx)):
        want = np.sort([p.distance(points[q]) for p in polys])[:k]
        assert np.allclose(dist[q], want, atol=1e-9), f"query {q}: {dist[q]} vs {want}"


def test_polygon_self_intersection_and_iou():
    # Square A (0,0)-(2,2) and square B (1,1)-(3,3) overlap in a 1x1 region.
    eng = _poly_engine([(0, 0, 2, 2), (1, 1, 3, 3)])
    pairs = eng.polygon_intersects_self_join().reshape(-1, 2)
    assert pairs.dtype == np.uint32
    assert pairs.shape[0] == 1
    i, j = int(pairs[0][0]), int(pairs[0][1])
    assert (i, j) == (0, 1)

    overlap = eng.polygon_pairs_intersection_area(
        np.array([i], dtype=np.uint32), np.array([j], dtype=np.uint32)
    )
    assert abs(float(overlap[0]) - 1.0) < 1e-9

    areas = eng.polygon_areas()
    union = areas[i] + areas[j] - overlap[0]
    iou = overlap[0] / union
    assert abs(iou - (1.0 / 7.0)) < 1e-9  # 1 / (4 + 4 - 1)


def test_polygon_pair_area_rejects_out_of_range_indices():
    eng = _poly_engine([(0, 0, 1, 1)])

    with pytest.raises(ValueError, match="less than the polygon count"):
        eng.polygon_pairs_intersection_area(
            np.array([0], dtype=np.uint32),
            np.array([1], dtype=np.uint32),
        )


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


def test_group_convex_hull_areas_rejects_mismatched_coordinates():
    xs = pl.Series([[0.0, 1.0, 0.0]])
    ys = pl.Series([[0.0]])

    with pytest.raises(ValueError, match="xs and ys must have the same length"):
        Engine.group_convex_hull_areas(xs, ys)


def test_wkb_point_distance():
    # 3-4-5 right triangle: distance should be exactly 5.0
    pts_a = pl.Series(shapely.to_wkb([shapely.Point(0.0, 0.0), shapely.Point(0.0, 0.0)]))
    pts_b = pl.Series(shapely.to_wkb([shapely.Point(3.0, 4.0), shapely.Point(0.0, 0.0)]))
    dists = wkb_point_distance(pts_a, pts_b)
    assert abs(dists[0] - 5.0) < 1e-9
    assert dists[1] == 0.0


def test_point_distance_planar_is_euclidean():
    # planar measures Euclidean in coordinate units (3-4-5 triangle -> 5.0)
    x1 = np.array([0.0, 0.0])
    y1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 0.0])
    y2 = np.array([4.0, 0.0])
    dists = point_distance(x1, y1, x2, y2, "planar")
    assert abs(dists[0] - 5.0) < 1e-9
    assert dists[1] == 0.0


def test_point_distance_geographic_matches_haversine_oracle():
    # JFK -> LAX great-circle distance, checked against an independent haversine reference
    lon1, lat1 = -73.7781, 40.6413
    lon2, lat2 = -118.4085, 33.9416
    dists = point_distance(
        np.array([lon1]), np.array([lat1]), np.array([lon2]), np.array([lat2]), "geographic"
    )
    assert abs(dists[0] - _haversine_ref(lon1, lat1, lon2, lat2)) < 1e-6


def test_distance_to_point_matches_pairwise():
    # column-to-center agrees with the pairwise form given a broadcast center
    lons = np.array([-118.4085, -87.9073])
    lats = np.array([33.9416, 41.9742])
    cx, cy = -73.7781, 40.6413
    to_pt = distance_to_point(lons, lats, cx, cy, "geographic")
    pair = point_distance(lons, lats, np.full_like(lons, cx), np.full_like(lats, cy), "geographic")
    assert np.allclose(to_pt, pair, atol=1e-6)


def test_distance_to_point_planar_is_euclidean():
    dists = distance_to_point(np.array([3.0]), np.array([4.0]), 0.0, 0.0, "planar")
    assert abs(dists[0] - 5.0) < 1e-9


def test_point_distance_accepts_polars_columns():
    # a polars Float64 column flows through the zero-copy path without a dtype error
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    dists = point_distance(df["x"], df["y"], df["x"], df["y"], "planar")
    assert np.allclose(dists, [0.0, 0.0])


@pytest.mark.parametrize("coordinate_system", ["planar", "geographic"])
def test_point_distance_rejects_mismatched_coordinate_lengths(coordinate_system):
    with pytest.raises(ValueError, match="all coordinate arrays must have the same length"):
        point_distance(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0]),
            np.array([0.0]),
            coordinate_system,
        )


@pytest.mark.parametrize("coordinate_system", ["planar", "geographic"])
def test_distance_to_point_rejects_mismatched_coordinate_lengths(coordinate_system):
    with pytest.raises(ValueError, match="same length"):
        distance_to_point(
            np.array([0.0, 1.0]),
            np.array([0.0]),
            0.0,
            0.0,
            coordinate_system,
        )


def test_wkb_point_distance_rejects_mismatched_lengths():
    left = pl.Series(shapely.to_wkb([shapely.Point(0, 0), shapely.Point(1, 1)]))
    right = pl.Series(shapely.to_wkb([shapely.Point(0, 0)]))

    with pytest.raises(ValueError, match="all coordinate arrays must have the same length"):
        wkb_point_distance(left, right)


def test_point_distance_rejects_unknown_coordinate_system():
    with pytest.raises(ValueError, match="planar"):
        point_distance(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), "sphere")


def test_distance_to_point_rejects_unknown_coordinate_system():
    with pytest.raises(ValueError, match="planar"):
        distance_to_point(np.array([0.0]), np.array([0.0]), 0.0, 0.0, "sphere")


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


def test_point_polygon_distance_planner_flips_pairs_and_aggregates():
    polygons = [shapely_box(i * 10, 0, i * 10 + 1, 1) for i in range(600)]
    hit_xs = np.arange(600, dtype=np.float64) * 10 + 0.5
    query_xs = np.concatenate([hit_xs, np.linspace(0, 5991, 4400)])
    query_ys = np.concatenate([np.full(600, 0.5), np.full(4400, 100.0)])
    values = np.arange(5000, dtype=np.float64)
    validities = np.ones(5000, dtype=np.uint8)
    auto = Engine.from_polygons(polygons)
    auto.set_index_mode("auto")
    forward = Engine.from_polygons(polygons)
    forward.set_index_mode("explicit:r_tree")

    auto_pairs = auto.batch_within_distance_to_polygons(query_xs, query_ys, 0.1).reshape(-1, 2)
    forward_pairs = forward.batch_within_distance_to_polygons(query_xs, query_ys, 0.1).reshape(
        -1, 2
    )
    auto_pairs = auto_pairs[np.lexsort((auto_pairs[:, 1], auto_pairs[:, 0]))]
    forward_pairs = forward_pairs[np.lexsort((forward_pairs[:, 1], forward_pairs[:, 0]))]
    assert np.array_equal(auto_pairs, forward_pairs)

    auto_agg = auto.batch_within_distance_to_polygons_aggregate(
        query_xs, query_ys, 0.1, [values], [validities]
    )
    forward_agg = forward.batch_within_distance_to_polygons_aggregate(
        query_xs, query_ys, 0.1, [values], [validities]
    )
    assert all(np.array_equal(actual, expected) for actual, expected in zip(auto_agg, forward_agg))

    operations = auto.take_metrics()["operations"]
    selected = {
        operation["name"]: operation["index"]
        for operation in operations
        if operation["name"]
        in {
            "batch_within_distance_to_polygons",
            "batch_within_distance_to_polygons_aggregate",
        }
    }
    assert selected == {
        "batch_within_distance_to_polygons": "grid",
        "batch_within_distance_to_polygons_aggregate": "grid",
    }


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


@pytest.mark.parametrize("sorted_output", [False, True])
def test_lazy_polygon_knn_join_is_exact_under_covering_mbrs(sorted_output):
    # Same covering-MBR trap as the Engine-level test, through the lazy join and both its paths
    polys = [_annulus(i * 0.1, 0.0, 100.0, 50.0) for i in range(4)]
    polys.append(shapely_box(1, 0, 2, 1))
    df = pl.DataFrame({"pid": list(range(5)), "geom": [p.wkb for p in polys]})
    sf = SpatialFrame.from_wkb_polygons(df, "geom")
    query = pl.DataFrame({"qx": [0.0], "qy": [0.0]})
    out = sf.lazy().polygon_knn_join(query, "qx", "qy", k=1, sorted_output=sorted_output).collect()
    assert out["pid"].to_list() == [4]
    assert abs(out["distance_to_polygon"][0] - 1.0) < 1e-9


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


def test_knn_to_polygons_counts_a_multipolygon_once():
    # Both parts of the MultiPolygon bracket the query, and the nearer part must not crowd the
    # far square out of the answer by occupying two of the k slots.
    mp = shapely.MultiPolygon([shapely_box(1, 0, 2, 1), shapely_box(-2, 0, -1, 1)])
    far = shapely_box(4, 0, 5, 1)
    eng = Engine.from_polygons([mp, far])
    idx, dist = eng.batch_knn_to_polygons(
        np.array([0.0], dtype=np.float64), np.array([0.5], dtype=np.float64), 2
    )

    assert idx.tolist() == [0, 1]
    assert np.allclose(dist, [1.0, 4.0], atol=1e-9)


def test_knn_to_polygons_pads_when_k_exceeds_polygon_count():
    eng = Engine.from_polygons([shapely_box(0, 0, 1, 1), shapely_box(3, 0, 4, 1)])
    idx, dist = eng.batch_knn_to_polygons(
        np.array([0.5], dtype=np.float64), np.array([0.5], dtype=np.float64), 5
    )

    assert idx[:2].tolist() == [0, 1]
    assert np.isinf(dist[2:]).all()


def test_knn_to_polygons_matches_brute_force_on_mixed_geometry():
    # Concave shapes holes and multi-part geometries together make the seed's MBR bound wrong
    # often enough to force the sweep, and k is wide enough to keep it running.
    rng = np.random.default_rng(11)
    polys: list = []
    for i in range(60):
        cx, cy = rng.uniform(0, 60, 2)
        kind = i % 3
        if kind == 0:
            polys.append(_annulus(cx, cy, rng.uniform(4.0, 14.0), rng.uniform(1.0, 3.0)))
        elif kind == 1:
            # L shape: its MBR corner sits far from any edge
            s = rng.uniform(3.0, 9.0)
            polys.append(
                shapely.Polygon(
                    [
                        (cx, cy),
                        (cx + s, cy),
                        (cx + s, cy + s / 3),
                        (cx + s / 3, cy + s / 3),
                        (cx + s / 3, cy + s),
                        (cx, cy + s),
                    ]
                )
            )
        else:
            off = rng.uniform(6.0, 18.0)
            polys.append(
                shapely.MultiPolygon(
                    [
                        shapely_box(cx, cy, cx + 2, cy + 2),
                        shapely_box(cx + off, cy + off, cx + off + 2, cy + off + 2),
                    ]
                )
            )

    eng = Engine.from_polygons(polys)
    qx = rng.uniform(0, 60, 120)
    qy = rng.uniform(0, 60, 120)
    k = 6
    _, dist = eng.batch_knn_to_polygons(qx, qy, k)
    dist = dist.reshape(-1, k)

    points = shapely.points(qx, qy)
    for q in range(len(qx)):
        want = np.sort([p.distance(points[q]) for p in polys])[:k]
        assert np.allclose(dist[q], want, atol=1e-9), f"query {q}: {dist[q]} vs {want}"
