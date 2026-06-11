"""Correctness checks shared by the SpatialBench query modules.

Each query's validate() reduces its PyCanopy result (Polars) and reference result
(pandas) to comparable Python values and calls one of these. The aim is to confirm
PyCanopy computes the same answer as the GeoPandas oracle, not identical formatting.
"""

from __future__ import annotations


def rowcount(pc_df, ref_df) -> tuple[bool, str]:
    """True when the two results have the same number of rows."""
    a, b = len(pc_df), len(ref_df)
    return a == b, f"rowcount pycanopy={a} reference={b}"


def scalar(pc_value, ref_value, rel_tol: float = 1e-6, abs_tol: float = 0.0) -> tuple[bool, str]:
    """True when two scalars match within a relative (or absolute) tolerance."""
    a, b = float(pc_value), float(ref_value)
    ok = abs(a - b) <= max(abs_tol, rel_tol * max(1.0, abs(b)))
    return ok, f"scalar pycanopy={a} reference={b}"


def grouped(pc_map: dict, ref_map: dict, rel_tol: float = 1e-6) -> tuple[bool, str]:
    """True when two key->value maps match (values within rel_tol).

    Args:
        pc_map: Mapping of group key to numeric value from the PyCanopy result.
        ref_map: Same mapping from the reference result.
        rel_tol: Relative tolerance for value comparison.

    Returns:
        (ok, detail). Detail lists up to five differing keys on mismatch.
    """
    keys = set(pc_map) | set(ref_map)
    diffs = []
    for k in keys:
        a = pc_map.get(k)
        b = ref_map.get(k)
        if a is None or b is None:
            diffs.append((k, a, b))
        elif abs(float(a) - float(b)) > rel_tol * max(1.0, abs(float(b))):
            diffs.append((k, a, b))
    if not diffs:
        return True, f"grouped match ({len(keys)} keys)"
    return False, f"grouped mismatch on {len(diffs)} keys: {diffs[:5]}"
