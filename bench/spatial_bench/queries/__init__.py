"""Registry of SpatialBench query modules.

Each query module exposes:
  id: str                      query identifier, e.g. "q1"
  title: str                   short description
  pycanopy(tables) -> pl.DataFrame      the PyCanopy + Polars implementation
  reference(paths) -> pd.DataFrame      the ported GeoPandas oracle / baseline
  validate(pc_df, ref_df) -> (bool, str)   correctness check vs the oracle
"""

from __future__ import annotations

from bench.spatial_bench.queries import (
    q01,
    q02,
    q03,
    q04,
    q05,
    q06,
    q07,
    q08,
    q09,
    q10,
    q11,
    q12,
)

# Ordered list of all implemented query modules.
ALL = [q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12]

_BY_ID = {q.id: q for q in ALL}


def select(ids: list[str] | None = None) -> list:
    """Return the requested query modules, or all of them when ids is falsy.

    Args:
        ids: Optional list of query ids (e.g. ["q1", "q4"]).

    Returns:
        Query modules in registry order.

    Raises:
        KeyError: If an unknown id is requested.
    """
    if not ids:
        return list(ALL)
    return [_BY_ID[i] for i in ids]
