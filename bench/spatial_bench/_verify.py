"""Verify SpatialBench results against the committed upstream answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

WORKLOAD_REVISION = "b9221a9c4b02b10db20611d79b4019d2b3c4b68e"
DATASET_VERSION = "v0.1.0"

_ANSWERS_DIR = Path(__file__).with_name("answers")
_LIMIT_QUERIES = {"q1", "q5", "q7", "q9", "q10", "q12"}
_LIMIT_CAP = 100
_RTOL = 1e-6
_ATOL = 1e-9
_DURATION_ATOL = 1e-3


def _as_frame(result) -> pl.DataFrame:
    # Materialize supported result types without changing their row order
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    if isinstance(result, pl.DataFrame):
        return result
    if hasattr(result, "to_arrow"):
        return pl.from_arrow(result.to_arrow())
    return pl.DataFrame(result)


def _column_kind(dtype: pl.DataType) -> str:
    # Classify the canonical answer type that controls comparison semantics
    if dtype.is_float():
        return "float"
    if dtype.is_integer():
        return "int"
    if dtype.is_temporal():
        return "temporal"
    return "exact"


def _compare(answer: pl.DataFrame, result: pl.DataFrame) -> list[dict]:
    # Compare columns by position while deriving semantics from the answer schema
    if answer.width != result.width:
        return [
            {
                "kind": "shape",
                "first_row": -1,
                "count": 0,
                "message": (
                    f"column count differs: answer {answer.width} ({answer.columns}) "
                    f"vs pycanopy {result.width} ({result.columns})"
                ),
            }
        ]

    issues: list[dict] = []
    if answer.height != result.height:
        issues.append(
            {
                "kind": "shape",
                "first_row": -1,
                "count": abs(answer.height - result.height),
                "message": (
                    f"row count differs: answer {answer.height} vs pycanopy {result.height}"
                ),
            }
        )

    size = min(answer.height, result.height)
    for position, answer_name in enumerate(answer.columns):
        dtype = answer.schema[answer_name]
        kind = _column_kind(dtype)
        expected = answer.get_column(answer_name).head(size)
        actual = result.get_column(result.columns[position]).head(size)

        if kind == "float":
            expected_values = expected.cast(pl.Float64).to_numpy()
            actual_values = actual.cast(pl.Float64, strict=False).to_numpy()
            atol = _DURATION_ATOL if answer_name.endswith("_seconds") else _ATOL
            bad = ~np.isclose(
                expected_values,
                actual_values,
                rtol=_RTOL,
                atol=atol,
                equal_nan=True,
            )
        else:
            actual = actual.cast(dtype, strict=False)
            bad = np.asarray(
                [left != right for left, right in zip(expected.to_list(), actual.to_list())],
                dtype=bool,
            )

        count = int(bad.sum())
        if count:
            first = int(np.flatnonzero(bad)[0])
            issues.append(
                {
                    "kind": kind,
                    "first_row": first,
                    "count": count,
                    "column": position,
                    "name": answer_name,
                    "message": (
                        f"column {position} ({answer_name!r}, {kind}): {count}/{size} differ "
                        f"starting at row {first}"
                    ),
                }
            )
    return issues


def _is_boundary_tie(query_id: str, issues: list[dict], answer: pl.DataFrame) -> bool:
    # Accept only a key or string swap in the final row of an eligible capped query
    if not issues or query_id not in _LIMIT_QUERIES or answer.height != _LIMIT_CAP:
        return False
    final_row = answer.height - 1
    return all(
        issue["kind"] != "shape"
        and issue["kind"] != "float"
        and issue["first_row"] == final_row
        and issue["count"] == 1
        for issue in issues
    )


def verify_output(
    result,
    query_id: str,
    scale_factor: int,
    answers_dir: Path | None = None,
) -> tuple[bool, str]:
    """Compare one result with the pinned upstream SpatialBench answer.

    Args:
        result: Materialized or lazy PyCanopy result frame.
        query_id: SpatialBench query identifier.
        scale_factor: Dataset scale factor, currently 1 or 10.
        answers_dir: Optional answer root used by tests and local tooling.

    Returns:
        Whether the result matches and a concise comparison detail.
    """
    root = answers_dir or _ANSWERS_DIR
    answer_path = root / f"sf{scale_factor}" / f"{query_id}.parquet"
    if not answer_path.is_file():
        raise FileNotFoundError(f"no committed SpatialBench answer at {answer_path}")

    answer = pl.read_parquet(answer_path)
    actual = _as_frame(result)
    issues = _compare(answer, actual)
    if not issues:
        return True, f"{actual.height} ordered rows match upstream answer"
    if _is_boundary_tie(query_id, issues, answer):
        return True, f"{actual.height} ordered rows match with final-row boundary tie"
    return False, "; ".join(issue["message"] for issue in issues[:3])
