"""
Aggregation specs for the fused aggregate-join (SpatialGroupBy.agg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from pycanopy.nodes import PolygonWithinDistanceJoinNode, WithinJoinNode

# Prefix for intermediate (partial) columns
_P = "__pc_agg__"


@dataclass(frozen=True)
class AggSpec:
    """One associative aggregation: a kind and the column it reads (None for count)."""

    kind: str
    column: str | None = None

    @property
    def inputs(self) -> set[str]:
        """Source columns this spec reads, for the join keep-set.

        Returns:
            The set of source column names, empty for count.
        """
        return set() if self.column is None else {self.column}

    def partial(self, name: str) -> list[pl.Expr]:
        """Build the per-morsel aggregation exprs for this spec.

        Args:
            name: Output column name this aggregation produces.

        Returns:
            Exprs producing this spec's prefixed intermediate columns.
        """
        col = pl.col(self.column) if self.column is not None else None
        if self.kind == "count":
            return [pl.len().alias(f"{_P}{name}__count")]
        if self.kind == "sum":
            return [col.sum().alias(f"{_P}{name}__sum")]
        if self.kind == "mean":
            return [
                col.sum().alias(f"{_P}{name}__sum"),
                col.count().alias(f"{_P}{name}__count"),
            ]
        if self.kind == "min":
            return [col.min().alias(f"{_P}{name}__min")]
        if self.kind == "max":
            return [col.max().alias(f"{_P}{name}__max")]
        raise ValueError(f"Unknown aggregation kind: {self.kind}")

    def combine(self, name: str) -> list[pl.Expr]:
        """Build the cross-morsel exprs that re-aggregate this spec's partials.

        Args:
            name: Output column name this aggregation produces.

        Returns:
            Exprs re-aggregating this spec's prefixed intermediate columns.
        """
        if self.kind == "count":
            return [pl.col(f"{_P}{name}__count").sum().alias(f"{_P}{name}__count")]
        if self.kind == "sum":
            return [pl.col(f"{_P}{name}__sum").sum().alias(f"{_P}{name}__sum")]
        if self.kind == "mean":
            return [
                pl.col(f"{_P}{name}__sum").sum().alias(f"{_P}{name}__sum"),
                pl.col(f"{_P}{name}__count").sum().alias(f"{_P}{name}__count"),
            ]
        if self.kind == "min":
            return [pl.col(f"{_P}{name}__min").min().alias(f"{_P}{name}__min")]
        if self.kind == "max":
            return [pl.col(f"{_P}{name}__max").max().alias(f"{_P}{name}__max")]
        raise ValueError(f"Unknown aggregation kind: {self.kind}")

    def finalize(self, name: str) -> pl.Expr:
        """Build the expr producing the named output from the combined partials.

        Args:
            name: Output column name this aggregation produces.

        Returns:
            Expr yielding the named output column.
        """
        if self.kind == "count":
            return pl.col(f"{_P}{name}__count").alias(name)
        if self.kind == "sum":
            return pl.col(f"{_P}{name}__sum").alias(name)
        if self.kind == "mean":
            count = pl.col(f"{_P}{name}__count")
            return (
                pl.when(count > 0)
                .then(pl.col(f"{_P}{name}__sum") / count)
                .otherwise(None)
                .alias(name)
            )
        if self.kind == "min":
            return pl.col(f"{_P}{name}__min").alias(name)
        if self.kind == "max":
            return pl.col(f"{_P}{name}__max").alias(name)
        raise ValueError(f"Unknown aggregation kind: {self.kind}")


def count() -> AggSpec:
    """Count rows (pairs) per group, like Polars pl.len().

    Returns:
        An AggSpec for the count aggregation.
    """
    return AggSpec("count")


def sum(column: str) -> AggSpec:
    """Sum a column per group.

    Args:
        column: Name of the column to sum.

    Returns:
        An AggSpec for the sum aggregation.
    """
    return AggSpec("sum", column)


def mean(column: str) -> AggSpec:
    """Mean of a column per group, ignoring nulls.

    Args:
        column: Name of the column to average.

    Returns:
        An AggSpec for the mean aggregation.
    """
    return AggSpec("mean", column)


def min(column: str) -> AggSpec:
    """Minimum of a column per group.

    Args:
        column: Name of the column to reduce.

    Returns:
        An AggSpec for the min aggregation.
    """
    return AggSpec("min", column)


def max(column: str) -> AggSpec:
    """Maximum of a column per group.

    Args:
        column: Name of the column to reduce.

    Returns:
        An AggSpec for the max aggregation.
    """
    return AggSpec("max", column)


def _partial_agg(frame: pl.DataFrame, keys: list[str], specs: dict[str, AggSpec]) -> pl.DataFrame:
    # Reduce one joined morsel to per-group partial columns
    return frame.group_by(keys).agg([e for name, spec in specs.items() for e in spec.partial(name)])


def _reduce_partials(
    partials: list[pl.DataFrame],
    keys: list[str],
    specs: dict[str, AggSpec],
) -> pl.DataFrame:
    # Combine per-morsel partial frames into the final grouped aggregate
    combine_exprs = [e for name, spec in specs.items() for e in spec.combine(name)]
    final_exprs = [spec.finalize(name) for name, spec in specs.items()]
    combined = pl.concat(partials).group_by(keys).agg(combine_exprs)
    return combined.select([*keys, *final_exprs])


def _try_fused_join_agg(
    sf, plan, keys: list[str], specs: dict[str, AggSpec]
) -> pl.DataFrame | None:
    # Run a supported polygon join aggregation without materializing match pairs
    if not keys:
        return None
    if len(plan) != 1 or not isinstance(plan[0], (WithinJoinNode, PolygonWithinDistanceJoinNode)):
        return None
    if any(spec.kind not in {"count", "sum", "mean"} for spec in specs.values()):
        return None

    node = plan[0]
    if isinstance(node.query_df, pl.LazyFrame):
        return None
    query_columns = set(node.query_df.columns)
    target_columns = set(sf.df.columns)
    if any(key not in target_columns or key in query_columns for key in keys):
        return None

    value_names = list(
        dict.fromkeys(spec.column for spec in specs.values() if spec.column is not None)
    )
    for name in value_names:
        if name not in query_columns or name in target_columns:
            return None
        dtype = node.query_df.schema[name]
        if not dtype.is_numeric():
            return None
        if any(spec.kind == "sum" and spec.column == name for spec in specs.values()):
            if dtype != pl.Float64:
                return None

    values: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    for name in value_names:
        series = node.query_df[name]
        values.append(series.cast(pl.Float64).fill_null(0.0).to_numpy())
        validities.append(series.is_not_null().cast(pl.UInt8).to_numpy())

    query_xs = node.query_df[node.x_col].to_numpy()
    query_ys = node.query_df[node.y_col].to_numpy()
    if isinstance(node, WithinJoinNode):
        groups, pair_counts, sums, value_counts = sf.engine.batch_contains_aggregate(
            query_xs, query_ys, values, validities
        )
    else:
        groups, pair_counts, sums, value_counts = (
            sf.engine.batch_within_distance_to_polygons_aggregate(
                query_xs, query_ys, node.distance, values, validities
            )
        )

    group_count = len(groups)
    sums = sums.reshape(len(value_names), group_count)
    value_counts = value_counts.reshape(len(value_names), group_count)
    group_idx = pl.Series("", groups)
    partial = sf.df.select(keys)[group_idx]
    value_position = {name: position for position, name in enumerate(value_names)}
    columns: list[pl.Series] = []
    for output, spec in specs.items():
        if spec.kind == "count":
            columns.append(pl.Series(f"{_P}{output}__count", pair_counts).cast(pl.UInt32))
        elif spec.kind == "sum":
            position = value_position[spec.column]
            columns.append(pl.Series(f"{_P}{output}__sum", sums[position]))
        else:
            position = value_position[spec.column]
            columns.extend(
                [
                    pl.Series(f"{_P}{output}__sum", sums[position]),
                    pl.Series(f"{_P}{output}__count", value_counts[position]).cast(pl.UInt32),
                ]
            )
    partial = partial.with_columns(columns)
    return _reduce_partials([partial], keys, specs)
