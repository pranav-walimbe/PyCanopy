"""Aggregation specs for the fused aggregate-join (SpatialGroupBy.agg).

Each spec is associative: a morsel reduces to per-group partials and the partials
combine exactly into the single-pass result, so a streamed join never materialises
the full pair frame. mean is carried as a sum and a count and divided at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Prefix for intermediate (partial) columns, kept distinct from user output names.
_P = "__pc_agg__"


@dataclass(frozen=True)
class AggSpec:
    """One associative aggregation: a kind and the column it reads (None for count)."""

    kind: str
    column: str | None = None

    @property
    def inputs(self) -> set[str]:
        """Source columns this spec reads, for the join keep-set."""
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
    """Count rows (pairs) per group, like Polars pl.len()."""
    return AggSpec("count")


def sum(column: str) -> AggSpec:
    """Sum a column per group."""
    return AggSpec("sum", column)


def mean(column: str) -> AggSpec:
    """Mean of a column per group, ignoring nulls."""
    return AggSpec("mean", column)


def min(column: str) -> AggSpec:
    """Minimum of a column per group."""
    return AggSpec("min", column)


def max(column: str) -> AggSpec:
    """Maximum of a column per group."""
    return AggSpec("max", column)


def _partial_agg(frame: pl.DataFrame, keys: list[str], specs: dict[str, AggSpec]) -> pl.DataFrame:
    """Reduce one joined morsel to per-group partial columns."""
    return frame.group_by(keys).agg([e for name, spec in specs.items() for e in spec.partial(name)])


def _reduce_partials(
    partials: list[pl.DataFrame],
    keys: list[str],
    specs: dict[str, AggSpec],
) -> pl.DataFrame:
    """Combine per-morsel partial frames into the final grouped aggregate.

    Args:
        partials: Per-morsel partial frames from _partial_agg.
        keys: Group-by key columns.
        specs: Output name to aggregation spec.

    Returns:
        The grouped aggregate with one row per key combination.
    """
    combine_exprs = [e for name, spec in specs.items() for e in spec.combine(name)]
    final_exprs = [spec.finalize(name) for name, spec in specs.items()]
    combined = pl.concat(partials).group_by(keys).agg(combine_exprs)
    return combined.select([*keys, *final_exprs])
