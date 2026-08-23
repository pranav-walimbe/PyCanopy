"""SpatialBench table fetching and the query-scoped frame builders used by the queries."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import shapely

from bench.spatial_bench.profiler_utils import StageProfiler
from pycanopy import SpatialFrame


def read_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one SpatialBench table as a Polars DataFrame (geometry stays WKB).

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        table: Table name (e.g. "trip").
        columns: Optional subset of columns to read.

    Returns:
        The table as a Polars DataFrame.
    """
    return pl.read_parquet(
        f"{data_dir.rstrip('/')}/{table}/**/*.parquet",
        columns=columns,
        storage_options={"skip_signature": "true"},
    )


def scan_table(data_dir: str, table: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazily scan one SpatialBench table as a LazyFrame (geometry stays WKB).

    Lazy sibling of read_table, for late materialization. A query that narrows rows on
    cheap columns never decodes the wide WKB column for the rows it later discards.

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        table: Table name (e.g. "trip").
        columns: Optional subset of columns to project.

    Returns:
        A LazyFrame over the table's parquet.
    """
    lf = pl.scan_parquet(
        f"{data_dir.rstrip('/')}/{table}/**/*.parquet",
        storage_options={"skip_signature": "true"},
    )
    return lf.select(columns) if columns is not None else lf


def wkb_to_polygons(series: pl.Series) -> list:
    """Decode a WKB polygon column to shapely Polygons / MultiPolygons.

    Args:
        series: A Polars Series of WKB-encoded polygon geometries.

    Returns:
        A list of shapely Polygon / MultiPolygon objects (each MultiPolygon kept whole).
    """
    return list(shapely.from_wkb(series.to_numpy()))


@dataclass
class SpatialBenchTables:
    """Projected SpatialBench table loader for one query run.

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
    """

    data_dir: str
    index_mode: str = "eager"

    def parallel_fetch(self, needs: dict[str, list[str] | None]) -> dict[str, pl.DataFrame]:
        """Fetch projected tables concurrently.

        Args:
            needs: Map of table name to the columns to fetch, or None for all columns.

        Returns:
            Query-scoped DataFrames keyed by table name.
        """
        names = list(needs)
        frames = self.collect_all([self.scan(name, needs[name]) for name in names])
        return dict(zip(names, frames, strict=True))

    def collect_all(self, frames: list[pl.LazyFrame]) -> list[pl.DataFrame]:
        """Collect lazy table plans concurrently.

        Args:
            frames: Lazy table plans to collect together.

        Returns:
            The collected DataFrames, in the order given.
        """
        return pl.collect_all(frames)

    def table(self, name: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Read one projected table into a query-scoped DataFrame.

        Args:
            name: Table name.
            columns: Optional subset of columns to read.

        Returns:
            The requested table as a Polars DataFrame.
        """
        return read_table(self.data_dir, name, columns)

    def scan(self, name: str, columns: list[str] | None = None) -> pl.LazyFrame:
        """Lazily scan table ``name`` (uncached, for late-materialization access).

        Returns a LazyFrame rather than a cached DataFrame because the point of a lazy
        scan is to defer reads until the collected plan decides what to read.

        Args:
            name: Table name.
            columns: Optional subset of columns to project.

        Returns:
            An uncached LazyFrame over the table.
        """
        return scan_table(self.data_dir, name, columns)

    def point_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a point SpatialFrame from a WKB point column of ``df``.

        Args:
            df: DataFrame holding the WKB point column.
            wkb_col: Name of the WKB point column.

        Returns:
            A point SpatialFrame over ``df``.
        """
        return SpatialFrame.from_wkb_points(df, wkb_col, index_mode=self.index_mode)

    def polygon_frame(self, df: pl.DataFrame, wkb_col: str) -> SpatialFrame:
        """Build a polygon SpatialFrame straight from the WKB column (decoded in Rust).

        Args:
            df: DataFrame holding the WKB polygon column.
            wkb_col: Name of the WKB polygon column.

        Returns:
            A polygon SpatialFrame over ``df``.
        """
        return SpatialFrame.from_wkb_polygons(df, wkb_col, index_mode=self.index_mode)


@dataclass
class ProfilingTables(SpatialBenchTables):
    """SpatialBench tables that identify fetch wall time without timing Engine work.

    Args:
        data_dir: Local or object-store SpatialBench dataset root.
        index_mode: PyCanopy index build policy ("eager" / "none" / "auto").
    """

    profiler: StageProfiler = field(default_factory=StageProfiler)

    def collect_all(self, frames: list[pl.LazyFrame]) -> list[pl.DataFrame]:
        """Collect lazy table plans while attributing the boundary to fetch.

        Args:
            frames: Lazy table plans to collect together.

        Returns:
            The collected DataFrames, in the order given.
        """
        with self.profiler.stage("fetch"):
            return super().collect_all(frames)

    def table(self, name: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Read one projected table while attributing the boundary to fetch.

        Args:
            name: Table name.
            columns: Optional subset of columns to read.

        Returns:
            The requested table as a Polars DataFrame.
        """
        with self.profiler.stage("fetch"):
            return super().table(name, columns)
