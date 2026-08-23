"""Combine per-engine transports into SpatialBench text and chart outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from bench.spatial_bench.config import DATASET_VERSION, DISPLAY_NAMES

_COLORS = {
    "pycanopy": "#2C7FB8",
    "sedonadb": "#DD8452",
    "duckdb": "#8C8C8C",
    "geopandas": "#C9BBA8",
}


def read_transport(path: Path) -> dict:
    """Read one engine's tab-separated cloud result."""
    result = {"metadata": {}, "queries": {}}
    with path.open(newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if not row:
                continue
            if row[0] == "engine":
                result["id"], result["version"] = row[1:3]
            elif row[0] == "metadata":
                result["metadata"][row[1]] = row[2]
            elif row[0] == "query":
                result["queries"][row[1]] = {
                    "status": row[2],
                    "seconds": float(row[3]) if row[3] else None,
                    "run_times": [float(value) for value in row[4].split(",") if value],
                    "error": row[5] if len(row) > 5 else "",
                }
    if "id" not in result:
        raise ValueError(f"invalid engine result: {path}")
    return result


def combine_transports(
    paths: list[Path], engines: list[str], scale_factor: int, index_mode: str
) -> dict:
    """Combine transports in the requested engine order."""
    parsed = {item["id"]: item for item in (read_transport(path) for path in paths)}
    combined = {
        "scale_factor": scale_factor,
        "index_mode": index_mode,
        "engine_order": engines,
        "engines": {},
        "metadata": {},
    }
    for engine in engines:
        if engine in parsed:
            combined["engines"][engine] = parsed[engine]
        else:
            combined["engines"][engine] = {
                "id": engine,
                "version": "not available",
                "metadata": {},
                "queries": {},
            }
    for engine in engines:
        if parsed.get(engine, {}).get("metadata"):
            combined["metadata"] = parsed[engine]["metadata"]
            break
    combined["metadata"]["engines"] = ", ".join(
        f"{DISPLAY_NAMES[engine]} {combined['engines'][engine]['version']}" for engine in engines
    )
    if "pycanopy" in engines:
        combined["metadata"]["PyCanopy index mode"] = index_mode
    return combined


def _query_ids(results: dict) -> list[str]:
    ids = {query for engine in results["engines"].values() for query in engine["queries"]}
    return sorted(ids, key=lambda query: int(query[1:]))


def write_results_txt(results: dict, out_path: Path) -> None:
    """Write the combined engine table and raw samples."""
    sf = results["scale_factor"]
    engines = results["engine_order"]
    query_ids = _query_ids(results)
    lines = [f"Apache SpatialBench SF{sf}", "", "Run metadata", "------------"]
    lines.extend(f"{key}: {value}" for key, value in results["metadata"].items())
    lines.extend(["", "Results", "-------"])

    widths = {engine: max(12, len(DISPLAY_NAMES[engine])) for engine in engines}
    header = f"{'query':<8}" + "".join(
        f"  {DISPLAY_NAMES[engine]:>{widths[engine]}}" for engine in engines
    )
    lines.extend((header, "-" * len(header)))
    for query_id in query_ids:
        row = f"{query_id:<8}"
        for engine in engines:
            query = results["engines"][engine]["queries"].get(query_id, {})
            value = query.get("seconds")
            cell = f"{value:.2f}" if value is not None else query.get("status", "ERROR").upper()
            row += f"  {cell:>{widths[engine]}}"
        lines.append(row)

    lines.extend(["", "Raw samples (seconds)", "---------------------"])
    for engine in engines:
        for query_id in query_ids:
            samples = results["engines"][engine]["queries"].get(query_id, {}).get("run_times", [])
            if samples:
                values = ", ".join(f"{sample:.2f}" for sample in samples)
                lines.append(f"{DISPLAY_NAMES[engine]} {query_id}: {values}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _nice_cap(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for multiplier in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if multiplier * magnitude >= value:
            return multiplier * magnitude
    return 10 * magnitude


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 1.0
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_chart(results: dict, out_path: Path) -> None:
    """Render the legacy grouped horizontal bar chart with live engine results."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    engines = results["engine_order"]
    query_ids = _query_ids(results)

    def value(engine: str, query_id: str):
        return results["engines"][engine]["queries"].get(query_id, {}).get("seconds")

    finite = [
        item
        for query_id in query_ids
        for engine in engines
        if isinstance(item := value(engine, query_id), (int, float))
    ]
    cap = _nice_cap(_percentile(finite, 0.90)) if finite else 1.0
    truncated = any(item > cap for item in finite)
    series_count = len(engines)
    figure, axis = plt.subplots(figsize=(8.2, 1.2 + 0.70 * len(query_ids)))
    axis.set_axisbelow(True)
    band = 0.82
    bar_height = band / series_count

    for position in range(len(query_ids)):
        if position % 2:
            axis.axhspan(position - 0.5, position + 0.5, color="#F4F7FA", zorder=0)
    for position in range(1, len(query_ids)):
        axis.axhline(position - 0.5, color="#DBDBDB", lw=0.6, ls=(0, (1, 2)), zorder=1)

    for series, engine in enumerate(engines):
        color = _COLORS[engine]
        for position, query_id in enumerate(query_ids):
            y = position + (series - (series_count - 1) / 2) * bar_height
            seconds = value(engine, query_id)
            if seconds is None:
                status = (
                    results["engines"][engine]["queries"].get(query_id, {}).get("status", "error")
                )
                axis.text(
                    cap * 0.012,
                    y,
                    status.lower(),
                    ha="left",
                    va="center",
                    fontsize=6.5,
                    color="#3C7FA6",
                    fontstyle="italic",
                )
                continue
            axis.barh(y, min(seconds, cap), height=bar_height * 0.9, color=color, zorder=2)
            if seconds > cap:
                axis.text(
                    cap * 1.015, y, f"... {seconds:.1f}", va="center", fontsize=6.5, color=color
                )
            elif seconds < cap * 0.03:
                label = f"{seconds:.2f}" if seconds < 1 else f"{seconds:.1f}"
                axis.text(
                    seconds + cap * 0.008, y, label, va="center", fontsize=6.5, color="#555555"
                )

    step = _nice_cap(cap / 6)
    ticks = []
    tick = 0.0
    while tick <= cap + 1e-9:
        ticks.append(round(tick, 6))
        tick += step
    axis.set_xticks(ticks)
    axis.set_xlim(0, cap * 1.16)
    axis.set_ylim(-0.5, len(query_ids) - 0.5)
    axis.invert_yaxis()
    axis.set_yticks(range(len(query_ids)))
    axis.set_yticklabels(query_ids)
    axis.set_xlabel("run time (seconds)")
    axis.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
    axis.tick_params(length=0)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    subtitle = f"dataset {DATASET_VERSION}    all engines measured on isolated matching nodes"
    if "pycanopy" in engines:
        subtitle += f"    PyCanopy index mode: {results['index_mode']}"
    if truncated:
        subtitle += f"    bars past {cap:g}s truncated"
    labels = " / ".join(DISPLAY_NAMES[engine] for engine in engines)
    axis.set_title(
        f"Apache SpatialBench SF{results['scale_factor']}: {labels}\n{subtitle}", fontsize=10
    )
    axis.legend(
        handles=[
            Patch(facecolor=_COLORS[engine], label=DISPLAY_NAMES[engine]) for engine in engines
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
