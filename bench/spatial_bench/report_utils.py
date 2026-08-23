"""SpatialBench result plumbing: the TSV transport plus the text, chart, and profile reports."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from bench.spatial_bench.config import (
    DATASET_VERSION,
    ENGINES,
    MIB,
    NS_PER_SECOND,
    RSS_SAMPLE_INTERVAL,
    WORKLOAD_REVISION,
)

_ENGINE_ROW = "engine"
_METADATA_ROW = "metadata"
_QUERY_ROW = "query"


def write_transport(
    path: Path,
    engine: str,
    engine_version: str,
    metadata: dict[str, str],
    results: dict[str, dict],
) -> None:
    """Write one engine's measured results to its transport file.

    Args:
        path: Destination transport path.
        engine: Engine id that produced the results.
        engine_version: Installed version of the measured engine.
        metadata: Run metadata recorded on the box.
        results: Per-query result dicts keyed by query id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow((_ENGINE_ROW, engine, engine_version))
        for key, value in metadata.items():
            writer.writerow((_METADATA_ROW, key, value))
        for query_id, result in results.items():
            samples = ",".join(f"{sample:.6f}" for sample in result.get("run_times", []))
            seconds = result.get("seconds")
            writer.writerow(
                (
                    _QUERY_ROW,
                    query_id,
                    result["status"],
                    f"{seconds:.6f}" if seconds is not None else "",
                    samples,
                    str(result.get("error", "")).replace("\n", " "),
                )
            )


def read_transport(path: Path) -> dict:
    """Read one engine's transport file back into a result dict.

    Args:
        path: Transport file downloaded from the results bucket.

    Returns:
        A dict with the engine id, version, metadata, and per-query results.
    """
    result = {"metadata": {}, "queries": {}}
    with path.open(newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if not row:
                continue
            if row[0] == _ENGINE_ROW:
                result["id"], result["version"] = row[1:3]
            elif row[0] == _METADATA_ROW:
                result["metadata"][row[1]] = row[2]
            elif row[0] == _QUERY_ROW:
                result["queries"][row[1]] = {
                    "status": row[2],
                    "seconds": float(row[3]) if row[3] else None,
                    "run_times": [float(value) for value in row[4].split(",") if value],
                    "error": row[5] if len(row) > 5 else "",
                }
    if "id" not in result:
        raise ValueError(f"invalid engine result: {path}")
    return result


_SEP = "=" * 76
_SUBSEP = "-" * 76


def combine_transports(paths: list[Path], engines: list[str], scale_factor: int) -> dict:
    """Combine transports in the requested engine order.

    Args:
        paths: Downloaded transport files, in any order.
        engines: Engine ids in the order they should be reported.
        scale_factor: Dataset scale factor for the run.

    Returns:
        One combined results dict covering every requested engine.
    """
    parsed = {item["id"]: item for item in (read_transport(path) for path in paths)}
    combined = {
        "scale_factor": scale_factor,
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
        f"{ENGINES[engine]['display_name']} {combined['engines'][engine]['version']}"
        for engine in engines
    )
    return combined


def _query_ids(results: dict) -> list[str]:
    # Every query id any engine reported, in numeric rather than lexical order
    ids = {query for engine in results["engines"].values() for query in engine["queries"]}
    return sorted(ids, key=lambda query: int(query[1:]))


def write_results_txt(results: dict, out_path: Path) -> None:
    """Write the combined engine table and raw samples.

    Args:
        results: Combined results dict from combine_transports.
        out_path: Destination text path.
    """
    sf = results["scale_factor"]
    engines = results["engine_order"]
    query_ids = _query_ids(results)
    lines = [f"Apache SpatialBench SF{sf}", "", "Run metadata", "------------"]
    lines.extend(f"{key}: {value}" for key, value in results["metadata"].items())
    lines.extend(["", "Results", "-------"])

    widths = {engine: max(12, len(ENGINES[engine]["display_name"])) for engine in engines}
    header = f"{'query':<8}" + "".join(
        f"  {ENGINES[engine]['display_name']:>{widths[engine]}}" for engine in engines
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
                lines.append(f"{ENGINES[engine]['display_name']} {query_id}: {values}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _nice_cap(value: float) -> float:
    # Round an axis bound up to the nearest readable 1/1.5/2/2.5/3/4/5/6/8/10 multiple
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for multiplier in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if multiplier * magnitude >= value:
            return multiplier * magnitude
    return 10 * magnitude


def _percentile(values: list[float], percentile: float) -> float:
    # Linear-interpolated percentile, used to keep one slow query from flattening the chart
    ordered = sorted(values)
    if not ordered:
        return 1.0
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_chart(results: dict, out_path: Path) -> None:
    """Render the grouped horizontal bar chart with live engine results.

    Args:
        results: Combined results dict from combine_transports.
        out_path: Destination PNG path.
    """
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
        color = ENGINES[engine]["color"]
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
    if truncated:
        subtitle += f"    bars past {cap:g}s truncated"
    labels = " / ".join(ENGINES[engine]["display_name"] for engine in engines)
    axis.set_title(
        f"Apache SpatialBench SF{results['scale_factor']}: {labels}\n{subtitle}", fontsize=10
    )
    axis.legend(
        handles=[
            Patch(facecolor=ENGINES[engine]["color"], label=ENGINES[engine]["display_name"])
            for engine in engines
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def _section(query_id: str, result: dict) -> str:
    # One profile block per query: wall boundaries, RSS, Engine work, and the verify verdict
    lines = [_SEP, f"{query_id}  {result.get('title', '')}".rstrip(), _SUBSEP]
    if result["status"] != "ok":
        return "\n".join(
            [*lines, f"status        {result['status']}  {result.get('error', '')}".rstrip()]
        )

    profile = result["profile"]
    wall = profile["time"]
    mib = {name: value / MIB for name, value in profile["mem"].items()}
    engine = profile["engine"]
    construction = engine["construction"]
    verdict = {"match": "PASS", "MISMATCH": "FAIL", "error": "ERROR"}.get(
        result["verify"], result["verify"]
    )
    lines += [
        f"wall (s)      total {wall['total']:7.3f}  fetch {wall['fetch']:7.3f}  "
        f"execute {wall['execute']:7.3f}  materialize {wall['materialize']:7.3f}",
        f"               non-engine wall after fetch {wall['non_engine']:7.3f}",
        f"memory (MiB)  peak {mib['peak']:8.1f}  baseline {mib['baseline']:8.1f}  "
        f"demand {mib['peak'] - mib['baseline']:+8.1f}",
        f"  stage peak  fetch {mib['fetch']:8.1f}  execute {mib['execute']:8.1f}  "
        f"materialize {mib['materialize']:8.1f}",
        f"engines       {len(engine['engines'])}  WKB decode "
        f"{construction['wkb_decode_ns'] / NS_PER_SECOND:7.3f}s  statistics "
        f"{construction['statistics_ns'] / NS_PER_SECOND:7.3f}s",
    ]
    if engine["index_builds"]:
        lines.append("index builds")
        for metric in engine["index_builds"]:
            lines.append(
                f"  {metric['index']:<26} calls {metric['build_count']:5,d}  "
                f"time {metric['elapsed_compute_ns'] / NS_PER_SECOND:9.4f}s"
            )
    if engine["operations"]:
        lines.append("engine operations")
        for metric in engine["operations"]:
            lines.append(
                f"  {metric['name']:<36} {metric['index']:<12} calls {metric['calls']:5,d}  "
                f"rows {metric['output_rows']:12,d}  "
                f"time {metric['elapsed_compute_ns'] / NS_PER_SECOND:9.4f}s"
            )
    lines.append(f"verify        {verdict}   {result['verify_detail']}")
    return "\n".join(lines)


def write_profile(results: dict, out_path: Path) -> None:
    """Write the human-readable SF1 profile report.

    Args:
        results: Per-query profile result dicts keyed by query id.
        out_path: Destination text path.
    """
    head = (
        "PyCanopy Apache SpatialBench SF1 profile (1 run/query)\n"
        f"Dataset {DATASET_VERSION}, workload revision {WORKLOAD_REVISION}\n"
        "Engine times are always-on production metrics. Harness stages are wall boundaries.\n"
        f"RSS is sampled every {int(RSS_SAMPLE_INTERVAL * 1000)} ms. Verification uses the "
        "committed upstream answers."
    )
    parts = [head, *[_section(query_id, result) for query_id, result in results.items()], _SEP]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")
