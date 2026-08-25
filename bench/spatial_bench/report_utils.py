"""SpatialBench result plumbing: the TSV transport plus the text, chart, and profile reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from bench.spatial_bench.config import (
    DATASET_VERSION,
    ENGINES,
    MIB,
    NS_PER_SECOND,
    PROFILE_VARIANT_LABELS,
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
_WIDE_SEP = "=" * 100
_WIDE_SUB = "-" * 100


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
    # One profile block per query covering wall and RSS and Engine work and the verdict
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
        f"wall (s)      total {wall['total']:7.3f}  non-engine {wall['non_engine']:7.3f}",
        f"memory (MiB)  peak {mib['peak']:8.1f}  baseline {mib['baseline']:8.1f}  "
        f"demand {mib['peak'] - mib['baseline']:+8.1f}",
        f"engines       {len(engine['engines'])}  WKB decode "
        f"{construction['wkb_decode_ns'] / NS_PER_SECOND:7.3f}s  statistics "
        f"{construction['statistics_ns'] / NS_PER_SECOND:7.3f}s",
    ]
    host = _host_times(result)
    if host:
        lines.append(
            "host stages   " + "  ".join(f"{name} {value:.3f}s" for name, value in host.items())
        )
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


def write_profile(results: dict, out_path: Path, metadata: dict[str, str] | None = None) -> None:
    """Write the human-readable SF1 profile report for a single build.

    Args:
        results: Per-query profile result dicts keyed by query id.
        out_path: Destination text path.
        metadata: Run metadata to record above the per-query sections, if collected.
    """
    parts = [_profile_head()]
    if metadata:
        parts.append(_metadata_block(metadata))
    parts.extend(_section(query_id, result) for query_id, result in results.items())
    parts.append(_SEP)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")


def write_profile_transport(
    path: Path, variant: str, metadata: dict[str, str], results: dict
) -> None:
    """Write one build's profile payload for the launcher to combine.

    Args:
        path: Destination JSON path.
        variant: Build variant id this box measured.
        metadata: Run metadata collected on the box.
        results: Per-query profile result dicts keyed by query id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"variant": variant, "metadata": metadata, "results": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def read_profile_transport(path: Path) -> dict:
    """Read one build's profile payload back.

    Args:
        path: Profile transport downloaded from the results bucket.

    Returns:
        A dict with the variant id, run metadata, and per-query results.
    """
    payload = json.loads(path.read_text())
    if "variant" not in payload:
        raise ValueError(f"invalid profile transport: {path}")
    return payload


def _profile_head() -> str:
    # Fixed preamble naming the workload and the units
    return (
        "PyCanopy Apache SpatialBench SF1 profile (1 run/query)\n"
        f"Dataset {DATASET_VERSION}, workload revision {WORKLOAD_REVISION}\n"
        "Engine times are always-on production metrics. Wall times are harness boundaries and\n"
        "include parquet reads. Memory is the whole-run RSS peak, sampled every "
        f"{int(RSS_SAMPLE_INTERVAL * 1000)} ms.\n"
        "Verification uses the committed upstream answers."
    )


def _metadata_block(metadata: dict[str, str]) -> str:
    # Run metadata lets a later reader tell a code change from a machine change
    lines = [_SEP, "Run metadata", _SUBSEP]
    lines.extend(f"{key:<22}{value}" for key, value in metadata.items())
    return "\n".join(lines)


def _summary(result: dict) -> dict | None:
    # Wall and memory and verdict for one query or None when it produced no profile
    if result.get("status") != "ok":
        return None
    profile = result["profile"]
    return {
        "wall": profile["time"]["total"],
        "non_engine": profile["time"]["non_engine"],
        "peak": profile["mem"]["peak"] / MIB,
        "verdict": {"match": "PASS", "MISMATCH": "FAIL", "error": "ERROR"}.get(
            result.get("verify", ""), str(result.get("verify", "?"))
        ),
    }


def _stage_times(result: dict) -> dict[str, float]:
    # Engine compute per stage, keyed alike for both builds and kept in pipeline order
    if result.get("status") != "ok":
        return {}
    engine = result["profile"]["engine"]
    stages: dict[str, float] = {}
    construction = engine["construction"]
    for name, label in (("wkb_decode_ns", "WKB decode"), ("statistics_ns", "statistics")):
        if construction.get(name):
            stages[label] = construction[name] / NS_PER_SECOND
    for metric in engine["index_builds"]:
        stages[f"build {metric['index']}"] = metric["elapsed_compute_ns"] / NS_PER_SECOND
    for metric in engine["operations"]:
        stages[f"{metric['name']} ({metric['index']})"] = (
            metric["elapsed_compute_ns"] / NS_PER_SECOND
        )
    return stages


# Host stage order in pipeline sequence with the derived remainder last
_HOST_ORDER = ("read", "wkb decode", "frame build", "polars compute")


def _host_times(result: dict) -> dict[str, float]:
    # Host wall per stage with everything unattributed folded into polars compute
    if result.get("status") != "ok":
        return {}
    profile = result["profile"]
    stages = dict(profile.get("stages", {}))
    if not stages:
        return {}
    engine = profile["engine"]
    outside = sum(metric["elapsed_compute_ns"] for metric in engine["index_builds"])
    outside += sum(metric["elapsed_compute_ns"] for metric in engine["operations"])
    accounted = sum(stages.values()) + outside / NS_PER_SECOND
    stages["polars compute"] = max(profile["time"]["total"] - accounted, 0.0)
    return {name: stages[name] for name in _HOST_ORDER if name in stages}


def _host_table(results: dict[str, dict], query_ids: list[str], labels: list[str]) -> str:
    # Per-query host wall by stage, covering the time no Engine metric reports
    head = f"{'query':<7}{'stage':<54}{labels[0]:>12}{labels[1]:>13}{'delta':>14}"
    lines = [_WIDE_SEP, "Host time by stage, seconds", _WIDE_SUB, head, "-" * len(head)]
    for query_id in query_ids:
        stages = [_host_times(results[label].get(query_id, {})) for label in labels]
        names = _stage_order(stages)
        if not names:
            continue
        for position, name in enumerate(names):
            new, old = stages[0].get(name, 0.0), stages[1].get(name, 0.0)
            lines.append(
                f"{query_id if position == 0 else '':<7}{name:<54}{new:>12.4f}"
                f"{old:>13.4f}{_delta(new, old):>14}"
            )
    lines.append(
        "\nHost stages are wall clock around the read and decode calls. frame build contains "
        "the Engine's own WKB decode and statistics. polars compute is the unattributed "
        "remainder after every host stage and Engine metric is subtracted."
    )
    return "\n".join(lines)


def _reports_metrics(results: dict) -> bool:
    # A wheel without the private metrics hook still profiles wall time and memory
    return any(
        result.get("profile", {}).get("metrics", True)
        for result in results.values()
        if result.get("status") == "ok"
    )


def _stage_order(stages: list[dict[str, float]]) -> list[str]:
    # Union both builds' stages in pipeline order and append whatever only one has
    ordered: list[str] = []
    for stage in stages:
        for name in stage:
            if name not in ordered:
                ordered.append(name)
    return ordered


def _delta(new: float, old: float) -> str:
    # Percent change of new against old, guarding a zero baseline
    if old == 0:
        return "new" if new > 0 else "same"
    return f"{100 * (new - old) / old:+.1f}%"


def _comparison_table(results: dict[str, dict], query_ids: list[str], labels: list[str]) -> str:
    # Both builds side by side with a totals row
    head = (
        f"{'query':<7}{labels[0] + ' wall':>13}{labels[1] + ' wall':>15}{'delta':>9}"
        f"{labels[0] + ' peak':>14}{labels[1] + ' peak':>15}{'delta':>9}{'verify':>18}"
    )
    lines = [
        _WIDE_SEP,
        f"Wall time and peak memory, {labels[0]} against {labels[1]}",
        _WIDE_SUB,
        head,
        "-" * len(head),
    ]
    totals = {label: [0.0, 0.0] for label in labels}
    for query_id in query_ids:
        pair = [_summary(results[label].get(query_id, {})) for label in labels]
        if not all(pair):
            missing = labels[pair.index(None)] if None in pair else "?"
            lines.append(f"{query_id:<7}no profile from the {missing} build")
            continue
        for label, item in zip(labels, pair):
            totals[label][0] += item["wall"]
            totals[label][1] += item["peak"]
        lines.append(
            f"{query_id:<7}{pair[0]['wall']:>13.3f}{pair[1]['wall']:>15.3f}"
            f"{_delta(pair[0]['wall'], pair[1]['wall']):>9}"
            f"{pair[0]['peak']:>14.1f}{pair[1]['peak']:>15.1f}"
            f"{_delta(pair[0]['peak'], pair[1]['peak']):>9}"
            f"{pair[0]['verdict'] + ' / ' + pair[1]['verdict']:>18}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'total':<7}{totals[labels[0]][0]:>13.3f}{totals[labels[1]][0]:>15.3f}"
        f"{_delta(totals[labels[0]][0], totals[labels[1]][0]):>9}"
        f"{totals[labels[0]][1]:>14.1f}{totals[labels[1]][1]:>15.1f}"
        f"{_delta(totals[labels[0]][1], totals[labels[1]][1]):>9}{'':>18}"
    )
    lines.append(
        f"\nA negative delta means the {labels[0]} build is faster or smaller than {labels[1]}."
    )
    return "\n".join(lines)


def _stage_table(results: dict[str, dict], query_ids: list[str], labels: list[str]) -> str:
    # Per-query engine compute by stage, showing where a wall change came from
    paired = _reports_metrics(results[labels[1]])
    head = f"{'query':<7}{'stage':<54}{labels[0]:>12}"
    if paired:
        head += f"{labels[1]:>13}{'delta':>14}"
    lines = [_WIDE_SEP, "Engine compute by stage, seconds", _WIDE_SUB, head, "-" * len(head)]
    for query_id in query_ids:
        stages = [_stage_times(results[label].get(query_id, {})) for label in labels]
        names = _stage_order(stages)
        if not names:
            continue
        for position, name in enumerate(names):
            new, old = stages[0].get(name, 0.0), stages[1].get(name, 0.0)
            row = f"{query_id if position == 0 else '':<7}{name:<54}{new:>12.4f}"
            lines.append(row + (f"{old:>13.4f}{_delta(new, old):>14}" if paired else ""))
        totals = [sum(stage.values()) for stage in stages]
        row = f"{'':<7}{'total engine compute':<54}{totals[0]:>12.4f}"
        lines.append(
            row + (f"{totals[1]:>13.4f}{_delta(totals[0], totals[1]):>14}" if paired else "")
        )
    if paired:
        lines.append(
            "\nStages absent from one build show as 0.0000. A stage marked new exists only in "
            f"the {labels[0]} build."
        )
    else:
        lines.append(
            f"\nThe {labels[1]} build reports no engine metrics. Its published wheel exports no "
            f"metrics hook, so only wall time and peak memory compare against it."
        )
    return "\n".join(lines)


def write_profile_comparison(transports: dict[str, dict], out_path: Path) -> None:
    """Write the SF1 profile report comparing the branch build against the released one.

    Args:
        transports: Profile payloads keyed by variant id, as read from each box.
        out_path: Destination text path.
    """
    labels = {variant: PROFILE_VARIANT_LABELS.get(variant, variant) for variant in transports}
    results = {labels[variant]: payload["results"] for variant, payload in transports.items()}
    order = [labels[variant] for variant in PROFILE_VARIANT_LABELS if variant in transports]

    metadata: dict[str, str] = {}
    for variant in PROFILE_VARIANT_LABELS:
        if variant in transports:
            metadata = dict(transports[variant]["metadata"])
            break
    for variant, payload in transports.items():
        metadata[f"{labels[variant]} build"] = payload["metadata"].get("pycanopy build", "unknown")
        metadata[f"{labels[variant]} run ID"] = payload["metadata"].get("run ID", "unknown")
    for key in ("pycanopy build", "run ID"):
        metadata.pop(key, None)

    primary = results[order[0]]
    query_ids = sorted(
        {query for build in results.values() for query in build}, key=lambda q: int(q[1:])
    )
    parts = [_profile_head(), _metadata_block(metadata)]
    if len(order) == 2:
        parts.append(_comparison_table(results, query_ids, order))
        parts.append(_stage_table(results, query_ids, order))
        parts.append(_host_table(results, query_ids, order))
        parts.append(f"{_WIDE_SEP}\nPer-query detail, {order[0]} build")
    parts.extend(
        _section(query_id, primary[query_id]) for query_id in query_ids if query_id in primary
    )
    parts.append(_SEP)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n")
