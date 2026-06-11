"""Render SpatialBench results: PyCanopy measured vs published reference engines.

Consumes a results JSON emitted by ``measure.py`` plus the baked-in published
numbers in ``references.py`` and produces a Markdown comparison table and grouped
bar charts. This layer is cheap and has no native dependencies, so it can run
anywhere — the heavy measurement happens once (on AWS) and the JSON is the handoff.

Usage:
    python -m bench.spatial_bench.report --results bench/spatial_bench/results/sf1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.spatial_bench.references import (
    REFERENCE_ENGINES,
    REFERENCE_META,
    reference_row,
)

_RESULTS_DIR = Path(__file__).parent / "results"
_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


def _fmt_seconds(value: float | str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):  # ERROR / TIMEOUT
        return value
    return f"{value:.2f}"


def load_results(path: str | Path) -> dict:
    """Load a results JSON produced by measure.py."""
    with open(path) as f:
        return json.load(f)


def build_markdown_table(results: dict) -> str:
    """Return a Markdown table: PyCanopy measured next to published engine numbers.

    Args:
        results: Parsed results JSON with keys ``scale_factor`` and ``queries``
            (a mapping of query id to {"pycanopy_seconds", "geopandas_seconds",
            "status", "row_count"}).

    Returns:
        Markdown string. All times in seconds.
    """
    sf = int(results["scale_factor"])
    engines = list(REFERENCE_ENGINES)
    header = ["Query", "PyCanopy", "PyCanopy (warm)", *engines]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for qid in sorted(results["queries"], key=lambda q: int(q[1:])):
        row = results["queries"][qid]
        ref = reference_row(sf, qid)
        cells = [
            qid,
            _fmt_seconds(row.get("pycanopy_seconds")),
            _fmt_seconds(row.get("pycanopy_warm_seconds")),
            *[_fmt_seconds(ref.get(e)) for e in engines],
        ]
        lines.append("| " + " | ".join(cells) + " |")

    caption = (
        f"\n_Scale factor {sf}. PyCanopy measured locally; "
        f"{', '.join(engines)} are published numbers "
        f"({REFERENCE_META['source']}, {REFERENCE_META['hardware']}). "
        "For a fair comparison PyCanopy must run on the same instance type — "
        "see spatial_bench/README.md._"
    )
    return "\n".join(lines) + "\n" + caption


def build_charts(results: dict, out_dir: Path | None = None) -> list[Path]:
    """Render a grouped bar chart per query (PyCanopy vs published engines).

    Skips silently if matplotlib is not installed. Numeric reference values only
    (ERROR/TIMEOUT entries are omitted from the bars).

    Args:
        results: Parsed results JSON.
        out_dir: Directory for PNGs. Defaults to the repo ``assets/`` directory.

    Returns:
        List of written chart paths (empty if matplotlib is unavailable).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir = out_dir or _ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = int(results["scale_factor"])

    qids = sorted(results["queries"], key=lambda q: int(q[1:]))
    labels = ["PyCanopy", *REFERENCE_ENGINES]

    fig, ax = plt.subplots(figsize=(max(8, len(qids)), 5))
    bar_w = 0.8 / len(labels)
    for li, label in enumerate(labels):
        xs, heights = [], []
        for qi, qid in enumerate(qids):
            if label == "PyCanopy":
                val = results["queries"][qid].get("pycanopy_seconds")
            else:
                val = reference_row(sf, qid).get(label)
            if isinstance(val, (int, float)):
                xs.append(qi + li * bar_w)
                heights.append(val)
        ax.bar(xs, heights, width=bar_w, label=label)

    ax.set_xticks([i + 0.4 for i in range(len(qids))])
    ax.set_xticklabels(qids)
    ax.set_ylabel("seconds (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"SpatialBench SF{sf}: PyCanopy vs published engines")
    ax.legend()
    fig.tight_layout()

    path = out_dir / f"spatialbench_sf{sf}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render SpatialBench comparison report.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a results JSON produced by measure.py.",
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip rendering matplotlib charts."
    )
    args = parser.parse_args(argv)

    results = load_results(args.results)
    print(build_markdown_table(results))

    if not args.no_charts:
        charts = build_charts(results)
        for c in charts:
            print(f"\nwrote chart: {c}")


if __name__ == "__main__":
    main()
