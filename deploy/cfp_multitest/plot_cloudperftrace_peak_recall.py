#!/usr/bin/env python3
"""Plot the paper's sample-ratio / top-5% peak-recall trade-off from results.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


CASE = "websearch_involuntary_ctx_switches"
METHODS = {
    "omniflow": ("OmniFlow", "#0072B2", "-"),
    "huang_wavelet_rate_inspired": ("Huang-inspired", "#D55E00", "--"),
    "magalhaes_truncated_exponential_inspired": (
        "Magalhaes-inspired", "#009E73", ":"
    ),
}


def load_points(path: Path) -> dict[str, list[dict[str, float]]]:
    points: dict[str, list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case"] != CASE:
                continue
            method = row["method"]
            if method not in METHODS and method not in {
                    "fixed_fast_boundary", "fixed_slow_boundary"}:
                continue
            points[method].append({
                "sample_pct": 100.0 * float(row["sample_ratio"]),
                "recall_pct": 100.0 * float(row["peak_recall_top5"]),
                "min_interval": float(row["min_interval"]),
            })
    return points


def _periodic_points(points: dict[str, list[dict[str, float]]]) -> list[dict[str, float]]:
    # The fixed-fast boundary visits I=1..19. At I_min=I_max=20 the runner
    # emits only one endpoint, labelled fixed_slow_boundary.
    candidates = list(points.get("fixed_fast_boundary", []))
    candidates.extend(
        row for row in points.get("fixed_slow_boundary", [])
        if int(row["min_interval"]) == 20
    )
    unique = {int(row["min_interval"]): row for row in candidates}
    return sorted(unique.values(), key=lambda row: row["sample_pct"])


def plot(results_csv: Path, output: Path) -> None:
    points = load_points(results_csv)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    norm = Normalize(vmin=1, vmax=20)
    cmap = plt.get_cmap("viridis")

    periodic = _periodic_points(points)
    if periodic:
        periodic = sorted(periodic, key=lambda row: row["sample_pct"])
        ax.plot(
            [row["sample_pct"] for row in periodic],
            [row["recall_pct"] for row in periodic],
            color="#CC79A7", linestyle="-.", linewidth=1.1,
            label="Periodic",
        )
        ax.scatter(
            [row["sample_pct"] for row in periodic],
            [row["recall_pct"] for row in periodic],
            c=[row["min_interval"] for row in periodic], cmap=cmap, norm=norm,
            edgecolors="#CC79A7", linewidths=0.35, s=17,
        )

    for method, (label, color, linestyle) in METHODS.items():
        rows = sorted(points.get(method, []), key=lambda row: row["sample_pct"])
        if not rows:
            continue
        ax.plot(
            [row["sample_pct"] for row in rows],
            [row["recall_pct"] for row in rows],
            color=color, linestyle=linestyle, linewidth=1.2, label=label,
        )
        ax.scatter(
            [row["sample_pct"] for row in rows],
            [row["recall_pct"] for row in rows],
            c=[row["min_interval"] for row in rows], cmap=cmap, norm=norm,
            edgecolors=color, linewidths=0.35, s=17,
        )

    default = next((row for row in points.get("omniflow", [])
                    if int(row["min_interval"]) == 1), None)
    if default:
        ax.scatter(
            [default["sample_pct"]], [default["recall_pct"]],
            marker="*", s=90, facecolors="white", edgecolors="black",
            linewidths=0.7, zorder=5, label="OmniFlow default",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4.7, 105)
    ax.set_ylim(4.7, 105)
    ax.set_xticks([5, 10, 20, 50, 100], labels=["5", "10", "20", "50", "100"])
    ax.set_yticks([5, 10, 20, 50, 100], labels=["5", "10", "20", "50", "100"])
    ax.set_xlabel("Sample ratio r (%)")
    ax.set_ylabel("Peak recall R5% (%)")
    ax.grid(True, which="major", color="black", alpha=0.09)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label("I_min")
    colorbar.set_ticks([1, 5, 10, 15, 20])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results_csv, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
