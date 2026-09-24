"""Generate the paper's I/O workload-shape figure from compact trace values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TITLES = {
    "postgres": "PostgreSQL (pgbench)",
    "redis": "Redis (AOF)",
    "latency": "fio (cyclic)",
}


def _representative_values(payload: dict[str, Any]) -> dict[str, list[float]]:
    workloads = payload.get("workloads")
    if not isinstance(workloads, dict):
        raise ValueError("trace input must contain a 'workloads' object")

    traces: dict[str, list[float]] = {}
    for name in TITLES:
        workload = workloads.get(name)
        runs = workload.get("runs") if isinstance(workload, dict) else None
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"trace input is missing runs for {name!r}")
        representative = next(
            (run for run in runs if str(run.get("repeat")) == "01"), runs[0]
        )
        values = representative.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"representative run for {name!r} has no values")
        traces[name] = [float(value) for value in values]
    return traces


def _sample_points(
        values: list[float],
        *,
        step: int = 10,
        skip_ticks: int = 100,
) -> list[tuple[float, float]]:
    if step < 1 or skip_ticks < 0:
        raise ValueError("step must be positive and skip_ticks non-negative")
    points = []
    for index in range(skip_ticks, len(values), step):
        # Original trace tick labels start at t=1, with 200 ms per tick.
        points.append((round((index + 1) * 0.2, 1), round(values[index])))
    if not points:
        raise ValueError("trace has no points after the warm-up interval")
    return points


def _coords(points: list[tuple[float, float]]) -> str:
    return " ".join(f"({time_s:.1f},{value:.0f})" for time_s, value in points)


def generate(payload: dict[str, Any]) -> str:
    traces = _representative_values(payload)
    tex = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=1 by 3, vertical sep=0.7cm},
    width=0.95\columnwidth,
    height=2.8cm,
    ymin=0,
    xlabel={time (s)},
    ylabel={syscalls},
    xtick distance=30,
    tick label style={font=\scriptsize},
    label style={font=\small},
    title style={font=\small, yshift=-2pt},
    no markers,
    every axis plot/.style={blue},
    grid=major,
    grid style={gray!20},
]
"""
    for name, title in TITLES.items():
        points = _sample_points(traces[name])
        start = points[0][0]
        end = points[-1][0]
        tex += rf"""
\nextgroupplot[xmin={start:.1f}, xmax={end:.1f}, title={{{title}}}]
\addplot coordinates {{ {_coords(points)} }};
"""
    tex += r"""
\end{groupplot}
\end{tikzpicture}
\caption{I/O syscall rate traces from representative runs of each workload. The signal shows write and fsync syscalls per 200ms tick.}
\label{fig:io-workloads}
\end{figure}
"""
    return tex


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "paper_results" / "io" / "replay_inputs.json",
        help="Compact dense-trace input produced by the paper-artifact generator.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("io_workloads.tex"),
        help="Output PGFPlots source file.",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate(json.loads(args.input.read_text())))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
