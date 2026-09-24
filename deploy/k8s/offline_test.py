#!/usr/bin/env python3
"""Replay interval candidates over dense nginx traces from HPA runs."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import evaluate_info_loss  # noqa: E402


CANDIDATES = {
    "linear_5_45": {"min_interval": 5, "max_interval": 45, "interval_mapping": "linear_rate"},
    "linear_5_30": {"min_interval": 5, "max_interval": 30, "interval_mapping": "linear_rate"},
    "ease_out_5_45": {"min_interval": 5, "max_interval": 45, "interval_mapping": "exponential_ease_out"},
    "ease_out_5_30": {"min_interval": 5, "max_interval": 30, "interval_mapping": "exponential_ease_out"},
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_dense_traces(repeat_dir: Path, label: str) -> tuple[PipelineConfig, list[np.ndarray]]:
    config = None
    traces = []
    for path in sorted(repeat_dir.glob(f"{label}_daemon_*.jsonl*")):
        records = load_jsonl(path)
        if config is None:
            event = next((r for r in records if r.get("event") == "daemon_start"), None)
            if event:
                config = PipelineConfig(**event["config"])
        nginx_ids = {
            r.get("pod")
            for r in records
            if r.get("event") == "track_start" and "nginx" in r.get("pod_name", "")
        }
        for pod_id in nginx_ids:
            values = [
                r["value"]
                for r in records
                if r.get("pod") == pod_id and "value" in r and "event" not in r
            ]
            if len(values) >= 2:
                traces.append(np.asarray(values, dtype=np.float64))
    return config or PipelineConfig(), traces


def replay_execution(config: PipelineConfig, traces: list[np.ndarray], overrides: dict) -> dict:
    candidate = replace(config, **overrides)
    reports = [
        evaluate_info_loss(trace, candidate._make_tracker(), candidate._make_poller())
        for trace in traces
    ]
    total = sum(report.n_total for report in reports)
    if not reports or total == 0:
        return {}

    def weighted(name: str) -> float:
        return sum(getattr(report, name) * report.n_total for report in reports) / total

    return {
        "n_pods": len(reports),
        "n_points": total,
        "sample_ratio": sum(report.n_sampled for report in reports) / total,
        "raw_recon_nrmse": weighted("raw_recon_nrmse"),
        "tracker_nrmse": weighted("nrmse_mean"),
        "peak_recall": weighted("peak_recall_top5"),
        "mean_gap": weighted("mean_gap"),
        "p95_gap": weighted("p95_gap"),
        "max_gap": max(report.max_gap for report in reports),
    }


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def evaluate_run(run_dir: Path) -> dict:
    params = json.loads((run_dir / "params.json").read_text())
    executions = []
    for label_dir in sorted((run_dir / "results").iterdir()):
        if not label_dir.is_dir():
            continue
        for repeat_dir in sorted(label_dir.glob("repeat_*")):
            config, traces = load_dense_traces(repeat_dir, label_dir.name)
            if traces:
                executions.append((label_dir.name, repeat_dir.name, config, traces))

    candidates = {}
    if not executions:
        raise ValueError(f"no daemon traces found under {run_dir}")
    for name, overrides in CANDIDATES.items():
        rows = []
        for label, repeat, config, traces in executions:
            row = replay_execution(config, traces, overrides)
            if row:
                row.update({"source_arm": label, "repeat": repeat})
                rows.append(row)
        metrics = (
            "sample_ratio", "raw_recon_nrmse", "tracker_nrmse", "peak_recall",
            "mean_gap", "p95_gap", "max_gap",
        )
        candidates[name] = {
            "config": overrides,
            "executions": len(rows),
            "pods": sum(row["n_pods"] for row in rows),
            "points": sum(row["n_points"] for row in rows),
            "summary": {metric: summarize([row[metric] for row in rows]) for metric in metrics},
            "rows": rows,
        }
    return {"shape": params["shape"], "run_dir": str(run_dir), "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"runs": [evaluate_run(path.resolve()) for path in args.run_dirs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for run in result["runs"]:
        print(f"\n{run['shape']}")
        for name, candidate in run["candidates"].items():
            summary = candidate["summary"]
            print(
                f"  {name:16s} r={summary['sample_ratio']['mean']:.3%} "
                f"raw_nrmse={summary['raw_recon_nrmse']['mean']:.4f} "
                f"peak={summary['peak_recall']['mean']:.1%} "
                f"gap={summary['mean_gap']['mean']:.1f}/"
                f"{summary['p95_gap']['mean']:.1f}/{summary['max_gap']['mean']:.1f} "
                f"n={candidate['executions']}"
            )


if __name__ == "__main__":
    main()
