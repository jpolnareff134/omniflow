#!/usr/bin/env python3
"""Recompute paper I/O replay metrics from compact dense trace values.

The input is a JSON object with ``workloads`` mapping workload names to a run
selection and compact values:

    {"workloads": {"postgres": {
        "paper_repeat_ids": ["01"],
        "runs": [{"repeat": "01", "values": [1, 2, ...]}]
    }}}

The paper metrics are calculated from the dense replay for the declared paper
run set. Means and standard deviations are over per-run values using population
standard deviation. The pooled median is retained as a diagnostic over every
dense replay tick in the selected runs. Latency repeats 01 and 40 are retained
in the input but excluded from the paper's 38-run workload-summary set because
their recorded workload arms failed the latency sample-alignment checks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "io_syscall"))

from evaluate import _build_pipe_config  # noqa: E402
from tracker.windowed import evaluate_info_loss_from_runs  # noqa: E402


DEFAULT_PAPER_RUNS = {
    "postgres": {f"{index:02d}" for index in range(1, 41)},
    "redis": {f"{index:02d}" for index in range(1, 41)},
    "latency": {f"{index:02d}" for index in range(2, 40)},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summarize_workload(
        name: str,
        runs: list[dict[str, Any]],
        paper_repeat_ids: set[str],
        config,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    selected_urgencies: list[np.ndarray] = []

    for item in sorted(runs, key=lambda row: str(row["repeat"])):
        repeat = str(item["repeat"])
        trace = np.asarray(item["values"], dtype=np.float64)
        full_results = config._make_tracker().track(trace)
        poll_results = config._make_poller().track(trace)
        urgency = np.asarray([result.urgency for result in poll_results], dtype=np.float64)
        info_loss = evaluate_info_loss_from_runs(trace, full_results, poll_results)
        included = repeat in paper_repeat_ids
        if included:
            selected_urgencies.append(urgency)
        per_run.append({
            "repeat": repeat,
            "included_in_paper_summary": included,
            "n_total": int(len(trace)),
            "n_sampled": int(sum(result.sampled for result in poll_results)),
            "sample_ratio": float(info_loss.sample_ratio),
            "mean_urgency": float(urgency.mean()) if urgency.size else None,
            "median_urgency": float(np.median(urgency)) if urgency.size else None,
            "nrmse_mean": float(info_loss.nrmse_mean),
            "correlation": float(info_loss.correlation),
            "max_gap": int(info_loss.max_gap),
        })

    selected_rows = [row for row in per_run if row["included_in_paper_summary"]]
    run_means = np.asarray([row["mean_urgency"] for row in selected_rows], dtype=np.float64)
    run_medians = np.asarray([row["median_urgency"] for row in selected_rows], dtype=np.float64)
    pooled = np.concatenate(selected_urgencies) if selected_urgencies else np.asarray([], dtype=np.float64)

    def _stats(values: np.ndarray) -> dict[str, float | int | None]:
        if not values.size:
            return {"count": 0, "mean": None, "stdev": None}
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "stdev": float(values.std(ddof=0)),
        }

    def _metric_stats(key: str) -> dict[str, float | int | None]:
        values = np.asarray(
            [row[key] for row in selected_rows if row[key] is not None],
            dtype=np.float64,
        )
        return _stats(values)

    return {
        "run_count": len(per_run),
        "paper_run_count": len(selected_rows),
        "paper_repeat_ids": sorted(paper_repeat_ids),
        "paper_metrics": {
            "mean_urgency": _metric_stats("mean_urgency"),
            "sample_ratio": _metric_stats("sample_ratio"),
            "nrmse_mean": _metric_stats("nrmse_mean"),
            "correlation": _metric_stats("correlation"),
            "max_gap": _metric_stats("max_gap"),
        },
        "paper_run_mean_urgency": _stats(run_means),
        "paper_run_median_urgency": _stats(run_medians),
        "paper_pooled_tick_urgency": {
            "count": int(pooled.size),
            "mean": None if not pooled.size else float(pooled.mean()),
            "median": None if not pooled.size else float(np.median(pooled)),
        },
        "runs": per_run,
    }


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute replay urgency and fidelity summaries for the selected runs."""
    config = _build_pipe_config(min_interval=1, max_interval=10, profile="default")
    workloads = payload.get("workloads")
    if not isinstance(workloads, dict):
        raise ValueError("input must contain a 'workloads' object")
    summaries = {}
    for name in ("postgres", "redis", "latency"):
        workload = workloads.get(name)
        if not isinstance(workload, dict):
            raise ValueError(f"input is missing runs for workload {name!r}")
        runs = workload.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"input is missing runs for workload {name!r}")
        selected_ids = workload.get("paper_repeat_ids")
        paper_repeat_ids = (
            {str(value) for value in selected_ids}
            if isinstance(selected_ids, list)
            else DEFAULT_PAPER_RUNS[name]
        )
        summaries[name] = _summarize_workload(
            name,
            runs,
            paper_repeat_ids,
            config,
        )
    return {
        "schema": "omniflow-io-replay-summary-v1",
        "configuration": config.to_dict(),
        "urgency_aggregation": (
            "mean/std are over per-run means with population standard deviation; "
            "pooled median is over all urgency values at dense trace ticks"
        ),
        "workloads": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Compact replay input JSON")
    parser.add_argument("--out", type=Path, required=True, help="Summary JSON destination")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    summary = summarize_payload(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
