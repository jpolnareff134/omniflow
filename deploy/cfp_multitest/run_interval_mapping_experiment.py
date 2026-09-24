#!/usr/bin/env python3
"""Rerun the CloudPerfTrace urgency-to-interval mapping ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.specific.cloudperftrace import load_parquet_trace  # noqa: E402
from support.log import make_run_folder  # noqa: E402
from sweep import (  # noqa: E402
    EvaluationResult,
    SweepResult,
    _build_shared_urgency_profile,
    _build_urgency_configs,
    _config_label,
)
from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import (  # noqa: E402
    compute_fixed_urgency_trace,
    evaluate_info_loss_from_runs,
    track_fixed_urgency,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudperftrace", required=True)
    parser.add_argument("--task", type=int, default=9)
    parser.add_argument("--column", default="tr_self")
    parser.add_argument("--metric", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--profile-points", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(
        make_run_folder(str(ROOT / "out"), application_type="interval_mapping"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = load_parquet_trace(
        args.cloudperftrace,
        task=args.task,
        column=args.column,
        metric=args.metric,
        max_rows=args.max_rows,
        max_points=args.max_points,
    )
    base_config = PipelineConfig()
    configs, sweep_params = _build_urgency_configs(base_config)
    label_params = list(sweep_params)
    urgency_trace = compute_fixed_urgency_trace(
        np.asarray(trace, dtype=np.float64),
        base_config._make_poller(),
    )
    full_results = base_config._make_tracker().track(trace)
    results = []
    profiles = []
    for index, config in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {config.interval_mapping}", flush=True)
        poll_results = track_fixed_urgency(trace, config._make_poller(), urgency_trace)
        results.append(EvaluationResult(
            config=config,
            info_loss=evaluate_info_loss_from_runs(trace, full_results, poll_results),
        ))
        point_count = len(poll_results)
        keep = np.arange(point_count, dtype=np.int64)
        if point_count > args.profile_points:
            keep = np.unique(np.linspace(
                0, point_count - 1, num=args.profile_points, dtype=np.int64,
            ))
        intervals = np.fromiter((result.interval for result in poll_results), dtype=np.int64)
        sampled = np.fromiter((result.sampled for result in poll_results), dtype=np.float64)
        cumulative_ratio = np.cumsum(sampled) / (np.arange(point_count) + 1)
        profiles.append({
            "config_id": index,
            "config_label": _config_label(config, label_params),
            "time_index": keep.tolist(),
            "interval": intervals[keep].tolist(),
            "sample_ratio": cumulative_ratio[keep].tolist(),
        })

    sweep = SweepResult(
        results=results,
        sweep_params={key: list(values) for key, values in sweep_params.items()},
        n_data_points=len(trace),
    )
    result_path = sweep.save_json(
        str(output_dir),
        filename="urgency_sweep.json",
        extra_manifest={
            "plot_kind": "urgency",
            "dataset": {
                "cloudperftrace": str(Path(args.cloudperftrace).resolve()),
                "task": args.task,
                "column": args.column,
                "metric": args.metric,
            },
            "shared_urgency_profile": _build_shared_urgency_profile(
                urgency_trace, max_points=args.profile_points,
            ),
            "config_profiles": profiles,
        },
    )
    (output_dir / "summary.txt").write_text(sweep.summary() + "\n")
    print(sweep.summary())

    if args.plot:
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "sweep.py"),
            "plot",
            "urgency",
            result_path,
        ], cwd=ROOT, check=True)

    metadata = {
        "dropped_logistic_variants": [
            {"midpoint": 0.4, "steepness": 6.0},
            {"midpoint": 0.6, "steepness": 6.0},
        ],
        "retained_logistic_variants": [
            {"midpoint": 0.4, "steepness": 12.0},
            {"midpoint": 0.6, "steepness": 12.0},
        ],
        "added_mapping": {"name": "exponential_ease_out", "steepness": 4.0},
    }
    (output_dir / "mapping_changes.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
