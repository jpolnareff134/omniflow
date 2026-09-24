#!/usr/bin/env python3
"""Reproduce the five CloudPerfTrace paper cases and comparison policies.

For every case this preserves the original two-pass design: OmniFlow adaptive
first, followed by the nearest reciprocal fixed interval derived from the
adaptive sample ratio. It also derives the shortest balanced N-of-M sampler
within a configurable budget tolerance and exhaustively evaluates all M phases.
Two DOI-backed
comparison controllers are evaluated through the exact same dense-reference
information-loss code as OmniFlow.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.specific.cloudperftrace import load_parquet_trace  # noqa: E402
from support.log import make_run_folder  # noqa: E402
from tracker.baselines import (  # noqa: E402
    budget_ratio_to_n_every_m,
    track_n_every_m,
    track_huang_wavelet_rate,
    track_magalhaes_exponential,
)
from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import evaluate_info_loss_from_runs  # noqa: E402


CASES = (
    {"name": "redis_tx_bytes", "task": 5, "column": "tr_self", "metric": 38},
    {"name": "websearch_involuntary_ctx_switches", "task": 6, "column": "tr_self", "metric": 21},
    {"name": "hbase_block_write_latency", "task": 11, "column": "tr_self", "metric": 51},
    {"name": "minio_block_read_bytes", "task": 14, "column": "tr_self", "metric": 47},
    {"name": "flink_memory_rss", "task": 16, "column": "tr_self", "metric": 22},
)

POLICIES = {
    "huang_wavelet_rate_inspired": {
        "doi": "10.1109/TCC.2016.2603473",
        "fidelity": "causal scalar adaptation of the CS-MON rate law",
        "observes_every_point": False,
        "parameters": {
            "window": 64,
            "retained_energy": 0.995,
            "measurements_per_coefficient": 5.0,
            "min_samples": 4,
            "initial_sparsity_fraction": 0.10,
            "kalman_process_variance": 0.0025,
            "kalman_measurement_variance": 0.01,
        },
        "runner": track_huang_wavelet_rate,
    },
    "magalhaes_truncated_exponential_inspired": {
        "doi": "10.1109/NCA.2011.30",
        "fidelity": "scalar adaptation of selective truncated-exponential profiling; uses deviation instead workload-response correlation",
        "observes_every_point": False,
        "parameters": {
            "min_interval": 1,
            "max_interval": 20,
            "reference_window": 20,
            "outlier_threshold": 3.0,
            "recovery_samples": 3,
        },
        "runner": track_magalhaes_exponential,
    },
}


def _report(full_results, trace, poll_results) -> dict:
    return asdict(evaluate_info_loss_from_runs(trace, full_results, poll_results))


def _summary(reports: list[dict]) -> dict:
    result = {}
    for key in reports[0]:
        values = [report[key] for report in reports]
        if isinstance(values[0], (int, float)):
            result[key] = {
                "mean": float(np.mean(values)),
                "stdev": float(np.std(values, ddof=0)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result


def _fixed_interval(sample_ratio: float) -> int:
    return max(1, int(math.floor((1.0 / sample_ratio) + 0.5)))


def _phase_reports(
        full_results,
        trace: np.ndarray,
        config: PipelineConfig,
        *,
        n: int,
        m: int,
) -> list[dict]:
    reports = []
    for phase in range(m):
        print(f"    phase {phase + 1}/{m}", flush=True)
        poll_results = track_n_every_m(
            trace,
            config._make_tracker(),
            n=n,
            m=m,
            phase=phase,
        )
        reports.append(_report(full_results, trace, poll_results))
        del poll_results
    return reports


def _run_case(
        case: dict,
        trace: np.ndarray,
        config: PipelineConfig,
        *,
        matched_budget_tolerance: float,
        matched_budget_max_period: int,
) -> dict:
    print(f"  dense reference: {len(trace):,} points", flush=True)
    full_results = config._make_tracker().track(trace)
    result = {
        "case": case,
        "trace_stats": {
            "n_points": len(trace),
            "mean": float(np.mean(trace)),
            "std": float(np.std(trace)),
            "min": float(np.min(trace)),
            "max": float(np.max(trace)),
        },
        "pipeline_config": config.to_dict(),
        "two_pass": {},
        "policies": {},
    }

    print("  pass 1/2: OmniFlow adaptive", flush=True)
    adaptive_results = config._make_poller().track(trace)
    adaptive_report = _report(full_results, trace, adaptive_results)
    del adaptive_results
    fixed_interval = _fixed_interval(adaptive_report["sample_ratio"])

    print(f"  pass 2/2: fixed interval {fixed_interval}", flush=True)
    fixed_config = replace(config, min_interval=fixed_interval, max_interval=fixed_interval)
    fixed_results = fixed_config._make_poller().track(trace)
    fixed_report = _report(full_results, trace, fixed_results)
    del fixed_results
    result["two_pass"] = {
        "adaptive": adaptive_report,
        "fixed": {
            "derived_from_adaptive_sample_ratio": adaptive_report["sample_ratio"],
            "interval": fixed_interval,
            "report": fixed_report,
        },
    }

    matched_n, matched_m = budget_ratio_to_n_every_m(
        adaptive_report["sample_ratio"],
        tolerance=matched_budget_tolerance,
        max_period=matched_budget_max_period,
    )
    matched_ratio = matched_n / matched_m
    print(
        f"  budget-matched baseline: {matched_n} of {matched_m} "
        f"({matched_ratio:.4%}) across all {matched_m} phases",
        flush=True,
    )
    matched_reports = _phase_reports(
        full_results,
        trace,
        config,
        n=matched_n,
        m=matched_m,
    )
    result["budget_matched"] = {
        "target_sample_ratio": adaptive_report["sample_ratio"],
        "budget_tolerance": matched_budget_tolerance,
        "max_period": matched_budget_max_period,
        "n": matched_n,
        "m": matched_m,
        "nominal_sample_ratio": matched_ratio,
        "budget_error": matched_ratio - adaptive_report["sample_ratio"],
        "phase_count": matched_m,
        "phase_reports": matched_reports,
        "summary": _summary(matched_reports),
    }

    for name, policy in POLICIES.items():
        print(f"  policy: {name}", flush=True)
        poll_results = policy["runner"](
            trace,
            config._make_tracker(),
            **policy["parameters"],
        )
        result["policies"][name] = {
            "doi": policy["doi"],
            "fidelity": policy["fidelity"],
            "observes_every_point": policy["observes_every_point"],
            "parameters": policy["parameters"],
            "report": _report(full_results, trace, poll_results),
        }
        del poll_results
        gc.collect()

    del full_results
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudperftrace", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--only-case", choices=[case["name"] for case in CASES], action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--matched-budget-tolerance",
        type=float,
        default=0.001,
        help="Maximum absolute difference from OmniFlow's sample ratio (default: 0.001 = 0.1 percentage points).",
    )
    parser.add_argument(
        "--matched-budget-max-period",
        type=int,
        default=128,
        help="Largest period considered for the exhaustive budget-matched sampler.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(
        make_run_folder(str(ROOT / "out"), application_type="cloudperftrace_paper"))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [case for case in CASES if not args.only_case or case["name"] in args.only_case]
    config = PipelineConfig()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cloudperftrace": str(Path(args.cloudperftrace).resolve()),
        "output_dir": str(output_dir.resolve()),
        "matched_budget": {
            "tolerance": args.matched_budget_tolerance,
            "max_period": args.matched_budget_max_period,
            "phase_policy": "all accumulator phases",
            "pattern": "balanced N-of-M",
        },
        "cases": [],
        "policy_sources": {
            name: {key: value for key, value in policy.items() if key != "runner"}
            for name, policy in POLICIES.items()
        },
    }

    for index, case in enumerate(selected, 1):
        case_path = output_dir / f"{case['name']}.json"
        print(f"[{index}/{len(selected)}] {case['name']}", flush=True)
        if args.resume and case_path.exists():
            print("  existing result retained", flush=True)
            manifest["cases"].append(json.loads(case_path.read_text()))
            continue
        trace = load_parquet_trace(
            args.cloudperftrace,
            task=case["task"],
            column=case["column"],
            metric=case["metric"],
            max_rows=args.max_rows,
            max_points=args.max_points,
        )
        case_result = _run_case(
            case,
            trace,
            config,
            matched_budget_tolerance=args.matched_budget_tolerance,
            matched_budget_max_period=args.matched_budget_max_period,
        )
        case_path.write_text(json.dumps(case_result, indent=2))
        manifest["cases"].append(case_result)
        (output_dir / "run.json").write_text(json.dumps(manifest, indent=2))
        del trace, case_result
        gc.collect()

    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"Results: {output_dir / 'run.json'}", flush=True)


if __name__ == "__main__":
    main()
