#!/usr/bin/env python3
"""Run a common interval-envelope sweep on the CloudPerfTrace paper cases.

The experiment gives every adaptive policy the same admissible temporal range
``[min_interval, max_interval]`` while keeping all other policy parameters
fixed across traces.  The default sweep fixes ``max_interval=20`` and varies
``min_interval`` over every integer value from ``20`` down to ``1``.  This
exhausts the discrete fastest-interval settings within the default envelope.

This is intentionally *not* an exact global sample-count quota.  Adaptive
policies may realize different total sample ratios inside the same interval
envelope; those realized ratios are part of the result.  Huang's blockwise
policy is clamped to the finite-block sample counts implied by the same
interval bounds without changing its sparsity predictor.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.specific.cloudperftrace import load_parquet_trace  # noqa: E402
from eval.sampling_quality import (  # noqa: E402
    PEAK_RECALL_FRACTIONS,
    peak_recalls_from_precomputed,
    precompute_peak_indices,
)
from support.log import make_run_folder  # noqa: E402
from tracker.baselines import (  # noqa: E402
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

HUANG_PARAMETERS = {
    "window": 64,
    "retained_energy": 0.995,
    "measurements_per_coefficient": 5.0,
    "min_samples": 4,
    "initial_sparsity_fraction": 0.10,
    "kalman_process_variance": 0.0025,
    "kalman_measurement_variance": 0.01,
}

MAGALHAES_PARAMETERS = {
    "reference_window": 20,
    "outlier_threshold": 3.0,
    "recovery_samples": 3,
}

REPORT_METRICS = (
    "n_total",
    "n_sampled",
    "sample_ratio",
    "max_gap",
    "mean_gap",
    "p95_gap",
    "nrmse_mean",
    "correlation",
    "peak_recall_top1",
    "peak_recall_top2",
    "peak_recall_top5",
    "peak_recall_top10",
    "peak_recall_top20",
)


def _report(full_results, trace, poll_results, peak_indices) -> dict:
    report = evaluate_info_loss_from_runs(trace, full_results, poll_results)
    result = {
        key: getattr(report, key)
        for key in report.__dataclass_fields__
    }
    recalls = peak_recalls_from_precomputed(
        poll_results,
        n=len(trace),
        peak_indices=peak_indices,
    )
    result.update({
        "peak_recall_top1": recalls[0.01],
        "peak_recall_top2": recalls[0.02],
        "peak_recall_top10": recalls[0.10],
        "peak_recall_top20": recalls[0.20],
    })
    if not np.isclose(result["peak_recall_top5"], recalls[0.05], rtol=0.0, atol=1e-12):
        raise AssertionError("top-5% peak recall implementations disagree")
    return result


def _parse_min_intervals(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one min interval is required")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("min intervals must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("min intervals must not contain duplicates")
    return values


def _envelope_metadata(min_interval: int, max_interval: int) -> dict:
    return {
        "min_interval": min_interval,
        "max_interval": max_interval,
        "nominal_min_rate": 1.0 / max_interval,
        "nominal_max_rate": 1.0 / min_interval,
        "interpretation": (
            "Common temporal envelope. Realized total sample ratios may differ "
            "between policies and from reciprocal interval bounds because of "
            "finite-trace startup and block quantization."
        ),
    }


def _run_envelope(
    trace: np.ndarray,
    full_results,
    peak_indices,
    config: PipelineConfig,
    *,
    min_interval: int,
    max_interval: int,
) -> dict:
    if min_interval > max_interval:
        raise ValueError(
            f"min_interval ({min_interval}) must be <= max_interval ({max_interval})"
        )

    envelope = _envelope_metadata(min_interval, max_interval)
    methods: dict[str, dict] = {}

    print("      OmniFlow", flush=True)
    omni_config = replace(
        config,
        min_interval=min_interval,
        max_interval=max_interval,
    )
    omni_results = omni_config._make_poller().track(trace)
    methods["omniflow"] = {
        "parameters": omni_config.to_dict(),
        "report": _report(full_results, trace, omni_results, peak_indices),
    }
    del omni_results

    print("      Huang-inspired", flush=True)
    huang_parameters = {
        **HUANG_PARAMETERS,
        "min_interval": min_interval,
        "max_interval": max_interval,
    }
    huang_results = track_huang_wavelet_rate(
        trace,
        config._make_tracker(),
        **huang_parameters,
    )
    methods["huang_wavelet_rate_inspired"] = {
        "parameters": huang_parameters,
        "report": _report(full_results, trace, huang_results, peak_indices),
    }
    del huang_results

    print("      Magalhaes-inspired", flush=True)
    magalhaes_parameters = {
        **MAGALHAES_PARAMETERS,
        "min_interval": min_interval,
        "max_interval": max_interval,
    }
    magalhaes_results = track_magalhaes_exponential(
        trace,
        config._make_tracker(),
        **magalhaes_parameters,
    )
    methods["magalhaes_truncated_exponential_inspired"] = {
        "parameters": magalhaes_parameters,
        "report": _report(full_results, trace, magalhaes_results, peak_indices),
    }
    del magalhaes_results

    # Two fixed endpoints make the envelope itself visible without being used as
    # adaptive competitors.  When the bounds coincide only one run is needed.
    print("      fixed endpoints", flush=True)
    fixed_reports: dict[str, dict] = {}
    for label, interval in (
        ("slow_boundary", max_interval),
        ("fast_boundary", min_interval),
    ):
        if interval in [item["interval"] for item in fixed_reports.values()]:
            continue
        fixed_config = replace(config, min_interval=interval, max_interval=interval)
        fixed_results = fixed_config._make_poller().track(trace)
        fixed_reports[label] = {
            "interval": interval,
            "report": _report(full_results, trace, fixed_results, peak_indices),
        }
        del fixed_results

    gc.collect()
    return {
        "envelope": envelope,
        "methods": methods,
        "fixed_endpoints": fixed_reports,
    }


def _run_case(
    case: dict,
    trace: np.ndarray,
    config: PipelineConfig,
    *,
    min_intervals: tuple[int, ...],
    max_interval: int,
) -> dict:
    print(f"  dense reference: {len(trace):,} points", flush=True)
    full_results = config._make_tracker().track(trace)
    peak_indices = precompute_peak_indices(
        trace,
        fractions=PEAK_RECALL_FRACTIONS,
    )
    result = {
        "case": case,
        "trace_stats": {
            "n_points": len(trace),
            "mean": float(np.mean(trace)),
            "std": float(np.std(trace)),
            "min": float(np.min(trace)),
            "max": float(np.max(trace)),
        },
        "base_pipeline_config": config.to_dict(),
        "envelopes": [],
    }

    for index, min_interval in enumerate(min_intervals, 1):
        print(
            f"    envelope {index}/{len(min_intervals)}: "
            f"I in [{min_interval}, {max_interval}] "
            f"(~{100 / max_interval:.1f}% to {100 / min_interval:.1f}% nominal)",
            flush=True,
        )
        result["envelopes"].append(
            _run_envelope(
                trace,
                full_results,
                peak_indices,
                config,
                min_interval=min_interval,
                max_interval=max_interval,
            )
        )

    del full_results
    return result


def _write_csv(payload: dict, path: Path) -> None:
    fields = [
        "case",
        "method",
        "min_interval",
        "max_interval",
        "nominal_min_rate",
        "nominal_max_rate",
        *REPORT_METRICS,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_result in payload["cases"]:
            case_name = case_result["case"]["name"]
            for envelope_result in case_result["envelopes"]:
                envelope = envelope_result["envelope"]
                common = {
                    "case": case_name,
                    "min_interval": envelope["min_interval"],
                    "max_interval": envelope["max_interval"],
                    "nominal_min_rate": envelope["nominal_min_rate"],
                    "nominal_max_rate": envelope["nominal_max_rate"],
                }
                for method, method_result in envelope_result["methods"].items():
                    report = method_result["report"]
                    writer.writerow({
                        **common,
                        "method": method,
                        **{metric: report[metric] for metric in REPORT_METRICS},
                    })
                for label, fixed_result in envelope_result["fixed_endpoints"].items():
                    report = fixed_result["report"]
                    writer.writerow({
                        **common,
                        "method": f"fixed_{label}",
                        **{metric: report[metric] for metric in REPORT_METRICS},
                    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudperftrace", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument(
        "--only-case",
        choices=[case["name"] for case in CASES],
        action="append",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-interval",
        type=int,
        default=20,
        help="Slowest allowed interval shared by every adaptive policy (default: 20).",
    )
    parser.add_argument(
        "--min-intervals",
        type=_parse_min_intervals,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated fastest-interval sweep. Default: every integer from "
            "--max-interval down to 1."
        ),
    )
    args = parser.parse_args()

    if args.max_interval < 1:
        parser.error("--max-interval must be positive")
    if args.min_intervals is None:
        args.min_intervals = tuple(range(args.max_interval, 0, -1))
    invalid = [value for value in args.min_intervals if value > args.max_interval]
    if invalid:
        parser.error(
            "every --min-intervals value must be <= --max-interval; "
            f"invalid: {invalid}"
        )

    output_dir = args.output_dir or Path(
        make_run_folder(
            str(ROOT / "out"),
            application_type="cloudperftrace_common_envelope",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        case
        for case in CASES
        if not args.only_case or case["name"] in args.only_case
    ]
    config = PipelineConfig()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cloudperftrace": str(Path(args.cloudperftrace).resolve()),
        "output_dir": str(output_dir.resolve()),
        "experiment": {
            "name": "common_interval_envelope",
            "max_interval": args.max_interval,
            "min_intervals": list(args.min_intervals),
            "policy": (
                "All adaptive policies receive the same interval envelope; "
                "all other parameters remain fixed across traces. Realized "
                "sample ratio is an outcome, not a matched target."
            ),
            "peak_recall_thresholds": [
                int(100 * fraction) for fraction in PEAK_RECALL_FRACTIONS
            ],
        },
        "policy_sources": {
            "omniflow": {
                "fidelity": "native OmniFlow controller with only interval bounds changed",
            },
            "huang_wavelet_rate_inspired": {
                "doi": "10.1109/TCC.2016.2603473",
                "fidelity": (
                    "causal scalar adaptation of the CS-MON rate law; common "
                    "interval bounds clamp only the block sample count"
                ),
                "base_parameters": HUANG_PARAMETERS,
            },
            "magalhaes_truncated_exponential_inspired": {
                "doi": "10.1109/NCA.2011.30",
                "fidelity": (
                    "scalar adaptation of selective truncated-exponential "
                    "profiling; native interval bounds are replaced by the "
                    "common envelope"
                ),
                "base_parameters": MAGALHAES_PARAMETERS,
            },
        },
        "cases": [],
    }

    for index, case in enumerate(selected, 1):
        case_path = output_dir / f"{case['name']}.json"
        print(f"[{index}/{len(selected)}] {case['name']}", flush=True)
        if args.resume and case_path.exists():
            print("  existing result retained", flush=True)
            case_result = json.loads(case_path.read_text())
        else:
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
                min_intervals=args.min_intervals,
                max_interval=args.max_interval,
            )
            case_path.write_text(json.dumps(case_result, indent=2))
            del trace
            gc.collect()

        manifest["cases"].append(case_result)
        (output_dir / "run.json").write_text(json.dumps(manifest, indent=2))
        _write_csv(manifest, output_dir / "results.csv")
        del case_result

    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    _write_csv(manifest, output_dir / "results.csv")
    print(f"Results: {output_dir / 'run.json'}", flush=True)
    print(f"Flat table: {output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
