#!/usr/bin/env python3
"""Evaluate original fixed-schedule runs for all five paper benchmarks.

The public table intentionally reports only the study's current outcomes:
sampling ratio and variable-cycle memory RMSE versus FUM. Throughput, wall-time
overshoot, controller rate, marker count, and paired same-run RMSE are retained
in diagnostic files rather than the paper-facing Markdown table.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from paper_benchmarks import BENCHMARKS, ORDER

PUBLIC_ORDER = ("full", "adp", "inv", "uni", "omni")
ALL_POLICIES = PUBLIC_ORDER + ("nom",)
SAMPLED = set(PUBLIC_ORDER)
DISPLAY = {"full": "FUM", "adp": "ADP", "inv": "INV", "uni": "UNI", "omni": "OmniFlow", "nom": "NOM"}
BENCHMARK_DISPLAY = {"tradebeans": "tradebeans-release*", "xalan": "xalan-release*"}


@dataclass(frozen=True)
class Cycle:
    marker: int
    final: bool
    selected_memory: float
    dense_memory: float | None
    selected_memory_observed: bool


def read_cycles(path: Path) -> list[Cycle]:
    if not path.is_file():
        raise ValueError(f"missing cycle file: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Cycle(
            marker=int(row["marker_second"]),
            final=row["final_cycle"].lower() == "true",
            # The archived evaluation discards negative heap deltas and treats
            # request types absent from the sampled positive-memory map as zero.
            # An entirely empty positive-memory sample is therefore a valid
            # zero estimate, not a missing monitoring-cycle marker.
            selected_memory=float(row["selected_memory_mean_bytes"]) if row["selected_memory_mean_bytes"] else 0.0,
            dense_memory=float(row["dense_memory_mean_bytes"]) if row["dense_memory_mean_bytes"] else None,
            selected_memory_observed=bool(row["selected_memory_mean_bytes"]),
        )
        for row in rows
    ]


def parse_log(path: Path, benchmark: str) -> tuple[int, float, int, int]:
    text = path.read_text(errors="replace")
    def last_int(pattern: str, label: str) -> int:
        values = re.findall(pattern, text)
        if not values:
            raise ValueError(f"{path}: missing {label}")
        return int(values[-1])
    wall = last_int(r"last sec:\s*(\d+)", "last sec")
    operations = last_int(r"Total operations:\s*(\d+)\s*;\s*Sec:\s*\d+", "operation total")
    last_bucket = last_int(r"Total operations:\s*\d+\s*;\s*Sec:\s*(\d+)", "last schedule bucket")
    tr = re.findall(r"TR per second:\s*(\d+(?:\.\d+)?)", text)
    tr_short = re.findall(r"(?m)^TR:\s*(\d+(?:\.\d+)?)\s*$", text)
    if tr:
        throughput = float(tr[-1])
    elif tr_short:
        throughput = float(tr_short[-1])
    else:
        # Throughput is an internal diagnostic. All released harnesses provide
        # an exact final cumulative operation count and wall duration even when
        # they omit an H2-style TR line (Cassandra) or use another spelling.
        throughput = operations / float(wall)
    return wall, throughput, operations, last_bucket


def read_summary_metrics(path: Path) -> tuple[float, int, float]:
    """Return selection ratio, request-type count, and positive heap-delta ratio."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    overall = next(row for row in rows if row["request_type"] == "__overall__")
    typed = [row for row in rows if row["request_type"] != "__overall__"]
    total = int(overall["total_requests"])
    selected = int(overall["selected_requests"])
    positive = sum(int(row["positive_total_count"]) for row in typed)
    return (
        selected / total if total else 0.0,
        len(typed),
        positive / total if total else 0.0,
    )


def read_positive_means(path: Path) -> dict[str, float]:
    """Read FUM positive-memory means by request type for reference stability."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, float] = {}
    for row in rows:
        if row["request_type"] == "__overall__":
            continue
        value = row.get("positive_selected_mean_bytes") or row.get("positive_total_mean_bytes")
        result[row["request_type"]] = float(value) if value else 0.0
    return result


def whole_run_rmse_kb(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return math.nan
    return math.sqrt(statistics.fmean(
        (left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys
    )) / 1024.0


def read_telemetry(path: Path, policy: str, horizon: int) -> tuple[float, int, int, int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if policy == "full":
        rate = 1.0
    else:
        values: list[float] = []
        for row in rows:
            value = row["next_rate"] if policy in {"adp", "inv", "uni"} else row["observed_sampling_ratio"]
            if value:
                number = float(value)
                if math.isfinite(number):
                    values.append(number)
        rate = statistics.fmean(values) if values else math.nan
    post = [row for row in rows if int(row["second"]) > horizon]
    return (
        rate,
        sum(int(row["agent_requests_per_second"]) for row in post),
        sum(int(row["selected_requests_per_second"]) for row in post),
        sum(int(row["top_level_operations_per_second"]) for row in post),
    )


def rmse_kb(errors: list[float]) -> float:
    return math.sqrt(statistics.fmean(error * error for error in errors)) / 1024.0


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def fmt(value: dict[str, float], *, percent: bool = False) -> str:
    return f"{value['mean']:.1%} ± {value['stdev']:.1%}" if percent else f"{value['mean']:.1f} ± {value['stdev']:.1f}"


def render_compact_summary(summary: dict[str, object]) -> str:
    """Render the paper-facing Markdown table from archived compact results."""
    results = summary.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("compact paper summary has no result rows")

    lines = [
        "| Benchmark | Policy | Reps | Sampling ratio | RMSE vs FUM (KB) |",
        "|---|---|---:|---:|---:|",
    ]
    benchmarks = set()
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("compact paper summary contains an invalid result row")
        benchmark = str(row["benchmark"])
        policy = str(row["policy"])
        if policy not in DISPLAY:
            raise ValueError(f"unknown paper policy in compact summary: {policy}")
        benchmarks.add(benchmark)
        benchmark_display = str(
            row.get("benchmark_display") or BENCHMARK_DISPLAY.get(benchmark, benchmark)
        )
        lines.append(
            f"| {benchmark_display} | {DISPLAY[policy]} | {int(row['reps'])} | "
            f"{fmt(row['selection_ratio'], percent=True)} | "
            f"{fmt(row['paper_cycle_rmse_kb'])} |"
        )

    if "tradebeans" in benchmarks:
        lines.extend([
            "",
            "* `tradebeans-release` is the exact released companion-artifact arm. "
            "It is TPCC-backed and is not claimed to be the paper's described DayTrader workload.",
        ])
    if "xalan" in benchmarks:
        lines.extend([
            "",
            "* `xalan-release` uses the XML input filename as the stable request identity. "
            "The released source queues 17 inputs; the paper reports 16 Xalan request types.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument(
        "--compact-summary-json",
        type=Path,
        help="Render the paper table from an archived compact paper-summary.json.",
    )
    parser.add_argument(
        "--markdown-out", type=Path, help="Write the compact table to this Markdown file."
    )
    parser.add_argument("--benchmarks", default="all", help="Comma-separated subset or all")
    parser.add_argument("--details", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--diagnostics-md", type=Path)
    args = parser.parse_args()

    if args.compact_summary_json:
        if args.root is not None:
            parser.error("do not combine a run root with --compact-summary-json")
        summary = json.loads(args.compact_summary_json.read_text(encoding="utf-8"))
        markdown = render_compact_summary(summary)
        if args.markdown_out:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(markdown, encoding="utf-8")
            print(args.markdown_out)
        else:
            print(markdown, end="")
        return
    if args.root is None:
        parser.error("provide a DaCapo run root or --compact-summary-json")

    requested = list(ORDER) if args.benchmarks.strip().lower() == "all" else [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(BENCHMARKS))
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}")

    details: list[dict[str, object]] = []
    fum_stability: list[dict[str, object]] = []
    for benchmark in ORDER:
        if benchmark not in requested:
            continue
        cfg = BENCHMARKS[benchmark]
        rep_dirs = sorted((args.root / benchmark).glob("rep-*"))
        if not rep_dirs:
            raise SystemExit(f"No repetitions found for {benchmark}")
        for rep_dir in rep_dirs:
            adp_cycles = read_cycles(rep_dir / "adp.csv.cycles.csv")
            markers = [c.marker for c in adp_cycles if not c.final]
            if not markers:
                raise SystemExit(f"{rep_dir}: no evaluable ADP markers")
            full = {c.marker: c for c in read_cycles(rep_dir / "full.csv.cycles.csv")}
            missing_full = [m for m in markers if m not in full]
            if missing_full:
                raise SystemExit(f"{rep_dir}: FUM missing ADP markers: {missing_full[:10]}")

            for policy in ALL_POLICIES:
                log = rep_dir / f"{policy}.log"
                if not log.is_file():
                    if policy == "nom":
                        continue
                    raise SystemExit(f"{rep_dir}: missing required policy {policy}")
                wall_seconds, throughput, operations, last_bucket = parse_log(log, benchmark)
                row: dict[str, object] = {
                    "benchmark": benchmark,
                    "benchmark_display": BENCHMARK_DISPLAY.get(benchmark, benchmark),
                    "artifact_mode": (
                        "released-tpcc-substitute" if benchmark == "tradebeans" else
                        "released-source-17-inputs" if benchmark == "xalan" else
                        "paper-runtime"
                    ),
                    "rep": rep_dir.name,
                    "policy": policy,
                    "nominal_seconds": cfg.seconds,
                    "wall_seconds": wall_seconds,
                    "overshoot_seconds": max(0, wall_seconds - cfg.seconds),
                    "last_schedule_bucket": last_bucket,
                    "throughput": throughput,
                    "operations": operations,
                    "selection_ratio": None,
                    "request_types": None,
                    "positive_heap_delta_ratio": None,
                    "controller_rate_mean": None,
                    "markers": len(markers),
                    "post_horizon_requests": 0,
                    "post_horizon_selected": 0,
                    "post_horizon_operations": 0,
                    "paper_cycle_rmse_kb": None,
                    "paired_cycle_rmse_kb": None,
                    "cross_to_paired_rmse_ratio": None,
                    "zero_positive_memory_cycles": 0,
                }
                if policy in SAMPLED:
                    ratio, request_types, positive_ratio = read_summary_metrics(
                        rep_dir / f"{policy}.csv.summary.csv"
                    )
                    row["selection_ratio"] = ratio
                    row["request_types"] = request_types
                    row["positive_heap_delta_ratio"] = positive_ratio
                    rate, late_req, late_sel, late_ops = read_telemetry(
                        rep_dir / f"{policy}.csv.telemetry.csv", policy, cfg.seconds
                    )
                    row["controller_rate_mean"] = rate
                    row["post_horizon_requests"] = late_req
                    row["post_horizon_selected"] = late_sel
                    row["post_horizon_operations"] = late_ops
                if policy == "full":
                    row["paper_cycle_rmse_kb"] = 0.0
                    row["paired_cycle_rmse_kb"] = 0.0
                elif policy in SAMPLED:
                    cycles = {c.marker: c for c in read_cycles(rep_dir / f"{policy}.csv.cycles.csv")}
                    missing = [m for m in markers if m not in cycles]
                    if missing:
                        raise SystemExit(f"{rep_dir}/{policy}: missing ADP markers {missing[:10]}")
                    row["zero_positive_memory_cycles"] = sum(
                        1 for m in markers if not cycles[m].selected_memory_observed
                    )
                    cross = [cycles[m].selected_memory - full[m].selected_memory for m in markers]
                    paired = [
                        cycles[m].selected_memory - cycles[m].dense_memory
                        for m in markers if cycles[m].dense_memory is not None
                    ]
                    row["paper_cycle_rmse_kb"] = rmse_kb(cross)
                    row["paired_cycle_rmse_kb"] = rmse_kb(paired)
                    paired_rmse = float(row["paired_cycle_rmse_kb"])
                    cross_rmse = float(row["paper_cycle_rmse_kb"])
                    row["cross_to_paired_rmse_ratio"] = (
                        cross_rmse / paired_rmse if paired_rmse > 0.0 else None
                    )
                details.append(row)

        fum_summaries = [rep_dir / "full.csv.summary.csv" for rep_dir in rep_dirs]
        if len(fum_summaries) > 1 and all(path.is_file() for path in fum_summaries):
            fum_maps = [read_positive_means(path) for path in fum_summaries]
            pair_values = [whole_run_rmse_kb(left, right)
                           for left, right in itertools.combinations(fum_maps, 2)]
            type_sets = [set(values) for values in fum_maps]
            fum_stability.append({
                "benchmark": benchmark,
                "runs": len(fum_maps),
                "pairs": len(pair_values),
                "type_sets_exact": all(value == type_sets[0] for value in type_sets[1:]),
                "pairwise_mean_rmse_kb": statistics.fmean(pair_values),
                "pairwise_stdev_rmse_kb": statistics.stdev(pair_values) if len(pair_values) > 1 else 0.0,
                "pairwise_median_rmse_kb": statistics.median(pair_values),
                "pairwise_max_rmse_kb": max(pair_values),
            })

    if not details:
        raise SystemExit("No complete publication repetitions found")

    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        fields = list(details[0].keys())
        with args.details.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(details)

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in details:
        groups.setdefault((str(row["benchmark"]), str(row["policy"])), []).append(row)

    print("| Benchmark | Policy | Reps | Sampling ratio | RMSE vs FUM (KB) |")
    print("|---|---|---:|---:|---:|")
    methodology_notes = [
        "after negative heap deltas are discarded, a sampled cycle with no remaining positive-memory observations is scored as a zero-memory estimate rather than treated as a missing marker",
    ]
    if "tradebeans" in requested:
        methodology_notes.append(
            "tradebeans-release is the exact TPCC-backed companion-artifact arm and is not claimed to be DayTrader"
        )
    if "xalan" in requested:
        methodology_notes.append(
            "xalan-release uses the stable XML input filename as request identity; the released source queues 17 inputs although the paper reports 16 types"
        )
    output: dict[str, object] = {
        "mode": "original-fixed-schedule-suite",
        "results": [],
        "diagnostic_warnings": [],
        "fum_reference_stability": fum_stability,
        "methodology_notes": methodology_notes,
    }
    for benchmark in ORDER:
        if benchmark not in requested:
            continue
        for policy in PUBLIC_ORDER:
            rows = groups.get((benchmark, policy), [])
            if not rows:
                continue
            ratio = stats([float(r["selection_ratio"]) for r in rows])
            rmse = stats([float(r["paper_cycle_rmse_kb"]) for r in rows])
            print(f"| {BENCHMARK_DISPLAY.get(benchmark, benchmark)} | {DISPLAY[policy]} | {len(rows)} | {fmt(ratio, percent=True)} | {fmt(rmse)} |")
            output["results"].append({
                "benchmark": benchmark,
                "benchmark_display": BENCHMARK_DISPLAY.get(benchmark, benchmark),
                "artifact_mode": (
                        "released-tpcc-substitute" if benchmark == "tradebeans" else
                        "released-source-17-inputs" if benchmark == "xalan" else
                        "paper-runtime"
                    ),
                "policy": policy,
                "reps": len(rows),
                "selection_ratio": ratio,
                "paper_cycle_rmse_kb": rmse,
            })
    if "tradebeans" in requested:
        print()
        print("* `tradebeans-release` is the exact released companion-artifact arm. "
              "It is TPCC-backed and is not claimed to be the paper's described DayTrader workload.")
    if "xalan" in requested:
        print()
        print("* `xalan-release` uses the XML input filename as the stable request identity. "
              "The released source queues 17 inputs; the paper reports 16 Xalan request types.")

    diagnostic_lines = [
        "# Internal Diagnostics", "",
        "These fields are retained for experiment validation and are not part of the paper-facing table.", "",
        "| Benchmark | Policy | Reps | Throughput (req/s) | Wall time (s) | Overshoot (s) | Request types | Positive heap deltas | Mean policy rate | ADP markers | Zero-positive cycles | Paired RMSE (KB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ORDER:
        if benchmark not in requested:
            continue
        for policy in ALL_POLICIES:
            rows = groups.get((benchmark, policy), [])
            if not rows:
                continue
            throughput = stats([float(r["throughput"]) for r in rows])
            wall = stats([float(r["wall_seconds"]) for r in rows])
            overshoot = stats([float(r["overshoot_seconds"]) for r in rows])
            rate_values = [float(r["controller_rate_mean"]) for r in rows if r["controller_rate_mean"] is not None]
            paired_values = [float(r["paired_cycle_rmse_kb"]) for r in rows if r["paired_cycle_rmse_kb"] is not None]
            rate = stats(rate_values) if rate_values else None
            paired = stats(paired_values) if paired_values else None
            marker = stats([float(r["markers"]) for r in rows])
            zero_positive = sum(int(r["zero_positive_memory_cycles"]) for r in rows)
            type_values = [float(r["request_types"]) for r in rows if r["request_types"] is not None]
            positive_values = [float(r["positive_heap_delta_ratio"]) for r in rows if r["positive_heap_delta_ratio"] is not None]
            request_types = stats(type_values) if type_values else None
            positive_ratio = stats(positive_values) if positive_values else None
            request_types_text = f"{request_types['mean']:.1f}" if request_types else "—"
            diagnostic_lines.append(
                f"| {BENCHMARK_DISPLAY.get(benchmark, benchmark)} | {DISPLAY[policy]} | {len(rows)} | {fmt(throughput)} | {fmt(wall)} | {fmt(overshoot)} | "
                f"{request_types_text} | {fmt(positive_ratio, percent=True) if positive_ratio else '—'} | "
                f"{fmt(rate, percent=True) if rate else '—'} | {marker['mean']:.1f} | {zero_positive} | {fmt(paired) if paired else '—'} |"
            )
            late_req = sum(int(r["post_horizon_requests"]) for r in rows)
            late_ops = sum(int(r["post_horizon_operations"]) for r in rows)
            if overshoot["mean"] > 0 or late_req > 0:
                output["diagnostic_warnings"].append({
                    "benchmark": benchmark, "policy": policy,
                    "mean_overshoot_seconds": overshoot["mean"],
                    "post_horizon_requests": late_req,
                    "post_horizon_operations": late_ops,
                })
            selection_values = [float(r["selection_ratio"]) for r in rows if r["selection_ratio"] is not None]
            if policy == "omni" and selection_values and statistics.fmean(selection_values) >= 0.95:
                output["diagnostic_warnings"].append({
                    "kind": "omniflow-dense-convergence",
                    "benchmark": benchmark,
                    "policy": policy,
                    "mean_selection_ratio": statistics.fmean(selection_values),
                    "mean_request_types": request_types["mean"] if request_types else None,
                    "mean_positive_heap_delta_ratio": positive_ratio["mean"] if positive_ratio else None,
                })
            cross_ratios = [
                float(r["cross_to_paired_rmse_ratio"])
                for r in rows if r["cross_to_paired_rmse_ratio"] is not None
            ]
            if policy != "full" and cross_ratios and statistics.fmean(cross_ratios) >= 4.0:
                output["diagnostic_warnings"].append({
                    "kind": "cross-run-rmse-dominates-paired",
                    "benchmark": benchmark,
                    "policy": policy,
                    "mean_cross_to_paired_rmse_ratio": statistics.fmean(cross_ratios),
                })

    dense = [w for w in output["diagnostic_warnings"] if w.get("kind") == "omniflow-dense-convergence"]
    gaps = [w for w in output["diagnostic_warnings"] if w.get("kind") == "cross-run-rmse-dominates-paired"]
    if dense:
        diagnostic_lines.extend(["", "## Dense OmniFlow convergence", ""])
        for warning in dense:
            diagnostic_lines.append(
                f"- {BENCHMARK_DISPLAY.get(str(warning['benchmark']), str(warning['benchmark']))}: "
                f"OmniFlow selected {float(warning['mean_selection_ratio']):.1%}; "
                f"request types={float(warning['mean_request_types']):.0f}, "
                f"positive heap deltas={float(warning['mean_positive_heap_delta_ratio']):.2%}."
            )
    if gaps:
        diagnostic_lines.extend(["", "## Cross-run versus paired RMSE", ""])
        diagnostic_lines.append(
            "The following policies have cross-run FUM RMSE at least four times their paired same-run RMSE. "
            "This indicates that between-execution heap variability materially contributes to the paper-style metric."
        )
        for warning in gaps:
            diagnostic_lines.append(
                f"- {BENCHMARK_DISPLAY.get(str(warning['benchmark']), str(warning['benchmark']))} "
                f"{DISPLAY[str(warning['policy'])]}: {float(warning['mean_cross_to_paired_rmse_ratio']):.1f}×."
            )

    if fum_stability:
        diagnostic_lines.extend(["", "## Independent FUM reproducibility", "",
            "This table estimates the between-JVM reference-noise floor using all pairwise comparisons among FUM repetitions. "
            "It is diagnostic and is not subtracted from the paper-compatible RMSE.", "",
            "| Benchmark | FUM runs | Pairwise RMSE mean ± SD (KB) | Median (KB) | Maximum (KB) | Exact type sets |",
            "|---|---:|---:|---:|---:|---:|"])
        for item in fum_stability:
            diagnostic_lines.append(
                f"| {BENCHMARK_DISPLAY.get(str(item['benchmark']), str(item['benchmark']))} | {item['runs']} | "
                f"{float(item['pairwise_mean_rmse_kb']):.1f} ± {float(item['pairwise_stdev_rmse_kb']):.1f} | "
                f"{float(item['pairwise_median_rmse_kb']):.1f} | {float(item['pairwise_max_rmse_kb']):.1f} | "
                f"{'yes' if item['type_sets_exact'] else 'no'} |"
            )

    if args.diagnostics_md:
        args.diagnostics_md.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_md.write_text("\n".join(diagnostic_lines) + "\n")
    if args.json:
        output["details"] = details
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
