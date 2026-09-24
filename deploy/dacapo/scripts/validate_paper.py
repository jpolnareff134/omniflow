#!/usr/bin/env python3
"""Strict validator for one original-artifact publication-policy run."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from paper_benchmarks import BENCHMARKS, MONITORED_POLICIES


def fail(message: str) -> None:
    raise SystemExit(f"invalid: {message}")


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty file: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"CSV contains no rows: {path}")
    return rows


def parse_last(pattern: str, text: str, label: str) -> int:
    values = re.findall(pattern, text)
    if not values:
        fail(f"{label} not found")
    return int(values[-1])


def validate_types(benchmark: str, count: int) -> None:
    config = BENCHMARKS[benchmark]
    if config.type_rule == "exact" and count != config.type_count:
        fail(f"expected exactly {config.type_count} {benchmark} request types, found {count}")
    if config.type_rule == "min" and count < config.type_count:
        fail(f"expected at least {config.type_count} {benchmark} request types, found {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--policy", required=True, choices=sorted(MONITORED_POLICIES | {"nom"}))
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--cycles", type=Path)
    parser.add_argument("--markers", type=Path)
    parser.add_argument("--reference-markers", type=Path)
    parser.add_argument("--uniform-rate", type=float, default=0.5)
    args = parser.parse_args()

    cfg = BENCHMARKS[args.benchmark]
    if not args.log.is_file():
        fail(f"missing log: {args.log}")
    text = args.log.read_text(errors="replace")
    passed = re.search(rf"===== DaCapo .*\b{re.escape(args.benchmark)}\b PASSED", text)
    failed = re.search(rf"===== DaCapo .*\b{re.escape(args.benchmark)}\b FAILED", text)
    if not passed:
        fail(f"DaCapo PASS marker not found for {args.benchmark}")
    if failed or "Validation FAILED" in text:
        fail("DaCapo failure marker found")

    wall_seconds = parse_last(r"last sec:\s*(\d+)", text, "last sec")
    if wall_seconds < cfg.seconds:
        fail(f"workload ended before nominal horizon {cfg.seconds}: {wall_seconds}")
    overshoot = wall_seconds - cfg.seconds
    total_ops = parse_last(
        r"Total operations:\s*(\d+)\s*;\s*Sec:\s*\d+", text, "final total operations"
    )
    throughput_values = re.findall(r"TR per second:\s*(\d+(?:\.\d+)?)", text)
    short_throughput_values = re.findall(r"(?m)^TR:\s*(\d+(?:\.\d+)?)\s*$", text)
    if throughput_values:
        throughput = float(throughput_values[-1])
        throughput_source = "TR per second"
    elif short_throughput_values:
        # Lusearch and Xalan in the released artifact print their final average
        # as `TR: N`, not the H2/Tradebeans `TR per second: N` spelling.
        throughput = float(short_throughput_values[-1])
        throughput_source = "TR"
    else:
        # Cassandra emits neither spelling. The exact cumulative operation count
        # and fixed-schedule wall duration are still available for every arm, so
        # this is also a safe compatibility fallback for a released harness that
        # omits its final TR line. Throughput is diagnostic only.
        throughput = total_ops / float(wall_seconds)
        throughput_source = "operations/wall_seconds"
    last_bucket = parse_last(
        r"Total operations:\s*\d+\s*;\s*Sec:\s*(\d+)", text, "last reported schedule bucket"
    )
    if last_bucket > cfg.seconds:
        fail(f"schedule continued beyond nominal horizon: {last_bucket} > {cfg.seconds}")
    if args.benchmark == "xalan":
        fatal_xalan_patterns = (
            "dist/dat/xalan/xmlspec.xsl (No such file or directory)",
            "FileNotFoundException: benchmarks/bms/xalan/dist/dat/xalan/xmlspec.xsl",
            "XalanWorker.run",
        )
        found = [pattern for pattern in fatal_xalan_patterns if pattern in text]
        if found:
            fail("Xalan workload emitted missing-data/worker errors: " + ", ".join(found))
    if throughput <= 0 or total_ops <= 0:
        fail("non-positive throughput or operation total")
    if args.benchmark == "cassandra":
        fatal_cassandra_patterns = (
            "Undefined column name",
            "InvalidQueryException",
            "Error inserting key",
            "Error reading key",
        )
        found = [pattern for pattern in fatal_cassandra_patterns if pattern in text]
        if found:
            fail("Cassandra workload emitted query/schema errors: " + ", ".join(found))

    status = "complete-paper" if overshoot == 0 else "complete-paper-wall-overshoot"
    if args.policy == "nom":
        if "[OmniFlowAgent]" in text:
            fail("NOM unexpectedly loaded the agent")
        print(
            f"status={status} benchmark={args.benchmark} policy=nom "
            f"nominal_seconds={cfg.seconds} wall_seconds={wall_seconds} "
            f"overshoot_seconds={overshoot} last_schedule_bucket={last_bucket} "
            f"operations={total_ops} throughput={throughput:.3f} throughput_source={throughput_source!r}"
        )
        return

    for path, name in ((args.summary, "summary"), (args.telemetry, "telemetry"),
                       (args.cycles, "cycles"), (args.markers, "markers")):
        if path is None:
            fail(f"--{name} is required for {args.policy}")

    new_hook = "[OmniFlowAgent] exact benchmark per-second hook enabled"
    legacy_hook = "[OmniFlowAgent] H2 exact per-second hook enabled"
    if new_hook not in text and legacy_hook not in text:
        fail("exact benchmark per-second hook marker not found")
    if "[OmniFlowAgent] H2 top-level throughput hook enabled" in text:
        fail("wall-clock throughput hook was enabled in paper mode")
    if "transform failed" in text:
        fail("agent transformation failure found")

    summary = read_csv(args.summary)
    typed = [row for row in summary if row.get("request_type") != "__overall__"]
    overall = next((row for row in summary if row.get("request_type") == "__overall__"), None)
    if overall is None:
        fail("summary overall row missing")
    validate_types(args.benchmark, len(typed))
    total = int(overall["total_requests"])
    selected = int(overall["selected_requests"])
    if total <= 0 or not 0 <= selected <= total:
        fail("invalid agent request totals")
    if args.policy == "full" and selected != total:
        fail("FUM is not dense")
    if args.policy == "uni":
        ratio = selected / total
        tolerance = max(0.01, 6.0 * math.sqrt(args.uniform_rate * (1.0 - args.uniform_rate) / total))
        if abs(ratio - args.uniform_rate) > tolerance:
            fail(f"UNI ratio {ratio:.6f} is outside tolerance around {args.uniform_rate}")

    telemetry = read_csv(args.telemetry)
    seconds_seen = [int(row["second"]) for row in telemetry]
    if min(seconds_seen) != 0 or max(seconds_seen) < cfg.seconds - 1:
        fail(f"telemetry does not span the fixed workload: {min(seconds_seen)}..{max(seconds_seen)}")
    telemetry_ops = sum(int(row["top_level_operations_per_second"]) for row in telemetry)
    if telemetry_ops != total_ops:
        fail(f"exact operation total mismatch: telemetry={telemetry_ops}, log={total_ops}")
    if int(telemetry[-1]["population_size"]) != total:
        fail("telemetry population differs from summary")
    if int(telemetry[-1]["sample_size"]) != selected:
        fail("telemetry sample differs from summary")

    post_rows = [row for row in telemetry if int(row["second"]) > cfg.seconds]
    post_ops = sum(int(row["top_level_operations_per_second"]) for row in post_rows)
    post_requests = sum(int(row["agent_requests_per_second"]) for row in post_rows)
    post_selected = sum(int(row["selected_requests_per_second"]) for row in post_rows)
    if post_ops != 0:
        fail(f"top-level operations were recorded after nominal horizon: {post_ops}")
    max_late = max(100, math.ceil(total * 0.0001))
    if post_requests > max_late:
        fail(f"too many post-horizon request completions: {post_requests} > {max_late}")

    cycles = read_csv(args.cycles)
    marker_lines = [int(line.strip()) for line in args.markers.read_text().splitlines() if line.strip()]
    cycle_markers = [int(row["marker_second"]) for row in cycles]
    if marker_lines != cycle_markers:
        fail("markers file differs from cycle CSV")
    if not marker_lines or marker_lines[-1] != cfg.seconds:
        fail(f"final marker must be {cfg.seconds}")
    if args.policy == "adp":
        nonfinal = [row for row in cycles if row["final_cycle"].lower() != "true"]
        if not nonfinal:
            fail("ADP emitted no completed monitoring cycle")
    elif args.reference_markers:
        expected = [int(line.strip()) for line in args.reference_markers.read_text().splitlines() if line.strip()]
        if marker_lines != expected:
            missing = [marker for marker in expected if marker not in set(marker_lines)]
            extra = [marker for marker in marker_lines if marker not in set(expected)]
            fail(
                "fixed policy marker set differs from ADP marker set "
                f"(missing={missing[:10]}, extra={extra[:10]})"
            )
        for index, row in enumerate(cycles):
            expected_readiness = "FINAL_MARKER_REBUILT" if index == len(cycles) - 1 else "FIXED_MARKER_REBUILT"
            if row.get("readiness") != expected_readiness:
                fail(
                    "fixed policy cycles were not rebuilt by the request-second segmenter: "
                    f"marker={row.get('marker_second')} readiness={row.get('readiness')!r}"
                )

    print(
        f"status={status} benchmark={args.benchmark} policy={args.policy} "
        f"nominal_seconds={cfg.seconds} wall_seconds={wall_seconds} overshoot_seconds={overshoot} "
        f"last_schedule_bucket={last_bucket} operations={total_ops} throughput={throughput:.3f} throughput_source={throughput_source!r} "
        f"agent_requests={total} selected={selected} ratio={selected / total:.6f} "
        f"types={len(typed)} markers={len(marker_lines)} post_horizon_requests={post_requests} "
        f"post_horizon_selected={post_selected}"
    )


if __name__ == "__main__":
    main()
