#!/usr/bin/env python3
"""Post-hoc analysis for the HPA autoscaling experiment.

Reads the log files collected by run-experiment.sh and produces a
unified analysis.json comparing baseline (metrics-server) vs OmniFlow
(custom metric) HPA performance.

Metrics computed:
  - Per-pod OmniFlow info-loss (from daemon logs)
  - Scaling event timeline (from K8s events)
  - Replica timeline analysis (from 5s polling snapshots)
  - Scaling reaction latency (time from load phase change -> first replica change)
  - Missed / delayed scale-ups and scale-downs
  - Flapping detection (scale-up -> rapid scale-down)

Usage:
    python collect-results.py --data-dir data/results --out data/results/analysis.json
"""

from __future__ import annotations

import argparse
import gzip
import glob
import json
import math
import os
import re
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
# Allow importing from src/
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "src"))

import numpy as np  # noqa: E402
from load_schedule import phase_boundaries  # noqa: E402
from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import evaluate_info_loss  # noqa: E402

MIN_CPU_COVERAGE = 0.95

# ------------------------------
# Load phase oracle - shape-specific
# ------------------------------

# Each phase: (name, duration_s, expected_direction)
#   "up"    = load is increasing, replicas should scale up
#   "stable"= load is constant, replicas should hold
#   "down"  = load is decreasing, replicas should scale down

SHAPE_PHASES = {
    "phased": [
        {"name": "warmup", "duration": 60, "direction": "stable"},
        {"name": "ramp", "duration": 120, "direction": "up"},
        {"name": "sustained", "duration": 60, "direction": "stable"},
        {"name": "spike", "duration": 90, "direction": "up"},
        {"name": "cooldown", "duration": 30, "direction": "down"},
        {"name": "pulse", "duration": 60, "direction": "up"},
        {"name": "cooldown2", "duration": 30, "direction": "down"},
        {"name": "tail", "duration": 60, "direction": "stable"},
    ],
    "staircase": [
        {"name": "baseline", "duration": 60, "direction": "stable"},
        {"name": "step1", "duration": 60, "direction": "up"},
        {"name": "step2", "duration": 60, "direction": "up"},
        {"name": "step3", "duration": 60, "direction": "up"},
        {"name": "step4", "duration": 60, "direction": "up"},
        {"name": "step5", "duration": 60, "direction": "up"},
        {"name": "sustained_peak", "duration": 90, "direction": "stable"},
        {"name": "hard_drop", "duration": 60, "direction": "down"},
    ],
    "oscillating": [
        {"name": "warmup", "duration": 60, "direction": "stable"},
        {"name": "spike1", "duration": 30, "direction": "up"},
        {"name": "valley1", "duration": 30, "direction": "down"},
        {"name": "spike2", "duration": 30, "direction": "up"},
        {"name": "valley2", "duration": 30, "direction": "down"},
        {"name": "spike3", "duration": 30, "direction": "up"},
        {"name": "valley3", "duration": 30, "direction": "down"},
        {"name": "spike4", "duration": 30, "direction": "up"},
        {"name": "valley4", "duration": 30, "direction": "down"},
        {"name": "tail", "duration": 60, "direction": "stable"},
    ],
    "flash_crowd": [
        {"name": "warmup", "duration": 30, "direction": "stable"},
        {"name": "flash1", "duration": 120, "direction": "up"},
        {"name": "drop1", "duration": 60, "direction": "down"},
        {"name": "rest", "duration": 30, "direction": "stable"},
        {"name": "flash2", "duration": 120, "direction": "up"},
        {"name": "drop2", "duration": 60, "direction": "down"},
        {"name": "tail", "duration": 30, "direction": "stable"},
    ],
}

# Fallback for unknown shapes
DEFAULT_PHASES = [
    {"name": "warmup", "duration": 60, "direction": "stable"},
    {"name": "ramp", "duration": 120, "direction": "up"},
    {"name": "sustained", "duration": 60, "direction": "stable"},
    {"name": "spike", "duration": 30, "direction": "up"},
    {"name": "cooldown", "duration": 30, "direction": "down"},
    {"name": "tail", "duration": 60, "direction": "down"},
]


def _get_phases_for_shape(shape: str) -> list[dict]:
    """Return the load phase oracle for the given shape."""
    return SHAPE_PHASES.get(shape, DEFAULT_PHASES)


def _phase_boundaries(shape: str = "phased") -> list[dict]:
    """Return phase boundaries as cumulative offsets for the given shape."""
    try:
        return phase_boundaries(shape)
    except ValueError:
        phases = _get_phases_for_shape(shape)
        result = []
        t = 0
        for phase in phases:
            result.append({
                "name": phase["name"],
                "start_s": t,
                "end_s": t + phase["duration"],
                "direction": phase["direction"],
            })
            t += phase["duration"]
        return result


# ------------------------------
# Parsing
# ------------------------------

def _load_jsonl(path: str) -> list[dict]:
    records = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


# ------------------------------
# Daemon trace analysis
# ------------------------------

def _load_daemon_config(data_dir: str, label: str) -> PipelineConfig:
    """Extract PipelineConfig from the daemon_start event in logs.

    The daemon emits a JSON line with event=daemon_start and a 'config'
    dict containing the full resolved PipelineConfig (all OMNIFLOW_* env
    vars already applied inside the container).  Falling back to
    PipelineConfig() would silently use host-side defaults which differ
    from the container's env vars.
    """
    pattern = os.path.join(data_dir, f"{label}_daemon_*.jsonl*")
    for path in sorted(glob.glob(pattern)):
        for record in _load_jsonl(path):
            if record.get("event") == "daemon_start" and "config" in record:
                try:
                    return PipelineConfig(**record["config"])
                except TypeError:
                    pass
    print(f"  WARNING: no daemon_start event found for '{label}'; "
          "using default PipelineConfig (env vars may not match container)",
          file=sys.stderr)
    return PipelineConfig()


def _load_tracker_offset(data_dir: str, label: str) -> float:
    """Read the Kubernetes-only tracker input offset from daemon metadata."""
    pattern = os.path.join(data_dir, f"{label}_daemon_*.jsonl*")
    for path in sorted(glob.glob(pattern)):
        for record in _load_jsonl(path):
            if record.get("event") == "daemon_start":
                return float(record.get("cpu_tracker_offset", 0.0))
    return 0.0


def _analyse_daemon_logs(data_dir: str, label: str,
                          config: PipelineConfig | None = None,
                          namespace: str | None = None,
                          load_events: list[dict] | None = None,
                          metadata: dict | None = None) -> list[dict]:
    """Parse daemon JSONL logs and run evaluate_info_loss per pod."""
    pattern = os.path.join(data_dir, f"{label}_daemon_*.jsonl*")
    results = []
    if config is None:
        config = _load_daemon_config(data_dir, label)
    tracker_offset = _load_tracker_offset(data_dir, label)
    load_start = next(
        (event for event in (load_events or []) if event.get("event") == "load_start"),
        None,
    )
    load_end = next(
        (event for event in (load_events or []) if event.get("event") == "load_end"),
        None,
    )
    guest_start_ns = (metadata or {}).get("locust_start_guest_ns")
    start_wall = (
        float(guest_start_ns) / 1e9
        if guest_start_ns is not None
        else float(load_start["observed_epoch_ns"]) / 1e9 if load_start else None
    )
    end_wall = (
        start_wall + float(load_end["observed_offset_s"])
        if start_wall is not None and load_end else None
    )

    for path in sorted(glob.glob(pattern)):
        records = _load_jsonl(path)
        nginx_cgroups = {
            record.get("pod")
            for record in records
            if record.get("event") == "track_start"
            and "nginx" in record.get("pod_name", "")
            and (namespace is None or record.get("namespace") == namespace)
        }
        for pod_name in sorted(nginx_cgroups):
            readings = [
                record for record in records
                if "value" in record
                and "event" not in record
                and record.get("pod") == pod_name
                and (start_wall is None or float(record.get("wall", 0)) >= start_wall)
                and (end_wall is None or float(record.get("wall", 0)) <= end_wall)
            ]
            if not readings:
                continue

            values = [record["value"] for record in readings]
            trace = np.array(values, dtype=np.float64) + tracker_offset
            info_loss = evaluate_info_loss(
                trace,
                full_tracker=config._make_tracker(),
                poller=config._make_poller(),
            )
            n_sampled = sum(bool(record.get("sampled")) for record in readings)
            results.append({
                "pod": pod_name,
                "n_readings": len(values),
                "n_sampled": n_sampled,
                "sample_ratio": n_sampled / len(readings),
                "info_loss": asdict(info_loss),
            })

    return results


# ------------------------------
# Scaling events
# ------------------------------

def _parse_scale_events(data_dir: str, label: str) -> list[dict]:
    """Extract scaling events from K8s events JSON."""
    path = os.path.join(data_dir, f"{label}_scale_events.json")
    if not os.path.exists(path):
        return []

    raw = _load_json(path)
    items = raw.get("items", []) if isinstance(raw, dict) else raw
    events = []
    for item in items:
        events.append({
            "reason": item.get("reason", ""),
            "message": item.get("message", ""),
            "timestamp": item.get("lastTimestamp", ""),
        })
    return events


# ------------------------------
# Replica timeline analysis
# ------------------------------

def _load_replica_timeline(
    data_dir: str, label: str, metadata: dict | None = None
) -> list[dict]:
    """Load the replica timeline and align it to the host release handshake."""
    path = os.path.join(data_dir, f"{label}_replica_timeline.jsonl")
    if not os.path.exists(path):
        return []
    timeline = _load_jsonl(path)
    origin_ns = (metadata or {}).get("locust_start_host_midpoint_ns")
    if origin_ns is None:
        origin_ns = (metadata or {}).get("release_midpoint_ns")
    if origin_ns is None:
        return timeline
    origin_ns = int(origin_ns)
    aligned = []
    for item in timeline:
        current = dict(item)
        if "wall_ns" in current:
            current["elapsed_s"] = (int(current["wall_ns"]) - origin_ns) / 1e9
        aligned.append(current)
    return aligned


def _parse_load_events(data_dir: str, label: str) -> list[dict]:
    path = os.path.join(data_dir, f"{label}_locust.log")
    if not os.path.exists(path):
        return []
    return [
        record for record in _load_jsonl(path)
        if record.get("event") in {"load_armed", "load_start", "phase_start", "phase_end", "load_end"}
    ]


def _parse_locust_stats(data_dir: str, label: str) -> dict:
    """Parse the Locust summary statistics from the tail of the log file.

    Returns the aggregated req/s, failures/s, and total counts reported by
    Locust at the end of the load test.
    """
    path = os.path.join(data_dir, f"{label}_locust.log")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        text = f.read()

    # Locust prints a block like:
    # Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
    # --------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
    # GET      /                                                                              23086     0(0.00%) |      3       1      36      3 |   41.42        0.00
    # --------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
    #          Aggregated                                                                     23086     0(0.00%) |      3       1      36      3 |   41.42        0.00
    in_stats = False
    for line in text.splitlines():
        if line.startswith("Type     Name") and "req/s" in line:
            in_stats = True
            continue
        if in_stats and line.startswith("         Aggregated"):
            parts = line.split("|")
            if len(parts) >= 3:
                counts_match = re.search(
                    r"Aggregated\s+(\d+)\s+(\d+)\(", parts[0]
                )
                rates = parts[2].strip().split()
                if counts_match and len(rates) >= 2:
                    try:
                        total_requests = int(counts_match.group(1))
                        total_failures = int(counts_match.group(2))
                        req_per_s = float(rates[0])
                        failures_per_s = float(rates[1])
                        return {
                            "total_requests": total_requests,
                            "total_failures": total_failures,
                            "req_per_s": req_per_s,
                            "failures_per_s": failures_per_s,
                        }
                    except ValueError:
                        pass
    return {}


def _phase_boundaries_from_events(events: list[dict]) -> list[dict]:
    starts = [event for event in events if event.get("event") == "phase_start"]
    ends = {
        event.get("phase"): event
        for event in events
        if event.get("event") == "phase_end"
    }
    phases = []
    for start in starts:
        end = ends.get(start.get("phase"))
        if end is None:
            continue
        phases.append({
            "name": start["phase"],
            "start_s": float(start["observed_offset_s"]),
            "end_s": float(end["observed_offset_s"]),
            "direction": start["direction"],
            "scheduled_start_s": float(start["scheduled_offset_s"]),
            "scheduled_end_s": float(end["scheduled_offset_s"]),
        })
    return phases


def _load_cpu_timeline(
        data_dir: str,
        label: str,
        load_events: list[dict],
        namespace: str = "omniflow-hpa",
        metadata: dict | None = None,
) -> list[dict]:
    """Load mean nginx CPU per second, aligned to Locust's observed start."""
    load_start = next(
        (event for event in load_events if event.get("event") == "load_start"),
        None,
    )
    load_end = next(
        (event for event in load_events if event.get("event") == "load_end"),
        None,
    )
    guest_start_ns = (metadata or {}).get("locust_start_guest_ns")
    if load_start is None and guest_start_ns is None:
        return []

    start_wall = (
        float(guest_start_ns) / 1e9
        if guest_start_ns is not None
        else float(load_start["observed_epoch_ns"]) / 1e9
    )
    end_s = float(load_end["observed_offset_s"]) if load_end else float("inf")
    buckets: dict[int, dict[str, float]] = {}

    pattern = os.path.join(data_dir, f"{label}_daemon_*.jsonl*")
    for path in sorted(glob.glob(pattern)):
        records = _load_jsonl(path)
        nginx_cgroups = {
            record.get("pod")
            for record in records
            if record.get("event") == "track_start"
            and record.get("namespace") == namespace
            and "nginx" in record.get("pod_name", "")
        }
        for record in records:
            if record.get("pod") not in nginx_cgroups or "value" not in record:
                continue
            elapsed_s = int(float(record["wall"]) - start_wall)
            if 0 <= elapsed_s <= end_s:
                buckets.setdefault(elapsed_s, {})[str(record["pod"])] = float(record["value"])

    return [
        {
            "elapsed_s": elapsed_s,
            "mean_cpu": statistics.mean(per_pod.values()),
            "n_pods": len(per_pod),
        }
        for elapsed_s, per_pod in sorted(buckets.items())
        if per_pod
    ]


def _transition_windows(
        phases: list[dict], horizon_s: float | None = None
) -> list[dict]:
    """Assign each transition until the next transition in that direction."""
    windows = []
    for index, phase in enumerate(phases):
        if phase["direction"] == "stable":
            continue
        end_s = horizon_s if horizon_s is not None else phases[-1]["end_s"]
        for following in phases[index + 1:]:
            if following["direction"] == phase["direction"]:
                end_s = following["start_s"]
                break
        windows.append({**phase, "window_end_s": end_s})
    return windows


def _directional_load_windows(phases: list[dict], direction: str) -> list[dict]:
    """Return directional load phases extended only through stable phases."""
    windows = []
    for index, phase in enumerate(phases):
        if phase["direction"] != direction:
            continue
        end_s = phase["end_s"]
        for following in phases[index + 1:]:
            if following["direction"] != "stable":
                break
            end_s = following["end_s"]
        windows.append({**phase, "window_end_s": end_s})
    return windows


def _parse_cpu_quantity(value: str) -> float:
    """Convert a Kubernetes CPU quantity to cores."""
    value = str(value)
    if value.endswith("m"):
        return float(value[:-1]) / 1000.0
    return float(value)


def _hpa_cpu_target(hpa_status: dict, default: float = 0.03) -> float:
    """Extract AverageValue CPU target from a kubectl HPA JSON document."""
    items = hpa_status.get("items", [hpa_status]) if isinstance(hpa_status, dict) else []
    for item in items:
        for metric in item.get("spec", {}).get("metrics", []):
            source = metric.get("resource") or metric.get("pods") or {}
            value = source.get("target", {}).get("averageValue")
            if value is not None:
                try:
                    return _parse_cpu_quantity(value)
                except ValueError:
                    pass
    return default


def _hpa_replica_bounds(hpa_status: dict) -> tuple[int, int | None]:
    """Extract the configured HPA replica bounds."""
    items = hpa_status.get("items", [hpa_status]) if isinstance(hpa_status, dict) else []
    if not items:
        return 1, None
    spec = items[0].get("spec", {})
    minimum = int(spec.get("minReplicas", 1))
    maximum = spec.get("maxReplicas")
    return minimum, int(maximum) if maximum is not None else None


def _compute_scaling_accuracy(
        timeline: list[dict],
        shape: str = "phased",
        phases: list[dict] | None = None,
        cpu_timeline: list[dict] | None = None,
        cpu_target: float = 0.03,
        min_replicas: int = 1,
        max_replicas: int | None = None,
) -> dict:
    """Analyse HPA decisions and pod readiness in transition windows.
    """
    result: dict = {
        "scale_up_detected": False,
        "scale_up_latency_s": None,
        "scale_up_latencies_s": [],
        "scale_up_transition_count": 0,
        "scale_up_detected_count": 0,
        "scale_up_detection_rate": None,
        "scale_down_detected": False,
        "scale_down_latency_s": None,
        "scale_down_latencies_s": [],
        "scale_down_transition_count": 0,
        "scale_down_detected_count": 0,
        "scale_down_detection_rate": None,
        "scale_up_decision_latency_s": None,
        "scale_up_decision_latencies_s": [],
        "scale_up_decision_transition_count": 0,
        "scale_up_decision_detected_count": 0,
        "scale_up_decision_detection_rate": None,
        "scale_down_decision_latency_s": None,
        "scale_down_decision_latencies_s": [],
        "scale_down_decision_transition_count": 0,
        "scale_down_decision_detected_count": 0,
        "scale_down_decision_detection_rate": None,
        "peak_replicas": 1,
        "under_provisioned_s": None,
        "cpu_target_excess_core_seconds": None,
        "cpu_expected_seconds": 0,
        "cpu_observed_seconds": 0,
        "cpu_telemetry_coverage": None,
        "flapping_events": 0,
        "scale_up_events": 0,
        "scale_down_events": 0,
        "hpa_scale_up_events": 0,
        "hpa_scale_down_events": 0,
        "replica_timeline_points": len(timeline),
    }

    if not timeline:
        return result

    ordered = sorted(timeline, key=lambda item: float(item.get("elapsed_s", 0)))
    elapsed = [float(item.get("elapsed_s", 0)) for item in ordered]

    def _forward_filled(key: str) -> list[int]:
        values: list[int] = []
        last = 1
        for item in ordered:
            value = item.get(key)
            # minReplicas=1, so zero/negative and null are failed observations.
            if isinstance(value, (int, float)) and value >= 1:
                last = int(value)
            values.append(last)
        return values

    ready = _forward_filled("ready")
    desired = _forward_filled("hpa_desired")
    phases = phases or _phase_boundaries(shape)
    windows = _transition_windows(phases, max(elapsed) + 1e-6)
    result["peak_replicas"] = max(ready)

    def _changes(values: list[int]) -> list[tuple[float, str]]:
        changes: list[tuple[float, str]] = []
        for i in range(1, len(values)):
            if values[i] > values[i - 1]:
                changes.append((elapsed[i], "up"))
            elif values[i] < values[i - 1]:
                changes.append((elapsed[i], "down"))
        return changes

    ready_changes = _changes(ready)
    desired_changes = _changes(desired)
    result["scale_up_events"] = sum(direction == "up" for _, direction in ready_changes)
    result["scale_down_events"] = sum(direction == "down" for _, direction in ready_changes)
    result["hpa_scale_up_events"] = sum(direction == "up" for _, direction in desired_changes)
    result["hpa_scale_down_events"] = sum(direction == "down" for _, direction in desired_changes)

    def _value_at(t: float, values: list[int]) -> int:
        best = 1
        for event_time, replicas in zip(elapsed, values):
            if event_time <= t:
                best = replicas
            else:
                break
        return best

    for window in windows:
        direction = window["direction"]
        ready_at_start = _value_at(float(window["start_s"]), ready)
        desired_at_start = _value_at(float(window["start_s"]), desired)

        ready_eligible = not (
            (direction == "down" and ready_at_start <= min_replicas)
            or (
                direction == "up"
                and max_replicas is not None
                and ready_at_start >= max_replicas
            )
        )
        decision_eligible = not (
            (direction == "down" and desired_at_start <= min_replicas)
            or (
                direction == "up"
                and max_replicas is not None
                and desired_at_start >= max_replicas
            )
        )

        if decision_eligible:
            result[f"scale_{direction}_decision_transition_count"] += 1
            decision_time = next((
                change_time for change_time, change_direction in desired_changes
                if change_direction == direction
                and window["start_s"] <= change_time < window["window_end_s"]
            ), None)
            if decision_time is not None:
                result[f"scale_{direction}_decision_detected_count"] += 1
                result[f"scale_{direction}_decision_latencies_s"].append(
                    decision_time - window["start_s"]
                )

        if ready_eligible:
            result[f"scale_{direction}_transition_count"] += 1
            ready_time = next((
                change_time for change_time, change_direction in ready_changes
                if change_direction == direction
                and window["start_s"] <= change_time < window["window_end_s"]
            ), None)
            if ready_time is not None:
                result[f"scale_{direction}_detected_count"] += 1
                result[f"scale_{direction}_latencies_s"].append(
                    ready_time - window["start_s"]
                )

    for direction in ("up", "down"):
        transitions = result[f"scale_{direction}_transition_count"]
        decision_transitions = result[f"scale_{direction}_decision_transition_count"]
        ready_latencies = result[f"scale_{direction}_latencies_s"]
        decision_latencies = result[f"scale_{direction}_decision_latencies_s"]
        result[f"scale_{direction}_detected"] = bool(ready_latencies)
        result[f"scale_{direction}_latency_s"] = (
            statistics.mean(ready_latencies) if ready_latencies else None
        )
        result[f"scale_{direction}_detection_rate"] = (
            result[f"scale_{direction}_detected_count"] / transitions
            if transitions else None
        )
        result[f"scale_{direction}_decision_latency_s"] = (
            statistics.mean(decision_latencies) if decision_latencies else None
        )
        result[f"scale_{direction}_decision_detection_rate"] = (
            result[f"scale_{direction}_decision_detected_count"] / decision_transitions
            if decision_transitions else None
        )

    if cpu_timeline:
        up_windows = _directional_load_windows(phases, "up")
        expected_seconds = {
            second
            for window in up_windows
            for second in range(
                math.ceil(float(window["start_s"])),
                math.ceil(float(window["window_end_s"])),
            )
        }
        relevant_points = [
            point for point in cpu_timeline
            if any(
                window["start_s"] <= point["elapsed_s"] < window["window_end_s"]
                for window in up_windows
            )
        ]
        observed_seconds = {
            int(point["elapsed_s"]) for point in relevant_points
        }
        coverage = (
            len(observed_seconds) / len(expected_seconds)
            if expected_seconds else None
        )
        result["cpu_expected_seconds"] = len(expected_seconds)
        result["cpu_observed_seconds"] = len(observed_seconds)
        result["cpu_telemetry_coverage"] = coverage
        if coverage is not None and coverage >= MIN_CPU_COVERAGE:
            result["under_provisioned_s"] = sum(
                1 for point in relevant_points if point["mean_cpu"] > cpu_target
            )
            result["cpu_target_excess_core_seconds"] = sum(
                max(0.0, point["mean_cpu"] - cpu_target)
                * _value_at(float(point["elapsed_s"]), ready)
                for point in relevant_points
            )

    # Flapping is based on actual Ready replicas, not transient desired counts.
    flaps = 0
    for i in range(1, len(ready)):
        if ready[i] > ready[i - 1]:
            up_time = elapsed[i]
            for j in range(i + 1, len(ready)):
                if elapsed[j] - up_time > 60:
                    break
                if ready[j] < ready[j - 1]:
                    flaps += 1
                    break
    result["flapping_events"] = flaps
    return result


# ------------------------------
# Main
# ------------------------------

def _mean_stdev(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.mean(values) if values else None
    stdev = statistics.stdev(values) if n > 1 else 0.0
    sem = stdev / np.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "stdev": stdev,
        "sem": sem,
        # Normal-approximation interval; paired intervals are computed separately.
        "ci95_half_width": 1.96 * sem,
        "median": statistics.median(values) if values else None,
        "n": n,
    }


def _summarize_repetition_latencies(
    scaling: list[dict], field: str
) -> dict:
    """Summarize one within-repetition latency mean per experimental run."""
    return _mean_stdev([
        statistics.mean(entry[field])
        for entry in scaling
        if entry.get(field)
    ])


def _format_mean_sd(value: dict | None, fmt: str = ".2f", unit: str = "") -> str:
    """Format a mean±stdev dict for a LaTeX table cell."""
    if value is None or value.get("mean") is None:
        return "-"
    mean = value["mean"]
    stdev = value.get("stdev", 0.0) or 0.0
    if stdev:
        return f"{mean:{fmt}}$\\pm${stdev:{fmt}}{unit}"
    return f"{mean:{fmt}}{unit}"


def _render_latex_table(analysis: dict) -> str:
    """Render the analysis as a booktabs-style LaTeX table."""
    experiments = analysis.get("experiments", {})

    # Two possible layouts: run-dir layout (metrics_60s/15s/omniflow)
    # vs. data-dir layout (baseline/omniflow).
    run_dir_layout = {
        "labels": ["metrics_60s", "metrics_15s", "omniflow"],
        "row_names": {
            "metrics_60s": "metrics-server 60s",
            "metrics_15s": "metrics-server 15s",
            "omniflow": "OmniFlow",
        },
    }
    data_dir_layout = {
        "labels": ["baseline", "omniflow"],
        "row_names": {"baseline": "metrics-server 15s", "omniflow": "OmniFlow"},
    }

    if "metrics_60s" in experiments:
        layout = run_dir_layout
    elif "baseline" in experiments:
        layout = data_dir_layout
    else:
        return "% No recognised experiment labels found for LaTeX table."

    def summary(experiment: dict) -> dict:
        return experiment.get("summary", {})

    def aggregate(experiment: dict) -> dict:
        return experiment.get("aggregate_info_loss", {})

    def scaling(experiment: dict) -> dict:
        return experiment.get("scaling_accuracy", {})

    rows = []
    for label in layout["labels"]:
        if label not in experiments:
            rows.append(
                f"{layout['row_names'][label]} & " + " & ".join(["-"] * 7) + " \\\\"
            )
            continue
        exp = experiments[label]
        s = summary(exp)
        agg = aggregate(exp)
        sa = scaling(exp)

        # Sampling ratio: use summary if available, else aggregate.
        sample_ratio = s.get("sample_ratio") if s.get("sample_ratio") else agg.get("sample_ratio_mean")
        if isinstance(sample_ratio, (int, float)):
            sample_ratio = {"mean": float(sample_ratio), "stdev": 0.0, "n": 1}

        # NRMSE: summary or aggregate.
        nrmse = s.get("nrmse") if s.get("nrmse") else agg.get("nrmse_mean")
        if isinstance(nrmse, (int, float)):
            nrmse = {"mean": float(nrmse), "stdev": 0.0, "n": 1}

        # Scale latencies and under-provisioning.
        scale_up = s.get("scale_up_latency_s") if s.get("scale_up_latency_s") else sa.get("scale_up_latency_s")
        if isinstance(scale_up, (int, float)):
            scale_up = {"mean": float(scale_up), "stdev": 0.0, "n": 1}
        scale_down = s.get("scale_down_latency_s") if s.get("scale_down_latency_s") else sa.get("scale_down_latency_s")
        if isinstance(scale_down, (int, float)):
            scale_down = {"mean": float(scale_down), "stdev": 0.0, "n": 1}
        under = s.get("under_provisioned_s") if s.get("under_provisioned_s") else sa.get("under_provisioned_s")
        if isinstance(under, (int, float)):
            under = {"mean": float(under), "stdev": 0.0, "n": 1}

        # Detection rates.
        up_rate = s.get("scale_up_detection_rate") if s.get("scale_up_detection_rate") is not None else sa.get("scale_up_detected")
        down_rate = s.get("scale_down_detection_rate") if s.get("scale_down_detection_rate") is not None else sa.get("scale_down_detected")

        # Flapping: run-dir has flapping in summary; data-dir has scalar in scaling_accuracy.
        flap = s.get("flapping_events")
        if flap is None:
            flap = sa.get("flapping_events")
        if isinstance(flap, (int, float)):
            flap = {"mean": float(flap), "stdev": 0.0, "n": 1}

        # sample_ratio is stored as a fraction; present as a percentage.
        if isinstance(sample_ratio, dict):
            sample_ratio_pct = {
                "mean": sample_ratio["mean"] * 100.0,
                "stdev": sample_ratio.get("stdev", 0.0) * 100.0,
                "n": sample_ratio["n"],
            }
        else:
            sample_ratio_pct = sample_ratio

        rows.append(
            f"{layout['row_names'][label]} & "
            f"{_format_mean_sd(sample_ratio_pct, '.2f', '\\%')} & "
            f"{_format_mean_sd(nrmse, '.3f')} & "
            f"{up_rate*100:.0f}\\% & "
            f"{_format_mean_sd(scale_up, '.1f')} & "
            f"{down_rate*100:.0f}\\% & "
            f"{_format_mean_sd(scale_down, '.1f')} & "
            f"{_format_mean_sd(under, '.1f')} & "
            f"{_format_mean_sd(flap, '.1f')} \\\\"
        )

    shape = analysis.get("shape", "phased")
    repetitions = max(
        (
            experiment.get("summary", {}).get("repetitions", 0)
            for experiment in experiments.values()
        ),
        default=0,
    )
    repetition_text = f", {repetitions} repetitions" if repetitions else ""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Kubernetes HPA autoscaling accuracy ({shape} workload{repetition_text}).}}",
        f"\\label{{tab:k8s_{shape}}}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Method & Sampling ratio & NRMSE & $\\uparrow$ detected & $\\uparrow$ latency (s) & "
        "$\\downarrow$ detected & $\\downarrow$ latency (s) & Under-prov. (s) & Flapping \\\\",
        "\\midrule",
    ]
    lines.extend(rows)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _shape_display(shape: str) -> str:
    """Return a display name for a load shape."""
    return {
        "phased": "Phased",
        "oscillating": "Oscillating",
        "staircase": "Staircase",
        "flash_crowd": "Flash crowd",
    }.get(shape, shape.replace("_", " ").title())


def _render_k8s_paper_table(analyses: list[dict]) -> str:
    """Render the multi-shape K8s autoscaling table from the paper.

    Expects one analysis dict per load shape (from --run-dir mode). Each shape
    contributes three rows: metrics-server 60s, metrics-server 15s, and OmniFlow.
    """
    ORDER = ["phased", "oscillating", "staircase", "flash_crowd"]
    HPA_ROWS = [
        ("metrics_60s", "MS 15/60"),
        ("metrics_15s", "MS 15/15"),
        ("omniflow", "OmniFlow"),
    ]

    def shape_sort_key(a: dict) -> int:
        shape = a.get("shape", "")
        return ORDER.index(shape) if shape in ORDER else len(ORDER)

    analyses = sorted(analyses, key=shape_sort_key)

    def fmt(value: dict | None, fmt: str = ".2f", unit: str = "", scale: float = 1.0) -> str:
        if value is None or value.get("mean") is None:
            return "--"
        mean = value["mean"] * scale
        stdev = (value.get("stdev", 0.0) or 0.0) * scale
        if stdev:
            return f"${mean:{fmt}}_{{\\pm{stdev:{fmt}}}}{unit}$"
        return f"${mean:{fmt}}{unit}$"

    def pct(value) -> str:
        if value is None:
            return "--"
        return f"${value*100:.0f}\\%$"

    def detected(summary: dict, direction: str) -> str:
        found = summary.get(
            f"scale_{direction}_decision_detected_count",
            summary.get(f"scale_{direction}_detected_count"),
        )
        total = summary.get(
            f"scale_{direction}_decision_transition_count",
            summary.get(f"scale_{direction}_transition_count"),
        )
        return f"${found}/{total}$" if found is not None and total is not None else "--"

    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\footnotesize",
        "    \\renewcommand{\\arraystretch}{1.15}",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\begin{tabular}{@{}l l r r r r r r r r r r@{}}",
        "        \\hline",
        "        \\multirow{2}{*}{Shape} & \\multirow{2}{*}{HPA} & \\makecell{Req/s}  & \\makecell{Fail.}   & $r$      & \\makecell{Scale-up} & \\makecell{Scale-down} & \\makecell{Avg $\\uparrow$} & \\makecell{Avg $\\downarrow$} & \\makecell{Down\\\\det.} & \\makecell{Under-} \\\\",
        "                               &                      &                   &                    &          & latency (s)         & latency (s)           & events                    & events                      &                       & prov. (s)         \\\\",
        "        \\hline",
    ]
    lines[-3] = (
        "        \\multirow{2}{*}{Shape} & \\multirow{2}{*}{HPA} & "
        "\\makecell{Req/s} & \\makecell{Fail.} & $r$ & \\makecell{Scale-up} & "
        "\\makecell{Scale-down} & \\makecell{Avg $\\uparrow$} & "
        "\\makecell{Avg $\\downarrow$} & \\makecell{Up det.} & "
        "\\makecell{Down det.} & \\makecell{Under-} \\\\"
    )
    lines[-2] = (
        "                               & & & & & latency (s) & latency (s) & "
        "events & events & & & prov. (s) \\\\"
    )

    for analysis in analyses:
        shape = analysis.get("shape", "unknown")
        shape_label = _shape_display(shape)
        experiments = analysis.get("experiments", {})
        for idx, (label, hpa_name) in enumerate(HPA_ROWS):
            exp = experiments.get(label, {})
            s = exp.get("summary", {})

            if idx == 0:
                row_prefix = f"        \\multirow{{3}}{{*}}{{{shape_label}}}"
            else:
                row_prefix = "                               "

            lines.append(
                f"{row_prefix} & {hpa_name} & "
                f"{fmt(s.get('req_per_s'), '.2f')} & "
                f"{fmt(s.get('failures_per_s'), '.2f')} & "
                f"{fmt(s.get('sample_ratio'), '.2f', '\\%', scale=100.0)} & "
                f"{fmt(s.get('scale_up_decision_latency_s') or s.get('scale_up_latency_s'), '.1f')} & "
                f"{fmt(s.get('scale_down_decision_latency_s') or s.get('scale_down_latency_s'), '.1f')} & "
                f"{fmt(s.get('hpa_scale_up_events') or s.get('scale_up_events'), '.2f')} & "
                f"{fmt(s.get('hpa_scale_down_events') or s.get('scale_down_events'), '.2f')} & "
                f"{detected(s, 'up')} & "
                f"{detected(s, 'down')} & "
                f"{fmt(s.get('under_provisioned_s'), '.1f')} \\\\"
            )
        lines.append("        \\hline")

    lines.extend([
        "    \\end{tabular}",
        "    \\caption{\\ac{k8s} HPA autoscaling results (mean $\\pm$ standard deviation across repetitions). MS 15/60 and MS 15/15 denote 15-second HPA synchronization with 60- and 15-second Metrics Server resolutions. HPA decision latencies are first averaged within each repetition and then summarized across repetitions; detection is reported as detected/eligible using desired replicas at transition start. Under-provisioning is the time with mean pod CPU above the 30m HPA target during rising-load windows; ``--'' indicates that no transition was detected.}",
        "    \\label{tab:k8s-results}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def _analyse_repeated_run(run_dir: str, out_path: str) -> None:
    """Analyse the three-case, per-repetition output layout from the runner."""
    params_path = os.path.join(run_dir, "params.json")
    params = _load_json(params_path) if os.path.exists(params_path) else {}
    shape = params.get("shape", "phased")
    namespace = params.get("namespace", "omniflow-hpa")
    base_interval = float(params.get("env", {}).get("BASE_INTERVAL", 1.0))
    metrics_resolution = float(str(params.get("metrics_server_resolution", "60s")).removesuffix("s"))
    analysis = {
        "shape": shape,
        "load_phases": _phase_boundaries(shape),
        "params": params,
        "experiments": {},
    }

    # Pre-load a reference daemon config from the metrics_60s arm (collected
    # earliest, so the daemon JSONL is smallest and daemon_start is most
    # likely to be present). Fall back to later arms if needed.
    ref_daemon_config: PipelineConfig | None = None
    for candidate_label in ("metrics_60s", "metrics_15s", "omniflow"):
        candidate_repeat_dirs = sorted(glob.glob(
            os.path.join(run_dir, "results", candidate_label, "repeat_*")
        ))
        for candidate_dir in candidate_repeat_dirs:
            cfg = _load_daemon_config(candidate_dir, candidate_label)
            if cfg.min_interval > 1:  # non-default => real config
                ref_daemon_config = cfg
                break
        if ref_daemon_config is not None:
            break

    for label in ("metrics_60s", "metrics_15s", "omniflow"):
        repeat_dirs = sorted(glob.glob(os.path.join(run_dir, "results", label, "repeat_*")))
        if not repeat_dirs:
            continue

        repeats = []
        for repeat_dir in repeat_dirs:
            metadata_path = os.path.join(repeat_dir, "metadata.json")
            metadata = _load_json(metadata_path) if os.path.exists(metadata_path) else {}
            load_events = _parse_load_events(repeat_dir, label)
            timeline = _load_replica_timeline(repeat_dir, label, metadata)
            daemon_config = _load_daemon_config(repeat_dir, label)
            # If the current arm's daemon files lack daemon_start
            # (truncated JSONL), reuse the reference config.
            if daemon_config.min_interval <= 1 and ref_daemon_config is not None:
                daemon_config = ref_daemon_config
            per_pod = _analyse_daemon_logs(
                repeat_dir,
                label,
                config=daemon_config,
                namespace=namespace,
                load_events=load_events,
                metadata=metadata,
            )
            locust_stats = _parse_locust_stats(repeat_dir, label)
            recorded_phases = _phase_boundaries_from_events(load_events)
            cpu_timeline = _load_cpu_timeline(
                repeat_dir, label, load_events, namespace, metadata
            )
            aggregate = {"n_pods": len(per_pod), "sample_ratio_mean": None, "nrmse_mean": None}
            if label == "metrics_60s":
                aggregate["sample_ratio_mean"] = base_interval / metrics_resolution
            elif label == "metrics_15s":
                aggregate["sample_ratio_mean"] = base_interval / 15.0
            elif per_pod:
                # Weight by trace length so short-lived scale-up pods do not
                # dominate the native adaptive aggregate.
                n_readings = sum(pod["n_readings"] for pod in per_pod)
                aggregate.update({
                    "sample_ratio_mean": sum(pod["n_sampled"] for pod in per_pod) / n_readings,
                    "nrmse_mean": sum(
                        pod["info_loss"]["raw_recon_nrmse"] * pod["n_readings"]
                        for pod in per_pod
                    ) / n_readings,
                })
            hpa_path = os.path.join(repeat_dir, f"{label}_hpa_status.json")
            hpa_status = _load_json(hpa_path) if os.path.exists(hpa_path) else {}
            repeats.append({
                "metadata": metadata,
                "pipeline_config": daemon_config.to_dict(),
                "aggregate_info_loss": aggregate,
                "per_pod": per_pod,
                "load_events": load_events,
                "locust_stats": locust_stats,
                "cpu_timeline": cpu_timeline,
                "load_phases": recorded_phases or _phase_boundaries(shape),
                "scale_events": _parse_scale_events(repeat_dir, label),
                "scaling_accuracy": _compute_scaling_accuracy(
                    timeline,
                    shape,
                    recorded_phases or None,
                    cpu_timeline,
                    _hpa_cpu_target(hpa_status),
                    *_hpa_replica_bounds(hpa_status),
                ),
                "hpa_status": hpa_status,
            })

        scaling = [repeat["scaling_accuracy"] for repeat in repeats]
        info = [repeat["aggregate_info_loss"] for repeat in repeats]
        analysis["experiments"][label] = {
            "summary": {
                "repetitions": len(repeats),
                "sample_ratio": _mean_stdev([
                    entry["sample_ratio_mean"] for entry in info
                    if entry["sample_ratio_mean"] is not None
                ]),
                "nrmse": _mean_stdev([
                    entry["nrmse_mean"] for entry in info
                    if entry["nrmse_mean"] is not None
                ]),
                "scale_up_latency_s": _summarize_repetition_latencies(
                    scaling, "scale_up_latencies_s"
                ),
                "scale_down_latency_s": _summarize_repetition_latencies(
                    scaling, "scale_down_latencies_s"
                ),
                "scale_up_decision_latency_s": _summarize_repetition_latencies(
                    scaling, "scale_up_decision_latencies_s"
                ),
                "scale_down_decision_latency_s": _summarize_repetition_latencies(
                    scaling, "scale_down_decision_latencies_s"
                ),
                "under_provisioned_s": _mean_stdev([
                    entry["under_provisioned_s"] for entry in scaling
                    if entry["under_provisioned_s"] is not None
                ]),
                "cpu_target_excess_core_seconds": _mean_stdev([
                    entry["cpu_target_excess_core_seconds"] for entry in scaling
                    if entry["cpu_target_excess_core_seconds"] is not None
                ]),
                "cpu_telemetry_coverage": _mean_stdev([
                    entry["cpu_telemetry_coverage"] for entry in scaling
                    if entry["cpu_telemetry_coverage"] is not None
                ]),
                "under_provisioned_valid_repetitions": sum(
                    entry["under_provisioned_s"] is not None for entry in scaling
                ),
                "scale_up_detection_rate": (
                    sum(entry["scale_up_detected_count"] for entry in scaling)
                    / sum(entry["scale_up_transition_count"] for entry in scaling)
                ) if sum(entry["scale_up_transition_count"] for entry in scaling) else None,
                "scale_up_detected_count": sum(
                    entry["scale_up_detected_count"] for entry in scaling
                ),
                "scale_up_transition_count": sum(
                    entry["scale_up_transition_count"] for entry in scaling
                ),
                "scale_down_detection_rate": (
                    sum(entry["scale_down_detected_count"] for entry in scaling)
                    / sum(entry["scale_down_transition_count"] for entry in scaling)
                ) if sum(entry["scale_down_transition_count"] for entry in scaling) else None,
                "scale_down_detected_count": sum(
                    entry["scale_down_detected_count"] for entry in scaling
                ),
                "scale_down_transition_count": sum(
                    entry["scale_down_transition_count"] for entry in scaling
                ),
                "scale_up_decision_detection_rate": (
                    sum(entry["scale_up_decision_detected_count"] for entry in scaling)
                    / sum(entry["scale_up_decision_transition_count"] for entry in scaling)
                ) if sum(entry["scale_up_decision_transition_count"] for entry in scaling) else None,
                "scale_up_decision_detected_count": sum(
                    entry["scale_up_decision_detected_count"] for entry in scaling
                ),
                "scale_up_decision_transition_count": sum(
                    entry["scale_up_decision_transition_count"] for entry in scaling
                ),
                "scale_down_decision_detection_rate": (
                    sum(entry["scale_down_decision_detected_count"] for entry in scaling)
                    / sum(entry["scale_down_decision_transition_count"] for entry in scaling)
                ) if sum(entry["scale_down_decision_transition_count"] for entry in scaling) else None,
                "scale_down_decision_detected_count": sum(
                    entry["scale_down_decision_detected_count"] for entry in scaling
                ),
                "scale_down_decision_transition_count": sum(
                    entry["scale_down_decision_transition_count"] for entry in scaling
                ),
                "scale_up_events": _mean_stdev([
                    entry["scale_up_events"] for entry in scaling
                ]),
                "scale_down_events": _mean_stdev([
                    entry["scale_down_events"] for entry in scaling
                ]),
                "hpa_scale_up_events": _mean_stdev([
                    entry["hpa_scale_up_events"] for entry in scaling
                ]),
                "hpa_scale_down_events": _mean_stdev([
                    entry["hpa_scale_down_events"] for entry in scaling
                ]),
                "flapping_events": _mean_stdev([
                    entry["flapping_events"] for entry in scaling
                ]),
                "req_per_s": _mean_stdev([
                    repeat["locust_stats"].get("req_per_s")
                    for repeat in repeats
                    if repeat.get("locust_stats", {}).get("req_per_s") is not None
                ]),
                "failures_per_s": _mean_stdev([
                    repeat["locust_stats"].get("failures_per_s")
                    for repeat in repeats
                    if repeat.get("locust_stats", {}).get("failures_per_s") is not None
                ]),
            },
            "repeats": repeats,
        }
        print(f"Analysed {label}: {len(repeats)} repetitions", file=sys.stderr)

    ms15 = analysis["experiments"].get("metrics_15s", {}).get("repeats", [])
    omni = analysis["experiments"].get("omniflow", {}).get("repeats", [])
    if ms15 and omni:
        def _by_repeat(entries: list[dict]) -> dict[int, dict]:
            result = {}
            for entry in entries:
                repeat = entry.get("metadata", {}).get("repeat")
                execution_mode = entry.get("metadata", {}).get("execution_mode")
                order = entry.get("metadata", {}).get("arm_order", [])
                if isinstance(repeat, int) and (
                    execution_mode == "isolated_arm_major" or len(order) == 3
                ):
                    result[repeat] = entry
            return result

        left = _by_repeat(ms15)
        right = _by_repeat(omni)
        common = sorted(set(left) & set(right))
        if common:
            def _paired(metric_getter) -> dict:
                deltas = []
                for repeat in common:
                    baseline_value = metric_getter(left[repeat])
                    omniflow_value = metric_getter(right[repeat])
                    if baseline_value is not None and omniflow_value is not None:
                        deltas.append(float(omniflow_value) - float(baseline_value))
                return _mean_stdev(deltas)

            analysis["paired_omniflow_minus_ms15"] = {
                "interpretation": "negative values favour OmniFlow",
                "under_provisioned_s": _paired(
                    lambda entry: entry["scaling_accuracy"].get("under_provisioned_s")
                ),
                "cpu_target_excess_core_seconds": _paired(
                    lambda entry: entry["scaling_accuracy"].get("cpu_target_excess_core_seconds")
                ),
                "scale_up_decision_latency_s": _paired(
                    lambda entry: entry["scaling_accuracy"].get("scale_up_decision_latency_s")
                ),
                "scale_down_decision_latency_s": _paired(
                    lambda entry: entry["scaling_accuracy"].get("scale_down_decision_latency_s")
                ),
                "req_per_s": _paired(
                    lambda entry: entry.get("locust_stats", {}).get("req_per_s")
                ),
            }

    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis saved to {out_path}", file=sys.stderr)

def _discover_run_dirs(parent_dir: str) -> list[str]:
    """Return run directories under parent_dir, sorted by shape order."""
    ORDER = ["phased", "oscillating", "staircase", "flash_crowd"]
    run_dirs = []
    for entry in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, entry)
        if not os.path.isdir(path):
            continue
        params_path = os.path.join(path, "params.json")
        if not os.path.exists(params_path):
            continue
        try:
            with open(params_path) as f:
                params = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if params.get("shape") in ORDER:
            run_dirs.append(path)
    shapes = [
        _load_json(os.path.join(path, "params.json")).get("shape", "")
        for path in run_dirs
    ]
    duplicates = sorted({shape for shape in shapes if shapes.count(shape) > 1})
    if duplicates:
        raise ValueError(
            "multiple campaigns found for shape(s) "
            f"{', '.join(duplicates)}; pass the intended directories with --run-dirs"
        )
    return sorted(run_dirs, key=lambda p: ORDER.index(_load_json(os.path.join(p, "params.json")).get("shape", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect HPA experiment results")
    parser.add_argument("--data-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--runs-dir", help="Parent directory containing one run directory per load shape.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="Exact run directories to combine, one per load shape.",
    )
    parser.add_argument(
        "--combined-json",
        help="Render the paper table from an existing combined analysis JSON file.",
    )
    parser.add_argument("--out", default="analysis.json")
    parser.add_argument(
        "--tex",
        metavar="FILE",
        help="Also write a LaTeX table of the analysis to FILE.",
    )
    args = parser.parse_args()

    if args.combined_json:
        if not args.tex:
            parser.error("--combined-json requires --tex")
        try:
            combined = _load_json(args.combined_json)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read combined analysis JSON: {exc}")
        analyses = combined.get("analyses") if isinstance(combined, dict) else None
        if not isinstance(analyses, list) or not analyses:
            parser.error("combined analysis JSON must contain a non-empty 'analyses' list")
        Path(args.tex).write_text(_render_k8s_paper_table(analyses))
        print(f"LaTeX table saved to {args.tex}", file=sys.stderr)
        return

    if args.run_dir:
        _analyse_repeated_run(os.path.abspath(args.run_dir), args.out)
        if args.tex:
            analysis = _load_json(args.out)
            Path(args.tex).write_text(_render_latex_table(analysis))
            print(f"LaTeX table saved to {args.tex}", file=sys.stderr)
        return

    if args.runs_dir or args.run_dirs:
        try:
            run_dirs = (
                [os.path.abspath(path) for path in args.run_dirs]
                if args.run_dirs else _discover_run_dirs(args.runs_dir)
            )
        except ValueError as exc:
            parser.error(str(exc))
        if not run_dirs:
            parser.error("no run directories found")
        shapes = [
            _load_json(os.path.join(path, "params.json")).get("shape")
            for path in run_dirs
        ]
        if len(set(shapes)) != len(shapes):
            parser.error("--run-dirs must contain exactly one campaign per shape")
        analyses = []
        for run_dir in run_dirs:
            out_path = os.path.join(run_dir, "analysis.json")
            _analyse_repeated_run(os.path.abspath(run_dir), out_path)
            analyses.append(_load_json(out_path))
        combined = {"shapes": [a.get("shape") for a in analyses], "analyses": analyses}
        with open(args.out, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"Combined analysis saved to {args.out}", file=sys.stderr)
        if args.tex:
            Path(args.tex).write_text(_render_k8s_paper_table(analyses))
            print(f"LaTeX table saved to {args.tex}", file=sys.stderr)
        return

    if not args.data_dir:
        parser.error("provide --run-dir, --run-dirs, --runs-dir, or --data-dir")

    # Read shape/namespace from params.json (parent of data-dir)
    params_path = os.path.join(os.path.dirname(args.data_dir), "params.json")
    shape = "phased"
    namespace = "omniflow-hpa"
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        shape = params.get("shape", "phased")
        namespace = params.get("namespace", "omniflow-hpa")

    analysis: dict = {"experiments": {}, "load_phases": _phase_boundaries(shape), "shape": shape}

    for label in ("baseline", "omniflow"):
        print(f"Analysing {label} ...", file=sys.stderr)

        # Scaling events
        scale_events = _parse_scale_events(args.data_dir, label)

        # Replica timeline + scaling accuracy
        load_events = _parse_load_events(args.data_dir, label)
        metadata_path = os.path.join(args.data_dir, "metadata.json")
        metadata = _load_json(metadata_path) if os.path.exists(metadata_path) else {}
        timeline = _load_replica_timeline(args.data_dir, label, metadata)
        recorded_phases = _phase_boundaries_from_events(load_events)
        daemon_config = _load_daemon_config(args.data_dir, label)
        daemon_results = _analyse_daemon_logs(
            args.data_dir,
            label,
            config=daemon_config,
            namespace=namespace,
            load_events=load_events,
            metadata=metadata,
        )

        # HPA status snapshot
        hpa_path = os.path.join(args.data_dir, f"{label}_hpa_status.json")
        hpa_status = _load_json(hpa_path) if os.path.exists(hpa_path) else {}
        cpu_timeline = _load_cpu_timeline(
            args.data_dir, label, load_events, namespace, metadata
        )
        scaling_accuracy = _compute_scaling_accuracy(
            timeline,
            shape,
            recorded_phases or None,
            cpu_timeline,
            _hpa_cpu_target(hpa_status),
            *_hpa_replica_bounds(hpa_status),
        )

        # Aggregate info-loss across pods
        # For baseline: HPA reads from metrics-server (fixed 15s interval)
        # For omniflow: HPA reads from daemon's adaptive output
        if daemon_results:
            nrmses = [d["info_loss"]["nrmse_mean"] for d in daemon_results]
            
            if label == "baseline":
                # Metrics-server scrapes every 15s, base interval is 1s
                # So baseline samples 1 out of every 15 points
                base_interval = float(params.get("env", {}).get("BASE_INTERVAL", "1.0"))
                metrics_server_interval = 15.0  # seconds, as documented
                sample_ratio = base_interval / metrics_server_interval
            else:
                # OmniFlow: use daemon's adaptive sample ratio
                ratios = [d["info_loss"]["sample_ratio"] for d in daemon_results]
                sample_ratio = float(np.mean(ratios))
            
            aggregate = {
                "n_pods": len(daemon_results),
                "sample_ratio_mean": round(sample_ratio, 4),
                "nrmse_mean": round(float(np.mean(nrmses)), 4),
                "nrmse_max": round(float(np.max(nrmses)), 4),
            }
        else:
            aggregate = {"n_pods": 0}

        analysis["experiments"][label] = {
            "pipeline_config": daemon_config.to_dict(),
            "aggregate_info_loss": aggregate,
            "per_pod": daemon_results,
            "load_events": load_events,
            "load_phases": recorded_phases or _phase_boundaries(shape),
            "scale_events": scale_events,
            "scaling_accuracy": scaling_accuracy,
            "replica_timeline": [
                {"elapsed_s": r.get("elapsed_s", 0), "replicas": r.get("replicas", 1)}
                for r in timeline
            ],
            "hpa_status": hpa_status,
        }

        print(f"  {label}: {aggregate.get('n_pods', 0)} pods, "
              f"{len(scale_events)} scale events, "
              f"{scaling_accuracy['replica_timeline_points']} timeline points",
              file=sys.stderr)
        if scaling_accuracy["scale_up_latency_s"] is not None:
            print(f"    scale_up_latency={scaling_accuracy['scale_up_latency_s']}s  "
                  f"peak={scaling_accuracy['peak_replicas']}  "
                  f"flapping={scaling_accuracy['flapping_events']}",
                  file=sys.stderr)

    # Comparison summary
    baseline_exp = analysis["experiments"].get("baseline", {})
    omniflow_exp = analysis["experiments"].get("omniflow", {})
    baseline_agg = baseline_exp.get("aggregate_info_loss", {})
    omniflow_agg = omniflow_exp.get("aggregate_info_loss", {})
    baseline_sa = baseline_exp.get("scaling_accuracy", {})
    omniflow_sa = omniflow_exp.get("scaling_accuracy", {})

    analysis["comparison"] = {
        "info_loss": {
            "baseline_sample_ratio": baseline_agg.get("sample_ratio_mean"),
            "omniflow_sample_ratio": omniflow_agg.get("sample_ratio_mean"),
            "baseline_nrmse": baseline_agg.get("nrmse_mean"),
            "omniflow_nrmse": omniflow_agg.get("nrmse_mean"),
        },
        "scaling": {
            "baseline_scale_up_detected": baseline_sa.get("scale_up_detected"),
            "omniflow_scale_up_detected": omniflow_sa.get("scale_up_detected"),
            "baseline_scale_up_latency_s": baseline_sa.get("scale_up_latency_s"),
            "omniflow_scale_up_latency_s": omniflow_sa.get("scale_up_latency_s"),
            "baseline_scale_down_detected": baseline_sa.get("scale_down_detected"),
            "omniflow_scale_down_detected": omniflow_sa.get("scale_down_detected"),
            "baseline_scale_down_latency_s": baseline_sa.get("scale_down_latency_s"),
            "omniflow_scale_down_latency_s": omniflow_sa.get("scale_down_latency_s"),
            "baseline_peak_replicas": baseline_sa.get("peak_replicas"),
            "omniflow_peak_replicas": omniflow_sa.get("peak_replicas"),
            "baseline_under_provisioned_s": baseline_sa.get("under_provisioned_s"),
            "omniflow_under_provisioned_s": omniflow_sa.get("under_provisioned_s"),
            "baseline_flapping_events": baseline_sa.get("flapping_events"),
            "omniflow_flapping_events": omniflow_sa.get("flapping_events"),
            "baseline_scale_up_events": baseline_sa.get("scale_up_events"),
            "omniflow_scale_up_events": omniflow_sa.get("scale_up_events"),
            "baseline_scale_down_events": baseline_sa.get("scale_down_events"),
            "omniflow_scale_down_events": omniflow_sa.get("scale_down_events"),
        },
    }

    with open(args.out, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {args.out}", file=sys.stderr)

    if args.tex:
        Path(args.tex).write_text(_render_latex_table(analysis))
        print(f"LaTeX table saved to {args.tex}", file=sys.stderr)


if __name__ == "__main__":
    main()
