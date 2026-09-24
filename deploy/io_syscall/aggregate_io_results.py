#!/usr/bin/env python3
"""Aggregate repeated I/O experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_json(path_str: str, pattern: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates = sorted(path.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} under {path}")
    return candidates[-1]


def _load_json(path_str: str, pattern: str) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_json(path_str, pattern)
    return resolved, json.loads(resolved.read_text())


def _summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _aggregate_metric_rows(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float] | None]:
    keys = sorted({key for row in rows for key in row})
    aggregated: dict[str, dict[str, float] | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        aggregated[key] = _summarize_values(values)
    return aggregated


def _status_counts(statuses: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _pct_saved(reference: float | None, value: float | None) -> float | None:
    if reference in (None, 0) or value is None:
        return None
    return (1.0 - value / reference) * 100.0


def _workload_warnings(payload: dict[str, Any]) -> list[str]:
    warnings = _get_nested(payload, "comparisons", "warnings") or []
    return [str(warning) for warning in warnings if isinstance(warning, str)]


def _workload_metric_row(payload: dict[str, Any]) -> dict[str, float | None]:
    return {
        "baseline_mean_throughput": _get_nested(payload, "scenarios", "baseline", "throughput", "mean"),
        "full_mean_throughput": _get_nested(payload, "scenarios", "full", "throughput", "mean"),
        "adaptive_mean_throughput": _get_nested(payload, "scenarios", "adaptive", "throughput", "mean"),
        "full_vs_baseline_pct": _get_nested(payload, "comparisons", "throughput_loss_pct", "full_vs_baseline"),
        "adaptive_vs_baseline_pct": _get_nested(payload, "comparisons", "throughput_loss_pct", "adaptive_vs_baseline"),
        "adaptive_vs_full_pct": _get_nested(payload, "comparisons", "throughput_loss_pct", "adaptive_vs_full"),
        "adaptive_sample_ratio_pct": _get_nested(payload, "comparisons", "data_capture", "adaptive_sample_ratio_pct"),
        "adaptive_data_saved_vs_full_pct": _get_nested(payload, "comparisons", "data_capture", "adaptive_data_saved_vs_full_pct"),
        "adaptive_peak_recall_top5_pct": _get_nested(payload, "comparisons", "data_capture", "adaptive_peak_recall_top5_pct"),
        "full_mean_cpu_pct": _get_nested(payload, "comparisons", "cpu", "full_mean_cpu_pct"),
        "adaptive_mean_cpu_pct": _get_nested(payload, "comparisons", "cpu", "adaptive_mean_cpu_pct"),
        "adaptive_mean_cpu_saved_vs_full_pct": _get_nested(payload, "comparisons", "cpu", "adaptive_mean_cpu_saved_vs_full_pct"),
        "full_total_cpu_s": _get_nested(payload, "comparisons", "cpu", "full_total_cpu_s"),
        "adaptive_total_cpu_s": _get_nested(payload, "comparisons", "cpu", "adaptive_total_cpu_s"),
        "adaptive_total_cpu_saved_vs_full_pct": _get_nested(payload, "comparisons", "cpu", "adaptive_total_cpu_saved_vs_full_pct"),
        "full_mean_urgency": _get_nested(payload, "comparisons", "urgency", "full_mean_urgency"),
        "adaptive_mean_urgency": _get_nested(payload, "comparisons", "urgency", "adaptive_mean_urgency"),
        "adaptive_replay_mean_urgency": _get_nested(payload, "comparisons", "urgency", "adaptive_replay_mean_urgency"),
        "adaptive_replay_median_urgency": _get_nested(payload, "comparisons", "urgency", "adaptive_replay_median_urgency"),
        "adaptive_replay_urgency_ticks": _get_nested(payload, "comparisons", "urgency", "adaptive_replay_urgency_ticks"),
        "full_active_mean_us": _get_nested(payload, "comparisons", "active_overhead", "full_active_mean_us"),
        "adaptive_active_mean_us": _get_nested(payload, "comparisons", "active_overhead", "adaptive_active_mean_us"),
        "adaptive_active_saved_vs_full_pct": _get_nested(payload, "comparisons", "active_overhead", "adaptive_active_saved_vs_full_pct"),
        "full_probe_fetch_mean_us": _get_nested(payload, "comparisons", "probe_fetch", "full_probe_fetch_mean_us"),
        "adaptive_probe_fetch_mean_us": _get_nested(payload, "comparisons", "probe_fetch", "adaptive_probe_fetch_mean_us"),
        "adaptive_probe_fetch_mean_saved_vs_full_pct": _get_nested(payload, "comparisons", "probe_fetch", "adaptive_probe_fetch_mean_saved_vs_full_pct"),
        "full_probe_fetch_total_s": _get_nested(payload, "comparisons", "probe_fetch", "full_probe_fetch_total_s"),
        "adaptive_probe_fetch_total_s": _get_nested(payload, "comparisons", "probe_fetch", "adaptive_probe_fetch_total_s"),
        "adaptive_probe_fetch_total_saved_vs_full_pct": _get_nested(payload, "comparisons", "probe_fetch", "adaptive_probe_fetch_total_saved_vs_full_pct"),
        "full_post_fetch_control_total_s": _get_nested(payload, "comparisons", "post_fetch_control", "full_post_fetch_control_total_s"),
        "adaptive_post_fetch_control_total_s": _get_nested(payload, "comparisons", "post_fetch_control", "adaptive_post_fetch_control_total_s"),
        "adaptive_post_fetch_control_saved_vs_full_pct": _get_nested(payload, "comparisons", "post_fetch_control", "adaptive_post_fetch_control_saved_vs_full_pct"),
        "full_publish_path_total_s": _get_nested(payload, "comparisons", "publish_path", "full_publish_path_total_s"),
        "adaptive_publish_path_total_s": _get_nested(payload, "comparisons", "publish_path", "adaptive_publish_path_total_s"),
        "adaptive_publish_path_saved_vs_full_pct": _get_nested(payload, "comparisons", "publish_path", "adaptive_publish_path_saved_vs_full_pct"),
        "tracked_metric_corr": _get_nested(payload, "comparisons", "tracked_metric_quality", "corr"),
        "tracked_metric_nrmse_p95": _get_nested(payload, "comparisons", "tracked_metric_quality", "nrmse_p95"),
        "tracked_metric_nrmse_mean": _get_nested(payload, "comparisons", "tracked_metric_quality", "nrmse_mean"),
        "tracked_metric_overlap_pct": _get_nested(payload, "comparisons", "tracked_metric_quality", "overlap_pct_of_adaptive_samples"),
        "tracked_metric_best_lag_ticks": _get_nested(payload, "comparisons", "tracked_metric_quality", "best_lag_ticks"),
    }


def _eval_metric_row(payload: dict[str, Any]) -> dict[str, float | None]:
    overhead = payload.get("overhead_summary") or {}
    cost_accounting = payload.get("cost_accounting") or {}

    def _stat(key: str, field: str) -> float | None:
        stats = overhead.get(key)
        if not isinstance(stats, dict):
            return None
        value = stats.get(field)
        return None if value is None else float(value)

    active_mean = _stat("active_loop_us", "mean")
    if active_mean is None:
        component_keys = [
            "read_overhead_us",
            "tracker_overhead_us",
            "adaptive_overhead_us",
            "metrics_overhead_us",
        ]
        if _stat("emit_overhead_us", "mean") is not None:
            component_keys.append("emit_overhead_us")
        if all(_stat(key, "mean") is not None for key in component_keys):
            active_mean = float(sum(_stat(key, "mean") or 0.0 for key in component_keys))

    cpu_normalization = payload.get("cpu_normalization") or {}
    total_cpu = _maybe_float(cpu_normalization.get("total_cpu_s"))
    if total_cpu is None:
        total_cpu = _stat("monitor_total_cpu_s_total", "max")
    if total_cpu is None:
        user_total = _stat("monitor_user_cpu_s_total", "max")
        system_total = _stat("monitor_system_cpu_s_total", "max")
        if user_total is not None or system_total is not None:
            total_cpu = float((user_total or 0.0) + (system_total or 0.0))

    mean_cpu_pct = _maybe_float(cpu_normalization.get("mean_cpu_pct_over_wall"))
    snapshot_mean_cpu_pct = _maybe_float(cpu_normalization.get("snapshot_mean_cpu_pct"))
    if mean_cpu_pct is None:
        mean_cpu_pct = _stat("monitor_cpu_pct", "mean")
    if snapshot_mean_cpu_pct is None:
        snapshot_mean_cpu_pct = _stat("monitor_cpu_pct", "mean")

    return {
        "live_sample_ratio": 1.0 if payload.get("sampling_summary") is None else _maybe_float(
            (payload.get("sampling_summary") or {}).get("sample_ratio")
        ),
        "live_max_gap": 1.0 if payload.get("sampling_summary") is None else _maybe_float(
            (payload.get("sampling_summary") or {}).get("max_gap")
        ),
        "mean_cpu_pct": mean_cpu_pct,
        "snapshot_mean_cpu_pct": snapshot_mean_cpu_pct,
        "total_cpu_s": total_cpu,
        "mean_urgency": _maybe_float(_get_nested(payload, "scheduler_summary", "mean_urgency"))
        or _maybe_float(_get_nested(payload, "sampling_summary", "mean_urgency")),
        "active_mean_us": active_mean,
        "read_mean_us": _stat("read_overhead_us", "mean"),
        "probe_fetch_mean_us": _maybe_float(cost_accounting.get("probe_fetch_mean_us")) or _stat("read_overhead_us", "mean"),
        "probe_fetch_total_s": _maybe_float(cost_accounting.get("probe_fetch_total_s_estimated")),
        "post_fetch_control_total_s": _maybe_float(cost_accounting.get("post_fetch_control_total_s_estimated")),
        "publish_path_total_s": _maybe_float(cost_accounting.get("publish_path_total_s_estimated")),
        "tracker_mean_us": _stat("tracker_overhead_us", "mean"),
        "adaptive_mean_us": _stat("adaptive_overhead_us", "mean"),
        "metrics_mean_us": _stat("metrics_overhead_us", "mean"),
        "emit_mean_us": _stat("emit_overhead_us", "mean"),
    }


def _eval_replay_metric_row(payload: dict[str, Any]) -> dict[str, float | None]:
    info_loss = payload.get("info_loss") or {}
    urgency = payload.get("replay_urgency_summary") or {}
    scheduler = payload.get("scheduler_summary") or {}
    mean_urgency = _maybe_float(urgency.get("mean"))
    median_urgency = _maybe_float(urgency.get("median"))
    if mean_urgency is None:
        mean_urgency = _maybe_float(scheduler.get("replay_mean_urgency"))
    if median_urgency is None:
        median_urgency = _maybe_float(scheduler.get("replay_median_urgency"))
    return {
        "sample_ratio": _maybe_float(info_loss.get("sample_ratio")),
        "peak_recall_top5": _maybe_float(info_loss.get("peak_recall_top5")),
        "nrmse_mean": _maybe_float(info_loss.get("nrmse_mean")),
        "correlation": _maybe_float(info_loss.get("correlation")),
        "max_gap": _maybe_float(info_loss.get("max_gap")),
        "mean_urgency": mean_urgency,
        "median_urgency": median_urgency,
    }


def _eval_stability(payload: dict[str, Any]) -> dict[str, Any]:
    assessment = payload.get("stability_assessment") or {}
    if not isinstance(assessment, dict):
        return {"status": "unknown", "score": None, "reasons": [], "hints": [], "dominant_components": []}
    status = assessment.get("status")
    return {
        "status": str(status) if isinstance(status, str) else "unknown",
        "score": assessment.get("score"),
        "reasons": assessment.get("reasons") or [],
        "hints": assessment.get("hints") or [],
        "dominant_components": assessment.get("dominant_components") or [],
    }


def _trace_path_for_entry(entry: str, resolved: Path) -> Path:
    candidate = Path(entry)
    if candidate.is_dir():
        return candidate / "trace.json"
    return resolved.parent / "trace.json"


def _run_label_for_trace(trace_path: Path) -> str:
    parent = trace_path.parent
    if parent.name in {"full", "adaptive", "adaptive_replay"} and parent.parent != parent:
        return parent.parent.name
    return parent.name


def _load_trace_series(trace_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(trace_path.read_text())
    max_t = max(int(row["t"]) for row in rows)
    values = np.full(max_t + 1, np.nan, dtype=np.float64)
    sampled = np.zeros(max_t + 1, dtype=bool)
    for row in rows:
        idx = int(row["t"])
        value = row.get("value")
        if value is not None:
            values[idx] = float(value)
            sampled[idx] = True
    return values, sampled


def _comparability_for_lag(
        full_values: np.ndarray,
        adaptive_values: np.ndarray,
        adaptive_sampled: np.ndarray,
        lag_ticks: int,
) -> dict[str, float] | None:
    if lag_ticks >= 0:
        full_segment = full_values[lag_ticks:]
        adaptive_segment = adaptive_values[:max(0, len(full_values) - lag_ticks)]
        sampled_segment = adaptive_sampled[:max(0, len(full_values) - lag_ticks)]
    else:
        shift = -lag_ticks
        full_segment = full_values[:max(0, len(full_values) - shift)]
        adaptive_segment = adaptive_values[shift:]
        sampled_segment = adaptive_sampled[shift:]

    n = min(len(full_segment), len(adaptive_segment), len(sampled_segment))
    full_segment = full_segment[:n]
    adaptive_segment = adaptive_segment[:n]
    sampled_segment = sampled_segment[:n]
    mask = sampled_segment & np.isfinite(full_segment) & np.isfinite(adaptive_segment)
    overlap_count = int(mask.sum())
    if overlap_count < 30:
        return None

    x = full_segment[mask]
    y = adaptive_segment[mask]
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    scale = float(np.percentile(np.abs(x), 95))
    nrmse_p95 = rmse / scale if scale > 0.0 else float("nan")
    return {
        "overlap_count": float(overlap_count),
        "corr": corr,
        "nrmse_p95": nrmse_p95,
    }


def _assess_live_comparability(
        full_trace_path: Path,
        adaptive_trace_path: Path,
) -> dict[str, Any]:
    full_values, _ = _load_trace_series(full_trace_path)
    adaptive_values, adaptive_sampled = _load_trace_series(adaptive_trace_path)
    adaptive_sampled_count = int(adaptive_sampled.sum())

    best: tuple[float, int, dict[str, float]] | None = None
    for lag_ticks in range(-12, 13):
        metrics = _comparability_for_lag(full_values, adaptive_values, adaptive_sampled, lag_ticks)
        if metrics is None:
            continue
        score = float(metrics["corr"] - metrics["nrmse_p95"])
        if best is None or score > best[0]:
            best = (score, lag_ticks, metrics)

    if best is None:
        return {
            "status": "unknown",
            "best_lag_ticks": None,
            "overlap_count": 0,
            "overlap_pct_of_adaptive_samples": None,
            "corr": None,
            "nrmse_p95": None,
            "notes": ["Not enough overlapping sampled points to assess comparability."],
        }

    _, lag_ticks, metrics = best
    corr = float(metrics["corr"])
    nrmse_p95 = float(metrics["nrmse_p95"])
    overlap_count = int(metrics["overlap_count"])

    notes: list[str] = []
    if abs(lag_ticks) >= 6:
        notes.append(f"Best alignment required a {lag_ticks}-tick shift, which suggests cycle-start skew between live runs.")
    if corr < 0.75:
        notes.append(f"Aligned waveform correlation stayed modest ({corr:.3f}).")
    if nrmse_p95 > 0.35:
        notes.append(f"Aligned waveform error remained elevated (nrmse_p95={nrmse_p95:.3f}).")

    status = "comparable"
    if corr >= 0.75 and nrmse_p95 <= 0.35:
        status = "comparable"
    elif corr >= 0.55 and nrmse_p95 <= 0.60:
        status = "suspect"
    else:
        status = "non-comparable"

    return {
        "status": status,
        "best_lag_ticks": lag_ticks,
        "overlap_count": overlap_count,
        "overlap_pct_of_adaptive_samples": None if adaptive_sampled_count == 0 else (overlap_count / adaptive_sampled_count) * 100.0,
        "corr": corr,
        "nrmse_p95": nrmse_p95,
        "notes": notes,
    }


def _aggregate_workload(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, float | None]] = []
    sources: list[str] = []
    warnings: list[str] = []
    excluded_sources: list[str] = []
    excluded_warnings: list[str] = []
    metric_meta: dict[str, Any] | None = None
    for summary_path in args.summary:
        resolved, payload = _load_json(summary_path, "workload_overhead_summary.json")
        payload_warnings = _workload_warnings(payload)
        if payload_warnings:
            excluded_sources.append(str(resolved))
            excluded_warnings.extend(payload_warnings)
            if metric_meta is None and isinstance(payload.get("metric"), dict):
                metric_meta = dict(payload["metric"])
            continue
        rows.append(_workload_metric_row(payload))
        sources.append(str(resolved))
        if metric_meta is None and isinstance(payload.get("metric"), dict):
            metric_meta = dict(payload["metric"])

    metrics = _aggregate_metric_rows(rows)

    def _mean(key: str) -> float | None:
        return _get_nested(metrics, key, "mean")

    return {
        "kind": args.kind,
        "metric": metric_meta,
        "count": len(rows),
        "input_count": len(args.summary),
        "excluded_count": len(excluded_sources),
        "sources": sources,
        "excluded_sources": excluded_sources,
        "metrics": metrics,
        "comparisons": {
            "throughput": {
                "baseline_mean_throughput": _mean("baseline_mean_throughput"),
                "full_mean_throughput": _mean("full_mean_throughput"),
                "adaptive_mean_throughput": _mean("adaptive_mean_throughput"),
                "full_vs_baseline_pct": _mean("full_vs_baseline_pct"),
                "adaptive_vs_baseline_pct": _mean("adaptive_vs_baseline_pct"),
                "adaptive_vs_full_pct": _mean("adaptive_vs_full_pct"),
            },
            "data_capture": {
                "adaptive_sample_ratio_pct": _mean("adaptive_sample_ratio_pct"),
                "adaptive_data_saved_vs_full_pct": _mean("adaptive_data_saved_vs_full_pct"),
                "adaptive_peak_recall_top5_pct": _mean("adaptive_peak_recall_top5_pct"),
            },
            "cpu": {
                "full_mean_cpu_pct": _mean("full_mean_cpu_pct"),
                "adaptive_mean_cpu_pct": _mean("adaptive_mean_cpu_pct"),
                "adaptive_mean_cpu_saved_vs_full_pct": _mean("adaptive_mean_cpu_saved_vs_full_pct"),
                "full_total_cpu_s": _mean("full_total_cpu_s"),
                "adaptive_total_cpu_s": _mean("adaptive_total_cpu_s"),
                "adaptive_total_cpu_saved_vs_full_pct": _mean("adaptive_total_cpu_saved_vs_full_pct"),
            },
            "urgency": {
                "full_mean_urgency": _mean("full_mean_urgency"),
                "adaptive_mean_urgency": _mean("adaptive_mean_urgency"),
                "adaptive_replay_mean_urgency": _mean("adaptive_replay_mean_urgency"),
                "adaptive_replay_median_urgency": _mean("adaptive_replay_median_urgency"),
                "adaptive_replay_urgency_ticks": _mean("adaptive_replay_urgency_ticks"),
                "adaptive_replay_urgency_ticks": _mean("adaptive_replay_urgency_ticks"),
            },
            "active_overhead": {
                "full_active_mean_us": _mean("full_active_mean_us"),
                "adaptive_active_mean_us": _mean("adaptive_active_mean_us"),
                "adaptive_active_saved_vs_full_pct": _mean("adaptive_active_saved_vs_full_pct"),
            },
            "probe_fetch": {
                "full_probe_fetch_mean_us": _mean("full_probe_fetch_mean_us"),
                "adaptive_probe_fetch_mean_us": _mean("adaptive_probe_fetch_mean_us"),
                "adaptive_probe_fetch_mean_saved_vs_full_pct": _mean("adaptive_probe_fetch_mean_saved_vs_full_pct"),
                "full_probe_fetch_total_s": _mean("full_probe_fetch_total_s"),
                "adaptive_probe_fetch_total_s": _mean("adaptive_probe_fetch_total_s"),
                "adaptive_probe_fetch_total_saved_vs_full_pct": _mean("adaptive_probe_fetch_total_saved_vs_full_pct"),
            },
            "post_fetch_control": {
                "full_post_fetch_control_total_s": _mean("full_post_fetch_control_total_s"),
                "adaptive_post_fetch_control_total_s": _mean("adaptive_post_fetch_control_total_s"),
                "adaptive_post_fetch_control_saved_vs_full_pct": _mean("adaptive_post_fetch_control_saved_vs_full_pct"),
            },
            "publish_path": {
                "full_publish_path_total_s": _mean("full_publish_path_total_s"),
                "adaptive_publish_path_total_s": _mean("adaptive_publish_path_total_s"),
                "adaptive_publish_path_saved_vs_full_pct": _mean("adaptive_publish_path_saved_vs_full_pct"),
            },
            "tracked_metric_quality": {
                "corr": _mean("tracked_metric_corr"),
                "nrmse_p95": _mean("tracked_metric_nrmse_p95"),
                "nrmse_mean": _mean("tracked_metric_nrmse_mean"),
                "overlap_pct_of_adaptive_samples": _mean("tracked_metric_overlap_pct"),
                "best_lag_ticks": _mean("tracked_metric_best_lag_ticks"),
            },
        },
        "warnings": sorted(set(warnings)),
        "excluded_warnings": sorted(set(excluded_warnings)),
    }


def _comparison_from_live_arm_means(arms: dict[str, dict[str, dict[str, float] | None]]) -> dict[str, float | None]:
    full = arms.get("full_live") or {}
    adaptive = arms.get("adaptive_live") or {}
    full_sample_ratio = _get_nested(full, "live_sample_ratio", "mean")
    adaptive_sample_ratio = _get_nested(adaptive, "live_sample_ratio", "mean")
    full_mean_cpu = _get_nested(full, "mean_cpu_pct", "mean")
    adaptive_mean_cpu = _get_nested(adaptive, "mean_cpu_pct", "mean")
    full_mean_urgency = _get_nested(full, "mean_urgency", "mean")
    adaptive_mean_urgency = _get_nested(adaptive, "mean_urgency", "mean")
    full_total_cpu = _get_nested(full, "total_cpu_s", "mean")
    adaptive_total_cpu = _get_nested(adaptive, "total_cpu_s", "mean")
    full_active = _get_nested(full, "active_mean_us", "mean")
    adaptive_active = _get_nested(adaptive, "active_mean_us", "mean")
    full_probe_fetch_mean = _get_nested(full, "probe_fetch_mean_us", "mean")
    adaptive_probe_fetch_mean = _get_nested(adaptive, "probe_fetch_mean_us", "mean")
    full_probe_fetch_total = _get_nested(full, "probe_fetch_total_s", "mean")
    adaptive_probe_fetch_total = _get_nested(adaptive, "probe_fetch_total_s", "mean")
    full_post_fetch_total = _get_nested(full, "post_fetch_control_total_s", "mean")
    adaptive_post_fetch_total = _get_nested(adaptive, "post_fetch_control_total_s", "mean")
    full_publish_total = _get_nested(full, "publish_path_total_s", "mean")
    adaptive_publish_total = _get_nested(adaptive, "publish_path_total_s", "mean")

    return {
        "full_live_sample_ratio_pct": None if full_sample_ratio is None else full_sample_ratio * 100.0,
        "adaptive_live_sample_ratio_pct": None if adaptive_sample_ratio is None else adaptive_sample_ratio * 100.0,
        "adaptive_live_vs_full_sample_ratio_pct_points": None
        if adaptive_sample_ratio is None or full_sample_ratio is None
        else (adaptive_sample_ratio - full_sample_ratio) * 100.0,
        "full_live_mean_cpu_pct": full_mean_cpu,
        "adaptive_live_mean_cpu_pct": adaptive_mean_cpu,
        "full_live_mean_urgency": full_mean_urgency,
        "adaptive_live_mean_urgency": adaptive_mean_urgency,
        "adaptive_live_mean_cpu_saved_vs_full_pct": _pct_saved(full_mean_cpu, adaptive_mean_cpu),
        "full_live_total_cpu_s": full_total_cpu,
        "adaptive_live_total_cpu_s": adaptive_total_cpu,
        "adaptive_live_total_cpu_saved_vs_full_pct": _pct_saved(full_total_cpu, adaptive_total_cpu),
        "full_live_active_mean_us": full_active,
        "adaptive_live_active_mean_us": adaptive_active,
        "adaptive_live_active_saved_vs_full_pct": _pct_saved(full_active, adaptive_active),
        "full_live_probe_fetch_mean_us": full_probe_fetch_mean,
        "adaptive_live_probe_fetch_mean_us": adaptive_probe_fetch_mean,
        "adaptive_live_probe_fetch_mean_saved_vs_full_pct": _pct_saved(full_probe_fetch_mean, adaptive_probe_fetch_mean),
        "full_live_probe_fetch_total_s": full_probe_fetch_total,
        "adaptive_live_probe_fetch_total_s": adaptive_probe_fetch_total,
        "adaptive_live_probe_fetch_total_saved_vs_full_pct": _pct_saved(full_probe_fetch_total, adaptive_probe_fetch_total),
        "full_live_post_fetch_control_total_s": full_post_fetch_total,
        "adaptive_live_post_fetch_control_total_s": adaptive_post_fetch_total,
        "adaptive_live_post_fetch_control_saved_vs_full_pct": _pct_saved(full_post_fetch_total, adaptive_post_fetch_total),
        "full_live_publish_path_total_s": full_publish_total,
        "adaptive_live_publish_path_total_s": adaptive_publish_total,
        "adaptive_live_publish_path_saved_vs_full_pct": _pct_saved(full_publish_total, adaptive_publish_total),
    }


def _fidelity_from_replay_means(
        replay_metrics: dict[str, dict[str, float] | None],
        pooled_urgency: dict[str, float] | None,
) -> dict[str, Any]:
    sample_ratio = _get_nested(replay_metrics, "sample_ratio", "mean")
    peak_recall = _get_nested(replay_metrics, "peak_recall_top5", "mean")
    nrmse_mean = _get_nested(replay_metrics, "nrmse_mean", "mean")
    correlation = _get_nested(replay_metrics, "correlation", "mean")
    max_gap = _get_nested(replay_metrics, "max_gap", "mean")
    return {
        "adaptive_replay_sample_ratio_pct": None if sample_ratio is None else sample_ratio * 100.0,
        "adaptive_replay_data_saved_vs_full_pct": None if sample_ratio is None else (1.0 - sample_ratio) * 100.0,
        "adaptive_replay_peak_recall_top5_pct": None if peak_recall is None else peak_recall * 100.0,
        "adaptive_replay_nrmse_mean": nrmse_mean,
        "adaptive_replay_correlation": correlation,
        "adaptive_replay_max_gap": max_gap,
        "adaptive_replay_mean_urgency": _get_nested(replay_metrics, "mean_urgency", "mean"),
        "adaptive_replay_mean_urgency_std": _get_nested(replay_metrics, "mean_urgency", "std"),
        "adaptive_replay_median_of_run_medians": _get_nested(replay_metrics, "median_urgency", "mean"),
        "adaptive_replay_pooled_urgency_count": None if pooled_urgency is None else pooled_urgency.get("count"),
        "adaptive_replay_pooled_mean_urgency": None if pooled_urgency is None else pooled_urgency.get("mean"),
        "adaptive_replay_pooled_median_urgency": None if pooled_urgency is None else pooled_urgency.get("median"),
    }


def _aggregate_eval(args: argparse.Namespace) -> dict[str, Any]:
    def _load_live_records(entries: list[str] | None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for entry in entries or []:
            resolved, payload = _load_json(entry, "io_syscall_results_*.json")
            profile = payload.get("config_profile")
            stability = _eval_stability(payload)
            records.append({
                "resolved": resolved,
                "source": str(resolved),
                "profile": str(profile) if isinstance(profile, str) else None,
                "metrics": _eval_metric_row(payload),
                "stability": {
                    "source": str(resolved),
                    "status": stability["status"],
                    "score": stability["score"],
                    "reasons": stability["reasons"],
                    "hints": stability["hints"],
                    "dominant_components": stability["dominant_components"],
                },
                "trace_path": _trace_path_for_entry(entry, resolved),
            })
        return records

    def _load_replay_records(entries: list[str] | None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for entry in entries or []:
            resolved, payload = _load_json(entry, "io_syscall_results_*.json")
            profile = payload.get("config_profile")
            records.append({
                "resolved": resolved,
                "source": str(resolved),
                "profile": str(profile) if isinstance(profile, str) else None,
                "metrics": _eval_replay_metric_row(payload),
                "urgency_values": list(
                    ((payload.get("replay_urgency_summary") or {}).get("values") or [])
                ),
            })
        return records

    def _summarize_live_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [record["metrics"] for record in records]
        profiles = sorted({profile for profile in (record["profile"] for record in records) if profile})
        stability_rows = [record["stability"] for record in records]
        stable_rows = [record["metrics"] for record in records if record["stability"]["status"] == "stable"]
        non_unstable_rows = [record["metrics"] for record in records if record["stability"]["status"] != "unstable"]
        summary: dict[str, Any] = {
            "count": len(records),
            "profiles": profiles,
            "sources": [record["source"] for record in records],
            "metrics": _aggregate_metric_rows(rows),
            "stability": {
                "counts": _status_counts([str(row["status"]) for row in stability_rows]),
                "sources": stability_rows,
            },
        }
        if stable_rows:
            summary["stable_only_metrics"] = _aggregate_metric_rows(stable_rows)
        if non_unstable_rows and len(non_unstable_rows) != len(records):
            summary["non_unstable_metrics"] = _aggregate_metric_rows(non_unstable_rows)
        return summary

    def _summarize_replay_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [record["metrics"] for record in records]
        profiles = sorted({profile for profile in (record["profile"] for record in records) if profile})
        urgency_values = [
            float(value)
            for record in records
            for value in record.get("urgency_values", [])
        ]
        return {
            "count": len(records),
            "profiles": profiles,
            "sources": [record["source"] for record in records],
            "metrics": _aggregate_metric_rows(rows),
            "pooled_urgency": (
                {
                    "count": len(urgency_values),
                    "mean": float(np.mean(urgency_values)),
                    "median": float(np.median(urgency_values)),
                }
                if urgency_values else None
            ),
        }

    full_live_records = _load_live_records(args.full)
    adaptive_live_records = _load_live_records(args.adaptive)
    adaptive_replay_records = _load_replay_records(args.adaptive_replay)

    arms: dict[str, dict[str, Any]] = {}
    if full_live_records:
        arms["full_live"] = _summarize_live_arm(full_live_records)
    if adaptive_live_records:
        arms["adaptive_live"] = _summarize_live_arm(adaptive_live_records)
    if adaptive_replay_records:
        arms["adaptive_replay"] = _summarize_replay_arm(adaptive_replay_records)

    live_comparability_pairs: list[dict[str, Any]] = []
    comparable_full_rows: list[dict[str, float | None]] = []
    comparable_adaptive_rows: list[dict[str, float | None]] = []
    for full_record, adaptive_record in zip(full_live_records, adaptive_live_records):
        comparability = _assess_live_comparability(full_record["trace_path"], adaptive_record["trace_path"])
        pair = {
            "pair_id": _run_label_for_trace(full_record["trace_path"]),
            "full_source": full_record["source"],
            "adaptive_source": adaptive_record["source"],
            **comparability,
        }
        live_comparability_pairs.append(pair)
        if comparability["status"] == "comparable":
            comparable_full_rows.append(full_record["metrics"])
            comparable_adaptive_rows.append(adaptive_record["metrics"])

    comparable_only_overhead_comparisons = None
    if comparable_full_rows and comparable_adaptive_rows:
        comparable_only_overhead_comparisons = _comparison_from_live_arm_means({
            "full_live": _aggregate_metric_rows(comparable_full_rows),
            "adaptive_live": _aggregate_metric_rows(comparable_adaptive_rows),
        })

    live_comparability = {
        "count": len(live_comparability_pairs),
        "counts": _status_counts([str(pair["status"]) for pair in live_comparability_pairs]),
        "metrics": _aggregate_metric_rows([
            {
                "best_lag_ticks": _maybe_float(pair["best_lag_ticks"]),
                "overlap_count": _maybe_float(pair["overlap_count"]),
                "overlap_pct_of_adaptive_samples": _maybe_float(pair["overlap_pct_of_adaptive_samples"]),
                "corr": _maybe_float(pair["corr"]),
                "nrmse_p95": _maybe_float(pair["nrmse_p95"]),
            }
            for pair in live_comparability_pairs
        ]),
        "pairs": live_comparability_pairs,
    }

    overhead_comparisons = None
    if "full_live" in arms and "adaptive_live" in arms:
        overhead_comparisons = _comparison_from_live_arm_means({
            "full_live": arms["full_live"]["metrics"],
            "adaptive_live": arms["adaptive_live"]["metrics"],
        })

    stable_only_overhead_comparisons = None
    if all(key in arms for key in ["full_live", "adaptive_live"]) and all(
            "stable_only_metrics" in arms[key] for key in ["full_live", "adaptive_live"]):
        stable_only_overhead_comparisons = _comparison_from_live_arm_means({
            "full_live": arms["full_live"]["stable_only_metrics"],
            "adaptive_live": arms["adaptive_live"]["stable_only_metrics"],
        })

    non_unstable_overhead_comparisons = None
    if all(key in arms for key in ["full_live", "adaptive_live"]) and all(
            "non_unstable_metrics" in arms[key] for key in ["full_live", "adaptive_live"]):
        non_unstable_overhead_comparisons = _comparison_from_live_arm_means({
            "full_live": arms["full_live"]["non_unstable_metrics"],
            "adaptive_live": arms["adaptive_live"]["non_unstable_metrics"],
        })

    fidelity = None
    if "adaptive_replay" in arms:
        fidelity = _fidelity_from_replay_means(
            arms["adaptive_replay"]["metrics"],
            arms["adaptive_replay"].get("pooled_urgency"),
        )

    return {
        "label": args.label,
        "arms": arms,
        "fidelity": fidelity,
        "live_comparability": live_comparability,
        "comparisons": overhead_comparisons,
        "overhead_comparisons": overhead_comparisons,
        "stable_only_overhead_comparisons": stable_only_overhead_comparisons,
        "non_unstable_overhead_comparisons": non_unstable_overhead_comparisons,
        "comparable_only_overhead_comparisons": comparable_only_overhead_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated I/O experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    workload = subparsers.add_parser("workload", help="Aggregate workload overhead summaries")
    workload.add_argument("--kind", choices=["postgres", "redis", "latency"], required=True)
    workload.add_argument(
        "--summary",
        nargs="+",
        required=True,
        help="Summary files or directories containing workload_overhead_summary.json",
    )
    workload.add_argument("--out", required=True, help="Output directory")

    eval_parser = subparsers.add_parser("eval", help="Aggregate io_syscall result files")
    eval_parser.add_argument("--label", required=True, help="Experiment label (e.g. latency, floor)")
    eval_parser.add_argument("--baseline", nargs="*", default=None,
                             help="Directories or result files for the baseline arm")
    eval_parser.add_argument("--full", nargs="*", default=None,
                             help="Directories or result files for the full live arm")
    eval_parser.add_argument("--fixed", nargs="*", default=None,
                             help="Directories or result files for the fixed arm")
    eval_parser.add_argument("--adaptive", nargs="*", default=None,
                             help="Directories or result files for the adaptive live arm")
    eval_parser.add_argument("--adaptive-replay", nargs="*", default=None,
                             help="Directories or result files for adaptive replay evaluated on the saved full trace")
    eval_parser.add_argument("--out", required=True, help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "workload":
        summary = _aggregate_workload(args)
        out_path = out_dir / "aggregate_workload_overhead_summary.json"
    else:
        summary = _aggregate_eval(args)
        out_path = out_dir / "aggregate_eval_summary.json"

    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "summary": str(out_path),
        "count": summary.get("count") or len(summary.get("arms", {})),
    }))


if __name__ == "__main__":
    main()
