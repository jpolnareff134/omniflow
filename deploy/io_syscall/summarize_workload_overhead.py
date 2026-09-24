#!/usr/bin/env python3
"""Summarize workload-side overhead across baseline, full, and adaptive runs."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class ThroughputSummary:
    count: int
    mean: float
    p50: float
    p95: float
    max: float


def _metric_meta(kind: str) -> dict[str, object]:
    if kind == "postgres":
        return {
            "series_label": "throughput",
            "summary_label": "throughput",
            "plot_title": "postgres workload comparison: baseline vs full vs adaptive",
            "xlabel": "Elapsed time (s)",
            "ylabel": "TPS",
            "relative_ylabel": "Mean throughput (% of baseline)",
            "higher_is_better": True,
        }
    if kind == "redis":
        return {
            "series_label": "throughput",
            "summary_label": "throughput",
            "plot_title": "redis workload comparison: baseline vs full vs adaptive",
            "xlabel": "Sample index",
            "ylabel": "Requests per second",
            "relative_ylabel": "Mean throughput (% of baseline)",
            "higher_is_better": True,
        }
    return {
        "series_label": "p99_latency_us",
        "summary_label": "p99 latency",
        "plot_title": "latency workload comparison: baseline vs full vs adaptive",
        "xlabel": "Sample index",
        "ylabel": "P99 latency (us)",
        "relative_ylabel": "Mean p99 latency (% of baseline)",
        "higher_is_better": False,
    }


def _parse_pgbench(text: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    pattern = re.compile(r"progress:\s*([0-9.]+)\s*s,\s*([0-9.]+)\s*tps")
    fallback = re.compile(r"tps\s*=\s*([0-9.]+)")

    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            xs.append(float(match.group(1)))
            ys.append(float(match.group(2)))

    if ys:
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    idx = 0.0
    for line in text.splitlines():
        match = fallback.search(line)
        if match:
            idx += 1.0
            xs.append(idx)
            ys.append(float(match.group(1)))

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _parse_redis(text: str) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    pattern = re.compile(
        r"^[A-Z_]+:\s+([0-9.]+)\s+requests per second", re.MULTILINE)
    for match in pattern.finditer(text):
        values.append(float(match.group(1)))

    xs: list[float] = []
    ys: list[float] = []
    idx = 0.0
    if values:
        for start in range(0, len(values), 2):
            chunk = values[start:start + 2]
            idx += 1.0
            xs.append(idx)
            ys.append(float(sum(chunk)))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _parse_latency(text: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    percentile_unit = None
    in_latency_percentiles = False
    unit_scale = {
        "nsec": 1e-3,
        "usec": 1.0,
        "msec": 1e3,
        "sec": 1e6,
    }
    prefix_scale = {
        "": 1.0,
        "k": 1e3,
        "m": 1e6,
        "g": 1e9,
    }
    unit_pattern = re.compile(
        r"^\s*lat percentiles \((nsec|usec|msec|sec)\):", re.IGNORECASE)
    p99_pattern = re.compile(
        r"99(?:\.0+)?th=\[\s*([0-9.]+)\s*([kKmMgG]?)\]", re.IGNORECASE)

    for line in text.splitlines():
        unit_match = unit_pattern.search(line)
        if unit_match:
            percentile_unit = unit_match.group(1).lower()
            in_latency_percentiles = True
            continue

        if in_latency_percentiles and "percentiles (" in line and not unit_match:
            in_latency_percentiles = False

        p99_match = p99_pattern.search(line)
        if in_latency_percentiles and p99_match and percentile_unit is not None:
            base_value = float(p99_match.group(1))
            prefix = p99_match.group(2).lower()
            xs.append(float(len(xs) + 1))
            ys.append(base_value * prefix_scale.get(prefix, 1.0)
                      * unit_scale[percentile_unit])
            in_latency_percentiles = False

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _trim_relative_prefix(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if xs.size == 0 or ys.size == 0:
        return xs, ys, 0.0
    if xs.size < 6:
        return xs - xs[0], ys, float(xs[0])

    p10 = float(np.percentile(ys, 10))
    p90 = float(np.percentile(ys, 90))
    dynamic_range = p90 - p10
    if not np.isfinite(dynamic_range) or dynamic_range <= max(abs(p90) * 0.05, 1e-6):
        return xs - xs[0], ys, float(xs[0])

    window = max(3, min(7, ys.size // 8))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth = np.convolve(ys, kernel, mode="valid")
    threshold = p10 + dynamic_range * 0.25
    start_idx = 0
    for idx, value in enumerate(smooth):
        if value >= threshold:
            start_idx = idx
            break

    return xs[start_idx:] - xs[start_idx], ys[start_idx:], float(xs[start_idx])


def _summarize(values: np.ndarray) -> ThroughputSummary | None:
    if values.size == 0:
        return None
    return ThroughputSummary(
        count=int(values.size),
        mean=float(np.mean(values)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        max=float(np.max(values)),
    )


def _pct_change(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or reference == 0:
        return None
    return ((value - reference) / reference) * 100.0


def _find_latest_eval_result(eval_dir: str | None) -> Path | None:
    if not eval_dir:
        return None
    base = Path(eval_dir)
    candidates = sorted(base.glob("io_syscall_results_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def _read_text(path_str: str | None) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text()


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _pct_saved(reference: float | None, value: float | None) -> float | None:
    if reference in (None, 0) or value is None:
        return None
    return (1.0 - value / reference) * 100.0


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


def _weighted_trace_mean(trace_path: Path, key: str) -> float | None:
    rows = json.loads(trace_path.read_text())
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        weight = max(_as_int(row.get("sample_window_ticks"), 1), 1)
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def _extract_replay_quality(eval_dir: str | None) -> dict[str, object] | None:
    result_path = _find_latest_eval_result(eval_dir)
    if result_path is None:
        return None

    payload = json.loads(result_path.read_text())
    info_loss = payload.get("info_loss") or {}
    return {
        "status": "replay",
        "corr": _maybe_float(info_loss.get("correlation")),
        "nrmse_p95": _maybe_float(info_loss.get("nrmse_p95")) or _maybe_float(info_loss.get("nrmse_mean")),
        "nrmse_mean": _maybe_float(info_loss.get("nrmse_mean")),
        "peak_recall_top5": _maybe_float(info_loss.get("peak_recall_top5")),
        "sample_ratio": _maybe_float(info_loss.get("sample_ratio")),
        "notes": [],
    }


def _extract_replay_urgency(eval_dir: str | None) -> dict[str, object] | None:
    """Read urgency computed by the offline poller, not from the source trace."""
    result_path = _find_latest_eval_result(eval_dir)
    if result_path is None:
        return None

    payload = json.loads(result_path.read_text())
    replay = payload.get("replay_urgency_summary") or {}
    scheduler = payload.get("scheduler_summary") or {}
    mean = _maybe_float(replay.get("mean"))
    median = _maybe_float(replay.get("median"))
    if mean is None:
        mean = _maybe_float(scheduler.get("replay_mean_urgency"))
    if median is None:
        median = _maybe_float(scheduler.get("replay_median_urgency"))
    if mean is None and median is None:
        return None
    return {
        "mean": mean,
        "median": median,
        "count": _as_int(replay.get("count"), 0),
        "aggregation": replay.get("aggregation"),
    }


def _comparability_for_lag(
        full_values: np.ndarray,
        adaptive_values: np.ndarray,
        adaptive_sampled: np.ndarray,
        lag_ticks: int,
) -> dict[str, float] | None:
    if lag_ticks >= 0:
        full_segment = full_values[lag_ticks:]
        adaptive_segment = adaptive_values[:max(
            0, len(full_values) - lag_ticks)]
        sampled_segment = adaptive_sampled[:max(
            0, len(full_values) - lag_ticks)]
    else:
        shift = -lag_ticks
        full_segment = full_values[:max(0, len(full_values) - shift)]
        adaptive_segment = adaptive_values[shift:]
        sampled_segment = adaptive_sampled[shift:]

    n = min(len(full_segment), len(adaptive_segment), len(sampled_segment))
    full_segment = full_segment[:n]
    adaptive_segment = adaptive_segment[:n]
    sampled_segment = sampled_segment[:n]
    mask = sampled_segment & np.isfinite(
        full_segment) & np.isfinite(adaptive_segment)
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
) -> dict[str, object]:
    full_values, _ = _load_trace_series(full_trace_path)
    adaptive_values, adaptive_sampled = _load_trace_series(adaptive_trace_path)
    adaptive_sampled_count = int(adaptive_sampled.sum())

    best: tuple[float, int, dict[str, float]] | None = None
    for lag_ticks in range(-12, 13):
        metrics = _comparability_for_lag(
            full_values, adaptive_values, adaptive_sampled, lag_ticks)
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
        notes.append(
            f"Best alignment required a {lag_ticks}-tick shift, which suggests cycle-start skew between live runs.")
    if corr < 0.75:
        notes.append(
            f"Aligned waveform correlation stayed modest ({corr:.3f}).")
    if nrmse_p95 > 0.35:
        notes.append(
            f"Aligned waveform error remained elevated (nrmse_p95={nrmse_p95:.3f}).")

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


def _extract_monitoring_summary(eval_dir: str | None) -> dict[str, object] | None:
    result_path = _find_latest_eval_result(eval_dir)
    if result_path is None:
        return None

    payload = json.loads(result_path.read_text())
    info_loss = payload.get("info_loss") or {}
    sampling_summary = payload.get("sampling_summary") or {}
    scheduler_summary = payload.get("scheduler_summary") or {}
    overhead = payload.get("overhead_summary") or {}
    config = payload.get("config") or {}
    cpu_normalization = payload.get("cpu_normalization") or {}
    cost_accounting = payload.get("cost_accounting") or {}
    data_source = info_loss if info_loss else sampling_summary

    def _stat(key: str, field: str) -> float | None:
        stats = overhead.get(key)
        if not isinstance(stats, dict):
            return None
        return _maybe_float(stats.get(field))

    def _active_stat(field: str) -> float | None:
        active_loop = _stat("active_loop_us", field)
        if active_loop is not None:
            return active_loop

        component_keys = [
            "read_overhead_us",
            "tracker_overhead_us",
            "adaptive_overhead_us",
            "metrics_overhead_us",
        ]
        if _stat("emit_overhead_us", field) is not None:
            component_keys.append("emit_overhead_us")
        if not all(_stat(key, field) is not None for key in component_keys):
            return None
        return float(sum(_stat(key, field) or 0.0 for key in component_keys))

    active_mean = _active_stat("mean")
    active_p95 = _active_stat("p95")
    active_max = _active_stat("max")

    user_total_s = _stat("monitor_user_cpu_s_total", "max")
    system_total_s = _stat("monitor_system_cpu_s_total", "max")
    cpu_total_s = _maybe_float(cpu_normalization.get("total_cpu_s"))
    if cpu_total_s is None and (user_total_s is not None or system_total_s is not None):
        cpu_total_s = float((user_total_s or 0.0) + (system_total_s or 0.0))

    mean_cpu_pct = _maybe_float(
        cpu_normalization.get("mean_cpu_pct_over_wall"))
    if mean_cpu_pct is None:
        mean_cpu_pct = _stat("monitor_cpu_pct", "mean")

    trace_path = result_path.parent / "trace.json"
    mean_urgency = _maybe_float(scheduler_summary.get("mean_urgency"))
    if mean_urgency is None and trace_path.exists():
        mean_urgency = _weighted_trace_mean(trace_path, "urgency")

    return {
        "eval_result": str(result_path),
        "trace_path": str(trace_path) if trace_path.exists() else None,
        "config": {
            "min_interval": config.get("min_interval"),
            "max_interval": config.get("max_interval"),
        },
        "data": {
            "sample_ratio": _maybe_float(data_source.get("sample_ratio")),
            "sampled_points": data_source.get("n_sampled"),
            "total_points": data_source.get("n_total"),
            "peak_recall_top5": _maybe_float(info_loss.get("peak_recall_top5")),
            "nrmse_mean": _maybe_float(info_loss.get("nrmse_mean")),
            "max_gap": data_source.get("max_gap"),
        },
        "cpu": {
            "mean_cpu_pct": mean_cpu_pct,
            "snapshot_mean_cpu_pct": _maybe_float(cpu_normalization.get("snapshot_mean_cpu_pct")),
            "p95_cpu_pct": _stat("monitor_cpu_pct", "p95"),
            "max_cpu_pct": _stat("monitor_cpu_pct", "max"),
            "user_cpu_total_s": user_total_s,
            "system_cpu_total_s": system_total_s,
            "cpu_total_s": cpu_total_s,
        },
        "scheduler": {
            "mean_urgency": mean_urgency,
            "source_trace_mean_urgency": _maybe_float(
                scheduler_summary.get("source_trace_mean_urgency")
            ) or mean_urgency,
            "mean_interval": _maybe_float(scheduler_summary.get("mean_interval")),
            "mean_sample_ratio_live": _maybe_float(scheduler_summary.get("mean_sample_ratio_live")),
        },
        "overhead": {
            "active_mean_us": active_mean,
            "active_p95_us": active_p95,
            "active_max_us": active_max,
            "active_loop_mean_us": _stat("active_loop_us", "mean"),
            "read_mean_us": _stat("read_overhead_us", "mean"),
            "probe_fetch_mean_us": _maybe_float(cost_accounting.get("probe_fetch_mean_us")) or _stat("read_overhead_us", "mean"),
            "probe_fetch_total_s": _maybe_float(cost_accounting.get("probe_fetch_total_s_estimated")),
            "post_fetch_control_total_s": _maybe_float(cost_accounting.get("post_fetch_control_total_s_estimated")),
            "publish_path_total_s": _maybe_float(cost_accounting.get("publish_path_total_s_estimated")),
            "tracker_mean_us": _stat("tracker_overhead_us", "mean"),
            "adaptive_mean_us": _stat("adaptive_overhead_us", "mean"),
            "metrics_mean_us": _stat("metrics_overhead_us", "mean"),
            "emit_mean_us": _stat("emit_overhead_us", "mean"),
        },
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _fmt_num(value: float | None, scale: float = 1.0, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value / scale:.3f}{suffix}"


def _sample_count(summary: ThroughputSummary | None) -> int:
    return 0 if summary is None else int(summary.count)


def _throughput_confidence_warnings(
        kind: str,
        baseline_summary: ThroughputSummary | None,
        full_summary: ThroughputSummary | None,
        adaptive_summary: ThroughputSummary | None,
) -> list[str]:
    warnings: list[str] = []
    counts = [
        _sample_count(baseline_summary),
        _sample_count(full_summary),
        _sample_count(adaptive_summary),
    ]
    min_count = min(
        counts,
    )
    threshold = 8
    if min_count < threshold:
        warnings.append(
            f"Low-confidence throughput comparison: only {min_count} workload samples in at least one arm;"
            f" need >= {threshold} for a stable mean."
        )
    if kind == "latency":
        max_count = max(counts)
        if min_count > 0 and max_count >= 2 * min_count:
            warnings.append(
                "Latency sample-count mismatch across arms: one arm contributed at least 2x more p99 samples"
                " than another. This usually indicates stale pod history or inconsistent log collection."
            )
    return warnings


def _latency_alignment_warnings(
        baseline_summary: ThroughputSummary | None,
        full_summary: ThroughputSummary | None,
        adaptive_summary: ThroughputSummary | None,
        baseline_trim_start: float,
        full_trim_start: float,
        adaptive_trim_start: float,
) -> list[str]:
    warnings: list[str] = []
    counts = [
        _sample_count(baseline_summary),
        _sample_count(full_summary),
        _sample_count(adaptive_summary),
    ]
    if max(counts) != min(counts):
        warnings.append(
            "Latency arms retained materially different sample counts after trimming;"
            " inspect workload log alignment before trusting mean p99 deltas."
        )
    trim_starts = [baseline_trim_start, full_trim_start, adaptive_trim_start]
    if max(trim_starts) - min(trim_starts) > 1.0:
        warnings.append(
            "Latency arms trimmed at different relative start points;"
            " the plotted phase alignment may be unreliable for this repeat."
        )
    return warnings


def _plot(
        kind: str,
        baseline_x: np.ndarray,
        baseline_y: np.ndarray,
        full_x: np.ndarray,
        full_y: np.ndarray,
        adaptive_x: np.ndarray,
        adaptive_y: np.ndarray,
        comparisons: dict[str, object],
        out_path: Path,
) -> None:
    metric_meta = _metric_meta(kind)
    metric_label = str(metric_meta["summary_label"])
    fig, axes = plt.subplots(3, 1, figsize=(
        13, 12), height_ratios=[2.4, 1.2, 1.6])

    ax = axes[0]
    if baseline_y.size:
        ax.plot(baseline_x, baseline_y, label="baseline (no monitoring)",
                linewidth=1.3, color="#2f7d32")
    if full_y.size:
        ax.plot(full_x, full_y, label="full (min=max=1)",
                linewidth=1.3, color="#c0392b")
    if adaptive_y.size:
        ax.plot(adaptive_x, adaptive_y, label="adaptive",
                linewidth=1.3, color="#1f77b4")
    ax.set_title(str(metric_meta["plot_title"]))
    ax.set_xlabel(str(metric_meta["xlabel"]))
    ax.set_ylabel(str(metric_meta["ylabel"]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    mean_relative = comparisons.get(
        "throughput_relative_to_baseline_pct") or {}
    categories = ["baseline", "full", "adaptive"]
    values = [
        mean_relative.get("baseline"),
        mean_relative.get("full"),
        mean_relative.get("adaptive"),
    ]
    plotted = [
        100.0 if name == "baseline" and value is None else (
            0.0 if value is None else value)
        for name, value in zip(categories, values)
    ]
    ax.bar(categories, plotted, color=[
           "#2f7d32", "#c0392b", "#1f77b4"], alpha=0.85)
    ax.axhline(100.0, color="#555", linewidth=0.9, linestyle="--")
    ax.set_ylabel(str(metric_meta["relative_ylabel"]))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        if value is None:
            ax.text(idx, plotted[idx] + 1.0, "n/a",
                    ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    ax.axis("off")
    throughput = comparisons.get("throughput_loss_pct") or {}
    overhead = comparisons.get("throughput_overhead_pct") or {}
    data_comp = comparisons.get("data_capture") or {}
    cpu_comp = comparisons.get("cpu") or {}
    overhead_comp = comparisons.get("active_overhead") or {}
    probe_comp = comparisons.get("probe_fetch") or {}
    publish_comp = comparisons.get("publish_path") or {}
    quality_comp = comparisons.get("tracked_metric_quality") or {}
    urgency_comp = comparisons.get("urgency") or {}
    warnings = comparisons.get("warnings") or []
    summary_lines = [
        f"Full vs baseline {metric_label} delta:{_fmt_pct(throughput.get('full_vs_baseline'))}",
        f"Adaptive vs baseline delta:       {_fmt_pct(throughput.get('adaptive_vs_baseline'))}",
        f"Adaptive vs full delta:           {_fmt_pct(throughput.get('adaptive_vs_full'))}",
        f"Full overhead vs baseline:        {_fmt_pct(overhead.get('full_vs_baseline'))}",
        f"Adaptive overhead vs baseline:    {_fmt_pct(overhead.get('adaptive_vs_baseline'))}",
        f"Adaptive overhead vs full:        {_fmt_pct(overhead.get('adaptive_vs_full'))}",
        "",
        f"Adaptive sample ratio:            {_fmt_pct(data_comp.get('adaptive_sample_ratio_pct'))}",
        f"Data saved vs full:               {_fmt_pct(data_comp.get('adaptive_data_saved_vs_full_pct'))}",
        f"Peak recall top5 (adaptive):      {_fmt_pct(data_comp.get('adaptive_peak_recall_top5_pct'))}",
        "",
        f"Full mean CPU:                    {_fmt_num(cpu_comp.get('full_mean_cpu_pct'), suffix='%')}",
        f"Adaptive mean CPU:                {_fmt_num(cpu_comp.get('adaptive_mean_cpu_pct'), suffix='%')}",
        f"Full mean urgency:                {_fmt_num(urgency_comp.get('full_mean_urgency'))}",
        f"Adaptive mean urgency:            {_fmt_num(urgency_comp.get('adaptive_mean_urgency'))}",
        f"Adaptive replay mean urgency:     {_fmt_num(urgency_comp.get('adaptive_replay_mean_urgency'))}",
        f"Adaptive replay median urgency:   {_fmt_num(urgency_comp.get('adaptive_replay_median_urgency'))}",
        f"Adaptive CPU saved vs full:       {_fmt_pct(cpu_comp.get('adaptive_mean_cpu_saved_vs_full_pct'))}",
        f"Adaptive total CPU saved vs full: {_fmt_pct(cpu_comp.get('adaptive_total_cpu_saved_vs_full_pct'))}",
        "",
        f"Full active overhead:             {_fmt_num(overhead_comp.get('full_active_mean_us'), scale=1000.0, suffix=' ms')}",
        f"Adaptive active overhead:         {_fmt_num(overhead_comp.get('adaptive_active_mean_us'), scale=1000.0, suffix=' ms')}",
        f"Adaptive active saved vs full:    {_fmt_pct(overhead_comp.get('adaptive_active_saved_vs_full_pct'))}",
        "",
        f"Full probe fetch total:           {_fmt_num(probe_comp.get('full_probe_fetch_total_s'), suffix=' s')}",
        f"Adaptive probe fetch total:       {_fmt_num(probe_comp.get('adaptive_probe_fetch_total_s'), suffix=' s')}",
        f"Adaptive probe saved vs full:     {_fmt_pct(probe_comp.get('adaptive_probe_fetch_total_saved_vs_full_pct'))}",
        f"Adaptive publish saved vs full:   {_fmt_pct(publish_comp.get('adaptive_publish_path_saved_vs_full_pct'))}",
        "",
        f"Tracked metric quality:           {quality_comp.get('status', 'n/a')}",
        f"Tracked metric corr:              {_fmt_num(quality_comp.get('corr'))}",
        f"Tracked metric nrmse_p95:         {_fmt_num(quality_comp.get('nrmse_p95'))}",
    ]
    if warnings:
        summary_lines.extend(["", "Warnings:"])
        summary_lines.extend(f"- {warning}" for warning in warnings)
    ax.text(
        0.01,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize workload slowdown across baseline, full, and adaptive runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind", choices=["postgres", "redis", "latency"], required=True)
    parser.add_argument("--baseline", required=True,
                        help="Baseline workload log file")
    parser.add_argument("--full", required=True,
                        help="Full-monitoring workload log file")
    parser.add_argument("--adaptive", required=True,
                        help="Adaptive workload log file")
    parser.add_argument("--full-eval-dir", default=None,
                        help="Directory containing io_syscall_results_*.json for full monitoring")
    parser.add_argument("--adaptive-eval-dir", default=None,
                        help="Directory containing io_syscall_results_*.json for adaptive monitoring")
    parser.add_argument("--adaptive-replay-eval-dir", default=None,
                        help="Directory containing io_syscall_results_*.json for adaptive replay")
    parser.add_argument(
        "--replay-urgency-eval-dir",
        default=None,
        help=(
            "Replay result directory used only for replay urgency summaries; "
            "live-comparability quality remains separate."
        ),
    )
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "postgres":
        parser_fn = _parse_pgbench
    elif args.kind == "redis":
        parser_fn = _parse_redis
    else:
        parser_fn = _parse_latency

    baseline_x, baseline_y = parser_fn(_read_text(args.baseline))
    full_x, full_y = parser_fn(_read_text(args.full))
    adaptive_x, adaptive_y = parser_fn(_read_text(args.adaptive))

    baseline_x, baseline_y, baseline_trim_start = _trim_relative_prefix(
        baseline_x, baseline_y)
    full_x, full_y, full_trim_start = _trim_relative_prefix(full_x, full_y)
    adaptive_x, adaptive_y, adaptive_trim_start = _trim_relative_prefix(
        adaptive_x, adaptive_y)

    baseline_summary = _summarize(baseline_y)
    full_summary = _summarize(full_y)
    adaptive_summary = _summarize(adaptive_y)

    full_monitoring = _extract_monitoring_summary(args.full_eval_dir)
    adaptive_monitoring = _extract_monitoring_summary(args.adaptive_eval_dir)
    replay_quality = _extract_replay_quality(args.adaptive_replay_eval_dir)
    replay_urgency = _extract_replay_urgency(
        args.replay_urgency_eval_dir or args.adaptive_replay_eval_dir
    )

    baseline_mean = baseline_summary.mean if baseline_summary else None
    full_mean = full_summary.mean if full_summary else None
    adaptive_mean = adaptive_summary.mean if adaptive_summary else None
    warnings = _throughput_confidence_warnings(
        args.kind,
        baseline_summary,
        full_summary,
        adaptive_summary,
    )
    if args.kind == "latency":
        warnings.extend(
            _latency_alignment_warnings(
                baseline_summary,
                full_summary,
                adaptive_summary,
                baseline_trim_start,
                full_trim_start,
                adaptive_trim_start,
            )
        )

    full_sample_ratio = None
    adaptive_sample_ratio = None
    adaptive_peak_recall = None
    full_mean_cpu = None
    adaptive_mean_cpu = None
    full_total_cpu = None
    adaptive_total_cpu = None
    full_active_mean = None
    adaptive_active_mean = None
    if full_monitoring:
        full_sample_ratio = full_monitoring["data"].get("sample_ratio")
        full_mean_cpu = full_monitoring["cpu"].get("mean_cpu_pct")
        full_total_cpu = full_monitoring["cpu"].get("cpu_total_s")
        full_active_mean = full_monitoring["overhead"].get("active_mean_us")
    if adaptive_monitoring:
        adaptive_sample_ratio = adaptive_monitoring["data"].get("sample_ratio")
        adaptive_peak_recall = adaptive_monitoring["data"].get(
            "peak_recall_top5")
        adaptive_mean_cpu = adaptive_monitoring["cpu"].get("mean_cpu_pct")
        adaptive_total_cpu = adaptive_monitoring["cpu"].get("cpu_total_s")
        adaptive_active_mean = adaptive_monitoring["overhead"].get(
            "active_mean_us")

    full_mean_urgency = None if not full_monitoring else full_monitoring["scheduler"].get(
        "mean_urgency")
    adaptive_mean_urgency = None if not adaptive_monitoring else adaptive_monitoring["scheduler"].get(
        "mean_urgency")

    full_probe_fetch_mean = None if not full_monitoring else full_monitoring["overhead"].get(
        "probe_fetch_mean_us")
    adaptive_probe_fetch_mean = None if not adaptive_monitoring else adaptive_monitoring["overhead"].get(
        "probe_fetch_mean_us")
    full_probe_fetch_total = None if not full_monitoring else full_monitoring["overhead"].get(
        "probe_fetch_total_s")
    adaptive_probe_fetch_total = None if not adaptive_monitoring else adaptive_monitoring["overhead"].get(
        "probe_fetch_total_s")
    full_post_fetch_total = None if not full_monitoring else full_monitoring["overhead"].get(
        "post_fetch_control_total_s")
    adaptive_post_fetch_total = None if not adaptive_monitoring else adaptive_monitoring["overhead"].get(
        "post_fetch_control_total_s")
    full_publish_total = None if not full_monitoring else full_monitoring["overhead"].get(
        "publish_path_total_s")
    adaptive_publish_total = None if not adaptive_monitoring else adaptive_monitoring["overhead"].get(
        "publish_path_total_s")

    tracked_metric_quality = None
    if replay_quality is not None:
        tracked_metric_quality = replay_quality
    elif full_monitoring and adaptive_monitoring:
        full_trace = full_monitoring.get("trace_path")
        adaptive_trace = adaptive_monitoring.get("trace_path")
        if isinstance(full_trace, str) and isinstance(adaptive_trace, str):
            full_trace_path = Path(full_trace)
            adaptive_trace_path = Path(adaptive_trace)
            if full_trace_path.exists() and adaptive_trace_path.exists():
                tracked_metric_quality = _assess_live_comparability(
                    full_trace_path, adaptive_trace_path)

    comparisons = {
        "throughput_loss_pct": {
            "full_vs_baseline": _pct_change(baseline_mean, full_mean),
            "adaptive_vs_baseline": _pct_change(baseline_mean, adaptive_mean),
            "adaptive_vs_full": _pct_change(full_mean, adaptive_mean),
        },
        "throughput_overhead_pct": {
            "full_vs_baseline": None if baseline_mean in (None, 0) or full_mean is None else ((
                baseline_mean - full_mean) / baseline_mean) * 100.0,
            "adaptive_vs_baseline": None if baseline_mean in (None, 0) or adaptive_mean is None else ((
                baseline_mean - adaptive_mean) / baseline_mean) * 100.0,
            "adaptive_vs_full": None if full_mean in (None, 0) or adaptive_mean is None else ((
                full_mean - adaptive_mean) / full_mean) * 100.0,
        },
        "throughput_relative_to_baseline_pct": {
            "baseline": 100.0 if baseline_mean is not None else None,
            "full": None if baseline_mean in (None, 0) or full_mean is None else (full_mean / baseline_mean) * 100.0,
            "adaptive": None if baseline_mean in (None, 0) or adaptive_mean is None else (
                adaptive_mean / baseline_mean) * 100.0,
        },
        "data_capture": {
            "full_sample_ratio_pct": None if full_sample_ratio is None else full_sample_ratio * 100.0,
            "adaptive_sample_ratio_pct": None if adaptive_sample_ratio is None else adaptive_sample_ratio * 100.0,
            "adaptive_data_saved_vs_full_pct": None if full_sample_ratio in (None,
                                                                             0) or adaptive_sample_ratio is None else (
                1.0 - adaptive_sample_ratio / full_sample_ratio) * 100.0,
            "adaptive_peak_recall_top5_pct": None if adaptive_peak_recall is None else adaptive_peak_recall * 100.0,
        },
        "cpu": {
            "full_mean_cpu_pct": full_mean_cpu,
            "adaptive_mean_cpu_pct": adaptive_mean_cpu,
            "adaptive_mean_cpu_saved_vs_full_pct": None if full_mean_cpu in (None,
                                                                             0) or adaptive_mean_cpu is None else (
                1.0 - adaptive_mean_cpu / full_mean_cpu) * 100.0,
            "full_total_cpu_s": full_total_cpu,
            "adaptive_total_cpu_s": adaptive_total_cpu,
            "adaptive_total_cpu_saved_vs_full_pct": None if full_total_cpu in (None,
                                                                               0) or adaptive_total_cpu is None else (
                1.0 - adaptive_total_cpu / full_total_cpu) * 100.0,
        },
        "urgency": {
            "full_mean_urgency": full_mean_urgency,
            "adaptive_mean_urgency": adaptive_mean_urgency,
            "adaptive_replay_mean_urgency": (
                None if replay_urgency is None else replay_urgency.get("mean")
            ),
            "adaptive_replay_median_urgency": (
                None if replay_urgency is None else replay_urgency.get("median")
            ),
            "adaptive_replay_urgency_ticks": (
                None if replay_urgency is None else replay_urgency.get("count")
            ),
        },
        "active_overhead": {
            "full_active_mean_us": full_active_mean,
            "adaptive_active_mean_us": adaptive_active_mean,
            "adaptive_active_saved_vs_full_pct": None if full_active_mean in (None,
                                                                              0) or adaptive_active_mean is None else (
                1.0 - adaptive_active_mean / full_active_mean) * 100.0,
        },
        "probe_fetch": {
            "full_probe_fetch_mean_us": full_probe_fetch_mean,
            "adaptive_probe_fetch_mean_us": adaptive_probe_fetch_mean,
            "adaptive_probe_fetch_mean_saved_vs_full_pct": _pct_saved(full_probe_fetch_mean, adaptive_probe_fetch_mean),
            "full_probe_fetch_total_s": full_probe_fetch_total,
            "adaptive_probe_fetch_total_s": adaptive_probe_fetch_total,
            "adaptive_probe_fetch_total_saved_vs_full_pct": _pct_saved(full_probe_fetch_total, adaptive_probe_fetch_total),
        },
        "post_fetch_control": {
            "full_post_fetch_control_total_s": full_post_fetch_total,
            "adaptive_post_fetch_control_total_s": adaptive_post_fetch_total,
            "adaptive_post_fetch_control_saved_vs_full_pct": _pct_saved(full_post_fetch_total, adaptive_post_fetch_total),
        },
        "publish_path": {
            "full_publish_path_total_s": full_publish_total,
            "adaptive_publish_path_total_s": adaptive_publish_total,
            "adaptive_publish_path_saved_vs_full_pct": _pct_saved(full_publish_total, adaptive_publish_total),
        },
        "tracked_metric_quality": tracked_metric_quality,
        "warnings": warnings,
    }

    summary = {
        "kind": args.kind,
        "metric": {
            "series_label": _metric_meta(args.kind)["series_label"],
            "summary_label": _metric_meta(args.kind)["summary_label"],
            "higher_is_better": _metric_meta(args.kind)["higher_is_better"],
        },
        "scenarios": {
            "baseline": {
                "throughput": asdict(baseline_summary) if baseline_summary else None,
                "trim_start": baseline_trim_start,
            },
            "full": {
                "throughput": asdict(full_summary) if full_summary else None,
                "monitoring": full_monitoring,
                "trim_start": full_trim_start,
            },
            "adaptive": {
                "throughput": asdict(adaptive_summary) if adaptive_summary else None,
                "monitoring": adaptive_monitoring,
                "trim_start": adaptive_trim_start,
            },
        },
        "comparisons": comparisons,
    }

    json_path = out_dir / "workload_overhead_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    plot_path = out_dir / "workload_overhead.png"
    if baseline_y.size or full_y.size or adaptive_y.size:
        _plot(
            args.kind,
            baseline_x,
            baseline_y,
            full_x,
            full_y,
            adaptive_x,
            adaptive_y,
            comparisons,
            plot_path,
        )

    print(json.dumps({
        "summary": str(json_path),
        "plot": str(plot_path),
        "comparisons": comparisons,
    }))


if __name__ == "__main__":
    main()
