#!/usr/bin/env python3
"""Adaptive I/O Syscall Rate Monitoring – Offline Evaluation.

Generates a synthetic I/O syscall rate trace that models three distinct
burst event types found in storage-intensive workloads:

  - **Checkpoint storm**: periodic fsync bursts (PostgreSQL WAL flush)
  - **Write saturation**: sustained high write rate (bulk inserts / ETL)
  - **Write flood**: sudden extreme spike 

Then runs OmniFlow's dense reference tracker + adaptive replay path, produces
the evaluation plot, automatically detects and highlights burst periods,
and reports the key metric: *peak recall (top-5%)*.

Usage - Load an existing JSON trace (e.g. captured from the live daemon)
--
python evaluate.py --json path/to/trace.json --out out/run_003
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from numpy.typing import NDArray

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import (  # noqa: E402
    TickResult, PollResult, Status,
    evaluate_info_loss,
)
from profile_loader import available_profiles, load_profile


# -----------------------------------------------------------------------
# Synthetic trace builder
# -----------------------------------------------------------------------

def build_io_trace(
        quiet_mean: float = 50.0,
        seed: int = 42,
) -> tuple[NDArray[np.float64], list[tuple[int, int, str]]]:
    """Build a synthetic I/O syscall rate trace (counts per 0.2 s interval).

    Returns
    -------
    trace
        1-D array of per-interval syscall counts.
    segments
        List of ``(start, end, label)`` tuples marking event regions.
    """
    rng = np.random.default_rng(seed)

    def _lognormal(n: int, mean: float, sigma: float) -> NDArray:
        mu = np.log(mean)
        vals = rng.lognormal(mu, sigma, size=n).astype(np.float64)
        return np.clip(vals, 0, None)

    # --- Segment lengths (at 5 polls/s = 0.2 s each) ---
    # Total ≈ 600 s (3 000 points) – matches a 120 s real-run window with
    # realistic PostgreSQL checkpoint_timeout=30s behaviour.
    #
    # Pattern: short quiet baseline -> 4 checkpoint storms (every ~30 s,
    # lasting 10–20 s each) -> brief inter-storm quiet -> 1 write-saturation
    # plateau -> 1 short write-flood spike -> recovery tail.
    #
    # Each checkpoint storm is a separate segment so the reference
    # labels are precise and peak-recall scoring is meaningful.

    seg_defs = [
        # n      label                       mean              sigma
        (125, "quiet baseline", quiet_mean, 0.50),
        (75, "checkpoint storm 1", quiet_mean * 10, 0.30),
        (100, "quiet", quiet_mean, 0.50),
        (75, "checkpoint storm 2", quiet_mean * 9, 0.30),
        (75, "quiet", quiet_mean, 0.50),
        (75, "checkpoint storm 3", quiet_mean * 11, 0.28),
        (75, "quiet", quiet_mean, 0.50),
        (75, "checkpoint storm 4", quiet_mean * 8, 0.32),
        (100, "quiet", quiet_mean, 0.50),
        (400, "batch write saturation", quiet_mean * 4, 0.35),
        (100, "quiet", quiet_mean, 0.50),
        (50, "write flood", quiet_mean * 25, 0.18),
        (175, "quiet recovery", quiet_mean, 0.50),
    ]

    parts: list[NDArray] = []
    segments: list[tuple[int, int, str]] = []
    pos = 0
    for n, label, mean, sigma in seg_defs:
        chunk = _lognormal(n, mean, sigma)
        parts.append(chunk)
        segments.append((pos, pos + n, label))
        pos += n

    trace = np.concatenate(parts)
    return trace, segments


# -----------------------------------------------------------------------
# Burst detection (threshold-based, signal-only)
# -----------------------------------------------------------------------

def detect_bursts(
        trace: NDArray[np.float64],
        threshold_percentile: float = 92.0,
        min_length: int = 10,
        merge_gap: int = 20,
) -> list[tuple[int, int, str]]:
    """Detect burst periods from the trace alone (no reference labels).

    A burst is a contiguous run of values above *threshold_percentile*
    that is at least *min_length* samples long.  Runs separated by fewer
    than *merge_gap* samples are merged.

    Returns list of ``(start, end, label)`` tuples.
    """
    threshold = np.percentile(trace, threshold_percentile)
    above = trace > threshold

    # Find contiguous runs
    raw_runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i, flag in enumerate(above):
        if flag and not in_run:
            in_run = True
            run_start = i
        elif not flag and in_run:
            in_run = False
            if i - run_start >= min_length:
                raw_runs.append((run_start, i))
    if in_run and len(trace) - run_start >= min_length:
        raw_runs.append((run_start, len(trace)))

    # Merge close runs
    merged: list[tuple[int, int]] = []
    for s, e in raw_runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    result = []
    for i, (s, e) in enumerate(merged):
        peak = float(trace[s:e].max())
        result.append((s, e, f"burst {i + 1}  (peak={peak:.0f})"))
    return result


# -----------------------------------------------------------------------
# 5-panel evaluation plot
# -----------------------------------------------------------------------

# Colour palette
_COL_SIGNAL = "#b0c4de"  # light steel blue – raw signal
_COL_FULL = "#e05c5c"  # tomato – dense reference mean
_COL_ADAPTIVE = "#2a7fcf"  # steel blue – adaptive mean
_COL_SAMPLED = "#1a5fa8"  # darker blue – sampled dots
_COL_INTERVAL = "#e07b39"  # orange – interval / polling freq
_COL_URGENCY = "#c0392b"  # crimson – urgency
_COL_DELTA = "#8e44ad"  # purple – d/dx urgency
_COL_RATIO = "#16a085"  # teal – sample ratio
_COL_BURST = "#f39c12"  # amber – burst highlight

_OVERHEAD_KEYS = [
    "read_overhead_us",
    "tracker_overhead_us",
    "adaptive_overhead_us",
    "metrics_overhead_us",
    "emit_overhead_us",
    "active_loop_us",
    "monitor_cpu_pct",
    "monitor_user_cpu_s_total",
    "monitor_system_cpu_s_total",
    "monitor_total_cpu_s_total",
]


def _as_int(value: object, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError("Expected integer-like value, got None")
        return default
    return int(cast(int | float | str, value))


def _as_float(value: object, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("Expected float-like value, got None")
        return default
    return float(cast(int | float | str, value))


def _row_ticks(rows: list[dict[str, object]]) -> NDArray[np.int64]:
    return np.array([
        _as_int(row.get("t"), i + 1)
        for i, row in enumerate(rows)
    ], dtype=np.int64)


def _weighted_row_mean(
        rows: list[dict[str, object]],
        value_key: str,
        weight_key: str = "sample_window_ticks",
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        weight = float(_as_int(row.get(weight_key), 1))
        if weight <= 0.0:
            continue
        weighted_sum += _as_float(value) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def _plot_finite_series(
        ax: plt.Axes,
        t_sec: NDArray[np.float64],
        series: NDArray[np.float64],
        *,
        label: str,
        color: str,
        linewidth: float = 0.9,
        marker: str = "o",
        markersize: float = 2.5,
) -> bool:
    finite = np.isfinite(series)
    if not finite.any():
        return False

    x = t_sec[finite]
    y = series[finite]
    line_kwargs = {
        "linewidth": linewidth,
        "label": label,
        "color": color,
        "marker": marker,
        "markersize": markersize,
        "markeredgewidth": 0.0,
    }
    if x.size == 1:
        ax.scatter(x, y, color=color, s=markersize * 8.0, label=label)
    else:
        ax.plot(x, y, **line_kwargs)
    return True

def plot_io_evaluation(
        trace: NDArray[np.float64],
        poll_results: list[PollResult],
        full_results: list[TickResult],
        burst_regions: list[tuple[int, int, str]],
        ground_truth_segments: list[tuple[int, int, str]] | None = None,
        info_loss_str: str = "",
        title: str = "I/O Syscall Adaptive Replay Evaluation",
        poll_interval_s: float = 0.2,
        save_path: str | None = None,
) -> None:
    """Produce the 5-panel evaluation figure.

    Panels
    ------
    1. Raw signal + dense reference mean + replay mean + sampled markers
    2. Poll interval (effective seconds between observations)
    3. Urgency score
    4. d/dx urgency (rate of urgency change)
    5. Cumulative sample ratio

    Burst periods detected from the signal are highlighted across all panels.
    """
    n = len(trace)
    t = np.arange(n)
    t_sec = t * poll_interval_s  # wall-clock seconds (x-axis label)

    # --- Derived arrays ---
    full_means = np.array([r.mean for r in full_results])

    s_idx = np.array([pr.time_index for pr in poll_results if pr.sampled])
    s_vals = trace[s_idx]
    s_means: list[float] = []
    s_stds: list[float] = []
    for pr in poll_results:
        if pr.sampled and pr.tick is not None:
            s_means.append(pr.tick.mean)
            s_stds.append(pr.tick.std)

    if len(s_idx) >= 2:
        adaptive_means = np.interp(t, s_idx, np.array(s_means))
        adaptive_stds = np.interp(t, s_idx, np.array(s_stds))
    elif len(s_idx) == 1:
        adaptive_means = np.full(n, s_means[0])
        adaptive_stds = np.full(n, s_stds[0])
    else:
        adaptive_means = np.full(n, trace[0])
        adaptive_stds = np.ones(n)

    intervals = np.array([pr.interval for pr in poll_results])
    urgencies = np.array([pr.urgency for pr in poll_results])
    d_urgency = np.gradient(urgencies)
    cum_ratio = np.cumsum([1.0 if pr.sampled else 0.0
                           for pr in poll_results]) / (t + 1)

    # --- Outlier / drift markers from adaptive poll results ---
    anom_idx = [pr.time_index for pr in poll_results
                if pr.sampled and pr.tick is not None
                and pr.tick.status == Status.OUTLIER]
    drift_idx = [pr.time_index for pr in poll_results
                 if pr.sampled and pr.tick is not None
                 and pr.tick.status == Status.DRIFT_RESET]

    # --- Layout ---
    height_ratios = [3, 1, 1, 1, 1]
    fig, axes = plt.subplots(
        5, 1, figsize=(16, 14), sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    def _shade_bursts(ax):
        for s, e, lbl in burst_regions:
            ax.axvspan(t_sec[s], t_sec[min(e, n - 1)],
                       alpha=0.12, color=_COL_BURST, zorder=0)

    def _add_gt_labels(ax):
        """Lightly mark reference segment boundaries (if given)."""
        if ground_truth_segments is None:
            return
        burst_labels = {"checkpoint storm", "batch write saturation", "write flood"}
        for s, e, lbl in ground_truth_segments:
            if lbl in burst_labels:
                mid = t_sec[(s + e) // 2]
                ymax = ax.get_ylim()[1]
                ax.text(mid, ymax * 0.97, lbl,
                        ha="center", va="top", fontsize=7,
                        color="dimgrey", fontstyle="italic")

    # ── Panel 1: signal ─────────────────────────────────────────────────
    ax = axes[0]
    _shade_bursts(ax)
    ax.plot(t_sec, trace, linewidth=0.5, color=_COL_SIGNAL,
            alpha=0.85, label="raw signal", zorder=2)
    ax.plot(t_sec, full_means, linewidth=1.0, color=_COL_FULL,
            alpha=0.7, linestyle="--", label="dense reference mean", zorder=3)
    ax.plot(t_sec, adaptive_means, linewidth=1.4, color=_COL_ADAPTIVE,
            label="replay mean", zorder=4)
    ax.fill_between(t_sec,
                    adaptive_means - 3 * adaptive_stds,
                    adaptive_means + 3 * adaptive_stds,
                    color=_COL_ADAPTIVE, alpha=0.06, zorder=1)
    ax.scatter(t_sec[s_idx], s_vals, s=10, color=_COL_SAMPLED,
               zorder=5, label=f"sampled ({len(s_idx)}/{n})")

    # Outlier / drift markers
    if anom_idx:
        ax.scatter([t_sec[i] for i in anom_idx],
                   trace[anom_idx],
                   marker="x", s=20, color="orange", zorder=6,
                   label=f"outlier ({len(anom_idx)})")
    if drift_idx:
        for di in drift_idx:
            ax.axvline(t_sec[di], color="purple",
                       linewidth=0.8, linestyle=":", alpha=0.5)
        ax.plot([], [], color="purple", linestyle=":",
                label=f"drift reset ({len(drift_idx)})")

    ax.set_ylabel("Syscall rate\n(counts / 0.2 s)", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.25)

    # Add burst region legend patch
    burst_patch = Patch(color=_COL_BURST, alpha=0.4,
                        label=f"detected burst ({len(burst_regions)})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [burst_patch],
              loc="upper right", fontsize=7, ncol=3)

    # ── Panel 2: poll interval ───────────────────────────────────────────
    ax = axes[1]
    _shade_bursts(ax)
    eff_interval_s = intervals * poll_interval_s
    ax.step(t_sec, eff_interval_s, where="post",
            linewidth=0.9, color=_COL_INTERVAL)
    ax.fill_between(t_sec, eff_interval_s, alpha=0.25,
                    color=_COL_INTERVAL, step="post")
    ax.set_ylabel("Poll interval (s)", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    # ── Panel 3: urgency ────────────────────────────────────────────────
    ax = axes[2]
    _shade_bursts(ax)
    ax.fill_between(t_sec, 0, urgencies, alpha=0.35,
                    color=_COL_URGENCY, step="post")
    ax.step(t_sec, urgencies, where="post",
            linewidth=0.9, color=_COL_URGENCY)
    ax.set_ylabel("Urgency", fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: d/dx urgency ───────────────────────────────────────────
    ax = axes[3]
    _shade_bursts(ax)
    ax.fill_between(t_sec, 0, d_urgency, alpha=0.30,
                    color=_COL_DELTA,
                    where=d_urgency >= 0)
    ax.fill_between(t_sec, 0, d_urgency, alpha=0.30,
                    color="steelblue",
                    where=d_urgency < 0)
    ax.step(t_sec, d_urgency, where="post",
            linewidth=0.7, color=_COL_DELTA)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Δ urgency / step", fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 5: cumulative sample ratio ────────────────────────────────
    ax = axes[4]
    _shade_bursts(ax)
    step = max(1, n // 5_000)
    ax.plot(t_sec[::step], cum_ratio[::step],
            linewidth=1.0, color=_COL_RATIO)
    ax.fill_between(t_sec[::step], cum_ratio[::step],
                    alpha=0.30, color=_COL_RATIO)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
    ax.set_ylabel("Cum. sample ratio", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(t_sec[0], t_sec[-1])
    ax.grid(True, alpha=0.25)

    # ── Burst region labels on panel 1 ──────────────────────────────────
    _add_gt_labels(axes[0])

    # ── Annotate detected burst regions with bracket + label ────────────
    ax0 = axes[0]
    for s, e, lbl in burst_regions:
        xs = t_sec[s]
        xe = t_sec[min(e, n - 1)]
        y_top = ax0.get_ylim()[1] * 0.88
        ax0.annotate(
            "", xy=(xe, y_top), xytext=(xs, y_top),
            arrowprops=dict(arrowstyle="<->", color=_COL_BURST, lw=1.2),
        )

    # ── Info-loss summary box ────────────────────────────────────────────
    if info_loss_str:
        fig.text(
            0.01, 0.01, info_loss_str,
            fontsize=7, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      alpha=0.7),
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------
# Peak-recall per burst region
# -----------------------------------------------------------------------

def burst_peak_recall(
        trace: NDArray[np.float64],
        poll_results: list[PollResult],
        burst_regions: list[tuple[int, int, str]],
        top_fraction: float = 0.10,
) -> list[dict]:
    """Compute peak recall within each detected burst region.

    For each burst region we compute the fraction of the top-*top_fraction*
    values (by rank) that were captured by the adaptive poller.

    Returns a list of dicts with keys: label, start, end, peak_recall,
    n_top, n_captured.
    """
    sampled_set = {pr.time_index for pr in poll_results if pr.sampled}
    results = []
    for s, e, lbl in burst_regions:
        region_trace = trace[s:e]
        k = max(1, int(np.ceil(len(region_trace) * top_fraction)))
        top_indices = np.argsort(region_trace)[-k:] + s  # global indices
        n_captured = sum(1 for idx in top_indices if idx in sampled_set)
        results.append({
            "label": lbl,
            "start": int(s),
            "end": int(e),
            "peak_recall": round(n_captured / k, 4),
            "n_top": int(k),
            "n_captured": int(n_captured),
        })
    return results


def summarize_live_sampling(
    raw: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize live sparse sampling metadata from daemon log rows."""
    tick_positions = _row_ticks(raw)
    start_tick = int(np.min(tick_positions))
    end_tick = int(np.max(tick_positions))
    total_points = end_tick - start_tick + 1
    sampled_indices = np.array([
        _as_int(row.get("t"), i + 1) - start_tick
        for i, row in enumerate(raw)
        if row.get("sampled") and row.get("value") is not None
    ], dtype=np.int64)
    sampled_values = np.array([
        _as_float(row["value"])
        for row in raw
        if row.get("sampled") and row.get("value") is not None
    ], dtype=np.float64)

    interval_series = np.full(total_points, np.nan, dtype=np.float64)
    urgency_series = np.full(total_points, np.nan, dtype=np.float64)
    ratio_series = np.full(total_points, np.nan, dtype=np.float64)
    for i, row in enumerate(raw):
        index = _as_int(row.get("t"), i + 1) - start_tick
        if index < 0 or index >= total_points:
            continue
        if row.get("interval") is not None:
            interval_series[index] = _as_float(row["interval"])
        if row.get("urgency") is not None:
            urgency_series[index] = _as_float(row["urgency"])
        if row.get("sample_ratio_live") is not None:
            ratio_series[index] = _as_float(row["sample_ratio_live"])

    if sampled_indices.size > 1:
        gaps = np.diff(sampled_indices).astype(np.float64)
        max_gap = int(np.max(gaps))
        mean_gap = float(np.mean(gaps))
        p95_gap = float(np.percentile(gaps, 95))
    elif sampled_indices.size == 1:
        max_gap = 1
        mean_gap = 1.0
        p95_gap = 1.0
    else:
        max_gap = 0
        mean_gap = 0.0
        p95_gap = 0.0

    if not np.isfinite(ratio_series).any():
        sampled_mask = np.zeros(total_points, dtype=np.float64)
        if sampled_indices.size:
            sampled_mask[sampled_indices] = 1.0
        ratio_series = np.cumsum(sampled_mask) / np.arange(1, total_points + 1)

    return {
        "start_tick": start_tick,
        "end_tick": end_tick,
        "n_total": int(total_points),
        "n_sampled": int(sampled_indices.size),
        "sample_ratio": float(sampled_indices.size / max(1, total_points)),
        "max_gap": max_gap,
        "mean_gap": mean_gap,
        "p95_gap": p95_gap,
        "sampled_indices": sampled_indices,
        "sampled_values": sampled_values,
        "interval_series": interval_series,
        "urgency_series": urgency_series,
        "ratio_series": ratio_series,
    }


def plot_live_sampling(
        sampling: dict[str, object],
        poll_interval_s: float = 0.2,
        save_path: str | None = None,
) -> None:
    """Plot live sparse sampling behavior when true probe reads are skipped."""
    total_points = int(cast(int, sampling["n_total"]))
    sampled_indices = np.asarray(cast(object, sampling["sampled_indices"]))
    sampled_values = np.asarray(cast(object, sampling["sampled_values"]))
    interval_series = np.asarray(cast(object, sampling["interval_series"]))
    urgency_series = np.asarray(cast(object, sampling["urgency_series"]))
    ratio_series = np.asarray(cast(object, sampling["ratio_series"]))

    t_sec = np.arange(total_points) * poll_interval_s
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)

    ax = axes[0]
    if sampled_indices.size:
        ax.plot(sampled_indices * poll_interval_s, sampled_values,
                linewidth=0.8, color=_COL_ADAPTIVE, alpha=0.8)
        ax.scatter(sampled_indices * poll_interval_s, sampled_values,
                   s=12, color=_COL_SAMPLED, label=f"sampled ({sampled_indices.size}/{total_points})")
    ax.set_ylabel("Observed rate\n(counts / 0.2 s)")
    ax.set_title("Live Adaptive I/O Collection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    if np.isfinite(interval_series).any():
        ax.step(t_sec, interval_series * poll_interval_s, where="post", color=_COL_INTERVAL, linewidth=0.9)
        ax.fill_between(t_sec, interval_series * poll_interval_s, alpha=0.25, color=_COL_INTERVAL, step="post")
    ax.set_ylabel("Poll interval (s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    if np.isfinite(urgency_series).any():
        ax.step(t_sec, urgency_series, where="post", color=_COL_URGENCY, linewidth=0.9)
        ax.fill_between(t_sec, 0, urgency_series, alpha=0.30, color=_COL_URGENCY, step="post")
    ax.set_ylabel("Urgency")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.plot(t_sec, ratio_series, color=_COL_RATIO, linewidth=1.0)
    ax.fill_between(t_sec, ratio_series, alpha=0.30, color=_COL_RATIO)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
    ax.set_ylabel("Cum. sample ratio")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def extract_trace_and_overhead(
        raw: list[dict[str, object]] | list[float],
) -> tuple[
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Extract the primary trace and optional overhead series."""
    if not raw:
        raise ValueError("Trace JSON is empty.")

    if not isinstance(raw[0], dict):
        return np.array(raw, dtype=np.float64), {}, None, None

    rows = cast(list[dict[str, object]], raw)
    timing_rows = sum(1 for row in rows if row.get("timing_measured") is True)
    sampled_rows = sum(
        1 for row in rows
        if row.get("sampled") is True or row.get("value") is not None
    )
    prometheus_updated_rows = sum(1 for row in rows if row.get("prometheus_updated") is True)
    prometheus_update_every_sampled = _as_int(rows[0].get("prometheus_update_every_sampled"), 1)
    if prometheus_updated_rows == 0 and "prometheus_updated" not in rows[0]:
        prometheus_updated_rows = sampled_rows if prometheus_update_every_sampled <= 1 else 1 + (sampled_rows // prometheus_update_every_sampled)
    overhead_metadata = {
        "cpu_accounting_source": rows[0].get("cpu_accounting_source"),
        "cpu_snapshot_interval_ticks": rows[0].get("cpu_snapshot_interval_ticks"),
        "prometheus_update_every_sampled": prometheus_update_every_sampled,
        "detail_timing_every_sampled": rows[0].get("detail_timing_every_sampled"),
        "total_rows": len(rows),
        "sampled_rows": sampled_rows,
        "prometheus_updated_rows": prometheus_updated_rows,
        "timing_measured_rows": timing_rows,
        "timing_measured_fraction": None if not rows else timing_rows / float(len(rows)),
        "mean_urgency": _weighted_row_mean(rows, "urgency"),
        "mean_interval": _weighted_row_mean(rows, "interval"),
        "mean_sample_ratio_live": _weighted_row_mean(rows, "sample_ratio_live"),
    }

    overhead: dict[str, NDArray[np.float64]] = {}
    for key in _OVERHEAD_KEYS:
        arr = np.array([
            _as_float(row[key], np.nan) if key in row else np.nan
            for row in rows
        ], dtype=np.float64)
        if np.isfinite(arr).any():
            overhead[key] = arr

    tick_positions = _row_ticks(rows)
    has_sparse_ticks = bool(
        tick_positions.size > 1 and np.any(np.diff(tick_positions) != 1)
    )
    has_sparse_values = any(row.get("value") is None for row in rows)
    if has_sparse_values or has_sparse_ticks:
        sampling = summarize_live_sampling(rows)
        return np.asarray(sampling["sampled_values"], dtype=np.float64), overhead, sampling, overhead_metadata

    trace = np.array([_as_float(row["value"]) for row in rows], dtype=np.float64)
    return trace, overhead, None, overhead_metadata


def summarize_replay_urgency(
        poll_results: list[PollResult],
        *,
        include_values: bool = False,
) -> dict[str, object]:
    """Summarize urgency over every dense replay tick.

    ``AdaptivePoller.track`` emits one ``PollResult`` per input tick, including
    skipped ticks where urgency is held constant.  Therefore the arithmetic
    mean and median here are time/tick-weighted replay summaries, unlike a
    summary over only sampled live trace rows.
    """
    values = np.asarray([result.urgency for result in poll_results], dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None}
    summary: dict[str, object] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "aggregation": (
            "arithmetic statistics over every dense input tick; urgency is "
            "held constant on skipped ticks"
        ),
    }
    if include_values:
        summary["values"] = values.tolist()
    return summary


def summarize_overhead(
        overhead: dict[str, NDArray[np.float64]],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for overhead time series."""
    summary: dict[str, dict[str, float]] = {}
    for key, values in overhead.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(finite)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(np.max(finite)),
        }
    return summary


def compute_cpu_normalization(
        overhead: dict[str, NDArray[np.float64]],
        overhead_summary: dict[str, dict[str, float]],
        n_points: int,
        poll_interval_s: float,
) -> dict[str, float] | None:
    """Normalize daemon CPU to wall-clock time for fair live comparisons."""
    wall_time_s = float(n_points) * float(poll_interval_s)
    if wall_time_s <= 0.0:
        return None

    total_cpu = None
    total_cpu_series = overhead.get("monitor_total_cpu_s_total")
    if total_cpu_series is not None:
        finite_total = total_cpu_series[np.isfinite(total_cpu_series)]
        if finite_total.size:
            total_cpu = float(np.max(finite_total))

    if total_cpu is None:
        user_total = _summary_field(overhead_summary, "monitor_user_cpu_s_total", "max")
        system_total = _summary_field(overhead_summary, "monitor_system_cpu_s_total", "max")
        if user_total is not None or system_total is not None:
            total_cpu = float((user_total or 0.0) + (system_total or 0.0))

    snapshot_mean_cpu_pct = _summary_field(overhead_summary, "monitor_cpu_pct", "mean")
    if total_cpu is None and snapshot_mean_cpu_pct is None:
        return None

    return {
        "wall_time_s": wall_time_s,
        "total_cpu_s": total_cpu,
        "mean_cpu_pct_over_wall": None if total_cpu is None else (total_cpu / wall_time_s) * 100.0,
        "snapshot_mean_cpu_pct": snapshot_mean_cpu_pct,
    }


def _estimated_total_seconds(mean_us: float | None, event_count: int | None) -> float | None:
    if mean_us is None or event_count is None:
        return None
    return (mean_us * float(event_count)) / 1_000_000.0


def _sum_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(sum(finite))


def compute_cost_accounting(
        overhead_summary: dict[str, dict[str, float]],
        overhead_metadata: dict[str, object] | None,
        sampling_summary: dict[str, object] | None,
) -> dict[str, float | int] | None:
    """Estimate probe-fetch cost separately from post-fetch Python control-plane cost."""
    if overhead_metadata is None or not overhead_summary:
        return None

    sampled_events = _as_int(
        overhead_metadata.get("sampled_rows"),
        _as_int(cast(dict[str, object], sampling_summary).get("n_sampled"), 0)
        if sampling_summary is not None
        else _as_int(overhead_metadata.get("total_rows"), 0),
    )
    emitted_events = _as_int(overhead_metadata.get("total_rows"), sampled_events)
    prometheus_updates = _as_int(overhead_metadata.get("prometheus_updated_rows"), sampled_events)

    read_mean_us = _summary_field(overhead_summary, "read_overhead_us", "mean")
    tracker_mean_us = _summary_field(overhead_summary, "tracker_overhead_us", "mean")
    adaptive_mean_us = _summary_field(overhead_summary, "adaptive_overhead_us", "mean")
    metrics_mean_us = _summary_field(overhead_summary, "metrics_overhead_us", "mean")
    emit_mean_us = _summary_field(overhead_summary, "emit_overhead_us", "mean")
    active_mean_us = _summary_field(overhead_summary, "active_loop_us", "mean")

    probe_fetch_total_s = _estimated_total_seconds(read_mean_us, sampled_events)
    tracker_total_s = _estimated_total_seconds(tracker_mean_us, sampled_events)
    adaptive_total_s = _estimated_total_seconds(adaptive_mean_us, sampled_events)
    metrics_total_s = _estimated_total_seconds(metrics_mean_us, prometheus_updates)
    emit_total_s = _estimated_total_seconds(emit_mean_us, emitted_events)
    active_total_s = _estimated_total_seconds(active_mean_us, sampled_events)
    post_fetch_total_s = _sum_optional([tracker_total_s, adaptive_total_s, metrics_total_s, emit_total_s])
    publish_total_s = _sum_optional([metrics_total_s, emit_total_s])

    return {
        "sampled_events": sampled_events,
        "emitted_events": emitted_events,
        "prometheus_updates": prometheus_updates,
        "probe_fetch_mean_us": read_mean_us,
        "probe_fetch_total_s_estimated": probe_fetch_total_s,
        "tracker_total_s_estimated": tracker_total_s,
        "adaptive_scheduler_total_s_estimated": adaptive_total_s,
        "metrics_total_s_estimated": metrics_total_s,
        "emit_total_s_estimated": emit_total_s,
        "publish_path_total_s_estimated": publish_total_s,
        "post_fetch_control_total_s_estimated": post_fetch_total_s,
        "active_loop_total_s_estimated": active_total_s,
        "probe_fetch_share_of_active_loop_pct_estimated": None
        if probe_fetch_total_s is None or active_total_s in (None, 0.0)
        else (probe_fetch_total_s / active_total_s) * 100.0,
        "post_fetch_control_share_of_active_loop_pct_estimated": None
        if post_fetch_total_s is None or active_total_s in (None, 0.0)
        else (post_fetch_total_s / active_total_s) * 100.0,
    }


def _summary_field(
        overhead_summary: dict[str, dict[str, float]],
        key: str,
        field: str,
) -> float | None:
    stats = overhead_summary.get(key)
    if not isinstance(stats, dict):
        return None
    value = stats.get(field)
    return None if value is None else float(value)


def _top_spike_indices(active_series: NDArray[np.float64]) -> NDArray[np.int64]:
    finite_idx = np.flatnonzero(np.isfinite(active_series))
    if finite_idx.size == 0:
        return np.array([], dtype=np.int64)

    finite_values = active_series[finite_idx]
    threshold = float(np.percentile(finite_values, 99))
    candidate_idx = finite_idx[finite_values >= threshold]
    if candidate_idx.size == 0:
        candidate_idx = finite_idx[np.argsort(finite_values)[-8:]]
    elif candidate_idx.size > 12:
        candidate_values = active_series[candidate_idx]
        candidate_idx = candidate_idx[np.argsort(candidate_values)[-12:]]
    return np.sort(candidate_idx.astype(np.int64))


def assess_run_stability(
        overhead: dict[str, NDArray[np.float64]],
        overhead_summary: dict[str, dict[str, float]],
        sampling_summary: dict[str, object] | None,
) -> dict[str, Any] | None:
    """Assess whether a run is stable and identify likely spike drivers."""
    active_series = overhead.get("active_loop_us")
    cpu_series = overhead.get("monitor_cpu_pct")
    if active_series is None or cpu_series is None or not overhead_summary:
        return None

    active_p95 = _summary_field(overhead_summary, "active_loop_us", "p95") or 0.0
    active_p99 = _summary_field(overhead_summary, "active_loop_us", "p99") or 0.0
    active_max = _summary_field(overhead_summary, "active_loop_us", "max") or 0.0
    cpu_p95 = _summary_field(overhead_summary, "monitor_cpu_pct", "p95") or 0.0
    cpu_p99 = _summary_field(overhead_summary, "monitor_cpu_pct", "p99") or 0.0
    cpu_max = _summary_field(overhead_summary, "monitor_cpu_pct", "max") or 0.0

    score = 0
    reasons: list[str] = []
    if active_max >= 100_000.0:
        score += 2
        reasons.append(
            f"active_loop_us max is extreme ({active_max:.0f} us), far above a normal tail."
        )
    if active_p99 >= max(10_000.0, active_p95 * 5.0):
        score += 1
        reasons.append(
            f"active_loop_us p99 is heavily inflated ({active_p99:.0f} us vs p95 {active_p95:.0f} us)."
        )
    if cpu_max >= 5.0:
        score += 2
        reasons.append(f"monitor_cpu_pct has a large spike ({cpu_max:.2f}%).")
    elif cpu_max >= 2.0:
        score += 1
        reasons.append(f"monitor_cpu_pct has a moderate spike ({cpu_max:.2f}%).")
    if cpu_p99 >= max(1.0, cpu_p95 * 3.0):
        score += 1
        reasons.append(
            f"monitor_cpu_pct p99 is much larger than p95 ({cpu_p99:.2f}% vs {cpu_p95:.2f}%)."
        )

    publish_max = max(
        _summary_field(overhead_summary, "metrics_overhead_us", "max") or 0.0,
        _summary_field(overhead_summary, "emit_overhead_us", "max") or 0.0,
    )
    core_max = max(
        _summary_field(overhead_summary, "read_overhead_us", "max") or 0.0,
        _summary_field(overhead_summary, "tracker_overhead_us", "max") or 0.0,
        _summary_field(overhead_summary, "adaptive_overhead_us", "max") or 0.0,
    )
    if publish_max >= 20_000.0:
        score += 1
        reasons.append(f"publish-path overhead spikes above 20 ms ({publish_max:.0f} us).")
    if core_max >= 10_000.0:
        score += 1
        reasons.append(f"core loop overhead spikes above 10 ms ({core_max:.0f} us).")

    spike_idx = _top_spike_indices(active_series)
    component_keys = [
        "read_overhead_us",
        "tracker_overhead_us",
        "adaptive_overhead_us",
        "metrics_overhead_us",
        "emit_overhead_us",
    ]
    component_totals: dict[str, float] = {}
    for key in component_keys:
        series = overhead.get(key)
        if series is None:
            continue
        component_totals[key] = float(np.nansum(np.abs(series[spike_idx])))

    total_component = sum(component_totals.values())
    dominant_components: list[dict[str, float | str]] = []
    if total_component > 0.0:
        dominant_components = [
            {
                "name": key,
                "share_pct": round((value / total_component) * 100.0, 2),
            }
            for key, value in sorted(component_totals.items(), key=lambda item: item[1], reverse=True)
            if value > 0.0
        ]

    spike_sampled_fraction = None
    spike_interval_mean = None
    spike_urgency_mean = None
    hints: list[str] = []
    if sampling_summary is not None and spike_idx.size:
        sampled_indices = set(np.asarray(cast(object, sampling_summary["sampled_indices"])).astype(np.int64).tolist())
        sampled_hits = sum(1 for idx in spike_idx if int(idx) in sampled_indices)
        spike_sampled_fraction = sampled_hits / float(spike_idx.size)

        interval_series = np.asarray(cast(object, sampling_summary["interval_series"]), dtype=np.float64)
        urgency_series = np.asarray(cast(object, sampling_summary["urgency_series"]), dtype=np.float64)
        if interval_series.size:
            spike_interval_mean = float(np.nanmean(interval_series[spike_idx]))
        if urgency_series.size:
            spike_urgency_mean = float(np.nanmean(urgency_series[spike_idx]))

    if dominant_components:
        top_names = [str(item["name"]) for item in dominant_components[:2]]
        publish_share = sum(
            float(item["share_pct"])
            for item in dominant_components
            if item["name"] in {"metrics_overhead_us", "emit_overhead_us"}
        )
        core_share = sum(
            float(item["share_pct"])
            for item in dominant_components
            if item["name"] in {"read_overhead_us", "tracker_overhead_us", "adaptive_overhead_us"}
        )
        if publish_share >= 50.0:
            hints.append(
                f"Top active-loop spikes were dominated by publish-path work ({', '.join(top_names)})."
            )
        elif core_share >= 50.0:
            hints.append(
                f"Top active-loop spikes were dominated by read/tracker/scheduler work ({', '.join(top_names)})."
            )

    if spike_sampled_fraction is not None:
        if spike_sampled_fraction <= 0.4:
            hints.append(
                "A large share of spike windows happened on unsampled ticks, pointing away from probe reads and toward publish overhead."
            )
        elif spike_sampled_fraction >= 0.6 and spike_interval_mean is not None and spike_urgency_mean is not None:
            if spike_interval_mean <= 2.0 and spike_urgency_mean >= 0.85:
                hints.append(
                    "Spike windows clustered while urgency was high and the scheduler had collapsed to short intervals."
                )

    status = "stable"
    if score >= 4:
        status = "unstable"
    elif score >= 2:
        status = "suspect"

    return {
        "status": status,
        "score": score,
        "reasons": reasons,
        "dominant_components": dominant_components,
        "signals": {
            "active_loop_p95_us": active_p95,
            "active_loop_p99_us": active_p99,
            "active_loop_max_us": active_max,
            "active_loop_max_over_p95": None if active_p95 <= 0.0 else active_max / active_p95,
            "cpu_p95_pct": cpu_p95,
            "cpu_p99_pct": cpu_p99,
            "cpu_max_pct": cpu_max,
            "cpu_max_over_p95": None if cpu_p95 <= 0.0 else cpu_max / cpu_p95,
            "spike_window_count": int(spike_idx.size),
            "spike_sampled_fraction": spike_sampled_fraction,
            "spike_interval_mean": spike_interval_mean,
            "spike_urgency_mean": spike_urgency_mean,
        },
        "hints": hints,
    }


def plot_io_overhead(
        overhead: dict[str, NDArray[np.float64]],
        burst_regions: list[tuple[int, int, str]],
        poll_interval_s: float = 0.2,
        save_path: str | None = None,
) -> None:
    """Plot daemon overhead telemetry collected in the live I/O run."""
    if not overhead:
        return

    n = len(next(iter(overhead.values())))
    t_sec = np.arange(n) * poll_interval_s

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)

    def _shade_bursts(ax):
        for s, e, _ in burst_regions:
            ax.axvspan(t_sec[s], t_sec[min(e, n - 1)], alpha=0.12,
                       color=_COL_BURST, zorder=0)

    ax = axes[0]
    _shade_bursts(ax)
    plotted_core = False
    for key, color in (
            ("read_overhead_us", "#1f77b4"),
            ("tracker_overhead_us", "#d62728"),
            ("adaptive_overhead_us", "#ff7f0e")):
        if key in overhead:
            plotted_core = _plot_finite_series(
                ax,
                t_sec,
                overhead[key],
                label=key,
                color=color,
            ) or plotted_core
    ax.set_ylabel("Core cost (us)")
    ax.grid(True, alpha=0.25)
    if plotted_core:
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    _shade_bursts(ax)
    plotted_publish = False
    if "metrics_overhead_us" in overhead:
        plotted_publish = _plot_finite_series(
            ax,
            t_sec,
            overhead["metrics_overhead_us"],
            label="metrics_overhead_us",
            color="#2ca02c",
        ) or plotted_publish
    if "emit_overhead_us" in overhead:
        plotted_publish = _plot_finite_series(
            ax,
            t_sec,
            overhead["emit_overhead_us"],
            label="emit_overhead_us",
            color="#8c564b",
        ) or plotted_publish
    active_series = overhead.get("active_loop_us")
    if active_series is None and {"read_overhead_us", "tracker_overhead_us", "adaptive_overhead_us", "metrics_overhead_us"}.issubset(overhead):
        active_series = (
                np.nan_to_num(overhead["read_overhead_us"])
                + np.nan_to_num(overhead["tracker_overhead_us"])
                + np.nan_to_num(overhead["adaptive_overhead_us"])
                + np.nan_to_num(overhead["metrics_overhead_us"])
        )
    if active_series is not None:
        plotted_publish = _plot_finite_series(
            ax,
            t_sec,
            active_series,
            label="active_loop_us",
            color="#9467bd",
            linewidth=1.0,
        ) or plotted_publish
    ax.set_ylabel("Publish cost (us)")
    ax.grid(True, alpha=0.25)
    if plotted_publish:
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    _shade_bursts(ax)
    if "monitor_cpu_pct" in overhead:
        ax.plot(t_sec, overhead["monitor_cpu_pct"], linewidth=1.0,
                color="#8c564b")
    ax.set_ylabel("Daemon CPU %")
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    _shade_bursts(ax)
    if "monitor_user_cpu_s_total" in overhead:
        ax.plot(t_sec, overhead["monitor_user_cpu_s_total"], linewidth=1.0,
                color="#17becf", label="user_cpu_s_total")
    if "monitor_system_cpu_s_total" in overhead:
        ax.plot(t_sec, overhead["monitor_system_cpu_s_total"], linewidth=1.0,
                color="#7f7f7f", label="system_cpu_s_total")
    ax.set_ylabel("Cumulative CPU (s)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Overhead plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def _build_pipe_config(
        min_interval: int = 1,
        max_interval: int = 25,
    profile: str = "default",
) -> PipelineConfig:
    """PipelineConfig tuned for I/O syscall rate signals.

    I/O rates are lognormally distributed with heavy tails; the tracker
    needs moderate drift tolerance (bursts are real, not sensor noise)
    and a wide min/max interval range to capture the overhead reduction.

    ``min_interval`` and ``max_interval`` control the adaptive poller range
    and must match what the live daemon was using so offline evaluation
    replays the same sampling policy.
    """
    params = load_profile(profile)
    return PipelineConfig(
        alpha_base=params["alpha_base"],
        beta=params["beta"],
        outlier_threshold=params["outlier_threshold"],
        drift_tolerance=params["drift_tolerance"],
        initial_variance_floor=params.get("initial_variance_floor", 0.0),
        warmup=params["warmup"],
        warmup_positive_std_only=params.get("warmup_positive_std_only", False),
        min_interval=min_interval,
        max_interval=max_interval,
        variance_sensitivity=params["variance_sensitivity"],
        interval_mapping=params["interval_mapping"],
        interval_power=params["interval_power"],
        interval_logistic_midpoint=params["interval_logistic_midpoint"],
        interval_logistic_steepness=params["interval_logistic_steepness"],
        interval_exponential_steepness=params.get("interval_exponential_steepness", 4.0),
        sigma_ref=params["sigma_ref"],
        sigma_ref_adapt=params["sigma_ref_adapt"],
        instability_window=params["instability_window"],
        instability_weight=params["instability_weight"],
        drift_boost=params["drift_boost"],
        drift_decay=params["drift_decay"],
        cooldown_rate=params["cooldown_rate"],
        outlier_decay=params["outlier_decay"],
        sigma_band=params["sigma_band"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="I/O Syscall Rate Monitoring – offline evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for plot + JSON. Displays interactively if omitted.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for synthetic trace generation.",
    )
    parser.add_argument(
        "--quiet-mean", type=float, default=50.0,
        dest="quiet_mean",
        help="Mean syscall rate during quiet periods (counts per 0.2 s).",
    )
    parser.add_argument(
        "--json", default=None,
        help="Path to a JSON file containing a pre-recorded trace "
             "(flat array of numbers).  Overrides synthetic generation.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.2,
        dest="poll_interval",
        help="eBPF poll interval in seconds (used for x-axis scaling only).",
    )
    parser.add_argument(
        "--min-interval", type=int, default=1,
        dest="min_interval",
        help="Adaptive poller minimum interval (must match the live daemon). "
             "Set to the same value as --max-interval to replay fixed-rate baselines.",
    )
    parser.add_argument(
        "--max-interval", type=int, default=25,
        dest="max_interval",
        help="Adaptive poller maximum interval (must match the live daemon).",
    )
    parser.add_argument(
        "--config-profile",
        choices=available_profiles(),
        default="default",
        help="Named adaptive policy profile to replay offline.",
    )
    parser.add_argument(
        "--include-replay-urgency-values",
        action="store_true",
        help=(
            "Store the per-tick replay urgency sequence in the result JSON. "
            "This permits exact pooled-median aggregation across traces."
        ),
    )
    args = parser.parse_args()

    # --- Load or generate trace ---
    ground_truth_segments = None
    sampling_summary = None
    overhead_metadata = None
    replay_urgency_summary = None
    if args.json:
        with open(args.json) as fh:
            raw = json.load(fh)
        trace, overhead, sampling_summary, overhead_metadata = extract_trace_and_overhead(raw)
        if sampling_summary is None:
            print(f"Loaded trace from {args.json}: {len(trace)} points")
        else:
            print(
                f"Loaded sparse live trace from {args.json}: "
                f"{sampling_summary['n_sampled']} sampled points over "
                f"{sampling_summary['n_total']} ticks"
            )
        if overhead_metadata is not None and overhead_metadata.get("cpu_accounting_source") is not None:
            print(
                "Overhead accounting: "
                f"cpu_source={overhead_metadata['cpu_accounting_source']}  "
                f"cpu_snapshot_every={overhead_metadata.get('cpu_snapshot_interval_ticks')} ticks  "
                f"timing_rows={overhead_metadata.get('timing_measured_rows')}/{overhead_metadata.get('total_rows')}"
            )
    else:
        trace, ground_truth_segments = build_io_trace(
            quiet_mean=args.quiet_mean,
            seed=args.seed,
        )
        overhead = {}
        print(f"Generated synthetic trace: {len(trace)} points  "
              f"({len(trace) * args.poll_interval:.0f} s @ "
              f"{args.poll_interval} s/poll)")

    # --- Pipeline ---
    cfg = _build_pipe_config(
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        profile=args.config_profile,
    )
    print(f"Poller config: min_interval={cfg.min_interval}  "
          f"max_interval={cfg.max_interval}  profile={args.config_profile}")

    if sampling_summary is None:
        t0 = time.monotonic()
        info_loss = evaluate_info_loss(
            trace,
            full_tracker=cfg._make_tracker(),
            poller=cfg._make_poller(),
        )
        elapsed = time.monotonic() - t0
        print(f"Pipeline evaluation done in {elapsed * 1000:.1f} ms")
        print(str(info_loss))

        full_tracker = cfg._make_tracker()
        full_results = full_tracker.track(trace)

        poller = cfg._make_poller()
        poll_results = poller.track(trace)
        replay_urgency_summary = summarize_replay_urgency(
            poll_results,
            include_values=args.include_replay_urgency_values,
        )

        burst_regions = detect_bursts(trace)
        print(f"\nDetected {len(burst_regions)} burst region(s):")
        for s, e, lbl in burst_regions:
            print(f"  [{s:5d}–{e:5d}]  {lbl}")

        per_burst = burst_peak_recall(trace, poll_results, burst_regions)
        print("\nPeak recall per burst (top 10%):")
        for r in per_burst:
            print(f"  {r['label']:<35s}  recall={r['peak_recall']:.1%}  "
                  f"({r['n_captured']}/{r['n_top']})")
        print(f"\nOverall peak recall (top 5%): {info_loss.peak_recall_top5:.1%}")
        print(f"Overall sample ratio:          {info_loss.sample_ratio:.1%}")
        info_str = (
            f"n={info_loss.n_total}  sampled={info_loss.n_sampled}"
            f"  ratio={info_loss.sample_ratio:.1%}"
            f"  peak_recall_top5={info_loss.peak_recall_top5:.1%}"
            f"  NRMSE_mean={info_loss.nrmse_mean:.4f}"
            f"  max_gap={info_loss.max_gap}"
        )
    else:
        info_loss = None
        full_results = []
        poll_results = []
        burst_regions = []
        per_burst = []
        print("\nLive sampling summary:")
        print(f"  Total ticks:       {sampling_summary['n_total']}")
        print(f"  Sampled:           {sampling_summary['n_sampled']} ({sampling_summary['sample_ratio']:.1%})")
        print(f"  Max gap:           {sampling_summary['max_gap']} samples")
        print(f"  Mean gap:          {sampling_summary['mean_gap']:.1f} samples")
        print(f"  P95 gap:           {sampling_summary['p95_gap']:.1f} samples")
        print("  Fidelity metrics:  unavailable (live probe reads were truly skipped)")
        info_str = (
            f"n={sampling_summary['n_total']}  sampled={sampling_summary['n_sampled']}"
            f"  ratio={sampling_summary['sample_ratio']:.1%}"
            f"  max_gap={sampling_summary['max_gap']}"
        )

    overhead_summary = summarize_overhead(overhead)
    n_points = int(sampling_summary["n_total"]) if sampling_summary is not None else len(trace)
    cpu_normalization = compute_cpu_normalization(
        overhead,
        overhead_summary,
        n_points=n_points,
        poll_interval_s=args.poll_interval,
    )
    cost_accounting = compute_cost_accounting(overhead_summary, overhead_metadata, sampling_summary)
    stability_assessment = assess_run_stability(overhead, overhead_summary, sampling_summary)
    if overhead_summary:
        print("\nOverhead summary:")
        for key, stats in overhead_summary.items():
            print(f"  {key:<28s} mean={stats['mean']:.3f}  p95={stats['p95']:.3f}  max={stats['max']:.3f}")
        if overhead_metadata is not None and overhead_metadata.get("timing_measured_rows"):
            timing_rows = int(cast(int, overhead_metadata["timing_measured_rows"]))
            total_rows = int(cast(int, overhead_metadata["total_rows"]))
            if 0 < timing_rows < total_rows:
                print(
                    "  Note: component micro-timings were sampled sparsely to reduce observer effect; "
                    f"summary stats come from {timing_rows}/{total_rows} emitted rows."
                )
    if cpu_normalization is not None:
        wall_pct = cpu_normalization.get("mean_cpu_pct_over_wall")
        total_cpu_s = cpu_normalization.get("total_cpu_s")
        snapshot_pct = cpu_normalization.get("snapshot_mean_cpu_pct")
        print("\nCPU normalization:")
        if wall_pct is not None and total_cpu_s is not None:
            print(
                f"  Wall-normalized mean CPU: {wall_pct:.3f}%  "
                f"({total_cpu_s:.3f} s over {cpu_normalization['wall_time_s']:.1f} s)"
            )
        if snapshot_pct is not None:
            print(f"  Snapshot mean CPU:        {snapshot_pct:.3f}%")
    scheduler_summary = None
    if overhead_metadata is not None:
        source_trace_mean_urgency = (
            None if overhead_metadata.get("mean_urgency") is None
            else float(overhead_metadata["mean_urgency"])
        )
        scheduler_summary = {
            # Keep mean_urgency for compatibility with live collectors. On a
            # replay result this is the urgency already present in the source
            # trace, not the replay poller's urgency.
            "mean_urgency": source_trace_mean_urgency,
            "source_trace_mean_urgency": source_trace_mean_urgency,
            "mean_interval": None if overhead_metadata.get("mean_interval") is None else float(overhead_metadata["mean_interval"]),
            "mean_sample_ratio_live": None if overhead_metadata.get("mean_sample_ratio_live") is None else float(overhead_metadata["mean_sample_ratio_live"]),
            "replay_mean_urgency": None if replay_urgency_summary is None else replay_urgency_summary["mean"],
            "replay_median_urgency": None if replay_urgency_summary is None else replay_urgency_summary["median"],
        }
    if scheduler_summary is not None:
        mean_urgency = scheduler_summary.get("mean_urgency")
        mean_interval = scheduler_summary.get("mean_interval")
        if mean_urgency is not None or mean_interval is not None:
            print("\nScheduler summary:")
            if replay_urgency_summary is not None:
                print(
                    f"  Replay mean urgency:      "
                    f"{replay_urgency_summary['mean']:.4f}"
                )
                print(
                    f"  Replay median urgency:    "
                    f"{replay_urgency_summary['median']:.4f}"
                )
            elif mean_urgency is not None:
                print(f"  Mean urgency:             {mean_urgency:.4f}")
            if mean_interval is not None:
                print(f"  Mean interval (ticks):    {mean_interval:.3f}")
    if cost_accounting is not None:
        print("\nSeparated cost accounting:")
        probe_total = cost_accounting.get("probe_fetch_total_s_estimated")
        probe_mean = cost_accounting.get("probe_fetch_mean_us")
        if probe_total is not None and probe_mean is not None:
            print(
                f"  Probe fetch total:       {probe_total:.6f} s est  "
                f"({probe_mean:.3f} us mean over {cost_accounting['sampled_events']} polls)"
            )
        post_total = cost_accounting.get("post_fetch_control_total_s_estimated")
        publish_total = cost_accounting.get("publish_path_total_s_estimated")
        if post_total is not None:
            print(f"  Post-fetch control:      {post_total:.6f} s est")
        if publish_total is not None:
            print(f"  Publish path:            {publish_total:.6f} s est")
    if stability_assessment is not None:
        print("\nStability assessment:")
        print(f"  Status: {stability_assessment['status']} (score={stability_assessment['score']})")
        for reason in stability_assessment["reasons"]:
            print(f"  Reason: {reason}")
        for hint in stability_assessment["hints"]:
            print(f"  Hint:   {hint}")

    # --- Output ---
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        plot_name = (
            f"io_syscall_live_collection_{run_id}.png"
            if sampling_summary is not None
            else f"io_syscall_adaptive_replay_{run_id}.png"
        )
        plot_path = os.path.join(args.out, plot_name)
        overhead_plot_path = os.path.join(args.out, f"io_syscall_monitoring_overhead_{run_id}.png")
        results_path = os.path.join(args.out, f"io_syscall_results_{run_id}.json")

        if sampling_summary is None:
            plot_io_evaluation(
                trace, poll_results, full_results,
                burst_regions=burst_regions,
                ground_truth_segments=ground_truth_segments,
                info_loss_str=info_str,
                poll_interval_s=args.poll_interval,
                save_path=plot_path,
            )
        else:
            plot_live_sampling(
                sampling_summary,
                poll_interval_s=args.poll_interval,
                save_path=plot_path,
            )

        results = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trace_source": args.json if args.json else "synthetic",
            "n_points": n_points,
            "poll_interval_s": args.poll_interval,
            "config_profile": args.config_profile,
            "config": cfg.to_dict(),
            "info_loss": None if info_loss is None else asdict(info_loss),
            "overhead_metadata": overhead_metadata,
            "scheduler_summary": scheduler_summary,
            "replay_urgency_summary": replay_urgency_summary,
            "cpu_normalization": cpu_normalization,
            "cost_accounting": cost_accounting,
            "sampling_summary": None if sampling_summary is None else {
                "start_tick": sampling_summary["start_tick"],
                "end_tick": sampling_summary["end_tick"],
                "n_total": sampling_summary["n_total"],
                "n_sampled": sampling_summary["n_sampled"],
                "sample_ratio": sampling_summary["sample_ratio"],
                "max_gap": sampling_summary["max_gap"],
                "mean_gap": sampling_summary["mean_gap"],
                "p95_gap": sampling_summary["p95_gap"],
                "mean_urgency": None if scheduler_summary is None else scheduler_summary.get("mean_urgency"),
                "mean_interval": None if scheduler_summary is None else scheduler_summary.get("mean_interval"),
            },
            "overhead_summary": overhead_summary,
            "stability_assessment": stability_assessment,
            "burst_regions": [
                {"start": s, "end": e, "label": lbl}
                for s, e, lbl in burst_regions
            ],
            "per_burst_peak_recall": per_burst,
        }
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2)
        if overhead:
            plot_io_overhead(
                overhead,
                burst_regions=burst_regions,
                poll_interval_s=args.poll_interval,
                save_path=overhead_plot_path,
            )
        print(f"\nResults saved to {results_path}")
    else:
        if sampling_summary is None:
            plot_io_evaluation(
                trace, poll_results, full_results,
                burst_regions=burst_regions,
                ground_truth_segments=ground_truth_segments,
                info_loss_str=info_str,
                poll_interval_s=args.poll_interval,
            )
        else:
            plot_live_sampling(
                sampling_summary,
                poll_interval_s=args.poll_interval,
            )
        if overhead:
            plot_io_overhead(
                overhead,
                burst_regions=burst_regions,
                poll_interval_s=args.poll_interval,
            )


if __name__ == "__main__":
    main()
