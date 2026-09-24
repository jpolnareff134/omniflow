from __future__ import annotations

import logging as log

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from tracker.windowed import Status, TickResult, PollResult


def plot_tracking(
        results: list[TickResult],
        title: str = "Dense Reference Tracking",
        figsize: tuple[int, int] = (14, 7),
        sigma_band: float = 3.0,
        save_path: str | None = None,
) -> None:
    """Visualise the tracker output: raw signal, running mean +- sigma band,
    and outlier / drift markers.

    Parameters
    ----------
    results : list[TickResult]
        Output of ``WindowedTracker.track()``.
    sigma_band : float
        Width of the confidence band in standard deviations.
    save_path : str or None
        If given, save figure to this path; otherwise ``plt.show()``.
    """
    n = len(results)
    t = np.arange(n)

    values = np.array([r.value for r in results])
    means = np.array([r.mean for r in results])
    stds = np.array([r.std for r in results])

    outlier_idx = [i for i, r in enumerate(results) if r.status == Status.OUTLIER]
    drift_idx = [i for i, r in enumerate(results) if r.status == Status.DRIFT_RESET]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(t, values, linewidth=0.6, color="steelblue", alpha=0.8,
            label="observed")
    ax.plot(t, means, linewidth=1.2, color="tomato", label="Tracker mean")
    ax.fill_between(t,
                    means - sigma_band * stds,
                    means + sigma_band * stds,
                    color="tomato", alpha=0.12,
                    label=f"+-{sigma_band:.0f}sigma band")

    if outlier_idx:
        ax.scatter(outlier_idx, values[outlier_idx],
                   marker="x", s=30, color="orange", zorder=5,
                   label=f"outlier ({len(outlier_idx)})")
    if drift_idx:
        for di in drift_idx:
            ax.axvline(di, color="purple", linewidth=1, linestyle="--",
                       alpha=0.3)
        # single legend entry
        ax.plot([], [], color="purple", linestyle="--",
                label=f"drift reset ({len(drift_idx)})")

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom panel: z-scores ---
    ax2 = axes[1]
    zscores = np.array([r.z_score for r in results])
    ax2.bar(t, zscores, width=1.0, color="grey", alpha=0.5)
    ax2.axhline(sigma_band, color="red", linestyle="--", linewidth=0.8)
    ax2.axhline(-sigma_band, color="red", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("z-score")
    ax2.set_xlabel("Time (sample index)")
    ax2.set_xlim(0, n - 1)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        log.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_adaptive_polling(
        data: NDArray[np.float64],
        poll_results: list[PollResult],
        full_results: list[TickResult] | None = None,
        title: str = "Adaptive Replay",
        figsize: tuple[int, int] = (14, 10),
        sigma_band: float = 3.0,
        save_path: str | None = None,
) -> None:
    """Four-panel visualisation of adaptive replay output.

    Panels (top -> bottom):
    1. Raw signal + replay mean (+ optional dense reference mean) + sampled markers
    2. Poll interval over time
    3. Urgency score over time
    4. Cumulative sample ratio

    Parameters
    ----------
    data : array
        The original dense trace.
    poll_results : list[PollResult]
        Output of ``AdaptivePoller.track()``.
    full_results : list[TickResult], optional
        Output of a dense-reference ``WindowedTracker.track()`` for comparison.
    """
    n = len(data)
    t = np.arange(n)

    # Collect sampled points for interpolation
    s_indices: list[int] = []
    s_means: list[float] = []
    s_stds: list[float] = []
    for i, pr in enumerate(poll_results):
        if pr.sampled and pr.tick is not None:
            s_indices.append(i)
            s_means.append(pr.tick.mean)
            s_stds.append(pr.tick.std)

    # Linearly interpolate replay means/stds back to dense resolution
    if len(s_indices) >= 2:
        si = np.array(s_indices)
        adaptive_means = np.interp(t, si, np.array(s_means))
        adaptive_stds = np.interp(t, si, np.array(s_stds))
        interpolated_signal = np.interp(t, si, data[si])
    elif len(s_indices) == 1:
        adaptive_means = np.full(n, s_means[0])
        adaptive_stds = np.full(n, s_stds[0])
        interpolated_signal = np.full(n, data[s_indices[0]])
    else:
        adaptive_means = np.full(n, data[0])
        adaptive_stds = np.ones(n)
        interpolated_signal = np.full(n, data[0])

    sampled_idx = [pr.time_index for pr in poll_results if pr.sampled]
    sampled_vals = data[sampled_idx]

    intervals = np.array([pr.interval for pr in poll_results])
    urgencies = np.array([pr.urgency for pr in poll_results])
    cum_ratio = np.cumsum([1 if pr.sampled else 0 for pr in poll_results]
                          ) / (t + 1)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    # ----- Signal -----
    ax = axes[0]
    ax.plot(t, data, linewidth=0.5, color="lightsteelblue", alpha=0.7,
            label="dense signal")
    ax.plot(t, interpolated_signal, linewidth=1.2, color="steelblue",
            alpha=0.8, label="interpolated replay")
    ax.scatter(sampled_idx, sampled_vals, s=12, color="steelblue", zorder=4,
               label=f"sampled ({len(sampled_idx)}/{n})")
    ax.plot(t, adaptive_means, linewidth=0.6, color="tomato",
            linestyle=":", alpha=0.6, label="replay mean")
    ax.fill_between(t,
                    adaptive_means - sigma_band * adaptive_stds,
                    adaptive_means + sigma_band * adaptive_stds,
                    color="tomato", alpha=0.08,
                    label=f"+-{sigma_band:.0f}sigma band")

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ----- Interval
    ax = axes[1]
    ax.step(t, intervals, where="post", linewidth=0.9, color="darkorange")
    ax.set_ylabel("Poll interval")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ----- Urgency + d/dx of urgency -----
    ax = axes[2]
    ax.fill_between(t, 0, urgencies, alpha=0.4, color="crimson",
                    step="post")
    ax.step(t, urgencies, where="post", linewidth=0.8, color="crimson")
    ax.set_ylabel("Urgency")
    # ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    # urgency_diff = np.diff(urgencies, prepend=urgencies[0])
    # ax.step(t, urgency_diff, where="post", linewidth=0.8, color="crimson")
    # ax.set_ylabel("Delta urgency")
    # ax.set_ylim(-0.5, 0.5)

    # ----- Panel 4: cumulative sample ratio -----
    ax = axes[3]
    ax.plot(t, cum_ratio, linewidth=1.0, color="teal")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
    ax.fill_between(t, cum_ratio, alpha=0.4, color="teal")
    ax.set_ylabel("Cum. sample ratio")
    ax.set_xlabel("Time (aka index of incoming data point)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, n - 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        log.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_composed_trace(
        data: NDArray[np.float64],
        boundaries: list[tuple[int, int, str]] | None = None,
        title: str = "Composed Trace",
        ylabel: str = "Value",
        figsize: tuple[int, int] = (14, 4),
        ylim: tuple[float, float] | None = None,
        show_mean: bool = True,
        show_stats: bool = True,
        save_path: str | None = None,
) -> None:
    """Plot a composed trace with optional segment-boundary shading.

    Parameters
    ----------
    boundaries : list[(start, end, label)]
        Output of :meth:`TraceComposer.segment_boundaries`.  When given,
        alternating segments are lightly shaded and labelled.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data, linewidth=0.7, color="steelblue", alpha=0.9)

    # Segment shading
    if boundaries:
        colors = ["#e0e0ff", "#ffe0e0", "#e0ffe0", "#fff0d0",
                  "#f0e0ff", "#e0ffff"]
        for i, (s, e, lbl) in enumerate(boundaries):
            c = colors[i % len(colors)]
            ax.axvspan(s, e, alpha=0.15, color=c)
            mid = (s + e) / 2
            ax.text(mid, ax.get_ylim()[1] if ylim is None else ylim[1],
                    lbl, ha="center", va="bottom", fontsize=8,
                    fontstyle="italic", color="grey")

    if show_mean:
        mean = float(np.mean(data))
        ax.axhline(mean, color="tomato", linestyle="--", linewidth=1,
                   label=f"mean = {mean:.1f}")
        ax.legend(loc="upper right")

    if show_stats:
        stats_text = (
            f"n    = {len(data)}\n"
            f"mean = {np.mean(data):.2f}\n"
            f"std  = {np.std(data):.2f}\n"
            f"min  = {np.min(data):.2f}\n"
            f"max  = {np.max(data):.2f}"
        )
        ax.text(0.01, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat",
                          alpha=0.5))

    ax.set_title(title)
    ax.set_xlabel("Time (sample index)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, len(data) - 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        log.info(f"Figure saved to {save_path}")
    else:
        plt.show()
