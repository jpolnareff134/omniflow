"""
Batch-mode plotting for very long traces.

Provides two complementary views:

1. **Aggregated overview** - bins the full trace into *n_bins* windows and
   plots per-bin summary statistics (mean value, mean urgency, outlier
   density, cumulative sample ratio).  Readable even for millions of
   points.

2. **Auto-zoom panels** - automatically identifies the *n_regions* most
   "interesting" windows (highest outlier density / urgency spikes) and
   renders a detailed sub-plot for each at full sample resolution.
"""

from __future__ import annotations

import logging as log
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from tracker.windowed import PollResult, TickResult, Status


# ----------------------------------------------------------------------
# Region-of-interest detection
# ----------------------------------------------------------------------

@dataclass
class Region:
    """An interesting slice of the trace."""
    start: int
    end: int
    score: float  # composite interestingness metric
    label: str = ""  # human-readable tag ("outlier burst", etc.)


def find_interesting_regions(
        poll_results: list[PollResult],
        n_regions: int = 5,
        window: int = 2000,
        stride: int | None = None,
        min_gap: int | None = None,
) -> list[Region]:
    """Slide a window over the poll results and rank by interestingness.

    *Interestingness* is a weighted combination of:
    - outlier + drift count in the window,
    - peak urgency in the window,
    - urgency variance (change).

    After scoring all windows, overlapping regions are merged and the
    top *n_regions* are returned.

    Parameters
    ----------
    poll_results : list[PollResult]
        The full list from the adaptive poller.
    n_regions : int
        How many top regions to return.
    window : int
        Width of the sliding window in samples.
    stride : int, optional
        Step between successive windows (default: window // 4).
    min_gap : int, optional
        Minimum gap between selected regions.  Regions closer than this
        are merged.  Default: ``window``.
    """
    n = len(poll_results)
    if stride is None:
        stride = max(1, window // 4)
    if min_gap is None:
        min_gap = window

    # Pre-compute arrays once
    urgencies = np.array([pr.urgency for pr in poll_results])
    is_anom = np.array([
        1.0 if (pr.sampled and pr.tick is not None
                and pr.tick.status in (Status.OUTLIER, Status.DRIFT_RESET))
        else 0.0
        for pr in poll_results
    ])

    candidates: list[Region] = []
    for start in range(0, n - window + 1, stride):
        end = start + window
        urg_slice = urgencies[start:end]
        anom_count = is_anom[start:end].sum()

        score = (
                anom_count * 2.0  # outlier density dominates
                + float(urg_slice.max()) * window  # peak urgency
                + float(urg_slice.std()) * window  # urgency variance -> change
        )
        candidates.append(Region(start=start, end=end, score=score))

    # Sort by score descending, then greedily pick non-overlapping ones
    candidates.sort(key=lambda r: r.score, reverse=True)
    selected: list[Region] = []
    for cand in candidates:
        if len(selected) >= n_regions:
            break
        # Check overlap / proximity
        if any(abs(cand.start - s.start) < min_gap for s in selected):
            continue
        cand.label = f"region @ {cand.start:,}-{cand.end:,}"
        selected.append(cand)

    # Return in trace order
    selected.sort(key=lambda r: r.start)
    return selected


# ----------------------------------------------------------------------
# Aggregated overview plot
# ----------------------------------------------------------------------

def plot_batch_overview(
        trace: NDArray[np.float64],
        poll_results: list[PollResult],
        full_results: list[TickResult] | None = None,
        n_bins: int = 1500,
        regions: list[Region] | None = None,
        title: str = "Batch Replay Overview",
        figsize: tuple[int, int] = (18, 12),
        save_path: str | None = None,
) -> None:
    """Produce a multi-panel aggregated view of a very long trace.

    Panels:
    1. Binned value (mean +- std) + optional dense-reference mean
    2. Mean error (|dense-reference mean - replay mean|) - only when *full_results* given
    3. Mean urgency per bin
    4. Outlier density per bin
    5. Cumulative sample ratio

    Optionally highlights discovered regions of interest.
    """
    n = len(trace)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Pre-compute per-point arrays
    urgencies = np.array([pr.urgency for pr in poll_results])
    sampled = np.array([1.0 if pr.sampled else 0.0 for pr in poll_results])

    # Dense-reference means (for error panel)
    has_full = full_results is not None
    if has_full:
        full_means = np.array([r.mean for r in full_results])
        # Reconstruct adaptive means via linear interpolation
        sampled_idx = []
        sampled_m = []
        for i, pr in enumerate(poll_results):
            if pr.sampled and pr.tick is not None:
                sampled_idx.append(i)
                sampled_m.append(pr.tick.mean)
        if len(sampled_idx) >= 2:
            adaptive_means = np.interp(
                np.arange(n), sampled_idx, sampled_m)
        elif len(sampled_idx) == 1:
            adaptive_means = np.full(n, sampled_m[0])
        else:
            adaptive_means = np.zeros(n)
        mean_error = np.abs(full_means - adaptive_means)
    else:
        mean_error = None

    # Bin them
    val_mean = np.empty(n_bins)
    val_std = np.empty(n_bins)
    urg_mean = np.empty(n_bins)
    urg_max = np.empty(n_bins)
    sample_ratio = np.empty(n_bins)
    err_mean = np.empty(n_bins) if has_full else None
    err_max = np.empty(n_bins) if has_full else None

    for i in range(n_bins):
        s, e = bin_edges[i], bin_edges[i + 1]
        if e <= s:
            e = s + 1
        sl = slice(s, e)
        val_mean[i] = trace[sl].mean()
        val_std[i] = trace[sl].std()
        urg_mean[i] = urgencies[sl].mean()
        urg_max[i] = urgencies[sl].max()
        sample_ratio[i] = sampled[sl].mean()
        if has_full:
            err_mean[i] = mean_error[sl].mean()
            err_max[i] = mean_error[sl].max()

    cum_sample = np.cumsum(sampled) / (np.arange(1, n + 1))

    n_panels = 4 if has_full else 3
    ratios = [3, 1, 1, 1] if has_full else [3, 1, 1]
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": ratios})

    # Helper to shade regions on an axis
    def _shade(__ax):
        if regions:
            for r in regions:
                __ax.axvspan(r.start, r.end, alpha=0.10, color="gold",
                             zorder=0)

    # --- Panel 1: value ---
    ax = axes[0]
    ax.plot(bin_centres, val_mean, linewidth=0.7, color="steelblue")
    ax.fill_between(bin_centres,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="steelblue")

    ax.set_ylabel("Value (binned mu +- sigma)")
    ax.set_title(f"{title}  -  {n:,} points, {n_bins} bins")
    ax.grid(True, alpha=0.3)
    _shade(ax)

    # --- Panel: dense-reference mean error (only when full_results given) ---
    panel_idx = 1
    if has_full:
        ax = axes[panel_idx]
        ax.fill_between(bin_centres, 0, err_max, alpha=0.2,
                        color="firebrick", step="mid", label="max")
        ax.fill_between(bin_centres, 0, err_mean, alpha=0.5,
                        color="firebrick", step="mid", label="mean")
        ax.set_ylabel("|dense mean error|")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        _shade(ax)
        panel_idx += 1

    # --- Panel: urgency ---
    ax = axes[panel_idx]
    ax.fill_between(bin_centres, 0, urg_mean, alpha=0.4, color="crimson",
                    step="mid")
    ax.plot(bin_centres, urg_max, linewidth=0.5, color="darkred",
            alpha=0.5, label="max")
    ax.set_ylabel("Urgency")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    _shade(ax)
    panel_idx += 1

    # --- Panel: cumulative sample ratio ---
    ax = axes[panel_idx]
    # Plot at a sub-sampled resolution (every ~1000th point) for speed
    step = max(1, n // 5000)
    ax.plot(np.arange(n)[::step], cum_sample[::step],
            linewidth=0.8, color="teal")
    ax.fill_between(np.arange(n)[::step], cum_sample[::step],
                    alpha=0.3, color="teal")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
    ax.set_ylabel("Cum. sample ratio")
    ax.set_xlabel("Time (sample index)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, n - 1)
    ax.grid(True, alpha=0.3)
    _shade(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        log.info("Batch replay overview saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Zoom panels for interesting regions
# ----------------------------------------------------------------------

def plot_batch_zooms(
        trace: NDArray[np.float64],
        poll_results: list[PollResult],
        regions: list[Region],
        full_results: list[TickResult] | None = None,
        sigma_band: float = 3.0,
        figsize_per_region: tuple[int, int] = (16, 6),
        save_path: str | None = None,
) -> None:
    """Produce detailed zoom panels for each region.

    Each region gets a 3-row subplot:
    1. Raw values + replay mean (+ dense-reference mean if given) + outlier/drift markers
    2. Urgency
    3. Poll interval

    If *save_path* is given the figure is saved as a multi-page PDF
    (one page per region).  Otherwise ``plt.show()`` is called for each.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(save_path) if save_path else None

    try:
        for idx, region in enumerate(regions):
            s, e = region.start, min(region.end, len(trace))
            sl = slice(s, e)
            t = np.arange(s, e)

            prs = poll_results[s:e]
            values = trace[sl]

            # Build adaptive mean and interpolated signal via linear interpolation
            s_idx, s_m, s_s = [], [], []
            for i, pr in enumerate(prs):
                if pr.sampled and pr.tick is not None:
                    s_idx.append(i)
                    s_m.append(pr.tick.mean)
                    s_s.append(pr.tick.std)
            x_full = np.arange(len(prs))
            if len(s_idx) >= 2:
                ad_mean = np.interp(x_full, s_idx, s_m)
                ad_std = np.interp(x_full, s_idx, s_s)
                interp_signal = np.interp(x_full, s_idx,
                                          [values[j] for j in s_idx])
            elif len(s_idx) == 1:
                ad_mean = np.full(len(prs), s_m[0])
                ad_std = np.full(len(prs), s_s[0])
                interp_signal = np.full(len(prs), values[s_idx[0]])
            else:
                ad_mean = np.full(len(prs), values[0])
                ad_std = np.ones(len(prs))
                interp_signal = np.full(len(prs), values[0])

            urgencies = np.array([pr.urgency for pr in prs])
            intervals = np.array([pr.interval for pr in prs])
            sampled_mask = np.array([pr.sampled for pr in prs])

            fig, axes = plt.subplots(
                3, 1, figsize=figsize_per_region, sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1]},
            )

            # --- Signal ---
            ax = axes[0]
            ax.plot(t, values, linewidth=0.5, color="lightsteelblue",
                    alpha=0.7, label="raw")
            ax.plot(t, interp_signal, linewidth=1.0, color="steelblue",
                    alpha=0.8, label="interpolated replay")
            ax.scatter(t[sampled_mask], values[sampled_mask], s=10,
                       color="steelblue", zorder=4,
                       label=f"sampled ({sampled_mask.sum()}/{len(prs)})")
            ax.plot(t, ad_mean, linewidth=0.6, color="tomato",
                    linestyle=":", alpha=0.6, label="replay mean")
            ax.fill_between(t,
                            ad_mean - sigma_band * ad_std,
                            ad_mean + sigma_band * ad_std,
                            color="tomato", alpha=0.08)
            ax.set_ylabel("Value")
            ax.set_title(
                f"Zoom {idx + 1}/{len(regions)}:  "
                f"index {s:,} - {e:,}  "
                f"(score {region.score:.0f})"
            )
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # --- Urgency ---
            ax = axes[1]
            ax.fill_between(t, 0, urgencies, alpha=0.4, color="crimson",
                            step="post")
            ax.step(t, urgencies, where="post", linewidth=0.8,
                    color="crimson")
            ax.set_ylabel("Urgency")
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)

            # --- Interval ---
            ax = axes[2]
            ax.step(t, intervals, where="post", linewidth=0.8,
                    color="darkorange")
            ax.set_ylabel("Poll interval (steps)")
            ax.set_xlabel("Time (sample index)")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()

            if pdf is not None:
                pdf.savefig(fig, dpi=200)
            else:
                plt.show()
            plt.close(fig)

    finally:
        if pdf is not None:
            pdf.close()
            log.info("Zoom panels saved to %s  (%d pages)",
                     save_path, len(regions))
