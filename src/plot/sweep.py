"""
Sweep result visualisation.

Handles millions of records efficiently by computing aggregates (median,
percentiles) per parameter value before plotting.  All data stays in
columnar numpy arrays - no pandas dependency.

Main entry points
-----------------
* :func:`plot_marginal`    - one subplot per parameter, shows how each
  parameter independently affects a chosen metric.
* :func:`plot_configuration_metric` - compare a chosen metric across
    explicit configuration labels.
* :func:`plot_pareto`      - hexbin density of two metrics with the
  Pareto frontier highlighted.
* :func:`plot_importance`  - bar chart ranking which parameters explain
  the most variance in a metric.
* :func:`plot_urgency_profiles` - one shared urgency trace and stacked
    per-configuration polling-interval traces for urgency sweeps.
* :func:`plot_heatmap`     - 2-D heatmap of a metric vs two parameters.
"""

from __future__ import annotations

import json
import logging as log
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

# -- Constants ----------------------------------------------------------

_DPI = 300
_GRID_ALPHA = 0.3
_PRIMARY = "steelblue"
_ACCENT = "tomato"

# Metrics where higher values are better (maximize).  Everything else
# is treated as lower-is-better (minimize).
_HIGHER_IS_BETTER = frozenset({
    "correlation",
    "peak_recall_top5",
    "coverage_3sigma",
    "jaccard_anomalies",
})


# -- I/O helpers --------------------------------------------------------

def _records_to_columns(records: list[dict[str, Any]]) -> dict[str, NDArray]:
    """Convert a list of flat dicts to columnar numpy arrays."""
    keys = list(records[0].keys())
    return {k: np.array([r[k] for r in records]) for k in keys}


def load_sweep(path: str | Path) -> tuple[dict[str, NDArray], dict[str, Any]]:
    """Load a sweep JSON and return ``(columnar_arrays, metadata)``.

    The *columnar_arrays* dict maps every field name (parameter or
    metric) to a 1-D numpy array with one entry per configuration.
    """
    log.info("Loading sweep from %s", path)
    with open(path) as f:
        data = json.load(f)

    records = data.pop("results")
    columns = _records_to_columns(records)
    log.info(
        "Loaded sweep with %d columns and metadata: %s",
        len(columns), data,
    )
    return columns, data


# -- Aggregation --------------------------------------------------------

def _grouped_percentiles(
        param_col: NDArray,
        metric_col: NDArray,
        percentiles: tuple[float, ...] = (25, 50, 75),
) -> tuple[NDArray, list[NDArray]]:
    """For each unique value in *param_col*, compute percentiles of *metric_col*.

    Returns ``(sorted_unique_values, [array_per_percentile])``.
    """
    uniques = np.unique(param_col)
    result = [np.empty(len(uniques)) for _ in percentiles]
    for i, val in enumerate(uniques):
        mask = param_col == val
        pcts = np.percentile(metric_col[mask], percentiles)
        for j, pct_val in enumerate(pcts):
            result[j][i] = pct_val
    return uniques, result


# -- Plots --------------------------------------------------------------

def plot_marginal(
        columns: dict[str, NDArray],
        sweep_params: dict[str, list],
        metric: str = "rmse_mean",
        figsize: tuple[int, int] | None = None,
        n_cols: int = 4,
        save_path: str | None = None,
) -> None:
    """Marginal effect of each parameter on *metric*.

    For every swept parameter a subplot shows how the **median** of
    *metric* varies as that parameter changes, with p25-p75 shading.
    This answers "which parameters matter, and in which direction?"
    irrespective of all other parameter values.
    """
    if metric not in columns:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(columns)}"
        )

    metric_col = columns[metric]

    # Use metadata to know which parameters were swept
    swept = [
        (p, columns[p]) for p in sorted(sweep_params)
        if p in columns and len(sweep_params[p]) > 1
    ]
    if not swept:
        log.warning("No swept parameters found - nothing to plot.")
        return

    n_params = len(swept)
    n_rows = (n_params + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes: list[list[plt.Axes]]

    for idx, (pname, pcol) in enumerate(swept):
        ax: plt.Axes = axes[idx // n_cols][idx % n_cols]
        uniques, (p25, median, p75) = _grouped_percentiles(pcol, metric_col)

        x = np.arange(len(uniques))
        ax.plot(x, median, marker="o", markersize=4,
                color=_PRIMARY, linewidth=1.4)
        ax.fill_between(x, p25, p75, color=_PRIMARY, alpha=0.2)

        labels = [f"{v:g}" if isinstance(v, float) else str(v)
                  for v in uniques]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7,
                           rotation=45 if len(labels) > 6 else 0)
        ax.set_title(pname, fontsize=9)
        ax.set_ylabel(metric, fontsize=7)
        ax.grid(True, alpha=_GRID_ALPHA)

    # Hide unused subplots
    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Marginal effect on '{metric}'  (median +- IQR, n={len(metric_col):,})",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI)
        log.info("Saved marginal plot to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_configuration_metric(
        columns: dict[str, NDArray],
        metric: str = "rmse_mean",
        figsize: tuple[int, int] | None = None,
        save_path: str | None = None,
) -> None:
    """Compare a metric across explicit configuration labels."""
    if metric not in columns:
        raise ValueError(f"Unknown metric {metric!r}. Available: {sorted(columns)}")
    if "config_id" not in columns or "config_label" not in columns:
        raise ValueError("Sweep results do not contain config_id/config_label columns.")

    config_ids = np.asarray(columns["config_id"])
    config_labels = np.asarray(columns["config_label"])
    metric_values = np.asarray(columns[metric], dtype=np.float64)

    try:
        order = np.argsort(config_ids.astype(np.int64))
    except (TypeError, ValueError):
        order = np.argsort(config_ids.astype(str))

    ordered_ids = config_ids[order]
    ordered_labels = config_labels[order]
    ordered_metric = metric_values[order]

    if figsize is None:
        figsize = (12, max(4, int(np.ceil(0.45 * len(order)))))

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(order))
    labels = [f"[{config_id}] {label}" for config_id, label in zip(ordered_ids, ordered_labels)]
    ax.barh(y, ordered_metric, color=_PRIMARY, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"Configuration comparison for '{metric}'  (n={len(ordered_metric):,})")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI)
        log.info("Saved configuration plot to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_pareto(
        columns: dict[str, NDArray],
        x_metric: str = "sample_ratio",
        y_metric: str = "rmse_mean",
        figsize: tuple[int, int] = (8, 6),
        gridsize: int = 45,
        save_path: str | None = None,
) -> None:
    """Hexbin plot of two metrics with a Pareto frontier overlay."""
    if x_metric not in columns:
        raise ValueError(f"Unknown x metric {x_metric!r}. Available: {sorted(columns)}")
    if y_metric not in columns:
        raise ValueError(f"Unknown y metric {y_metric!r}. Available: {sorted(columns)}")

    x = np.asarray(columns[x_metric], dtype=np.float64)
    y = np.asarray(columns[y_metric], dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)

    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="Blues", mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="count")

    maximize_y = y_metric in _HIGHER_IS_BETTER
    maximize_x = x_metric in _HIGHER_IS_BETTER

    x_sorted_idx = np.argsort(x)
    x_s, y_s = x[x_sorted_idx], y[x_sorted_idx]

    n_bins = min(200, gridsize)
    bin_edges = np.linspace(x_s[0], x_s[-1], n_bins + 1)
    bin_idx = np.clip(np.digitize(x_s, bin_edges) - 1, 0, n_bins - 1)

    best_y = np.max if maximize_y else np.min
    pareto_x, pareto_y = [], []
    for bin_number in range(n_bins):
        mask = bin_idx == bin_number
        if mask.any():
            pareto_x.append(np.median(x_s[mask]))
            pareto_y.append(best_y(y_s[mask]))

    pareto_x_arr = np.array(pareto_x)
    pareto_y_arr = np.array(pareto_y)

    accumulator = np.maximum.accumulate if maximize_y else np.minimum.accumulate
    if maximize_x:
        running_best = accumulator(pareto_y_arr[::-1])[::-1]
    else:
        running_best = accumulator(pareto_y_arr)

    direction_label = "max" if maximize_y else "min"
    ax.plot(
        pareto_x_arr,
        running_best,
        color=_ACCENT,
        linewidth=2,
        label=f"Pareto frontier ({direction_label})",
        zorder=5,
    )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{y_metric} vs {x_metric}  (n={len(x):,})")
    ax.legend()
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI)
        log.info("Saved Pareto plot to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def _find_interesting_region(
        rate_arrays: list[NDArray[np.float64]],
        window: int,
) -> tuple[int, int]:
    """Return the most interesting window based on summed polling-rate variance."""
    if not rate_arrays:
        return 0, 0

    n = len(rate_arrays[0])
    if window >= n:
        return 0, n - 1

    scores = np.zeros(n - window + 1, dtype=np.float64)
    for rates in rate_arrays:
        csum = np.concatenate(([0.0], np.cumsum(rates)))
        csum_sq = np.concatenate(([0.0], np.cumsum(rates ** 2)))
        sums = csum[window:] - csum[:-window]
        sums_sq = csum_sq[window:] - csum_sq[:-window]
        means = sums / window
        variances = np.maximum(0.0, sums_sq / window - means ** 2)
        scores += variances

    start = int(np.argmax(scores))
    return start, start + window - 1


def plot_importance(
        columns: dict[str, NDArray],
        sweep_params: dict[str, list],
        metric: str = "rmse_mean",
        figsize: tuple[int, int] = (10, 5),
        save_path: str | None = None,
) -> None:
    """Bar chart of parameter importance for *metric*.

    Importance is proportional to eta-squared: the fraction of total metric variance
    explained by grouping on each parameter independently.  Higher bars
    mean the parameter has a larger marginal effect.
    """
    params = sorted(p for p in sweep_params if p in columns)
    metric_col = columns[metric]
    total_var = np.var(metric_col)

    if total_var == 0:
        log.warning("Metric '%s' has zero variance - nothing to plot.", metric)
        return

    grand_mean = metric_col.mean()
    n = len(metric_col)
    importances: dict[str, float] = {}

    for pname in params:
        pcol = columns[pname]
        uniques = np.unique(pcol)
        if len(uniques) < 2:
            continue
        group_means = np.array([metric_col[pcol == v].mean() for v in uniques])
        group_sizes = np.array([np.sum(pcol == v) for v in uniques])
        between_var = np.sum(
            group_sizes * (group_means - grand_mean) ** 2
        ) / n
        importances[pname] = between_var / total_var

    sorted_params = sorted(importances, key=lambda p: importances[p], reverse=True)
    vals = [importances[p] for p in sorted_params]
    labels = list(sorted_params)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(vals)), vals, color=_PRIMARY,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"eta-squared (fraction of '{metric}' variance explained)")
    ax.set_title(
        f"Parameter importance for '{metric}'  (n={len(metric_col):,})"
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=_GRID_ALPHA)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI)
        log.info("Saved importance plot to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_urgency_profiles(
        meta: dict[str, Any],
        figsize: tuple[int, int] = (42, 20),
        combined_save_path: str | None = None,
        zoom_save_path: str | None = None,
) -> None:
    """Plot one shared urgency trace plus stacked polling intervals, full and zoomed."""
    shared_urgency = meta.get("shared_urgency_profile")
    profiles = meta.get("config_profiles")
    if not shared_urgency or not profiles:
        log.warning("No config_profiles found in sweep metadata - skipping urgency profile plot.")
        return

    n_profiles = len(profiles)
    urgency_t = np.asarray(shared_urgency["time_index"], dtype=np.int64)
    urgency_values = np.asarray(shared_urgency["urgency"], dtype=np.float64)

    interval_arrays = [np.asarray(profile["interval"], dtype=np.int64) for profile in profiles]
    rate_arrays = [1.0 / interval.astype(np.float64) for interval in interval_arrays]

    window = min(1000, len(urgency_t))
    interesting_start, interesting_end = _find_interesting_region(rate_arrays, window)

    def _make_combined_figure(
            xlim: tuple[int, int] | None,
            title: str,
            highlight_region: tuple[int, int] | None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(
            n_profiles + 1,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1.4] + [1.0] * n_profiles},
        )

        urgency_ax = axes[0]
        urgency_ax.plot(urgency_t, urgency_values, color="crimson", linewidth=1.0)
        urgency_ax.fill_between(urgency_t, 0.0, urgency_values, color="crimson", alpha=0.18)
        urgency_ax.set_ylabel("Urgency")
        urgency_ax.set_ylim(-0.02, 1.02)
        urgency_ax.grid(True, alpha=_GRID_ALPHA)

        for ax, profile, interval in zip(axes[1:], profiles, interval_arrays):
            t = np.asarray(profile["time_index"], dtype=np.int64)
            ax.step(t, interval, where="post", color=_PRIMARY, linewidth=1.0)
            ax.set_ylabel(f"[{profile['config_id']}]\nint")
            ax.set_title(profile["config_label"], fontsize=9, loc="left")
            ax.grid(True, alpha=_GRID_ALPHA)

        if highlight_region is not None:
            lo, hi = highlight_region
            for ax in axes:
                ax.axvspan(lo, hi, color="gold", alpha=0.14)

        if xlim is not None:
            for ax in axes:
                ax.set_xlim(*xlim)

        axes[-1].set_xlabel("Time (sample index)")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        return fig

    full_fig = _make_combined_figure(
        xlim=None,
        title="Shared urgency and interval remappings",
        highlight_region=(interesting_start, interesting_end),
    )
    if combined_save_path:
        full_fig.savefig(combined_save_path, dpi=_DPI)
        log.info("Saved combined urgency/interval plot to %s", combined_save_path)
    else:
        plt.show()
    plt.close(full_fig)

    zoom_fig = _make_combined_figure(
        xlim=(interesting_start, interesting_end),
        title=(
            "Shared urgency and interval remappings "
            f"(zoomed interesting region [{interesting_start}, {interesting_end}])"
        ),
        highlight_region=None,
    )
    if zoom_save_path:
        zoom_fig.savefig(zoom_save_path, dpi=_DPI)
        log.info("Saved zoomed urgency/interval plot to %s", zoom_save_path)
    else:
        plt.show()
    plt.close(zoom_fig)


def plot_heatmap(
        columns: dict[str, NDArray],
        param_x: str,
        param_y: str,
        metric: str = "rmse_mean",
        figsize: tuple[int, int] = (9, 7),
        save_path: str | None = None,
) -> None:
    """2-D heatmap: median *metric* for each *(param_x, param_y)* cell.

    Aggregates over all other parameters.  Useful for spotting
    interactions between two parameters.
    """
    px, py, m = columns[param_x], columns[param_y], columns[metric]
    ux, uy = np.unique(px), np.unique(py)

    grid = np.full((len(uy), len(ux)), np.nan)
    for i, vy in enumerate(uy):
        for j, vx in enumerate(ux):
            mask = (px == vx) & (py == vy)
            if mask.any():
                grid[i, j] = np.median(m[mask])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower")

    xlabels = [f"{v:g}" if isinstance(v, float) else str(v) for v in ux]
    ylabels = [f"{v:g}" if isinstance(v, float) else str(v) for v in uy]
    ax.set_xticks(range(len(ux)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(range(len(uy)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"median({metric})  by  ({param_x}, {param_y})")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=_DPI)
        log.info("Saved heatmap to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
