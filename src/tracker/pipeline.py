"""
One-shot analysis pipeline.

Wraps dense reference tracking, adaptive replay, information-loss
evaluation, and plotting behind a single ``Pipeline`` object so callers
do not need to manually instantiate duplicated trackers or pollers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

import config
from tracker.windowed import (
    WindowedTracker, AdaptivePoller, TickResult, PollResult,
    InfoLossReport, evaluate_info_loss,
)


# ------------------------------
# Config dataclass - single place for every parameter
# ------------------------------

@dataclass
class PipelineConfig:
    """All tunables for the tracker + adaptive poller.

    Grouping them here avoids repeating the same kwargs in multiple places.
    Defaults are loaded from config.py and can be overridden via environment
    variables or at instantiation time.
    """
    # --- Tracker ---
    alpha_base: float = config.ALPHA_BASE
    beta: float = config.BETA
    outlier_threshold: float = config.OUTLIER_THRESHOLD
    drift_tolerance: int = config.DRIFT_TOLERANCE
    initial_variance_floor: float = config.TRACKER_VARIANCE_FLOOR

    # --- Adaptive poller ---
    min_interval: int = config.MIN_INTERVAL
    max_interval: int = config.MAX_INTERVAL
    variance_sensitivity: float = config.VARIANCE_SENSITIVITY
    interval_mapping: str = config.INTERVAL_MAPPING
    interval_power: float = config.INTERVAL_POWER
    interval_logistic_midpoint: float = config.INTERVAL_LOGISTIC_MIDPOINT
    interval_logistic_steepness: float = config.INTERVAL_LOGISTIC_STEEPNESS
    interval_exponential_steepness: float = config.INTERVAL_EXPONENTIAL_STEEPNESS
    sigma_ref: float = config.SIGMA_REF
    warmup: int = config.WARMUP
    warmup_positive_std_only: bool = config.WARMUP_POSITIVE_STD_ONLY
    sigma_ref_adapt: float = config.SIGMA_REF_ADAPT
    instability_window: int = config.INSTABILITY_WINDOW
    instability_weight: float = config.INSTABILITY_WEIGHT
    drift_boost: float = config.DRIFT_BOOST
    drift_decay: float = config.DRIFT_DECAY
    cooldown_rate: float = config.COOLDOWN_RATE
    urgency_smoothing: float = config.URGENCY_SMOOTHING
    cooldown_threshold: float = config.COOLDOWN_THRESHOLD
    outlier_decay: float = config.OUTLIER_DECAY

    # --- Plot defaults ---
    sigma_band: float = config.SIGMA_BAND

    def to_dict(self) -> dict[str, Any]:
        """Return all tunables as a plain dict (JSON-safe)."""
        return asdict(self)

    def _make_tracker(self) -> WindowedTracker:
        return WindowedTracker(
            alpha_base=self.alpha_base,
            beta=self.beta,
            outlier_threshold=self.outlier_threshold,
            drift_tolerance=self.drift_tolerance,
            initial_variance_floor=self.initial_variance_floor,
        )

    def _make_poller(self) -> AdaptivePoller:
        return AdaptivePoller(
            self._make_tracker(),
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            variance_sensitivity=self.variance_sensitivity,
            interval_mapping=self.interval_mapping,
            interval_power=self.interval_power,
            interval_logistic_midpoint=self.interval_logistic_midpoint,
            interval_logistic_steepness=self.interval_logistic_steepness,
            interval_exponential_steepness=self.interval_exponential_steepness,
            sigma_ref=self.sigma_ref,
            warmup=self.warmup,
            warmup_positive_std_only=self.warmup_positive_std_only,
            sigma_ref_adapt=self.sigma_ref_adapt,
            instability_window=self.instability_window,
            instability_weight=self.instability_weight,
            drift_boost=self.drift_boost,
            drift_decay=self.drift_decay,
            cooldown_rate=self.cooldown_rate,
            urgency_smoothing=self.urgency_smoothing,
            cooldown_threshold=self.cooldown_threshold,
            outlier_decay=self.outlier_decay,
        )


# ------------------------------
# Result bundle
# ------------------------------

@dataclass
class PipelineResult:
    """Everything produced by a single :meth:`Pipeline.run` call."""
    trace: NDArray[np.float64]
    config: PipelineConfig

    full_results: list[TickResult]
    poll_results: list[PollResult]
    info_loss: InfoLossReport

    # -- convenience accessors --

    @property
    def n_total(self) -> int:
        return len(self.trace)

    @property
    def n_sampled(self) -> int:
        return self.info_loss.n_sampled

    @property
    def sample_ratio(self) -> float:
        return self.info_loss.sample_ratio

    def status_counts(self, adaptive: bool = False) -> dict[str, int]:
        """Count occurrences of each status.

        Parameters
        ----------
        adaptive : bool
            If *True*, count statuses from the adaptive poller (sampled
            points only). Otherwise count from the dense reference run.
        """
        if adaptive:
            statuses = [pr.tick.status for pr in self.poll_results
                        if pr.sampled and pr.tick is not None]
        else:
            statuses = [r.status for r in self.full_results]
        counts: dict[str, int] = {}
        for s in statuses:
            counts[s.name] = counts.get(s.name, 0) + 1
        return counts

    # -- one-liner plotting --

    def plot_full(self, title: str = "Dense Reference Windowed Tracker",
                  save_path: str | None = None, **kwargs: Any) -> None:
        """Plot the dense reference tracking result."""
        from plot.plot import plot_tracking  # noqa: PLC0415
        plot_tracking(self.full_results, title=title,
                      sigma_band=self.config.sigma_band,
                      save_path=save_path, **kwargs)

    def plot_adaptive(self, title: str = "Adaptive Replay",
                      save_path: str | None = None, **kwargs: Any) -> None:
        """Plot the adaptive replay result (4-panel)."""
        from plot.plot import plot_adaptive_polling  # noqa: PLC0415
        plot_adaptive_polling(
            self.trace, self.poll_results,
            full_results=self.full_results,
            title=title,
            sigma_band=self.config.sigma_band,
            save_path=save_path, **kwargs,
        )

    def plot(self, save_dir: str | None = None, **kwargs: Any) -> None:
        """Plot both dense reference and adaptive replay views.

        If *save_dir* is given, figures are saved there instead of
        shown interactively.
        """
        full_path = f"{save_dir}/dense_reference_tracking.png" if save_dir else None
        adaptive_path = f"{save_dir}/adaptive_replay.png" if save_dir else None
        self.plot_full(save_path=full_path, **kwargs)
        self.plot_adaptive(save_path=adaptive_path, **kwargs)

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        il = self.info_loss
        scnad = [f"{str(s)}: {c}" for s, c in self.status_counts(adaptive=True).items()]
        scad = [f"{s}: {c}" for s, c in self.status_counts(adaptive=False).items()]
        lines = [
            "Pipeline Summary",
            str(il),
            "   ==> Point breakdown:",
            f"  Full statuses:      {scnad}",
            f"  Adaptive statuses:  {scad}",
        ]
        return "\n".join(lines)

    def summary_json(self) -> dict[str, Any]:
        """Return a compact machine-readable summary."""
        il: InfoLossReport = self.info_loss
        return {
            "info_loss": {
                "n_total": il.n_total,
                "n_sampled": il.n_sampled,
                "sample_ratio": il.sample_ratio,
                "max_gap": il.max_gap,
                "mean_gap": il.mean_gap,
                "p95_gap": il.p95_gap,
                "mae_mean": il.mae_mean,
                "rmse_mean": il.rmse_mean,
                "nrmse_mean": il.nrmse_mean,
                "max_mean_error": il.max_mean_error,
                "p50_error": il.p50_error,
                "p90_error": il.p90_error,
                "p95_error": il.p95_error,
                "p99_error": il.p99_error,
                "correlation": il.correlation,
                # "lag_samples": il.lag_samples,
                "raw_recon_rmse": il.raw_recon_rmse,
                "raw_recon_nrmse": il.raw_recon_nrmse,
                "peak_recall_top5": il.peak_recall_top5,
                "mae_std": il.mae_std,
                "rmse_std": il.rmse_std,
                "coverage_3sigma": il.coverage_3sigma,
                # "missed_outliers": il.missed_outliers,
                # "extra_outliers": il.extra_outliers,
                # "missed_drifts": il.missed_drifts,
                # "jaccard_outliers": il.jaccard_outliers,
            },
            "status_counts": {
                "full": self.status_counts(adaptive=False),
                "adaptive": self.status_counts(adaptive=True),
            },
        }

    def save_json(
            self,
            save_dir: str,
            generator: dict[str, Any] | None = None,
            filename: str = "experiment.json",
    ) -> str:
        """Persist a full experiment manifest as JSON.

        The file contains everything needed to understand and reproduce
        the experiment:

        * **timestamp** - ISO-8601 UTC datetime.
        * **pipeline_config** - all tracker / poller tunables.
        * **generator** - trace-builder description (segments, bridges, ...).
        * **trace_stats** - basic statistics of the input trace.
        * **result_summary** - info-loss metrics and status counts.

        Parameters
        ----------
        save_dir : str
            Directory where the JSON file is written.
        generator : dict, optional
            Serialised description of the :class:`TraceComposer` /
            :class:`Segment` list used to produce the trace.
        filename : str
            Name of the output file (default ``"experiment.json"``).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        manifest: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": self.config.to_dict(),
            "generator": generator,
            "trace_stats": {
                "n_points": int(len(self.trace)),
                "mean": float(np.mean(self.trace)),
                "std": float(np.std(self.trace)),
                "min": float(np.min(self.trace)),
                "max": float(np.max(self.trace)),
            },
            "result_summary": self.summary_json(),
        }
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # Also save the raw trace as a standalone JSON array.
        trace_path = os.path.join(save_dir, "trace.json")
        with open(trace_path, "w") as f:
            json.dump(
                [{"t": i, "value": float(v)} for i, v in enumerate(self.trace)],
                f,
            )

        return path


# ------------------------------
# Pipeline
# ------------------------------

class Pipeline:
    """
    Dense reference plus adaptive replay analysis in one call.

    Parameters are accepted either as a :class:`PipelineConfig` object
    or as keyword arguments (which build one for you).
    """

    def __init__(self, __config: PipelineConfig | None = None, **kwargs: Any):
        if __config is not None:
            self.config = __config
        else:
            self.config = PipelineConfig(**kwargs)

    def run(self, trace: NDArray[np.float64]) -> PipelineResult:
        """Run dense reference tracking, adaptive replay, and info-loss
        evaluation on *trace*.

        Returns a :class:`PipelineResult` bundling all outputs.
        """
        trace = np.asarray(trace, dtype=np.float64)

        # Dense reference path
        full_tracker = self.config._make_tracker()
        full_results = full_tracker.track(trace)

        # Adaptive replay path
        poller = self.config._make_poller()
        poll_results = poller.track(trace)

        # Info loss (needs fresh instances internally)
        info_loss = evaluate_info_loss(
            trace,
            full_tracker=self.config._make_tracker(),
            poller=self.config._make_poller(),
        )

        return PipelineResult(
            trace=trace,
            config=self.config,
            full_results=full_results,
            poll_results=poll_results,
            info_loss=info_loss,
        )
