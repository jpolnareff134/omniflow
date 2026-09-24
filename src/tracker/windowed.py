import logging as log
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ------------------------------
# Status enum returned for every observation
# ------------------------------

class Status(Enum):
    INIT = auto()  # First point - baseline initialised
    NORMAL = auto()  # Within expected range
    OUTLIER = auto()  # Outlier rejected (not incorporated)
    DRIFT_RESET = auto()  # Consecutive outliers -> forced baseline reset

    def __str__(self) -> str:
        if self == Status.INIT:
            return "Initial point"
        elif self == Status.NORMAL:
            return "Normal"
        elif self == Status.OUTLIER:
            return "Outlier"
        elif self == Status.DRIFT_RESET:
            return "Drift reset"
        else:
            return "Unknown status"


class IntervalMapping(str, Enum):
    ORIGINAL_INTERVAL = "original_interval"
    LINEAR_RATE = "linear_rate"
    SMOOTHSTEP_INTERVAL = "smoothstep_interval"
    POWER_RATE = "power_rate"
    GEOMETRIC_RATE = "geometric_rate"
    LOGISTIC_RATE = "logistic_rate"
    EXPONENTIAL_EASE_OUT = "exponential_ease_out"

    @classmethod
    def from_value(cls, value: str) -> "IntervalMapping":
        if value == "linear_interval":
            return cls.ORIGINAL_INTERVAL
        try:
            return cls(value)
        except ValueError as exc:
            supported = ", ".join(mapping.value for mapping in cls)
            raise ValueError(
                f"Unknown interval mapping '{value}'. Supported mappings: {supported}"
            ) from exc


# ------------------------------
# Result produced for each time step
# ------------------------------

@dataclass
class TickResult:
    """Diagnostic snapshot after processing one observation."""
    value: float
    status: Status
    mean: float
    std: float
    z_score: float
    alpha: float  # effective learning rate used


@dataclass
class PollResult:
    """One step of the adaptive poller."""
    time_index: int  # position in the original trace
    sampled: bool  # True -> point was consumed by the tracker
    tick: TickResult | None  # tracker output (None when skipped)
    interval: int  # polling interval that was in effect
    urgency: float  # urgency score that determined the interval


# ------------------------------
# Core tracker
# ------------------------------

class WindowedTracker:
    """Adaptive moving average tracker with drift detection.

    Parameters
    ----------
    alpha_base : float
        Base rate for the exponential moving average (0 < alpha < 1).
        Smaller -> slower adaptation, more inertia.
    beta : float
        Dampening factor controlling how much the probability *p(z)* reduces
        the effective rate. 
        ``alpha_eff = alpha_base * (1 - beta * p_t)``.
    outlier_threshold : float
        Number of standard deviations beyond which a point is flagged as an
        outlier.
    drift_tolerance : int
        Number of consecutive outliers before the tracker declares concept
        drift and hard-resets the baseline.
    """

    def __init__(
            self,
            alpha_base: float = 0.1,
            beta: float = 0.5,
            outlier_threshold: float = 3.0,
            drift_tolerance: int = 5,
            initial_variance_floor: float = 0.0,
    ) -> None:
        self.alpha_base = alpha_base
        self.beta = beta
        self.outlier_threshold = outlier_threshold
        self.drift_tolerance = drift_tolerance
        if not np.isfinite(initial_variance_floor) or initial_variance_floor < 0.0:
            raise ValueError("initial_variance_floor must be finite and non-negative")
        self.initial_variance_floor = float(initial_variance_floor)

        # Running statistics
        self.mean: float = 0.0
        self.var: float = 0.0
        self.std: float = 0.0
        self._initialised: bool = False

        # Drift bookkeeping
        self._consecutive_outliers: int = 0

    # ----- public API -----

    def update(self, x: float) -> TickResult:
        """Ingest a single observation and return diagnostics."""

        if not self._initialised:
            self.mean = x
            # Seed variance proportional to signal magnitude so the
            # z-threshold is meaningful from the very next point.
            self.var = max(self.initial_variance_floor, (abs(x) * 0.1) ** 2)
            self.std = np.sqrt(self.var)
            self._initialised = True
            return TickResult(x, Status.INIT, self.mean, self.std,
                              z_score=0.0, alpha=1.0)

        # Z-score
        z = (x - self.mean) / (self.std + 1e-8)
        is_outlier = abs(z) > self.outlier_threshold

        if is_outlier:
            self._consecutive_outliers += 1

            # Concept drift: too many consecutive rejections -> hard reset
            if self._consecutive_outliers >= self.drift_tolerance:
                self.mean = x
                # Also reset variance so the tracker can re-learn from
                # the new baseline instead of staying stuck.
                self.var = max(self.initial_variance_floor, (abs(x) * 0.1) ** 2)
                self.std = np.sqrt(self.var)
                self._consecutive_outliers = 0
                return TickResult(x, Status.DRIFT_RESET, self.mean,
                                  self.std, z_score=z, alpha=1.0)

            return TickResult(x, Status.OUTLIER, self.mean, self.std,
                              z_score=z, alpha=0.0)

        # Normal point -> tracker update
        self._consecutive_outliers = 0

        p_t = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
        alpha_t = self.alpha_base * (1.0 - self.beta * p_t)

        old_mean = self.mean
        self.mean = (1.0 - alpha_t) * self.mean + alpha_t * x
        self.var = (1.0 - alpha_t) * self.var + alpha_t * (x - old_mean) ** 2
        self.std = np.sqrt(self.var)

        return TickResult(x, Status.NORMAL, self.mean, self.std,
                          z_score=z, alpha=alpha_t)

    def track(self, data: Sequence[float] | NDArray) -> list[TickResult]:
        """Process a full trace and return per-step diagnostics."""
        return [self.update(float(x)) for x in data]

    def reset(self) -> None:
        """Reset the tracker to its initial (un-initialised) state."""
        self.mean = 0.0
        self.var = 0.0
        self.std = 0.0
        self._initialised = False
        self._consecutive_outliers = 0


# ------------------------------
# Adaptive scheduling controller
# ------------------------------

class AdaptivePoller:
    """Wraps a :class:`WindowedTracker` tracker with an adaptive sampling strategy.

    The poller maintains a *polling interval* (number of time steps between
    consecutive samples). The interval shrinks when the signal is volatile
    and grows when it is calm, reducing collection cost during stable
    periods while reacting quickly to outliers or drift.

    The mapping from signal state to interval is governed by an **urgency
    score** taking values in [0, 1], where 1 = maximum urgency 
    (poll every point) and 0 = minimum urgency (poll every *max_interval*-th point).  

     Urgency itself is a weighted combination of four components:

     1. **Variance component** - how large the current std is relative to a
       reference value.
     2. **Instability component** - rolling fraction of recent points flagged
       as outliers.
     3. **Drift component** - a burst of urgency after every drift-reset event,
       decaying exponentially.
     4. **Outlier component** - an immediate, magnitude-proportional urgency
        impulse when an outlier is detected, decaying exponentially.
       The bigger the z-score, the bigger the kick.

     The urgency-to-interval mapping itself is configurable so alternative
     response curves can be compared in ablation studies. The supported
     mappings are:

     1. ``original_interval`` - interpolate directly in interval space.
         This is the legacy behaviour.
     2. ``linear_rate`` - interpolate in sampling-rate space, then invert
         back to an interval.
     3. ``geometric_rate`` - interpolate multiplicatively in sampling-rate
         space, which is useful when the interval span is large.
     4. ``power_rate`` - same as ``linear_rate`` but with
         ``urgency ** interval_power``.
     5. ``logistic_rate`` - apply a normalized logistic curve before the
         rate-space interpolation, giving an explicit urgency knee.
     6. ``exponential_ease_out`` - react strongly at low urgency and flatten
        toward the fastest interval as urgency approaches one.

    Parameters
    ----------
    tracker : WindowedTracker
        A (possibly fresh) WindowedTracker instance.
    min_interval : int
        Fastest polling rate (sample every *min_interval*-th point).
        ``1`` = no skipping.
    max_interval : int
        Slowest polling rate during calm periods.
        variance_sensitivity : float
        Gain applied to the variance component. Higher -> the poller
        reacts more aggressively to variance changes.
        sigma_ref : float
        Reference standard deviation used to normalise the variance component.
        Set to **0** (default) to auto-calibrate from the signal's own
        stable-period std after the first few normal points.
    instability_window : int
        Number of recent *sampled* points over which the instability rate is
        computed.
    instability_weight : float
        Weight of the instability component in the urgency score.
    drift_boost : float
        Instantaneous value injected into the drift component when a
        drift-reset occurs.
    drift_decay : float
        Exponential decay factor for the drift component (per sample step).
    cooldown_rate : float
        How quickly urgency decays toward 0 when all inputs are calm.
        Applied as ``urgency *= (1 - cooldown_rate)`` each step.
    urgency_smoothing : float
        Exponential smoothing factor used to blend the previous urgency
        with the current raw urgency.
    cooldown_threshold : float
        Threshold below which cooldown is applied to the smoothed
        urgency.
    outlier_decay : float
        Exponential decay for the outlier component. After an outlier
        injects ``min(1, (|z| - outlier_threshold) / outlier_threshold)``, the
        component decays by ``outlier_decay`` each sample step. Set to 0
        to disable.
    interval_mapping : str
        Strategy used to convert urgency in [0, 1] into a polling
        interval.  See the list above for the supported values.
    interval_power : float
        Shape parameter used by the ``power_*`` mappings. Values greater
        than 1 bias the poller toward slower reactions at low urgency and
        faster collapse toward ``min_interval`` at high urgency.
    interval_logistic_midpoint : float
        Midpoint of the normalized logistic curve used by
        ``logistic_rate``. Lower values move the knee earlier.
    interval_logistic_steepness : float
        Steepness of the normalized logistic curve used by
        ``logistic_rate``. Higher values make the response more
        threshold-like.
    """

    def __init__(
            self,
            tracker: WindowedTracker,
            *,
            min_interval: int,
            max_interval: int,
            variance_sensitivity: float,
            sigma_ref: float,
            warmup: int,
            sigma_ref_adapt: float = 0.01,
            instability_window: int,
            instability_weight: float,
            drift_boost: float,
            drift_decay: float,
            cooldown_rate: float,
            urgency_smoothing: float = 0.6,
            cooldown_threshold: float = 0.2,
            outlier_decay: float = 0.85,
            interval_mapping: str = IntervalMapping.ORIGINAL_INTERVAL.value,
            interval_power: float = 2.0,
             interval_logistic_midpoint: float = 0.5,
             interval_logistic_steepness: float = 8.0,
             interval_exponential_steepness: float = 4.0,
             warmup_positive_std_only: bool = False,
    ) -> None:
        self.tracker = tracker
        self.min_interval = max(1, min_interval)
        self.max_interval = max(self.min_interval, max_interval)
        self.variance_sensitivity = variance_sensitivity
        self.sigma_ref = sigma_ref
        self._initial_sigma_ref = sigma_ref
        self._sigma_ref_auto = (sigma_ref == 0.0)  # auto-calibrate mode
        self._warmup = max(1, warmup)
        self._sigma_ref_adapt = sigma_ref_adapt
        self._warmup_positive_std_only = warmup_positive_std_only
        self._warmup_stds: list[float] = []
        self.instability_window = instability_window
        self.instability_weight = instability_weight
        self.drift_boost = drift_boost
        self.drift_decay = drift_decay
        self.cooldown_rate = cooldown_rate
        if not 0.0 <= urgency_smoothing <= 1.0:
            raise ValueError(
                f"urgency_smoothing must be in [0, 1], got {urgency_smoothing}"
            )
        self.urgency_smoothing = urgency_smoothing
        if not 0.0 <= cooldown_threshold <= 1.0:
            raise ValueError(
                f"cooldown_threshold must be in [0, 1], got {cooldown_threshold}"
            )
        self.cooldown_threshold = cooldown_threshold
        self.outlier_decay = outlier_decay
        self.interval_mapping = IntervalMapping.from_value(interval_mapping)
        if interval_power <= 0.0:
            raise ValueError(f"interval_power must be > 0, got {interval_power}")
        self.interval_power = interval_power
        if not 0.0 < interval_logistic_midpoint < 1.0:
            raise ValueError(
                "interval_logistic_midpoint must be in (0, 1), got "
                f"{interval_logistic_midpoint}"
            )
        if interval_logistic_steepness <= 0.0:
            raise ValueError(
                "interval_logistic_steepness must be > 0, got "
                f"{interval_logistic_steepness}"
            )
        self.interval_logistic_midpoint = interval_logistic_midpoint
        self.interval_logistic_steepness = interval_logistic_steepness
        if interval_exponential_steepness <= 0.0:
            raise ValueError(
                "interval_exponential_steepness must be > 0, got "
                f"{interval_exponential_steepness}"
            )
        self.interval_exponential_steepness = interval_exponential_steepness

        # Internal state
        self._urgency: float = 1.0  # start fast to learn baseline
        self._drift_component: float = 0.0
        self._outlier_component: float = 0.0
        self._recent_statuses: deque[Status] = deque(maxlen=instability_window)
        self._steps_since_sample: int = 0

    # -- properties --

    @property
    def interval(self) -> int:
        """Current polling interval (derived from urgency)."""
        return self._urgency_to_interval(self._urgency)

    def _urgency_to_interval(self, urgency: float) -> int:
        urgency = float(np.clip(urgency, 0.0, 1.0))
        span = self.max_interval - self.min_interval

        if span == 0:
            return self.min_interval

        if self.interval_mapping in {
            IntervalMapping.ORIGINAL_INTERVAL,
            IntervalMapping.LINEAR_RATE,
        }:
            shaped = urgency
            interval = self.max_interval - span * shaped
        elif self.interval_mapping == IntervalMapping.SMOOTHSTEP_INTERVAL:
            shaped = urgency * urgency * (3.0 - 2.0 * urgency)
            interval = self.max_interval - span * shaped
        elif self.interval_mapping == IntervalMapping.GEOMETRIC_RATE:
            interval = self.min_interval * ((self.max_interval / self.min_interval) ** (1.0 - urgency))
        elif self.interval_mapping == IntervalMapping.POWER_RATE:
            shaped = urgency ** self.interval_power
            interval = self.max_interval - span * shaped
        elif self.interval_mapping == IntervalMapping.LOGISTIC_RATE:
            shaped = self._normalize_logistic(urgency)
            interval = self.max_interval - span * shaped
        elif self.interval_mapping == IntervalMapping.EXPONENTIAL_EASE_OUT:
            steepness = self.interval_exponential_steepness
            shaped = -np.expm1(-steepness * urgency) / -np.expm1(-steepness)
            interval = self.max_interval - span * shaped
        else:
            raise AssertionError(
                f"Unhandled interval mapping: {self.interval_mapping.value}"
            )

        return int(np.clip(round(interval), self.min_interval, self.max_interval))

    def _normalize_logistic(self, urgency: float) -> float:
        midpoint = self.interval_logistic_midpoint
        steepness = self.interval_logistic_steepness

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        lower = _sigmoid(-steepness * midpoint)
        upper = _sigmoid(steepness * (1.0 - midpoint))
        denom = upper - lower
        if denom <= 1e-12:
            return urgency

        shaped = (_sigmoid(steepness * (urgency - midpoint)) - lower) / denom
        return float(np.clip(shaped, 0.0, 1.0))

    # -- core loop --

    def step(self, x: float, time_index: int = 0) -> PollResult:
        """Decide whether to sample *x* and, if so, feed it to the tracker.

        Returns a :class:`PollResult` regardless of whether the point was
        consumed or skipped.
        """
        current_interval = self.interval
        self._steps_since_sample += 1

        # Forces sampling of first point to avoid spikes in error metrics
        is_first_point = not self.tracker._initialised

        if self._steps_since_sample >= current_interval or is_first_point:
            # --- SAMPLE this point ---
            tick = self.tracker.update(x)
            self._steps_since_sample = 0

            # Update recent-status window
            self._recent_statuses.append(tick.status)

            # Recompute urgency
            self._update_urgency(tick)

            return PollResult(time_index=time_index, sampled=True,
                              tick=tick, interval=current_interval,
                              urgency=self._urgency)
        else:
            # --- SKIP ---
            return PollResult(time_index=time_index, sampled=False,
                              tick=None, interval=current_interval,
                              urgency=self._urgency)

    def track(self, data: Sequence[float] | NDArray) -> list[PollResult]:
        """Run the adaptive poller over a full trace."""
        return [self.step(float(x), i) for i, x in enumerate(data)]

    def feed(self, x: float, time_index: int = 0) -> PollResult:
        """Feed a live observation - always sampled (no skipping).

        Use this in **live mode** where every probe read should be
        processed.  The urgency (and thus :attr:`interval`) is updated
        based on the observation.

        Returns a :class:`PollResult` with ``sampled=True``.
        """
        tick = self.tracker.update(x)
        self._steps_since_sample = 0

        # Update recent-status window
        self._recent_statuses.append(tick.status)

        self._update_urgency(tick)

        return PollResult(
            time_index=time_index,
            sampled=True,
            tick=tick,
            interval=self.interval,
            urgency=self._urgency,
        )

    # -- urgency computation --

    def _update_urgency(self, tick: TickResult) -> None:
        # --- Auto-calibration / warmup phase ---
        # Collect tracker stds over _warmup NORMAL points, then
        # lock in the converged value.  During warmup, sample every point.
        if self._sigma_ref_auto:
            if tick.status == Status.NORMAL and (
                    not self._warmup_positive_std_only or self.tracker.std > 0.0):
                self._warmup_stds.append(self.tracker.std)
                if len(self._warmup_stds) >= self._warmup:
                    self.sigma_ref = self._warmup_stds[-1]
                    self._sigma_ref_auto = False
            # During warmup keep urgency pinned to 1 (sample every point).
            self._urgency = 1.0
            return

        # --- Slowly adapt sigma_ref toward current variability ---
        if self._sigma_ref_adapt > 0 and self.tracker.std > 0:
            self.sigma_ref = ((1.0 - self._sigma_ref_adapt) * self.sigma_ref
                              + self._sigma_ref_adapt * self.tracker.std)

        # 1. Variance component: how much current std exceeds the baseline.
        variance_component = float(np.clip(
            (self.tracker.std / (self.sigma_ref + 1e-8) - 1.0)
            * self.variance_sensitivity,
            0.0,
            1.0,
        ))

        # 2. Instability component: fraction of recent samples flagged.
        if self._recent_statuses:
            n_unstable = sum(1 for s in self._recent_statuses
                             if s in (Status.OUTLIER, Status.DRIFT_RESET))
            instability_component = float(np.clip(
                (n_unstable / len(self._recent_statuses)) * self.instability_weight,
                0.0,
                1.0,
            ))
        else:
            instability_component = 0.0

        # 3. Drift component: impulse on drift, then exponential decay.
        if tick.status == Status.DRIFT_RESET:
            self._drift_component = float(np.clip(self.drift_boost, 0.0, 1.0))
        else:
            self._drift_component = float(np.clip(
                self._drift_component * self.drift_decay,
                0.0,
                1.0,
            ))

        # 4. Outlier component: immediate, magnitude-proportional impulse
        #    on any outlier. Strength is how far z exceeds threshold,
        #    normalised so that 2*threshold -> component = 1.0.
        if tick.status in (Status.OUTLIER, Status.DRIFT_RESET):
            kick = min(1.0, (abs(tick.z_score) - self.tracker.outlier_threshold)
                       / self.tracker.outlier_threshold)
            self._outlier_component = max(self._outlier_component, kick)
        else:
            self._outlier_component = float(np.clip(
                self._outlier_component * self.outlier_decay,
                0.0,
                1.0,
            ))

        # Combine components.
        raw = float(np.clip(max(
            variance_component,
            instability_component,
            self._drift_component,
            self._outlier_component,
        ), 0.0, 1.0))

        # Smooth transition: blend toward new urgency, apply cooldown
        self._urgency = (
            self.urgency_smoothing * self._urgency
            + (1.0 - self.urgency_smoothing) * raw
        )
        # Only apply cooldown when raw signal is actually low
        if raw < self.cooldown_threshold:
            self._urgency *= (1.0 - self.cooldown_rate)

        self._urgency = float(np.clip(self._urgency, 0.0, 1.0))

    def reset(self) -> None:
        """Reset poller and underlying tracker."""
        self.tracker.reset()
        self._urgency = 1.0
        self._drift_component = 0.0
        self._outlier_component = 0.0
        self._recent_statuses.clear()
        self._steps_since_sample = 0
        self._warmup_stds.clear()
        self.sigma_ref = self._initial_sigma_ref
        self._sigma_ref_auto = (self._initial_sigma_ref == 0.0)
        # self._samples_since_calibration = 0  # unused, kept for reference


# ------------------------------
# Information-loss evaluation
# ------------------------------

@dataclass
class InfoLossReport:
    """Quantifies the information lost by adaptive (sub-)sampling.

    Error metrics compare the adaptive tracker's state, linearly
    interpolated back to dense resolution, against a dense reference
    tracker run on every point.
    """
    # --- Sampling efficiency ---
    n_total: int  # total points in the trace
    n_sampled: int  # points actually consumed
    sample_ratio: float  # n_sampled / n_total
    max_gap: int  # longest interval between consecutive samples
    mean_gap: float  # mean interval between consecutive samples
    p95_gap: float  # 95th percentile gap between consecutive samples

    # --- Mean tracking quality ---
    mae_mean: float  # mean-absolute-error of the running mean
    rmse_mean: float  # root-mean-square-error of the running mean
    nrmse_mean: float  # normalised RMSE (RMSE / signal range)
    max_mean_error: float  # worst-case instantaneous mean deviation
    p50_error: float  # 50th percentile (median) error
    p90_error: float  # 90th percentile error
    p95_error: float  # 95th percentile error
    p99_error: float  # 99th percentile error
    correlation: float  # Pearson correlation (full vs adaptive means)
    # lag_samples: int  # temporal lag via cross-correlation

    # --- Raw signal reconstruction quality ---
    raw_recon_rmse: float  # RMSE of interpolated-subsample vs true signal
    raw_recon_nrmse: float  # normalised RMSE of reconstruction
    peak_recall_top5: float  # fraction of top-5% signal peaks that were sampled

    # --- Std/variance tracking quality ---
    mae_std: float  # mean-absolute-error of the running std
    rmse_std: float  # root-mean-square-error of the running std

    # --- Confidence band quality ---
    coverage_3sigma: float  # % of data within adaptive +-3sigma band

    # --- Outlier/drift detection (optional) ---
    # missed_outliers: int  # outliers detected by full but not adaptive
    # extra_outliers: int  # outliers detected by adaptive but not full
    # missed_drifts: int  # drift resets detected by full but not adaptive
    # jaccard_outliers: float  # Jaccard similarity of outlier index sets

    def __str__(self) -> str:
        return "Information Loss Report:" + \
            f"\n  ==> Sampling Efficiency" + \
            f"\n  Total points:       {self.n_total}" + \
            f"\n  Sampled:            {self.n_sampled} ({self.sample_ratio:.1%})" + \
            f"\n  Max gap:            {self.max_gap} samples" + \
            f"\n  Mean gap:           {self.mean_gap:.1f} samples" + \
            f"\n  P95 gap:            {self.p95_gap:.0f} samples" + \
            f"\n" + \
            f"\n  ==> Mean Tracking Quality" + \
            f"\n  MAE:                {self.mae_mean:.4f}" + \
            f"\n  RMSE:               {self.rmse_mean:.4f}" + \
            f"\n  NRMSE:              {self.nrmse_mean:.4f}" + \
            f"\n  Max error:          {self.max_mean_error:.4f}" + \
            f"\n  P50/90/95/99 error:     {self.p50_error:.4f} / {self.p90_error:.4f} / " + \
            f"{self.p95_error:.4f} / {self.p99_error:.4f}" + \
            f"\n  Correlation:        {self.correlation:.4f}" + \
            f"\n" + \
            f"\n  ==> Raw Signal Reconstruction" + \
            f"\n  Recon RMSE:         {self.raw_recon_rmse:.4f}" + \
            f"\n  Recon NRMSE:        {self.raw_recon_nrmse:.4f}" + \
            f"\n  Peak recall (top5%): {self.peak_recall_top5:.1%}" + \
            f"\n" + \
            f"\n  ==> Std Tracking Quality" + \
            f"\n  MAE(std):           {self.mae_std:.4f}" + \
            f"\n  RMSE(std):          {self.rmse_std:.4f}" + \
            f"\n" + \
            f"\n  ==> Confidence Band" + \
            f"\n  Coverage (+-3sigma):     {self.coverage_3sigma:.1%}"
        # f"\n  Lag:                {self.lag_samples} samples" + \
        # f"\n" + \
        # f"\n  ==> Anomaly/Drift" + \
        # f"\n  Missed outliers:    {self.missed_outliers}" + \
        # f"\n  Extra outliers:     {self.extra_outliers}" + \
        # f"\n  Missed drifts:      {self.missed_drifts}" + \
        # f"\n  Jaccard:            {self.jaccard_outliers:.3f}"


def evaluate_info_loss(
        data: Sequence[float] | NDArray,
        full_tracker: WindowedTracker,
        poller: AdaptivePoller,
) -> InfoLossReport:
    """Run both a dense reference tracker and the adaptive poller on *data*,
    then compare their outputs.

    .. note:: Both ``full_tracker`` and ``poller`` (including its inner
       tracker) are **reset** before the evaluation so results are
       deterministic.

    Parameters
    ----------
    data : array-like
        The trace to evaluate on.
    full_tracker : WindowedTracker
        A WindowedTracker instance (will be reset) used as the dense reference path.
    poller : AdaptivePoller
        The adaptive poller (will be reset) to evaluate.
    """
    data = np.asarray(data, dtype=np.float64)

    # --- Dense reference run ---
    log.debug("Running dense reference tracker...")
    full_tracker.reset()
    full_results = full_tracker.track(data)
    log.debug("Running adaptive poller...")
    poller.reset()
    poll_results = poller.track(data)

    return evaluate_info_loss_from_runs(data, full_results, poll_results)


def compute_fixed_urgency_trace(
        data: Sequence[float] | NDArray,
        poller: AdaptivePoller,
) -> NDArray[np.float64]:
    """Run the urgency function on every point to obtain a deterministic urgency trace."""
    data = np.asarray(data, dtype=np.float64)
    poller.reset()
    urgencies = np.empty(len(data), dtype=np.float64)
    for i, x in enumerate(data):
        urgencies[i] = poller.feed(float(x), time_index=i).urgency
    return urgencies


def track_fixed_urgency(
        data: Sequence[float] | NDArray,
        poller: AdaptivePoller,
        urgency_trace: Sequence[float] | NDArray,
) -> list[PollResult]:
    """Replay a fixed urgency trace through a pure urgency-to-interval mapping."""
    data = np.asarray(data, dtype=np.float64)
    urgency_trace = np.asarray(urgency_trace, dtype=np.float64)
    if len(data) != len(urgency_trace):
        raise ValueError(
            "urgency_trace length must match data length, got "
            f"{len(urgency_trace)} vs {len(data)}"
        )

    poller.reset()
    poll_results: list[PollResult] = []
    steps_since_sample = 0

    for i, (x, urgency) in enumerate(zip(data, urgency_trace)):
        urgency = float(np.clip(urgency, 0.0, 1.0))
        current_interval = poller._urgency_to_interval(urgency)
        steps_since_sample += 1
        is_first_point = not poller.tracker._initialised

        if steps_since_sample >= current_interval or is_first_point:
            tick = poller.tracker.update(float(x))
            steps_since_sample = 0
            poll_results.append(PollResult(
                time_index=i,
                sampled=True,
                tick=tick,
                interval=current_interval,
                urgency=urgency,
            ))
        else:
            poll_results.append(PollResult(
                time_index=i,
                sampled=False,
                tick=None,
                interval=current_interval,
                urgency=urgency,
            ))

    return poll_results


def evaluate_info_loss_fixed_urgency(
        data: Sequence[float] | NDArray,
        full_tracker: WindowedTracker,
        poller: AdaptivePoller,
        urgency_trace: Sequence[float] | NDArray,
) -> InfoLossReport:
    """Evaluate a mapping when urgency is fixed and deterministic across configurations."""
    data = np.asarray(data, dtype=np.float64)

    log.debug("Running dense reference tracker...")
    full_tracker.reset()
    full_results = full_tracker.track(data)

    log.debug("Replaying fixed urgency trace through interval mapping...")
    poll_results = track_fixed_urgency(data, poller, urgency_trace)

    return evaluate_info_loss_from_runs(data, full_results, poll_results)


def evaluate_info_loss_from_runs(
        data: Sequence[float] | NDArray,
        full_results: list[TickResult],
        poll_results: list[PollResult],
) -> InfoLossReport:
    """Compute information-loss metrics from a baseline run and poller output."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    full_means = np.array([r.mean for r in full_results])
    full_stds = np.array([r.std for r in full_results])
    full_outlier_set = {i for i, r in enumerate(full_results)
                        if r.status == Status.OUTLIER}
    full_drift_set = {i for i, r in enumerate(full_results)
                      if r.status == Status.DRIFT_RESET}

    # Collect sampled indices and their tracker outputs
    log.debug("Collecting sampled points and reconstructing adaptive track...")
    sampled_indices: list[int] = []
    sampled_means: list[float] = []
    sampled_stds: list[float] = []
    sampled_values: list[float] = []  # raw signal at sampled points

    for i, pr in enumerate(poll_results):
        if pr.sampled and pr.tick is not None:
            sampled_indices.append(i)
            sampled_means.append(pr.tick.mean)
            sampled_stds.append(pr.tick.std)
            sampled_values.append(data[i])

    # Reconstruct adaptive means/stds at full resolution via linear
    # interpolation between sampled points.
    log.debug("Interpolating adaptive track to full resolution...")
    adaptive_means = np.empty(n, dtype=np.float64)
    adaptive_stds = np.empty(n, dtype=np.float64)

    if len(sampled_indices) >= 2:
        si = np.array(sampled_indices)
        adaptive_means = np.interp(np.arange(n), si,
                                   np.array(sampled_means))
        adaptive_stds = np.interp(np.arange(n), si,
                                  np.array(sampled_stds))
    elif len(sampled_indices) == 1:
        adaptive_means[:] = sampled_means[0]
        adaptive_stds[:] = sampled_stds[0]
    else:
        adaptive_means[:] = 0.0
        adaptive_stds[:] = 0.0

    adaptive_outlier_set = {pr.time_index for pr in poll_results
                            if pr.sampled and pr.tick is not None
                            and pr.tick.status == Status.OUTLIER}
    adaptive_drift_set = {pr.time_index for pr in poll_results
                          if pr.sampled and pr.tick is not None
                          and pr.tick.status == Status.DRIFT_RESET}

    # --- Compute metrics ---
    log.debug("Computing information-loss metrics...")
    n_sampled = len(sampled_indices)

    # Signal range for normalisation (guard against constant signals)
    signal_range = float(np.ptp(data))  # max - min
    if signal_range < 1e-8:
        signal_range = 1.0

    # Sampling gaps
    log.debug("Computing sampling gap metrics (1/5)...")
    if n_sampled > 1:
        gaps = np.diff(sampled_indices).astype(np.float64)
        max_gap = int(np.max(gaps))
        mean_gap = float(np.mean(gaps))
        p95_gap = float(np.percentile(gaps, 95))
    else:
        max_gap = n
        mean_gap = float(n)
        p95_gap = float(n)

    # Mean tracking errors
    mean_err = np.abs(full_means - adaptive_means)
    mae_mean = float(np.mean(mean_err))
    rmse_mean = float(np.sqrt(np.mean(mean_err ** 2)))
    nrmse_mean = rmse_mean / signal_range
    max_mean_error = float(np.max(mean_err))
    p50_error = float(np.percentile(mean_err, 50))
    p90_error = float(np.percentile(mean_err, 90))
    p95_error = float(np.percentile(mean_err, 95))
    p99_error = float(np.percentile(mean_err, 99))

    # Correlation
    log.debug("Computing correlation and lag metrics (2/5)...")
    # if np.std(full_means) > 1e-8 and np.std(adaptive_means) > 1e-8:
    correlation = float(np.corrcoef(full_means, adaptive_means)[0, 1])
    # else:
    #     correlation = 1.0  # both flat

    # Lag via cross-correlation (search +-50 samples)
    # max_lag = min(50, n // 4)
    # if max_lag > 0:
    #     xcorr = np.correlate(
    #         full_means - np.mean(full_means),
    #         adaptive_means - np.mean(adaptive_means),
    #         mode='full'
    #     )
    #     center = len(xcorr) // 2
    #     search_slice = xcorr[center - max_lag:center + max_lag + 1]
    #     lag_offset = int(np.argmax(search_slice) - max_lag)
    # else:
    #     lag_offset = 0

    # --- Raw signal reconstruction ---
    # Linearly interpolate the raw signal values at sampled points to
    # full resolution; compare against the true signal.
    log.debug("Computing raw signal reconstruction metrics (3/5)...")
    if n_sampled >= 2:
        si = np.array(sampled_indices)
        reconstructed = np.interp(np.arange(n), si,
                                  np.array(sampled_values))
    elif n_sampled == 1:
        reconstructed = np.full(n, sampled_values[0])
    else:
        reconstructed = np.zeros(n)

    recon_err = data - reconstructed
    raw_recon_rmse = float(np.sqrt(np.mean(recon_err ** 2)))
    raw_recon_nrmse = raw_recon_rmse / signal_range

    # Peak recall: fraction of top-5% signal peaks that coincide with
    # a sampled point.  A "peak" is any index whose |value| is in the
    # top 5% by magnitude.
    log.debug("Computing peak recall metric (4/5)...")
    n_top = max(1, n // 20)
    top_indices = set(np.argpartition(np.abs(data), -n_top)[-n_top:])
    sampled_set = set(sampled_indices)
    peak_recall = len(top_indices & sampled_set) / len(top_indices)

    # Std tracking errors
    log.debug("Computing std tracking error metrics and confidence band coverage (5/5)...")
    std_err = np.abs(full_stds - adaptive_stds)
    mae_std = float(np.mean(std_err))
    rmse_std = float(np.sqrt(np.mean(std_err ** 2)))

    # Confidence band coverage (+-3sigma)
    lower_band = adaptive_means - 3 * adaptive_stds
    upper_band = adaptive_means + 3 * adaptive_stds
    within_band = (data >= lower_band) & (data <= upper_band)
    coverage_3sigma = float(np.mean(within_band))

    # Anomaly/drift sets (Jaccard)
    union = full_outlier_set | adaptive_outlier_set
    inter = full_outlier_set & adaptive_outlier_set
    # jaccard = len(inter) / len(union) if union else 1.0

    return InfoLossReport(
        n_total=n,
        n_sampled=n_sampled,
        sample_ratio=n_sampled / n,
        max_gap=max_gap,
        mean_gap=mean_gap,
        p95_gap=p95_gap,
        mae_mean=mae_mean,
        rmse_mean=rmse_mean,
        nrmse_mean=nrmse_mean,
        max_mean_error=max_mean_error,
        p50_error=p50_error,
        p90_error=p90_error,
        p95_error=p95_error,
        p99_error=p99_error,
        correlation=correlation,
        # lag_samples=lag_offset,
        raw_recon_rmse=raw_recon_rmse,
        raw_recon_nrmse=raw_recon_nrmse,
        peak_recall_top5=peak_recall,
        mae_std=mae_std,
        rmse_std=rmse_std,
        coverage_3sigma=coverage_3sigma
        # missed_outliers=len(full_outlier_set - adaptive_outlier_set),
        # extra_outliers=len(adaptive_outlier_set - full_outlier_set),
        # missed_drifts=len(full_drift_set - adaptive_drift_set),
        # jaccard_outliers=jaccard,
    )
