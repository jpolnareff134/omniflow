"""Temporal sampling baselines used in the OmniFlow evaluation.

The three literature-derived policies below are deliberately labelled as
*scalar adaptations*.  The cited systems operate under different observation
models:

* Huang et al. use adaptive-rate compressive sampling with random linear
  measurements, cross-validation, and sparsity prediction.
* Magalhaes and Silva use the correlation between workload and response time to
  control selective fine-grained profiling.
* Daoud et al. continuously inspect low-level memory events and emit a memory
  sample when a timer or accumulated-variation threshold fires.

The implementations preserve the rate-control idea of each paper while mapping
it to a single scalar trace.  They are not claimed to reproduce the original
systems or their cost models exactly.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tracker.windowed import (
    InfoLossReport,
    PollResult,
    WindowedTracker,
    evaluate_info_loss_from_runs,
)

_EPS = 1e-12


def _as_float_array(data: Sequence[float] | NDArray) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a one-dimensional trace, got shape {values.shape}")
    return values


def _poll_results_from_mask(
        data: NDArray[np.float64],
        tracker: WindowedTracker,
        mask: NDArray[np.bool_],
        intervals: NDArray[np.integer] | None = None,
) -> list[PollResult]:
    """Feed ``tracker`` only at timestamps selected by ``mask``."""
    if len(mask) != len(data):
        raise ValueError("mask length must match data length")
    if intervals is not None and len(intervals) != len(data):
        raise ValueError("interval array length must match data length")

    tracker.reset()
    results: list[PollResult] = []
    for time_index, value in enumerate(data):
        sampled = bool(mask[time_index])
        tick = tracker.update(float(value)) if sampled else None
        interval = int(intervals[time_index]) if intervals is not None else 1
        interval = max(1, interval)
        results.append(PollResult(
            time_index=time_index,
            sampled=sampled,
            tick=tick,
            interval=interval,
            urgency=1.0 / interval,
        ))
    return results


def _evenly_spaced_indices(start: int, stop: int, count: int) -> NDArray[np.int64]:
    """Return ``count`` distinct, approximately uniform indices in [start, stop)."""
    length = stop - start
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(length, max(1, int(count)))
    if count == 1:
        return np.asarray([start], dtype=np.int64)
    return np.unique(np.rint(np.linspace(start, stop - 1, count)).astype(np.int64))


def _haar_sparsity(values: NDArray[np.float64], retained_energy: float) -> int:
    """Return the Haar coefficients needed to retain ``retained_energy``.

    Huang et al. define sparsity as the number of transform coefficients needed
    to retain a fixed fraction of signal energy.  Their CS-MON experiments use
    99.5 percent for the adaptive rate law.
    """
    if not 0.0 < retained_energy <= 1.0:
        raise ValueError("retained_energy must be in (0, 1]")
    if len(values) == 0:
        return 1

    n = 1 << max(0, (len(values) - 1).bit_length())
    work = np.pad(values, (0, n - len(values)), mode="edge").astype(np.float64)
    coefficients: list[float] = []
    while len(work) > 1:
        averages = (work[0::2] + work[1::2]) / np.sqrt(2.0)
        details = (work[0::2] - work[1::2]) / np.sqrt(2.0)
        coefficients.extend(details.tolist())
        work = averages
    coefficients.extend(work.tolist())

    energy = np.square(np.asarray(coefficients, dtype=np.float64))
    total = float(np.sum(energy))
    if total <= _EPS:
        return 1
    ordered = np.sort(energy)[::-1]
    return int(np.searchsorted(np.cumsum(ordered), retained_energy * total) + 1)




# ---------------------------------------------------------------------------
# Huang et al.: causal scalar analogue of the CS-MON rate law
# ---------------------------------------------------------------------------


def track_huang_wavelet_rate(
        data: Sequence[float] | NDArray,
        tracker: WindowedTracker,
        *,
        window: int = 64,
        retained_energy: float = 0.995,
        measurements_per_coefficient: float = 5.0,
        min_samples: int = 4,
        initial_sparsity_fraction: float = 0.10,
        kalman_process_variance: float = 0.0025,
        kalman_measurement_variance: float = 0.01,
        min_interval: int | None = None,
        max_interval: int | None = None,
) -> list[PollResult]:
    """Causal temporal adaptation inspired by Huang et al.'s CS-MON.

    The original paper collects random linear measurements, periodically
    estimates sparsity with cross-validation, and predicts future sparsity with
    a Kalman filter.  A point sampler cannot reproduce that sensing model.

    This scalar analogue preserves its blockwise rate law without consulting
    unsampled values:

    1. the predicted sparsity determines the point budget for the next block;
    2. that budget is spread approximately uniformly over the block;
    3. after the block, a Haar coefficient fraction is estimated from the
       sampled sequence;
    4. a scalar Kalman update predicts the sparsity used by the next block;
    5. the paper's empirical ``M = 5 s`` rule maps predicted sparsity to budget.

    ``min_interval`` and ``max_interval`` optionally impose a common temporal
    sampling envelope for controlled comparisons.  They do not alter the
    sparsity predictor: they only clamp each block's sample count to the count
    implied by those interval bounds.  With neither bound set, behavior is
    identical to the original scalar adaptation.

    Consequently, this function is a meaningful compressibility-driven temporal
    baseline, not an implementation of CS-MON's random projections or HTP
    reconstruction.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if not 0.0 < initial_sparsity_fraction <= 1.0:
        raise ValueError("initial_sparsity_fraction must be in (0, 1]")
    if measurements_per_coefficient <= 0.0:
        raise ValueError("measurements_per_coefficient must be positive")
    if kalman_process_variance < 0.0 or kalman_measurement_variance <= 0.0:
        raise ValueError("Kalman variances must be non-negative, with measurement variance > 0")
    if min_interval is not None:
        min_interval = max(1, int(min_interval))
    if max_interval is not None:
        max_interval = max(1, int(max_interval))
    if (
        min_interval is not None
        and max_interval is not None
        and max_interval < min_interval
    ):
        raise ValueError("max_interval must be >= min_interval")

    values = _as_float_array(data)
    n_total = len(values)
    if n_total == 0:
        return []

    mask = np.zeros(n_total, dtype=np.bool_)
    reported_intervals = np.full(n_total, window, dtype=np.int64)

    sparsity_estimate = float(initial_sparsity_fraction)
    estimate_variance = float(kalman_measurement_variance)

    for start in range(0, n_total, window):
        stop = min(n_total, start + window)
        block_len = stop - start

        predicted_sparsity = float(np.clip(sparsity_estimate, 1.0 / block_len, 1.0))
        # Huang et al. express sparsity as a fraction/percentage of the N
        # transform coefficients and use a sample fraction of approximately 5s.
        budget = int(np.ceil(measurements_per_coefficient * predicted_sparsity * block_len))
        budget = min(block_len, max(min_samples, budget))

        # Optional common-envelope clamp.  A sampler that takes the first point
        # and then samples every I ticks selects ceil(block_len / I) points in a
        # finite block, so use the same count convention here.
        if max_interval is not None:
            min_budget = int(math.ceil(block_len / max_interval))
            budget = max(min_budget, budget)
        if min_interval is not None:
            max_budget = int(math.ceil(block_len / min_interval))
            budget = min(max_budget, budget)
        budget = min(block_len, max(1, budget))

        selected = _evenly_spaced_indices(start, stop, budget)
        mask[selected] = True

        effective_interval = max(1, int(math.ceil(block_len / len(selected))))
        reported_intervals[start:stop] = effective_interval

        sampled_values = values[selected]
        sampled_sparsity = _haar_sparsity(sampled_values, retained_energy)
        padded_sample_length = 1 << max(0, (len(sampled_values) - 1).bit_length())
        measured_sparsity = sampled_sparsity / padded_sample_length
        measured_sparsity = float(np.clip(measured_sparsity, 1.0 / block_len, 1.0))

        # Random-walk scalar Kalman filter: prediction followed by measurement
        # update.  This mirrors CS-MON's use of sparsity prediction while keeping
        # the adaptation self-contained and causal.
        predicted_variance = estimate_variance + kalman_process_variance
        gain = predicted_variance / (predicted_variance + kalman_measurement_variance)
        sparsity_estimate = predicted_sparsity + gain * (measured_sparsity - predicted_sparsity)
        estimate_variance = (1.0 - gain) * predicted_variance

    return _poll_results_from_mask(values, tracker, mask, reported_intervals)


# ---------------------------------------------------------------------------
# Generic predictive baseline retained for backwards compatibility
# ---------------------------------------------------------------------------


def track_liu_predictive_frequency(
        data: Sequence[float] | NDArray,
        tracker: WindowedTracker,
        *,
        intervals: tuple[int, ...] = (1, 2, 4, 8, 16, 20),
        ewma_alpha: float = 0.2,
        error_fraction: float = 0.1,
        hysteresis: float = 0.2,
) -> list[PollResult]:
    """Causal prediction-error frequency controller.

    This helper is not one of the three literature-derived baselines reported in
    the CloudPerfTrace table; it is retained for existing experiment scripts.
    """
    if not 0.0 < ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1]")
    if error_fraction < 0.0 or hysteresis < 0.0:
        raise ValueError("error_fraction and hysteresis must be non-negative")

    values = _as_float_array(data)
    allowed = tuple(sorted(set(max(1, int(item)) for item in intervals)))
    if not allowed:
        raise ValueError("intervals must not be empty")

    current_interval = allowed[0]
    next_sample = 0
    previous: float | None = None
    innovation_ewma = 0.0
    scale_ewma = 0.0
    results: list[PollResult] = []
    tracker.reset()

    for time_index, raw_value in enumerate(values):
        sampled = time_index >= next_sample
        tick = None
        if sampled:
            value = float(raw_value)
            tick = tracker.update(value)
            if previous is not None:
                innovation = abs(value - previous) / max(1, current_interval)
                innovation_ewma = (
                    ewma_alpha * innovation
                    + (1.0 - ewma_alpha) * innovation_ewma
                )
            scale_ewma = ewma_alpha * abs(value) + (1.0 - ewma_alpha) * scale_ewma
            error_limit = error_fraction * max(scale_ewma, 1e-8)

            candidate = allowed[0]
            for interval in allowed:
                if innovation_ewma * interval <= error_limit:
                    candidate = interval
            relative_change = abs(candidate - current_interval) / max(1, current_interval)
            if relative_change >= hysteresis:
                current_interval = candidate

            previous = value
            next_sample = time_index + current_interval

        results.append(PollResult(
            time_index=time_index,
            sampled=sampled,
            tick=tick,
            interval=current_interval,
            urgency=1.0 / current_interval,
        ))
    return results


# ---------------------------------------------------------------------------
# Magalhaes and Silva: scalar correlation-degradation analogue
# ---------------------------------------------------------------------------


def track_magalhaes_exponential(
        data: Sequence[float] | NDArray,
        tracker: WindowedTracker,
        *,
        min_interval: int = 1,
        max_interval: int = 20,
        reference_window: int = 20,
        outlier_threshold: float = 3.0,
        recovery_samples: int = 3,
) -> list[PollResult]:
    """Scalar outlier-triggered controller inspired by Magalhaes and Silva.

    The original method changes fine-grained profiling intervals from the
    deterioration of Pearson correlation between workload and transaction
    response time.  A univariate trace does not contain that pair of signals.

    This adaptation preserves the paper's selective, truncated-exponential
    response: an independently computed robust scalar deviation is used as the
    outlier symptom, the interval is halved when the symptom is present, and it
    is doubled after ``recovery_samples`` stable sampled observations.  The
    decision does not reuse OmniFlow's tracker status or urgency components.
    """
    min_interval = max(1, int(min_interval))
    max_interval = max(min_interval, int(max_interval))
    if reference_window < 3:
        raise ValueError("reference_window must be at least 3")
    if outlier_threshold <= 0.0:
        raise ValueError("outlier_threshold must be positive")
    if recovery_samples < 1:
        raise ValueError("recovery_samples must be positive")

    values = _as_float_array(data)
    tracker.reset()
    reference: deque[float] = deque(maxlen=reference_window)
    current_interval = min_interval  # dense warm-up for the scalar detector
    next_sample = 0
    stable_samples = 0
    results: list[PollResult] = []

    for time_index, raw_value in enumerate(values):
        sampled = time_index >= next_sample
        tick = None
        if sampled:
            value = float(raw_value)
            tick = tracker.update(value)

            if len(reference) < reference_window:
                reference.append(value)
                current_interval = min_interval
                stable_samples = 0
            else:
                history = np.asarray(reference, dtype=np.float64)
                median = float(np.median(history))
                mad = float(np.median(np.abs(history - median)))
                robust_scale = 1.4826 * mad
                if robust_scale <= _EPS:
                    robust_scale = max(1e-8, 0.01 * max(1.0, abs(median)))
                deviation = abs(value - median) / robust_scale
                is_outlier = deviation > outlier_threshold

                if is_outlier:
                    current_interval = max(
                        min_interval,
                        int(math.ceil(current_interval / 2.0)),
                    )
                    stable_samples = 0
                else:
                    stable_samples += 1
                    if stable_samples >= recovery_samples:
                        current_interval = min(max_interval, current_interval * 2)
                        stable_samples = 0

                reference.append(value)

            next_sample = time_index + current_interval

        results.append(PollResult(
            time_index=time_index,
            sampled=sampled,
            tick=tick,
            interval=current_interval,
            urgency=1.0 / current_interval,
        ))
    return results


# ---------------------------------------------------------------------------
# Daoud et al.: time-or-accumulated-variation emission trigger
# ---------------------------------------------------------------------------


def track_daoud_time_variation(
        data: Sequence[float] | NDArray,
        tracker: WindowedTracker,
        *,
        max_interval: int = 20,
        variation_threshold: float | None = None,
        threshold_scale: float = 1.0,
        threshold_quantile: float = 0.90,
        warmup: int = 64,
) -> list[PollResult]:
    """Time-or-variation trigger adapted from Daoud et al. Algorithm 1.

    The original kernel module observes every page allocation/free event,
    accumulates signed memory variation, and emits a memory-usage record when a
    periodic timer fires or the accumulated variation exceeds a threshold.

    This scalar analogue therefore inspects every base-trace increment.  It is
    an event-triggered *emission* baseline, not a pure polling baseline with the
    same observation cost as OmniFlow.  If no threshold is supplied, a causal
    warm-up estimates it from the distribution of absolute cumulative changes
    over windows no longer than ``max_interval``.

    After a threshold-triggered emission, accumulated variation is reset so a
    continuously exceeded threshold does not emit every subsequent point.  The
    periodic timer remains independent, matching the architecture in the paper.
    """
    max_interval = max(1, int(max_interval))
    warmup = max(2, int(warmup))
    if variation_threshold is not None and variation_threshold <= 0.0:
        raise ValueError("variation_threshold must be positive")
    if threshold_scale <= 0.0:
        raise ValueError("threshold_scale must be positive")
    if not 0.0 < threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in (0, 1]")

    values = _as_float_array(data)
    n_total = len(values)
    if n_total == 0:
        return []

    tracker.reset()
    mask = np.zeros(n_total, dtype=np.bool_)
    intervals = np.full(n_total, max_interval, dtype=np.int64)

    threshold = float(variation_threshold) if variation_threshold is not None else None
    calibration_changes: deque[float] = deque(maxlen=warmup)

    mask[0] = True
    intervals[0] = 1
    previous_value = float(values[0])
    accumulated_variation = 0.0
    calibration_variation = 0.0
    calibration_elapsed = 0
    next_timer = max_interval
    last_emission = 0

    for time_index in range(1, n_total):
        value = float(values[time_index])
        delta = value - previous_value
        previous_value = value

        accumulated_variation += delta

        if threshold is None:
            calibration_variation += delta
            calibration_elapsed += 1
            # Observe the magnitude of cumulative changes over partial periods.
            # This yields a scale estimate after ``warmup`` base observations,
            # without publishing those observations as samples.
            calibration_changes.append(abs(calibration_variation))
            if calibration_elapsed >= max_interval:
                calibration_variation = 0.0
                calibration_elapsed = 0
            if len(calibration_changes) >= warmup:
                candidate = float(np.quantile(
                    np.asarray(calibration_changes, dtype=np.float64),
                    threshold_quantile,
                ))
                threshold = max(1e-8, threshold_scale * candidate)

        timer_fired = time_index >= next_timer
        threshold_fired = threshold is not None and abs(accumulated_variation) >= threshold

        if timer_fired or threshold_fired:
            mask[time_index] = True
            intervals[time_index] = max(1, time_index - last_emission)
            last_emission = time_index
            accumulated_variation = 0.0

            if timer_fired:
                # Keep the timer periodic even if threshold events occur between
                # ticks.  Advance far enough to be strictly after this timestamp.
                while next_timer <= time_index:
                    next_timer += max_interval

    return _poll_results_from_mask(values, tracker, mask, intervals)


# ---------------------------------------------------------------------------
# Matched-budget deterministic baseline
# ---------------------------------------------------------------------------


def track_n_every_m(
        data: Sequence[float] | NDArray,
        tracker: WindowedTracker,
        *,
        n: int,
        m: int,
        phase: int = 0,
) -> list[PollResult]:
    """Run a balanced deterministic ``N every M`` temporal sampling pattern."""
    if m < 1 or not 1 <= n <= m:
        raise ValueError(f"expected 1 <= n <= m, got n={n}, m={m}")
    if not 0 <= phase < m:
        raise ValueError(f"phase must be in [0, {m}), got {phase}")

    values = _as_float_array(data)
    tracker.reset()
    results: list[PollResult] = []
    urgency = n / m
    max_gap = int(math.ceil(m / n))

    for time_index, value in enumerate(values):
        sampled = (time_index * n + phase) % m < n
        tick = tracker.update(float(value)) if sampled else None
        results.append(PollResult(
            time_index=time_index,
            sampled=sampled,
            tick=tick,
            interval=max_gap,
            urgency=urgency,
        ))
    return results


def budget_ratio_to_n_every_m(
        sample_ratio: float,
        *,
        tolerance: float = 0.001,
        max_period: int = 128,
) -> tuple[int, int]:
    """Find the shortest periodic budget within ``tolerance`` of a ratio."""
    if not 0.0 < sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if max_period < 1:
        raise ValueError(f"max_period must be positive, got {max_period}")

    closest: tuple[float, int, int] | None = None
    for period in range(1, max_period + 1):
        samples = min(period, max(1, int(math.floor(sample_ratio * period + 0.5))))
        error = abs(samples / period - sample_ratio)
        if closest is None or error < closest[0]:
            closest = (error, samples, period)
        if error <= tolerance + 1e-12:
            divisor = math.gcd(samples, period)
            return samples // divisor, period // divisor

    assert closest is not None
    raise ValueError(
        f"no periodic budget within {tolerance} of {sample_ratio:.12f} "
        f"using a period <= {max_period}; closest is {closest[1]}/{closest[2]} "
        f"with absolute error {closest[0]:.12f}"
    )


def evaluate_info_loss_n_every_m(
        data: Sequence[float] | NDArray,
        full_tracker: WindowedTracker,
        sampled_tracker: WindowedTracker,
        *,
        n: int,
        m: int,
        phase: int = 0,
) -> InfoLossReport:
    """Compare an ``N every M`` run against the dense tracker reference."""
    values = _as_float_array(data)
    full_tracker.reset()
    full_results = full_tracker.track(values)
    sampled_results = track_n_every_m(
        values,
        sampled_tracker,
        n=n,
        m=m,
        phase=phase,
    )
    return evaluate_info_loss_from_runs(values, full_results, sampled_results)
