"""Sampling-quality metrics that do not depend on OmniFlow's controller state.

These helpers operate only on the original scalar trace and on the indices
selected by a sampling policy.  They are used by the budget-matched adaptive
comparison so operating points can be selected by sample ratio first and then
assessed with several independent fidelity views.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tracker.windowed import PollResult


PEAK_RECALL_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class SamplingQualityReport:
    """Policy-independent quality metrics derived from selected observations."""

    causal_raw_recon_rmse: float
    causal_raw_recon_nrmse: float
    peak_recall_top1: float
    peak_recall_top2: float
    peak_recall_top5: float
    peak_recall_top10: float
    peak_recall_top20: float
    peak_event_count_top5: int
    peak_event_hits_top5: int
    peak_event_recall_top5: float
    peak_event_mean_delay_top5: float
    peak_event_p95_delay_top5: float

    def to_dict(self) -> dict:
        return asdict(self)


def _as_float_array(data: Sequence[float] | NDArray) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a one-dimensional trace, got shape {values.shape}")
    return values


def _sampled_indices(poll_results: Sequence[PollResult], n: int) -> NDArray[np.int64]:
    indices = np.asarray(
        [result.time_index for result in poll_results if result.sampled],
        dtype=np.int64,
    )
    if len(indices) == 0:
        return indices
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("sampled index outside trace bounds")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("sampled indices must be strictly increasing")
    return indices


def _top_indices(values: NDArray[np.float64], fraction: float) -> NDArray[np.int64]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)

    # Keep exactly floor(fraction * n) observations, matching the paper's
    # existing top-5% implementation (with at least one point on short traces).
    n_top = max(1, int(np.floor(len(values) * fraction)))
    return np.argpartition(np.abs(values), -n_top)[-n_top:].astype(np.int64)


def precompute_peak_indices(
    data: Sequence[float] | NDArray,
    *,
    fractions: Sequence[float] = PEAK_RECALL_FRACTIONS,
) -> dict[float, NDArray[np.int64]]:
    """Precompute exact top-value index sets for several recall thresholds.

    This is useful for parameter sweeps: the target peak sets depend only on the
    dense trace, so they can be computed once per trace and reused for every
    sampling policy and operating point.
    """
    values = _as_float_array(data)
    return {
        float(fraction): _top_indices(values, float(fraction))
        for fraction in fractions
    }


def peak_recalls_from_precomputed(
    poll_results: Sequence[PollResult],
    *,
    n: int,
    peak_indices: dict[float, NDArray[np.int64]],
) -> dict[float, float]:
    """Compute peak recall from precomputed target index sets."""
    if n == 0:
        return {fraction: 1.0 for fraction in peak_indices}

    sampled = _sampled_indices(poll_results, n)
    sampled_mask = np.zeros(n, dtype=bool)
    sampled_mask[sampled] = True

    recalls: dict[float, float] = {}
    for fraction, indices in peak_indices.items():
        recalls[fraction] = (
            float(np.count_nonzero(sampled_mask[indices])) / len(indices)
            if len(indices)
            else 1.0
        )
    return recalls


def peak_recall(
    data: Sequence[float] | NDArray,
    poll_results: Sequence[PollResult],
    *,
    fraction: float,
) -> float:
    """Fraction of the highest-magnitude observations that were selected."""
    values = _as_float_array(data)
    targets = precompute_peak_indices(values, fractions=(fraction,))
    return peak_recalls_from_precomputed(
        poll_results,
        n=len(values),
        peak_indices=targets,
    )[float(fraction)]


def causal_raw_reconstruction(
    data: Sequence[float] | NDArray,
    poll_results: Sequence[PollResult],
) -> NDArray[np.float64]:
    """Reconstruct the raw trace using the most recent selected observation.

    This zero-order hold is causal: the value reconstructed at time ``t`` uses
    no observation collected after ``t``.
    """
    values = _as_float_array(data)
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    sampled = _sampled_indices(poll_results, n)
    if len(sampled) == 0:
        return np.zeros(n, dtype=np.float64)

    reconstructed = np.empty(n, dtype=np.float64)
    first = int(sampled[0])
    reconstructed[:first] = values[first]
    for pos, start in enumerate(sampled):
        stop = int(sampled[pos + 1]) if pos + 1 < len(sampled) else n
        reconstructed[int(start):stop] = values[int(start)]
    return reconstructed


def _contiguous_events(indices: NDArray[np.int64]) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    ordered = np.sort(indices)
    events: list[tuple[int, int]] = []
    start = int(ordered[0])
    previous = start
    for raw_index in ordered[1:]:
        index = int(raw_index)
        if index != previous + 1:
            events.append((start, previous))
            start = index
        previous = index
    events.append((start, previous))
    return events


def top_event_metrics(
    data: Sequence[float] | NDArray,
    poll_results: Sequence[PollResult],
    *,
    fraction: float = 0.05,
) -> tuple[int, int, float, float, float]:
    """Measure whether contiguous top-quantile excursions were observed.

    Consecutive top-``fraction`` indices are treated as one event.  An event is
    hit when at least one selected observation falls inside it.  Detection delay
    is measured from the event start to the first selected point in that event.
    Delay statistics are conditional on detected events.
    """
    values = _as_float_array(data)
    if len(values) == 0:
        return 0, 0, 1.0, 0.0, 0.0

    events = _contiguous_events(_top_indices(values, fraction))
    sampled = _sampled_indices(poll_results, len(values))
    delays: list[int] = []

    for start, stop in events:
        left = int(np.searchsorted(sampled, start, side="left"))
        if left < len(sampled) and int(sampled[left]) <= stop:
            delays.append(int(sampled[left]) - start)

    n_events = len(events)
    n_hits = len(delays)
    recall = n_hits / n_events if n_events else 1.0
    if delays:
        mean_delay = float(np.mean(delays))
        p95_delay = float(np.percentile(delays, 95))
    else:
        mean_delay = float("nan")
        p95_delay = float("nan")
    return n_events, n_hits, recall, mean_delay, p95_delay


def evaluate_sampling_quality(
    data: Sequence[float] | NDArray,
    poll_results: Sequence[PollResult],
) -> SamplingQualityReport:
    """Return independent fidelity metrics for a set of selected observations."""
    values = _as_float_array(data)
    reconstructed = causal_raw_reconstruction(values, poll_results)

    if len(values) == 0:
        rmse = 0.0
        nrmse = 0.0
    else:
        error = values - reconstructed
        rmse = float(np.sqrt(np.mean(error ** 2)))
        signal_range = float(np.ptp(values))
        if signal_range < 1e-8:
            signal_range = 1.0
        nrmse = rmse / signal_range

    peak_indices = precompute_peak_indices(values)
    recalls = peak_recalls_from_precomputed(
        poll_results,
        n=len(values),
        peak_indices=peak_indices,
    )

    event_count, event_hits, event_recall, mean_delay, p95_delay = top_event_metrics(
        values,
        poll_results,
        fraction=0.05,
    )

    return SamplingQualityReport(
        causal_raw_recon_rmse=rmse,
        causal_raw_recon_nrmse=nrmse,
        peak_recall_top1=recalls[0.01],
        peak_recall_top2=recalls[0.02],
        peak_recall_top5=recalls[0.05],
        peak_recall_top10=recalls[0.10],
        peak_recall_top20=recalls[0.20],
        peak_event_count_top5=event_count,
        peak_event_hits_top5=event_hits,
        peak_event_recall_top5=event_recall,
        peak_event_mean_delay_top5=mean_delay,
        peak_event_p95_delay_top5=p95_delay,
    )
