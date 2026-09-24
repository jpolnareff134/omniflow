"""Anomaly detection and comparison utilities for adaptive replay evaluation.

Provides a simple threshold-based anomaly detector that can be applied to
both dense-reference and adaptively reconstructed signals. The comparison
function measures how many reference anomalies are detected or missed in
the replayed sparse signal.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def detect_anomalies(
        signal: NDArray,
        z: float = 3.0,
        min_consecutive: int = 3,
        warmup: int = 50,
) -> list[tuple[int, int]]:
    """Detect anomalous windows in *signal* using a z-score threshold.

    An anomaly window is a contiguous run of >= *min_consecutive* points
    where ``|x - mean| > z * std``.  Mean and std are computed over
    the first *warmup* points (or the full signal if shorter).

    Returns a list of ``(start_index, end_index)`` inclusive.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < warmup:
        warmup = n

    mu = float(np.mean(signal[:warmup]))
    sigma = float(np.std(signal[:warmup]))
    if sigma < 1e-8:
        sigma = 1.0

    above = np.abs(signal - mu) > z * sigma

    windows: list[tuple[int, int]] = []
    start = None
    for i in range(n):
        if above[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_consecutive:
                    windows.append((start, i - 1))
                start = None
    if start is not None and n - start >= min_consecutive:
        windows.append((start, n - 1))

    return windows


def compare_anomalies(
        ground_truth: list[tuple[int, int]],
        detected: list[tuple[int, int]],
        tolerance: int = 5,
) -> dict:
    """Compare detected anomaly windows against a reference signal.

    A reference anomaly window is "detected" if any detected window overlaps
    it or starts within *tolerance* ticks of its start.

    Returns a dict with recall, missed count, false positive count,
    and mean detection latency.
    """
    matched = 0
    latencies: list[int] = []

    gt_matched = set()
    det_matched = set()

    for gi, (gs, ge) in enumerate(ground_truth):
        for di, (ds, de) in enumerate(detected):
            # Overlap or within tolerance
            if ds <= ge + tolerance and de >= gs - tolerance:
                if gi not in gt_matched:
                    matched += 1
                    latencies.append(max(0, ds - gs))
                    gt_matched.add(gi)
                    det_matched.add(di)
                break

    n_gt = len(ground_truth)
    n_det = len(detected)
    missed = n_gt - matched
    false_positives = len([i for i in range(n_det) if i not in det_matched])

    return {
        "ground_truth_count": n_gt,
        "detected_count": n_det,
        "detected": matched,
        "missed": missed,
        "false_positives": false_positives,
        "recall": round(matched / n_gt, 4) if n_gt > 0 else 1.0,
        "mean_latency_ticks": round(float(np.mean(latencies)), 1) if latencies else 0.0,
    }
