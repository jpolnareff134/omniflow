import math

import numpy as np

from eval.sampling_quality import (
    causal_raw_reconstruction,
    evaluate_sampling_quality,
    peak_recall,
    top_event_metrics,
)
from tracker.windowed import PollResult


def _poll_results(n, sampled):
    sampled = set(sampled)
    return [
        PollResult(
            time_index=i,
            sampled=i in sampled,
            tick=None,
            interval=1,
            urgency=1.0,
        )
        for i in range(n)
    ]


def test_causal_reconstruction_uses_last_selected_value_only():
    data = np.asarray([1.0, 2.0, 8.0, 4.0, 5.0])
    results = _poll_results(len(data), sampled=[0, 2, 4])

    reconstructed = causal_raw_reconstruction(data, results)

    np.testing.assert_allclose(reconstructed, [1.0, 1.0, 8.0, 8.0, 5.0])


def test_peak_recall_uses_exact_top_fraction_count():
    data = np.arange(1.0, 101.0)
    results = _poll_results(len(data), sampled=[95, 96, 97, 98, 99])

    assert peak_recall(data, results, fraction=0.05) == 1.0
    assert peak_recall(data, results, fraction=0.10) == 0.5


def test_sampling_quality_reports_all_requested_peak_recall_thresholds():
    data = np.arange(1.0, 101.0)
    # Select the largest ten observations.
    results = _poll_results(len(data), sampled=list(range(90, 100)))

    report = evaluate_sampling_quality(data, results)

    assert report.peak_recall_top1 == 1.0
    assert report.peak_recall_top2 == 1.0
    assert report.peak_recall_top5 == 1.0
    assert report.peak_recall_top10 == 1.0
    assert report.peak_recall_top20 == 0.5


def test_top_event_metrics_group_contiguous_peaks_and_measure_first_hit_delay():
    data = np.zeros(20)
    data[2:5] = [10.0, 11.0, 12.0]
    data[12:14] = [13.0, 14.0]
    # Top 25% = five points, forming two contiguous events.
    results = _poll_results(len(data), sampled=[3, 12])

    n_events, n_hits, recall, mean_delay, p95_delay = top_event_metrics(
        data,
        results,
        fraction=0.25,
    )

    assert n_events == 2
    assert n_hits == 2
    assert recall == 1.0
    assert mean_delay == 0.5
    assert math.isclose(p95_delay, 0.95)


def test_sampling_quality_reports_peak_threshold_sensitivity_and_causal_error():
    data = np.arange(1.0, 101.0)
    results = _poll_results(len(data), sampled=list(range(0, 100, 2)))

    report = evaluate_sampling_quality(data, results)

    assert 0.0 <= report.peak_recall_top1 <= 1.0
    assert 0.0 <= report.peak_recall_top2 <= 1.0
    assert 0.0 <= report.peak_recall_top5 <= 1.0
    assert 0.0 <= report.peak_recall_top10 <= 1.0
    assert 0.0 <= report.peak_recall_top20 <= 1.0
    assert report.causal_raw_recon_rmse > 0.0
    assert report.causal_raw_recon_nrmse > 0.0
