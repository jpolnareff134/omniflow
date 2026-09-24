import math

import numpy as np
import pytest

from tracker.baselines import track_huang_wavelet_rate
from tracker.windowed import WindowedTracker


def _sample_count(results, start, stop):
    return sum(result.sampled for result in results[start:stop])


def test_huang_default_behavior_is_unchanged_when_envelope_is_omitted():
    trace = np.sin(np.linspace(0.0, 8.0, 128))

    default = track_huang_wavelet_rate(trace, WindowedTracker())
    explicit_none = track_huang_wavelet_rate(
        trace,
        WindowedTracker(),
        min_interval=None,
        max_interval=None,
    )

    assert [result.sampled for result in default] == [
        result.sampled for result in explicit_none
    ]
    assert [result.interval for result in default] == [
        result.interval for result in explicit_none
    ]


def test_huang_common_envelope_clamps_each_full_block_sample_count():
    window = 64
    min_interval = 5
    max_interval = 20
    trace = np.sin(np.linspace(0.0, 20.0, 2 * window))

    results = track_huang_wavelet_rate(
        trace,
        WindowedTracker(),
        window=window,
        min_interval=min_interval,
        max_interval=max_interval,
    )

    lower = math.ceil(window / max_interval)
    upper = math.ceil(window / min_interval)
    for start in range(0, len(trace), window):
        count = _sample_count(results, start, start + window)
        assert lower <= count <= upper


def test_huang_equal_interval_envelope_fixes_block_budget():
    window = 64
    interval = 10
    trace = np.linspace(0.0, 1.0, 2 * window)

    results = track_huang_wavelet_rate(
        trace,
        WindowedTracker(),
        window=window,
        min_interval=interval,
        max_interval=interval,
    )

    expected = math.ceil(window / interval)
    assert _sample_count(results, 0, window) == expected
    assert _sample_count(results, window, 2 * window) == expected


def test_huang_rejects_inverted_interval_envelope():
    trace = np.arange(64, dtype=float)
    with pytest.raises(ValueError, match="max_interval must be >= min_interval"):
        track_huang_wavelet_rate(
            trace,
            WindowedTracker(),
            min_interval=10,
            max_interval=5,
        )
