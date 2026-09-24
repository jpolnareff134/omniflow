import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracker.pipeline import PipelineConfig  # noqa: E402


class AdaptivePollerTest(unittest.TestCase):
    def test_zero_variance_normals_complete_automatic_calibration(self) -> None:
        poller = PipelineConfig(warmup=3)._make_poller()

        results = [poller.feed(0.0) for _ in range(4)]

        self.assertFalse(poller._sigma_ref_auto)
        self.assertEqual(poller.sigma_ref, 0.0)
        self.assertEqual([result.urgency for result in results], [1.0] * 4)

        next_result = poller.feed(0.0)
        self.assertLess(next_result.urgency, 1.0)

    def test_reset_restarts_automatic_calibration(self) -> None:
        trace = np.concatenate((np.zeros(10), np.ones(20)))
        poller = PipelineConfig(warmup=3)._make_poller()
        first = [result.time_index for result in poller.track(trace) if result.sampled]

        poller.reset()
        second = [result.time_index for result in poller.track(trace) if result.sampled]

        self.assertEqual(first, second)

    def test_legacy_variance_floor_is_opt_in(self) -> None:
        current = PipelineConfig()._make_tracker()
        io_legacy = PipelineConfig(initial_variance_floor=1.0)._make_tracker()

        self.assertEqual(current.update(0.0).std, 0.0)
        self.assertEqual(io_legacy.update(0.0).std, 1.0)

    def test_legacy_warmup_ignores_zero_standard_deviation(self) -> None:
        legacy = PipelineConfig(
            warmup=1,
            warmup_positive_std_only=True,
        )._make_poller()
        current = PipelineConfig(warmup=1)._make_poller()

        for poller in (legacy, current):
            poller.feed(0.0)
            poller.feed(0.0)

        self.assertTrue(legacy._sigma_ref_auto)
        self.assertFalse(current._sigma_ref_auto)


if __name__ == "__main__":
    unittest.main()
