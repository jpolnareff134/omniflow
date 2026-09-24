import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy" / "io_syscall"))

from evaluate import _build_pipe_config, summarize_replay_urgency  # noqa: E402
from aggregate_io_results import _eval_replay_metric_row  # noqa: E402
from replay_paper_traces import summarize_payload  # noqa: E402
from tracker.pipeline import PipelineConfig  # noqa: E402


class ReplayUrgencySummaryTest(unittest.TestCase):
    def test_io_default_profile_preserves_paper_replay_semantics(self) -> None:
        config = _build_pipe_config(min_interval=1, max_interval=10, profile="default")

        self.assertEqual(config.initial_variance_floor, 1.0)
        self.assertTrue(config.warmup_positive_std_only)

    def test_replay_aggregator_uses_replay_not_source_trace_urgency(self) -> None:
        row = _eval_replay_metric_row({
            "info_loss": {"sample_ratio": 0.5},
            "scheduler_summary": {
                "mean_urgency": 0.2,
                "replay_mean_urgency": 0.8,
                "replay_median_urgency": 0.9,
            },
        })

        self.assertEqual(row["mean_urgency"], 0.8)
        self.assertEqual(row["median_urgency"], 0.9)

    def test_summary_uses_every_dense_tick_including_skips(self) -> None:
        trace = np.full(100, 10.0, dtype=np.float64)
        poll_results = PipelineConfig(
            min_interval=1,
            max_interval=10,
        )._make_poller().track(trace)
        values = np.asarray([result.urgency for result in poll_results])

        self.assertLess(sum(result.sampled for result in poll_results), len(trace))
        summary = summarize_replay_urgency(poll_results, include_values=True)

        self.assertEqual(summary["count"], len(trace))
        self.assertAlmostEqual(summary["mean"], float(np.mean(values)))
        self.assertAlmostEqual(summary["median"], float(np.median(values)))
        self.assertEqual(summary["values"], values.tolist())

    def test_paper_replay_summary_uses_declared_run_sets(self) -> None:
        trace = np.linspace(10.0, 30.0, 60).tolist()
        payload = {
            "workloads": {
                name: {
                    "paper_repeat_ids": ["01", "02"] if name != "latency" else ["02"],
                    "runs": [
                        {"repeat": repeat, "values": trace}
                        for repeat in ("01", "02")
                    ],
                }
                for name in ("postgres", "redis", "latency")
            }
        }

        summary = summarize_payload(payload)

        self.assertEqual(summary["workloads"]["postgres"]["paper_run_count"], 2)
        self.assertEqual(summary["workloads"]["redis"]["paper_run_count"], 2)
        self.assertEqual(summary["workloads"]["latency"]["paper_run_count"], 1)
        self.assertEqual(
            summary["workloads"]["latency"]["paper_pooled_tick_urgency"]["count"],
            60,
        )
        paper_metrics = summary["workloads"]["postgres"]["paper_metrics"]
        self.assertEqual(paper_metrics["sample_ratio"]["count"], 2)
        self.assertEqual(paper_metrics["nrmse_mean"]["count"], 2)
        self.assertEqual(paper_metrics["correlation"]["count"], 2)
        self.assertEqual(paper_metrics["max_gap"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
