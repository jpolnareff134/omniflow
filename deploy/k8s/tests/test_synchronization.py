import importlib.util
import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


K8S_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K8S_DIR))

from experiment_sync import (  # noqa: E402
    daemon_sample_versions,
    fresh_metrics,
    fresh_daemon_samples,
    metric_versions,
    metrics_published_after_samples,
    ready_pod_names,
    render_job,
    stable_hpa,
)
from load_schedule import crossed_events, phase_boundaries, total_duration  # noqa: E402


def _load_collector():
    spec = importlib.util.spec_from_file_location("collect_results", K8S_DIR / "collect-results.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_daemon():
    prometheus = types.ModuleType("prometheus_client")

    class Gauge:
        def __init__(self, *_args, **_kwargs):
            self.removed = []

        def labels(self, **_labels):
            return self

        def set(self, _value):
            pass

        def remove(self, *_labels):
            self.removed.append(_labels)

    prometheus.Gauge = Gauge
    prometheus.start_http_server = lambda _port: None
    previous = sys.modules.get("prometheus_client")
    sys.modules["prometheus_client"] = prometheus
    try:
        spec = importlib.util.spec_from_file_location("omniflow_daemon", K8S_DIR / "omniflow_daemon.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("prometheus_client", None)
        else:
            sys.modules["prometheus_client"] = previous


class LoadScheduleTest(unittest.TestCase):
    def test_shape_durations_are_unchanged(self) -> None:
        self.assertEqual(total_duration("phased"), 510)
        self.assertEqual(total_duration("oscillating"), 360)
        self.assertEqual(total_duration("staircase"), 510)
        self.assertEqual(total_duration("flash_crowd"), 450)

    def test_crossed_boundaries_are_not_lost(self) -> None:
        events = crossed_events("flash_crowd", -1.0, 151.0)
        names = [(event["event"], event.get("phase")) for event in events]
        self.assertIn(("load_start", None), names)
        self.assertIn(("phase_start", "warmup"), names)
        self.assertIn(("phase_end", "warmup"), names)
        self.assertIn(("phase_start", "flash1"), names)
        self.assertIn(("phase_end", "flash1"), names)
        self.assertIn(("phase_start", "drop1"), names)

    def test_phase_boundaries_cover_entire_shape(self) -> None:
        phases = phase_boundaries("phased")
        self.assertEqual(phases[0]["start_s"], 0)
        self.assertEqual(phases[-1]["end_s"], total_duration("phased"))


class ExperimentSyncTest(unittest.TestCase):
    def test_fresh_metrics_require_same_ready_pods_and_new_timestamps(self) -> None:
        ready = ["nginx-a"]
        self.assertEqual(fresh_metrics(ready, {"nginx-a": "1"}, {"nginx-a": "1"})[0], False)
        self.assertEqual(fresh_metrics(ready, {"nginx-a": "1"}, {"nginx-a": "2"})[0], True)
        self.assertEqual(
            fresh_metrics(ready, {"nginx-a": "1"}, {"nginx-a": "2", "nginx-old": "3"})[0],
            False,
        )

    def test_fresh_daemon_sample_requires_scheduler_counter_advance(self) -> None:
        payload = {"nodes": [{"tracks": [{
            "pod_name": "nginx-a",
            "namespace": "test-ns",
            "n_sampled": 4,
            "last_sample_wall": 10.0,
        }]}]}
        baseline = daemon_sample_versions(payload, "test-ns", {"nginx-a"})
        self.assertFalse(fresh_daemon_samples(["nginx-a"], baseline, baseline)[0])
        payload["nodes"][0]["tracks"][0].update(
            n_sampled=5, last_sample_wall=20.0
        )
        current = daemon_sample_versions(payload, "test-ns", {"nginx-a"})
        self.assertTrue(fresh_daemon_samples(["nginx-a"], baseline, current)[0])

    def test_custom_metric_must_follow_fresh_sample_and_scrape_delay(self) -> None:
        samples = {"nginx-a": {"n_sampled": 5, "last_sample_wall": 100.0}}
        self.assertFalse(metrics_published_after_samples(
            ["nginx-a"], {"nginx-a": "1970-01-01T00:01:55Z"}, samples, 17
        )[0])
        self.assertTrue(metrics_published_after_samples(
            ["nginx-a"], {"nginx-a": "1970-01-01T00:01:57Z"}, samples, 17
        )[0])

    def test_metric_payload_validation(self) -> None:
        pods = {"items": [{
            "metadata": {"name": "nginx-a", "labels": {"app": "nginx"}},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }]}
        metrics = {"items": [{
            "metadata": {"name": "nginx-a"},
            "timestamp": "2026-01-01T00:00:00Z",
            "containers": [{"usage": {"cpu": "1m"}}],
        }]}
        self.assertEqual(ready_pod_names(pods), ["nginx-a"])
        self.assertEqual(metric_versions(metrics, "resource"), {
            "nginx-a": "2026-01-01T00:00:00Z",
        })

    def test_metric_payload_ignores_other_namespace_pods(self) -> None:
        metrics = {"items": [
            {
                "metadata": {"name": "nginx-a"},
                "timestamp": "2026-01-01T00:00:00Z",
                "containers": [{"usage": {"cpu": "1m"}}],
            },
            {
                "metadata": {"name": "omniflow-daemon-a"},
                "timestamp": "2026-01-01T00:00:00Z",
                "containers": [{"usage": {"cpu": "1m"}}],
            },
        ]}
        self.assertEqual(
            metric_versions(metrics, "resource", {"nginx-a"}),
            {"nginx-a": "2026-01-01T00:00:00Z"},
        )

    def test_hpa_requires_stable_ready_deployment(self) -> None:
        hpa = {"status": {
            "conditions": [
                {"type": "ScalingActive", "status": "True"},
                {"type": "AbleToScale", "status": "True"},
            ],
            "currentMetrics": [{}],
            "currentReplicas": 1,
            "desiredReplicas": 1,
        }}
        deployment = {
            "metadata": {"generation": 2},
            "spec": {"replicas": 1},
            "status": {
                "observedGeneration": 2,
                "replicas": 1,
                "updatedReplicas": 1,
                "readyReplicas": 1,
                "availableReplicas": 1,
            },
        }
        self.assertTrue(stable_hpa(hpa, deployment)[0])
        deployment["status"]["readyReplicas"] = 0
        self.assertFalse(stable_hpa(hpa, deployment)[0])

    def test_job_renderer_sets_shape_seed_and_release_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job.yaml"
            render_job(K8S_DIR / "manifests" / "locust-job.yaml", output, "staircase", 99)
            text = output.read_text()
        self.assertIn("staircase", text)
        self.assertIn("'99'", text)
        self.assertIn("locust-armed", text)
        self.assertIn("locust-release", text)


class CollectorEventTest(unittest.TestCase):
    def test_latency_summary_uses_one_mean_per_repetition(self) -> None:
        collector = _load_collector()
        summary = collector._summarize_repetition_latencies(
            [
                {"scale_up_decision_latencies_s": [1.0, 3.0, 5.0]},
                {"scale_up_decision_latencies_s": [100.0]},
                {"scale_up_decision_latencies_s": []},
            ],
            "scale_up_decision_latencies_s",
        )
        self.assertEqual(summary["mean"], 51.5)
        self.assertEqual(summary["n"], 2)

    def test_replica_timeline_aligns_to_host_release_midpoint(self) -> None:
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "omniflow_replica_timeline.jsonl"
            path.write_text(json.dumps({
                "wall_ns": 105_000_000_000,
                "elapsed_s": 99,
                "ready": 1,
            }) + "\n")
            timeline = collector._load_replica_timeline(
                directory,
                "omniflow",
                {"locust_start_host_midpoint_ns": 100_000_000_000},
            )
        self.assertEqual(timeline[0]["elapsed_s"], 5.0)

    def test_recorded_events_override_static_phase_timing(self) -> None:
        collector = _load_collector()
        events = [
            {"event": "phase_start", "phase": "warmup", "direction": "stable",
             "observed_offset_s": 0.2, "scheduled_offset_s": 0},
            {"event": "phase_end", "phase": "warmup",
             "observed_offset_s": 60.4, "scheduled_offset_s": 60},
        ]
        phases = collector._phase_boundaries_from_events(events)
        self.assertEqual(phases[0]["start_s"], 0.2)
        self.assertEqual(phases[0]["end_s"], 60.4)

    def test_transition_windows_end_at_next_same_direction(self) -> None:
        collector = _load_collector()
        phases = [
            {"name": "up1", "start_s": 0, "end_s": 10, "direction": "up"},
            {"name": "hold", "start_s": 10, "end_s": 20, "direction": "stable"},
            {"name": "down1", "start_s": 20, "end_s": 30, "direction": "down"},
            {"name": "rest", "start_s": 30, "end_s": 40, "direction": "stable"},
            {"name": "up2", "start_s": 40, "end_s": 50, "direction": "up"},
        ]
        windows = collector._transition_windows(phases)
        self.assertEqual(
            [(window["name"], window["window_end_s"]) for window in windows],
            [("up1", 40), ("down1", 50), ("up2", 50)],
        )

    def test_scaling_events_stay_inside_direction_windows(self) -> None:
        collector = _load_collector()
        phases = [
            {"name": "up1", "start_s": 0, "end_s": 10, "direction": "up"},
            {"name": "hold", "start_s": 10, "end_s": 20, "direction": "stable"},
            {"name": "down1", "start_s": 20, "end_s": 30, "direction": "down"},
            {"name": "rest", "start_s": 30, "end_s": 40, "direction": "stable"},
            {"name": "up2", "start_s": 40, "end_s": 50, "direction": "up"},
        ]
        timeline = [
            {"elapsed_s": 0, "ready": 1},
            {"elapsed_s": 15, "ready": 2},
            {"elapsed_s": 35, "ready": 1},
            {"elapsed_s": 45, "ready": 2},
        ]
        cpu = [
            {
                "elapsed_s": second,
                "mean_cpu": 0.04 if second in {5, 45} else 0.01,
            }
            for second in range(50)
        ]
        result = collector._compute_scaling_accuracy(
            timeline, phases=phases, cpu_timeline=cpu, cpu_target=0.03
        )
        self.assertEqual(result["scale_up_latencies_s"], [15, 5])
        self.assertEqual(result["scale_down_latencies_s"], [15])
        self.assertEqual(result["scale_up_detection_rate"], 1.0)
        self.assertEqual(result["scale_down_detection_rate"], 1.0)
        self.assertEqual(result["under_provisioned_s"], 2)
        self.assertEqual(result["cpu_telemetry_coverage"], 1.0)

    def test_under_provisioning_requires_dense_cpu_coverage(self) -> None:
        collector = _load_collector()
        phases = [
            {"name": "up", "start_s": 0, "end_s": 100, "direction": "up"},
        ]
        timeline = [
            {"elapsed_s": 0, "ready": 1},
            {"elapsed_s": 100, "ready": 1},
        ]
        cpu = [
            {"elapsed_s": second, "mean_cpu": 0.04}
            for second in range(90)
        ]
        result = collector._compute_scaling_accuracy(
            timeline, phases=phases, cpu_timeline=cpu, cpu_target=0.03
        )
        self.assertEqual(result["cpu_telemetry_coverage"], 0.9)
        self.assertIsNone(result["under_provisioned_s"])

    def test_transition_eligibility_respects_hpa_replica_bounds(self) -> None:
        collector = _load_collector()
        phases = [
            {"name": "up", "start_s": 0, "end_s": 10, "direction": "up"},
            {"name": "down", "start_s": 10, "end_s": 20, "direction": "down"},
        ]
        at_max = collector._compute_scaling_accuracy(
            [
                {"elapsed_s": 0, "ready": 5},
                {"elapsed_s": 20, "ready": 5},
            ],
            phases=phases,
            min_replicas=1,
            max_replicas=5,
        )
        self.assertEqual(at_max["scale_up_transition_count"], 0)
        self.assertEqual(at_max["scale_down_transition_count"], 1)

        at_min = collector._compute_scaling_accuracy(
            [
                {"elapsed_s": 0, "ready": 1},
                {"elapsed_s": 20, "ready": 1},
            ],
            phases=phases,
            min_replicas=1,
            max_replicas=5,
        )
        self.assertEqual(at_min["scale_up_transition_count"], 1)
        self.assertEqual(at_min["scale_down_transition_count"], 0)

    def test_decision_eligibility_uses_desired_not_ready_replicas(self) -> None:
        collector = _load_collector()
        phases = [
            {"name": "up", "start_s": 0, "end_s": 10, "direction": "up"},
            {"name": "down", "start_s": 10, "end_s": 20, "direction": "down"},
        ]
        result = collector._compute_scaling_accuracy(
            [
                {"elapsed_s": 0, "ready": 1, "hpa_desired": 5},
                {"elapsed_s": 10, "ready": 5, "hpa_desired": 1},
                {"elapsed_s": 20, "ready": 1, "hpa_desired": 1},
            ],
            phases=phases,
            min_replicas=1,
            max_replicas=5,
        )
        self.assertEqual(result["scale_up_transition_count"], 1)
        self.assertEqual(result["scale_up_decision_transition_count"], 0)
        self.assertEqual(result["scale_down_transition_count"], 1)
        self.assertEqual(result["scale_down_decision_transition_count"], 0)

    def test_collector_reads_tracker_offset(self) -> None:
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "omniflow_daemon_node.jsonl"
            path.write_text(json.dumps({
                "event": "daemon_start",
                "cpu_tracker_offset": 0.2,
            }) + "\n")
            self.assertEqual(collector._load_tracker_offset(directory, "omniflow"), 0.2)

    def test_collector_reads_compressed_daemon_jsonl(self) -> None:
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "omniflow_daemon_node.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(json.dumps({
                    "event": "daemon_start",
                    "cpu_tracker_offset": 0.2,
                }) + "\n")
            self.assertEqual(collector._load_tracker_offset(directory, "omniflow"), 0.2)

    def test_locust_counts_are_parsed_from_count_columns(self) -> None:
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "omniflow_locust.log"
            path.write_text(
                "Type     Name  # reqs  # fails | Avg Min Max Med | req/s failures/s\n"
                "         Aggregated 23086 0(0.00%) | 3 1 36 3 | 41.42 0.00\n"
            )
            result = collector._parse_locust_stats(directory, "omniflow")
        self.assertEqual(result["total_requests"], 23086)
        self.assertEqual(result["total_failures"], 0)
        self.assertEqual(result["req_per_s"], 41.42)

    def test_daemon_analysis_filters_namespace_and_load_window(self) -> None:
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "omniflow_daemon_node.jsonl"
            records = [
                {"event": "track_start", "pod": "target", "pod_name": "nginx-a",
                 "namespace": "target-ns"},
                {"event": "track_start", "pod": "other", "pod_name": "nginx-b",
                 "namespace": "other-ns"},
            ]
            for wall in range(98, 107):
                records.append({"pod": "target", "wall": wall, "value": 0.01,
                                "sampled": True})
                records.append({"pod": "other", "wall": wall, "value": 0.02,
                                "sampled": True})
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            result = collector._analyse_daemon_logs(
                directory,
                "omniflow",
                namespace="target-ns",
                load_events=[
                    {"event": "load_start", "observed_epoch_ns": 100_000_000_000},
                    {"event": "load_end", "observed_offset_s": 4},
                ],
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pod"], "target")
        self.assertEqual(result[0]["n_readings"], 5)


class DaemonOffsetTest(unittest.TestCase):
    def test_trace_reset_and_snapshot_rotate_under_daemon_control(self) -> None:
        daemon = _load_daemon()
        with tempfile.TemporaryDirectory() as directory:
            daemon.TRACE_PATH = str(Path(directory) / "trace.jsonl")
            daemon.TRACE_STARTUP_PATH = str(Path(directory) / "startup.jsonl")
            Path(daemon.TRACE_STARTUP_PATH).write_text('{"event":"daemon_start"}\n')
            daemon._TRACE_FILE = open(daemon.TRACE_PATH, "a", encoding="utf-8")
            try:
                daemon._TRACE_FILE.write('{"value":1}\n')
                daemon._TRACE_FILE.flush()
                daemon._reset_trace_file()
                self.assertEqual(Path(daemon.TRACE_PATH).read_text(), "")

                daemon._TRACE_FILE.write('{"value":2}\n')
                daemon._TRACE_FILE.flush()
                snapshot = daemon._rotate_trace_snapshot()
                self.assertEqual(Path(snapshot).read_text(), '{"value":2}\n')
                self.assertEqual(Path(daemon.TRACE_PATH).read_text(), "")
            finally:
                if daemon._TRACE_FILE is not None:
                    daemon._TRACE_FILE.close()
                    daemon._TRACE_FILE = None

    def test_track_removal_cleans_all_prometheus_labels(self) -> None:
        daemon = _load_daemon()

        class Track:
            cgroup_path = "/sys/fs/cgroup/pod-id"
            name = "pod-id"
            pod_name = "nginx-a"
            pod_namespace = "omniflow-hpa-phased"
            stopped = False

            def stop(self):
                self.stopped = True

        track = Track()
        daemon._remove_track(track, emit_stop=False)
        self.assertTrue(track.stopped)
        for gauge in (
            daemon.g_cpu_mean, daemon.g_cpu_std, daemon.g_urgency,
            daemon.g_interval, daemon.g_ratio, daemon.g_latest,
        ):
            self.assertEqual(
                gauge.removed,
                [("nginx-a", "omniflow-hpa-phased")],
            )

    def test_tracker_is_shifted_but_published_cpu_is_raw(self) -> None:
        daemon = _load_daemon()
        from tracker.pipeline import PipelineConfig

        class Probe:
            def __init__(self):
                self.values = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.04])

            def read(self):
                return next(self.values)

        track = object.__new__(daemon._PodTrack)
        track.cgroup_path = "/tmp/pod"
        track.name = "pod"
        track.pod_name = ""
        track.pod_namespace = ""
        track.probe = Probe()
        config = PipelineConfig(min_interval=5, max_interval=30)
        track.full_tracker = config._make_tracker()
        track.poller = config._make_poller()
        track.tick = 0
        track.n_sampled = 0
        track.hpa_value = 0.0
        track.tracker_offset = 0.2
        track.last_sample_wall = None

        first = track.step()
        self.assertEqual(first["value"], 0.0)
        self.assertEqual(first["mean"], 0.0)
        self.assertAlmostEqual(track.poller.tracker.mean, 0.2)

        for _ in range(4):
            track.step()
        sampled = track.step()
        self.assertTrue(sampled["sampled"])
        self.assertEqual(sampled["value"], 0.04)
        self.assertEqual(sampled["hpa_value"], 0.04)
        self.assertIsNotNone(track.last_sample_wall)


if __name__ == "__main__":
    unittest.main()
