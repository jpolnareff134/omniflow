#!/usr/bin/env python3
"""OmniFlow K8s DaemonSet daemon.

Runs on every node, discovers pod cgroups via ``_discover_container_cgroups``,
and maintains a per-container ``_PodTrack`` (dense reference tracker + adaptive
scheduler). Exposes per-pod Prometheus metrics on ``/metrics`` and writes
structured JSON-line traces to a bounded pod-local volume for post-hoc analysis.
Trace reset and snapshot operations use a pod-local control endpoint so they are
atomic with respect to daemon writes.

Architecture mirrors ``FleetReader`` but does NOT use its blocking
``run()``; instead the scan loop runs in a background thread while the
main thread serves the prometheus_client HTTP server.

Prometheus metrics (labelled by ``pod``):
   - ``omniflow_cpu_mean``        adaptive tracker running mean
   - ``omniflow_cpu_std``         adaptive tracker running std
   - ``omniflow_cpu_latest``      latest adaptively sampled raw CPU reading
   - ``omniflow_urgency``         adaptive scheduler urgency
   - ``omniflow_interval``        current adaptive interval (steps)
  - ``omniflow_sample_ratio``    sampled / total

Environment variables:
  BASE_INTERVAL          seconds between read sweeps (default: 1.0)
  REFRESH_INTERVAL       seconds between cgroup re-scans (default: 10.0)
  DAEMON_PORT            Prometheus /metrics port (default: 9100)
  CGROUP_ROOT            cgroup v2 mount point (default: /sys/fs/cgroup)
  OMNIFLOW_CPU_TRACKER_OFFSET constant added only to tracker inputs (default: 0)
  OMNIFLOW_*             standard OmniFlow env-var overrides
"""

from __future__ import annotations

import json
import os
import re
import signal
import ssl
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from prometheus_client import Gauge, start_http_server  # noqa: E402
from probe.cgroup import (  # noqa: E402
    CgroupCpuProbe,
    _discover_container_cgroups,
    _K8S_CGROUP_GLOBS,
)
from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import WindowedTracker, AdaptivePoller  # noqa: E402

# ------------------------------
# Config
# ------------------------------

BASE_INTERVAL = float(os.environ.get("BASE_INTERVAL", "1.0"))
REFRESH_INTERVAL = float(os.environ.get("REFRESH_INTERVAL", "10.0"))
DAEMON_PORT = int(os.environ.get("DAEMON_PORT", "9100"))
CONTROL_PORT = int(os.environ.get("OMNIFLOW_CONTROL_PORT", "9101"))
CPU_TRACKER_OFFSET = float(os.environ.get("OMNIFLOW_CPU_TRACKER_OFFSET", "0.0"))
TRACE_PATH = os.environ.get("OMNIFLOW_TRACE_PATH", "/var/run/omniflow/trace.jsonl")
TRACE_STARTUP_PATH = os.environ.get(
    "OMNIFLOW_TRACE_STARTUP_PATH", "/var/run/omniflow/startup.jsonl"
)
TRACE_NAMESPACE_PREFIX = os.environ.get(
    "OMNIFLOW_TRACE_NAMESPACE_PREFIX", "omniflow-hpa-"
)
_TRACE_LOCK = threading.Lock()
_TRACE_FILE = None
_LAST_POD_MAP_ERROR = ""
_LAST_POD_MAP_ERROR_AT = 0.0
_TRACK_STATE_LOCK = threading.Lock()
_TRACK_STATE: dict[str, dict] = {}


def _write_json_line(
        record: dict, *, trace: bool = False, stdout: bool = True
) -> None:
    """Emit one JSON record to the selected diagnostics sinks."""
    line = json.dumps(record, separators=(",", ":"))
    if stdout:
        print(line, flush=True)
    if trace and _TRACE_FILE is not None:
        with _TRACE_LOCK:
            _TRACE_FILE.write(line + "\n")
            _TRACE_FILE.flush()


def _trace_namespace(namespace: str) -> bool:
    return bool(namespace) and namespace.startswith(TRACE_NAMESPACE_PREFIX)


def _reset_trace_file() -> None:
    """Atomically discard the active trace while coordinating with writers."""
    global _TRACE_FILE
    with _TRACE_LOCK:
        if _TRACE_FILE is not None:
            _TRACE_FILE.flush()
            _TRACE_FILE.close()
        _TRACE_FILE = open(TRACE_PATH, "w", encoding="utf-8", buffering=1)


def _rotate_trace_snapshot() -> str:
    """Rotate the active trace and return an immutable snapshot path."""
    global _TRACE_FILE
    snapshot = f"{TRACE_PATH}.snapshot.{time.time_ns()}"
    with _TRACE_LOCK:
        if _TRACE_FILE is not None:
            _TRACE_FILE.flush()
            _TRACE_FILE.close()
        if os.path.exists(TRACE_PATH):
            os.replace(TRACE_PATH, snapshot)
        else:
            open(snapshot, "w", encoding="utf-8").close()
        _TRACE_FILE = open(TRACE_PATH, "a", encoding="utf-8", buffering=1)
    return snapshot


def _track_state_snapshot() -> list[dict]:
    with _TRACK_STATE_LOCK:
        return [dict(item) for item in _TRACK_STATE.values()]


class _TraceControlHandler(BaseHTTPRequestHandler):
    """Pod-local atomic trace reset and snapshot interface."""

    def log_message(self, _format: str, *_args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/trace/reset":
            self.send_error(404)
            return
        _reset_trace_file()
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/tracks":
            payload = json.dumps(
                {"tracks": _track_state_snapshot()}, separators=(",", ":")
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path != "/trace/snapshot":
            self.send_error(404)
            return
        snapshot = _rotate_trace_snapshot()
        try:
            paths = [TRACE_STARTUP_PATH, snapshot]
            length = sum(os.path.getsize(path) for path in paths if os.path.exists(path))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            for path in paths:
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self.wfile.write(chunk)
        finally:
            try:
                os.unlink(snapshot)
            except FileNotFoundError:
                pass

# ------------------------------
# Prometheus metrics
# ------------------------------

g_cpu_mean = Gauge("omniflow_cpu_mean", "Adaptive tracker CPU running mean", ["pod", "namespace"])
g_cpu_std = Gauge("omniflow_cpu_std", "Adaptive tracker CPU running std", ["pod", "namespace"])
g_urgency = Gauge("omniflow_urgency", "Adaptive scheduler urgency", ["pod", "namespace"])
g_interval = Gauge("omniflow_interval", "Current adaptive interval (steps)", ["pod", "namespace"])
g_ratio = Gauge("omniflow_sample_ratio", "Cumulative sample ratio", ["pod", "namespace"])
g_latest = Gauge("omniflow_cpu_latest", "Latest adaptively sampled raw CPU reading", ["pod", "namespace"])


# ------------------------------
# Per-container state
# ------------------------------

class _PodTrack:
    """Mutable state for one pod's dual-tracking."""
    __slots__ = (
        "cgroup_path", "name", "pod_name", "pod_namespace", "probe",
        "full_tracker", "poller",
        "tick", "n_sampled", "hpa_value", "tracker_offset",
        "last_sample_wall",
    )

    def __init__(self, cgroup_path: str, config: PipelineConfig,
                 pod_name: str = "", pod_namespace: str = "",
                 tracker_offset: float = CPU_TRACKER_OFFSET) -> None:
        self.cgroup_path = cgroup_path
        self.name = os.path.basename(cgroup_path.rstrip("/"))
        self.pod_name = pod_name
        self.pod_namespace = pod_namespace
        self.probe = CgroupCpuProbe(cgroup_path=cgroup_path)
        self.full_tracker: WindowedTracker = config._make_tracker()
        self.poller: AdaptivePoller = config._make_poller()
        self.tick: int = 0
        self.n_sampled: int = 0
        self.hpa_value: float = 0.0
        self.tracker_offset = tracker_offset
        self.last_sample_wall: float | None = None

    def start(self) -> None:
        self.probe.start()

    def stop(self) -> None:
        self.probe.stop()

    def step(self) -> dict | None:
        """Read CPU, feed both tracks, return a log record."""
        try:
            value = self.probe.read()
        except (FileNotFoundError, RuntimeError):
            return None

        # Keep the HPA signal raw while giving the tracker a nonzero origin for
        # near-idle CPU values. A constant shift preserves all differences.
        tracker_value = value + self.tracker_offset
        self.full_tracker.update(tracker_value)
        # Adaptive scheduler path
        pr = self.poller.step(tracker_value, time_index=self.tick)
        self.tick += 1
        if pr.sampled:
            self.n_sampled += 1
            self.hpa_value = value
            self.last_sample_wall = time.time()

        # Update Prometheus gauges (only when pod identity is known)
        if self.pod_name and self.pod_namespace:
            lbl = {"pod": self.pod_name, "namespace": self.pod_namespace}
            g_cpu_mean.labels(**lbl).set(self.poller.tracker.mean - self.tracker_offset)
            g_cpu_std.labels(**lbl).set(self.poller.tracker.std)
            g_urgency.labels(**lbl).set(pr.urgency)
            g_interval.labels(**lbl).set(pr.interval)
            ratio = self.n_sampled / self.tick if self.tick > 0 else 1.0
            g_ratio.labels(**lbl).set(ratio)
            if pr.sampled:
                # Hold the most recent raw sample until OmniFlow samples again.
                g_latest.labels(**lbl).set(self.hpa_value)
        return {
            "pod": self.name,
            "pod_name": self.pod_name,
            "namespace": self.pod_namespace,
            "t": self.tick,
            "value": round(value, 4),
            "mean": round(self.poller.tracker.mean - self.tracker_offset, 4),
            "std": round(self.poller.tracker.std, 4),
            "full_mean": round(self.full_tracker.mean - self.tracker_offset, 4),
            "full_std": round(self.full_tracker.std, 4),
            "urgency": round(pr.urgency, 4),
            "interval": pr.interval,
            "sampled": pr.sampled,
            "published": pr.sampled,
            "hpa_value": round(self.hpa_value, 4),
            "wall": time.time(),
        }


# ------------------------------
# K8s pod UID -> (name, namespace) lookup
# ------------------------------

def _get_pod_map() -> dict[str, tuple[str, str]]:
    """Return {uid: (pod_name, namespace)} for all pods via in-cluster API."""
    global _LAST_POD_MAP_ERROR, _LAST_POD_MAP_ERROR_AT
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    if not os.path.exists(token_path):
        return {}
    try:
        with open(token_path) as f:
            token = f.read().strip()
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        url = f"https://{host}:{port}/api/v1/pods"
        ctx = ssl.create_default_context(cafile=ca_path)
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read())
        return {
            item["metadata"]["uid"]: (
                item["metadata"]["name"],
                item["metadata"]["namespace"],
            )
            for item in data.get("items", [])
        }
    except Exception as exc:
        error = str(exc)
        now = time.monotonic()
        if error != _LAST_POD_MAP_ERROR or now - _LAST_POD_MAP_ERROR_AT >= 60:
            _write_json_line(
                {"event": "pod_map_error", "error": error, "wall": time.time()}
            )
            _LAST_POD_MAP_ERROR = error
            _LAST_POD_MAP_ERROR_AT = now
        return {}


def _pod_uid_from_cgroup_name(name: str) -> str:
    """Extract a Kubernetes pod UID from cgroupfs or systemd cgroup names."""
    match = re.search(r"pod([0-9a-f_-]+)(?:\.slice)?$", name)
    if not match:
        return ""
    return match.group(1).replace("_", "-")


# ------------------------------
# Fleet scan loop
# ------------------------------

def _remove_track(track: _PodTrack, *, emit_stop: bool = True) -> None:
    """Stop one track and remove every exported Prometheus labelset."""
    track.stop()
    with _TRACK_STATE_LOCK:
        _TRACK_STATE.pop(track.cgroup_path, None)
    if track.pod_name and track.pod_namespace:
        for gauge in (
            g_cpu_mean, g_cpu_std, g_urgency, g_interval, g_ratio, g_latest,
        ):
            try:
                gauge.remove(track.pod_name, track.pod_namespace)
            except KeyError:
                pass
    if emit_stop:
        stop_record = {
            "event": "track_stop",
            "pod": track.name,
            "pod_name": track.pod_name,
            "namespace": track.pod_namespace,
            "wall": time.time(),
        }
        _write_json_line(
            stop_record,
            trace=_trace_namespace(track.pod_namespace),
            stdout=_trace_namespace(track.pod_namespace),
        )

def _scan_loop(
        config: PipelineConfig,
        stop: threading.Event,
) -> None:
    """Background thread: discover cgroups, track pods, log readings."""
    tracks: dict[str, _PodTrack] = {}
    last_refresh = 0.0
    pod_map: dict[str, tuple[str, str]] = {}

    while not stop.is_set():
        now = time.monotonic()

        # --- periodic re-scan ---
        if now - last_refresh >= REFRESH_INTERVAL:
            pod_map = _get_pod_map()
            current = set(_discover_container_cgroups(_K8S_CGROUP_GLOBS))
            # Start new pods
            for path in current - set(tracks):
                cgroup_id = os.path.basename(path.rstrip("/"))
                uid = _pod_uid_from_cgroup_name(cgroup_id)
                pod_name, pod_ns = pod_map.get(uid, ("", ""))
                # The daemon has cluster-wide visibility only to resolve pod
                # identities. Never retain or export data outside experiment
                # namespaces.
                if not _trace_namespace(pod_ns):
                    continue
                track = _PodTrack(
                    path,
                    config,
                    pod_name=pod_name,
                    pod_namespace=pod_ns,
                    tracker_offset=CPU_TRACKER_OFFSET,
                )
                try:
                    track.start()
                    tracks[path] = track
                    with _TRACK_STATE_LOCK:
                        _TRACK_STATE[path] = {
                            "cgroup_path": path,
                            "pod_name": pod_name,
                            "namespace": pod_ns,
                            "tick": track.tick,
                            "n_sampled": track.n_sampled,
                            "last_sample_wall": track.last_sample_wall,
                        }
                    start_record = {"event": "track_start", "pod": track.name,
                                    "pod_name": pod_name, "namespace": pod_ns,
                                    "wall": time.time()}
                    _write_json_line(start_record, trace=True)
                except Exception:
                    pass
            # Prune gone pods
            for path in set(tracks) - current:
                track = tracks.pop(path)
                _remove_track(track)
            last_refresh = now

        # --- read all tracked pods ---
        dead: list[str] = []
        for path, track in tracks.items():
            record = track.step()
            if record is None:
                dead.append(path)
                continue
            with _TRACK_STATE_LOCK:
                _TRACK_STATE[path] = {
                    "cgroup_path": path,
                    "pod_name": track.pod_name,
                    "namespace": track.pod_namespace,
                    "tick": track.tick,
                    "n_sampled": track.n_sampled,
                    "last_sample_wall": track.last_sample_wall,
                }
            _write_json_line(
                record,
                trace=_trace_namespace(track.pod_namespace),
                stdout=False,
            )

        for path in dead:
            track = tracks.pop(path)
            _remove_track(track)

        # --- sleep until next sweep ---
        elapsed = time.monotonic() - now
        remaining = BASE_INTERVAL - elapsed
        if remaining > 0:
            stop.wait(remaining)

    # Cleanup
    for track in tracks.values():
        _remove_track(track, emit_stop=False)


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    global _TRACE_FILE
    config = PipelineConfig()
    startup_info = {
        "event": "daemon_start",
        "port": DAEMON_PORT,
        "control_port": CONTROL_PORT,
        "base_interval": BASE_INTERVAL,
        "refresh_interval": REFRESH_INTERVAL,
        "cpu_tracker_offset": CPU_TRACKER_OFFSET,
        "trace_path": TRACE_PATH,
        "trace_namespace_prefix": TRACE_NAMESPACE_PREFIX,
        "config": config.to_dict(),
        "wall": time.time(),
    }
    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    startup_tmp = TRACE_STARTUP_PATH + ".tmp"
    with open(startup_tmp, "w", encoding="utf-8") as startup_file:
        startup_file.write(json.dumps(startup_info, separators=(",", ":")) + "\n")
        startup_file.flush()
        os.fsync(startup_file.fileno())
    os.replace(startup_tmp, TRACE_STARTUP_PATH)
    _TRACE_FILE = open(TRACE_PATH, "a", encoding="utf-8", buffering=1)
    _write_json_line(startup_info)
    print(f"OmniFlow daemon starting: port={DAEMON_PORT} "
          f"base_interval={BASE_INTERVAL}s "
          f"min_interval={config.min_interval} "
          f"max_interval={config.max_interval} "
          f"cpu_tracker_offset={CPU_TRACKER_OFFSET}",
          file=sys.stderr)

    start_http_server(DAEMON_PORT)
    print(f"Prometheus metrics on :{DAEMON_PORT}/metrics", file=sys.stderr)
    control_server = ThreadingHTTPServer(
        ("127.0.0.1", CONTROL_PORT), _TraceControlHandler
    )
    control_thread = threading.Thread(
        target=control_server.serve_forever, daemon=True
    )
    control_thread.start()
    print(
        f"Trace control on 127.0.0.1:{CONTROL_PORT}", file=sys.stderr
    )

    stop = threading.Event()

    def _sigterm(_sig: int, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _sigterm)

    t = threading.Thread(target=_scan_loop, args=(config, stop), daemon=True)
    t.start()

    try:
        while not stop.is_set():
            stop.wait(1)
    except KeyboardInterrupt:
        pass

    stop.set()
    t.join(timeout=10)
    control_server.shutdown()
    control_server.server_close()
    control_thread.join(timeout=10)
    if _TRACE_FILE is not None:
        with _TRACE_LOCK:
            _TRACE_FILE.flush()
            _TRACE_FILE.close()
        _TRACE_FILE = None
    print("Daemon stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
