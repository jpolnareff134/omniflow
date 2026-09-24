#!/usr/bin/env python3
"""OmniFlow I/O Syscall DaemonSet daemon 

Runs on every Kubernetes node and monitors the aggregate rate of I/O
system calls (``write`` + ``pwrite64`` + ``fsync``) using an eBPF
tracepoint attached via BCC.

Applies OmniFlow's adaptive scheduler to decide when to read the eBPF map,
reducing userspace collection overhead during calm periods while staying
responsive during I/O bursts.

Exposes per-node Prometheus metrics on ``/metrics``:

  omniflow_io_syscall_rate        – latest raw I/O rate (counts per poll)
  omniflow_io_mean                – tracker running mean
  omniflow_io_std                 – tracker running std
    omniflow_io_urgency             – adaptive scheduler urgency [0..1]
    omniflow_io_interval            – current adaptive interval (eBPF polls)
  omniflow_io_sample_ratio        – cumulative sampled / total ratio
  omniflow_io_anomaly_total       – cumulative outlier count
  omniflow_io_drift_total         – cumulative drift-reset count

Environment variables
---------------------
  SYSCALLS            Comma-separated syscall names to aggregate
                      (default: ``write,pwrite64,fsync``)
  BASE_INTERVAL       Seconds between eBPF map reads (default: 0.2)
  DAEMON_PORT         Prometheus /metrics port (default: 9101)
  NODE_NAME           Node label injected by k8s downward API
  OMNIFLOW_*          Standard OmniFlow parameter overrides
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass

# Allow importing from src/ (in the Docker image, WORKDIR=/app and
# OmniFlow source is under /app/src/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from prometheus_client import Gauge, Counter, start_http_server  # noqa: E402
from probe.syscall_counter import SyscallCounter  # noqa: E402
from tracker.pipeline import PipelineConfig  # noqa: E402
from tracker.windowed import AdaptivePoller, Status  # noqa: E402

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

_SYSCALLS_ENV = os.environ.get("SYSCALLS", "write,pwrite64,fsync")
SYSCALLS = [s.strip() for s in _SYSCALLS_ENV.split(",") if s.strip()]
BASE_INTERVAL = float(os.environ.get("BASE_INTERVAL", "0.2"))
DAEMON_PORT = int(os.environ.get("DAEMON_PORT", "9101"))
NODE_NAME = os.environ.get("NODE_NAME", "unknown")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


EMIT_UNSAMPLED_LOGS = _env_bool("EMIT_UNSAMPLED_LOGS", False)
UPDATE_PROMETHEUS_ON_UNSAMPLED = _env_bool("UPDATE_PROMETHEUS_ON_UNSAMPLED", False)
PROMETHEUS_UPDATE_EVERY_SAMPLED = max(1, int(os.environ.get("PROMETHEUS_UPDATE_EVERY_SAMPLED", "1")))
CPU_ACCOUNTING_SOURCE = os.environ.get("CPU_ACCOUNTING_SOURCE", "auto").strip().lower()
CPU_ACCOUNTING_INTERVAL_TICKS = max(1, int(os.environ.get("CPU_ACCOUNTING_INTERVAL_TICKS", "10")))
DETAIL_TIMING_EVERY_SAMPLED = max(0, int(os.environ.get("DETAIL_TIMING_EVERY_SAMPLED", "16")))


@dataclass
class CpuSnapshot:
    total_cpu_s: float
    user_cpu_s: float | None
    system_cpu_s: float | None


class CpuUsageTracker:
    def __init__(self, source: str) -> None:
        self._source_name = source
        self._cgroup_cpu_stat_path = self._resolve_cgroup_cpu_stat_path()
        self._clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        self._proc_stat_path = "/proc/self/stat"
        self._last_snapshot = self._read_snapshot(source)
        self._last_wall = time.monotonic()
        self._cpu_pct = 0.0
        self._cpu_user_total = self._last_snapshot.user_cpu_s or 0.0
        self._cpu_system_total = self._last_snapshot.system_cpu_s or 0.0
        self._cpu_total = self._last_snapshot.total_cpu_s

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def cpu_pct(self) -> float:
        return self._cpu_pct

    @property
    def cpu_user_total(self) -> float:
        return self._cpu_user_total

    @property
    def cpu_system_total(self) -> float:
        return self._cpu_system_total

    @property
    def cpu_total(self) -> float:
        return self._cpu_total

    def maybe_refresh(self, force: bool = False) -> bool:
        if not force:
            return False
        now_wall = time.monotonic()
        snapshot = self._read_snapshot(self._source_name)
        wall_dt = max(now_wall - self._last_wall, 1e-9)
        cpu_dt = max(snapshot.total_cpu_s - self._last_snapshot.total_cpu_s, 0.0)
        self._cpu_pct = max(0.0, (cpu_dt / wall_dt) * 100.0)
        self._cpu_total = snapshot.total_cpu_s
        self._cpu_user_total = snapshot.user_cpu_s if snapshot.user_cpu_s is not None else max(snapshot.total_cpu_s, 0.0)
        self._cpu_system_total = snapshot.system_cpu_s if snapshot.system_cpu_s is not None else max(snapshot.total_cpu_s - self._cpu_user_total, 0.0)
        self._last_snapshot = snapshot
        self._last_wall = now_wall
        return True

    def _read_snapshot(self, requested_source: str) -> CpuSnapshot:
        candidates = [requested_source]
        if requested_source == "auto":
            candidates = ["cgroup", "proc"]

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                if candidate == "cgroup":
                    snapshot = self._read_cgroup_snapshot()
                elif candidate == "proc":
                    snapshot = self._read_proc_snapshot()
                else:
                    raise ValueError(f"Unsupported CPU accounting source: {candidate}")
                self._source_name = candidate
                return snapshot
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
                continue

        raise RuntimeError(f"Unable to initialize CPU accounting source {requested_source!r}: {last_error}")

    def _resolve_cgroup_cpu_stat_path(self) -> str | None:
        try:
            with open("/proc/self/cgroup") as handle:
                for line in handle:
                    parts = line.strip().split(":", 2)
                    if len(parts) == 3 and parts[0] == "0":
                        path = os.path.join("/sys/fs/cgroup", parts[2].lstrip("/"), "cpu.stat")
                        if os.path.exists(path):
                            return path
        except OSError:
            return None
        return None

    def _read_cgroup_snapshot(self) -> CpuSnapshot:
        if not self._cgroup_cpu_stat_path:
            raise FileNotFoundError("cpu.stat path unavailable for current cgroup")

        values: dict[str, int] = {}
        with open(self._cgroup_cpu_stat_path) as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        values[parts[0]] = int(parts[1])
                    except ValueError:
                        continue

        usage_usec = values.get("usage_usec")
        if usage_usec is None:
            raise RuntimeError(f"usage_usec not found in {self._cgroup_cpu_stat_path}")
        user_usec = values.get("user_usec")
        system_usec = values.get("system_usec")
        return CpuSnapshot(
            total_cpu_s=usage_usec / 1_000_000.0,
            user_cpu_s=None if user_usec is None else user_usec / 1_000_000.0,
            system_cpu_s=None if system_usec is None else system_usec / 1_000_000.0,
        )

    def _read_proc_snapshot(self) -> CpuSnapshot:
        with open(self._proc_stat_path) as handle:
            parts = handle.read().split()

        if len(parts) < 15:
            raise RuntimeError(f"Unexpected /proc/self/stat format in {self._proc_stat_path}")

        user_ticks = int(parts[13])
        system_ticks = int(parts[14])
        return CpuSnapshot(
            total_cpu_s=(user_ticks + system_ticks) / self._clk_tck,
            user_cpu_s=user_ticks / self._clk_tck,
            system_cpu_s=system_ticks / self._clk_tck,
        )

# ─────────────────────────────────────────────
# Prometheus metrics (labelled by node)
# ─────────────────────────────────────────────

_LABELS = ["node", "syscalls"]

g_rate = Gauge("omniflow_io_syscall_rate",
               "Latest raw I/O syscall rate (counts per poll interval)",
               _LABELS)
g_mean = Gauge("omniflow_io_mean",
               "OmniFlow tracker running mean of I/O syscall rate",
               _LABELS)
g_std = Gauge("omniflow_io_std",
              "OmniFlow tracker running std of I/O syscall rate",
              _LABELS)
g_urgency = Gauge("omniflow_io_urgency",
                  "OmniFlow adaptive scheduler urgency [0..1]",
                  _LABELS)
g_interval = Gauge("omniflow_io_interval",
                   "Current adaptive interval (number of eBPF polls)",
                   _LABELS)
g_sample_ratio = Gauge("omniflow_io_sample_ratio",
                       "Cumulative fraction of eBPF polls consumed by the tracker",
                       _LABELS)
g_read_seconds = Gauge("omniflow_io_read_seconds",
                       "Latest probe read duration in seconds",
                       _LABELS)
g_tracker_seconds = Gauge("omniflow_io_tracker_seconds",
                          "Latest full-tracker update duration in seconds",
                          _LABELS)
g_adaptive_seconds = Gauge("omniflow_io_adaptive_seconds",
                           "Latest adaptive-step duration in seconds",
                           _LABELS)
g_metrics_seconds = Gauge("omniflow_io_metrics_seconds",
                          "Latest Prometheus metric update duration in seconds",
                          _LABELS)
g_emit_seconds = Gauge("omniflow_io_emit_seconds",
                       "Latest structured-log emission duration in seconds",
                       _LABELS)
g_active_loop_seconds = Gauge("omniflow_io_active_loop_seconds",
                              "Latest active monitoring loop duration in seconds",
                              _LABELS)
g_cpu_pct = Gauge("omniflow_io_monitor_cpu_pct",
                  "Latest daemon CPU percentage over the last interval",
                  _LABELS)
g_cpu_user_total = Gauge("omniflow_io_monitor_user_cpu_seconds_total",
                         "Cumulative daemon user CPU seconds",
                         _LABELS)
g_cpu_system_total = Gauge("omniflow_io_monitor_system_cpu_seconds_total",
                           "Cumulative daemon system CPU seconds",
                           _LABELS)
c_outlier = Counter("omniflow_io_anomaly_total",
                    "Cumulative anomaly detections",
                    _LABELS)
c_drift = Counter("omniflow_io_drift_total",
                  "Cumulative drift-reset events",
                  _LABELS)


# ─────────────────────────────────────────────
# Main daemon loop
# ─────────────────────────────────────────────

def _build_pipeline() -> AdaptivePoller:
    cfg = PipelineConfig()
    return cfg._make_poller()


def run() -> None:
    syscall_label = "+".join(SYSCALLS)
    lbl = {"node": NODE_NAME, "syscalls": syscall_label}
    metrics = {
        "rate": g_rate.labels(**lbl),
        "mean": g_mean.labels(**lbl),
        "std": g_std.labels(**lbl),
        "urgency": g_urgency.labels(**lbl),
        "interval": g_interval.labels(**lbl),
        "sample_ratio": g_sample_ratio.labels(**lbl),
        "read_seconds": g_read_seconds.labels(**lbl),
        "tracker_seconds": g_tracker_seconds.labels(**lbl),
        "adaptive_seconds": g_adaptive_seconds.labels(**lbl),
        "metrics_seconds": g_metrics_seconds.labels(**lbl),
        "emit_seconds": g_emit_seconds.labels(**lbl),
        "active_loop_seconds": g_active_loop_seconds.labels(**lbl),
        "cpu_pct": g_cpu_pct.labels(**lbl),
        "cpu_user_total": g_cpu_user_total.labels(**lbl),
        "cpu_system_total": g_cpu_system_total.labels(**lbl),
    }
    outlier_counter = c_outlier.labels(**lbl)
    drift_counter = c_drift.labels(**lbl)

    print(f"[omniflow-io] node={NODE_NAME}  syscalls={syscall_label}  "
          f"base_interval={BASE_INTERVAL}s  port={DAEMON_PORT}",
          flush=True)

    # Start Prometheus HTTP server
    start_http_server(DAEMON_PORT)
    print(f"[omniflow-io] Prometheus metrics on :{DAEMON_PORT}/metrics",
          flush=True)

    # Set up eBPF probe
    probe = SyscallCounter(
        syscalls=SYSCALLS,
        poll_interval=BASE_INTERVAL,
    )
    probe.start()
    print(f"[omniflow-io] eBPF probe attached for: {syscall_label}", flush=True)

    # OmniFlow adaptive pipeline. In live mode the interval now controls
    # the actual probe.read() cadence instead of only post-hoc evaluation.
    poller = _build_pipeline()
    cpu_tracker = CpuUsageTracker(CPU_ACCOUNTING_SOURCE)
    print(
        f"[omniflow-io] cpu_accounting_source={cpu_tracker.source_name}  "
        f"cpu_snapshot_every={CPU_ACCOUNTING_INTERVAL_TICKS} ticks  "
        f"prometheus_update_every_sampled={PROMETHEUS_UPDATE_EVERY_SAMPLED}  "
        f"detail_timing_every_sampled={DETAIL_TIMING_EVERY_SAMPLED}",
        flush=True,
    )

    tick = 0
    n_sampled = 0
    ticks_since_sample = 0
    sampled_ticks = 0
    last_value = 0.0
    _running = True
    last_emit_ns = 0
    last_active_ns = 0
    last_cpu_pct = 0.0
    last_cpu_user_total = cpu_tracker.cpu_user_total
    last_cpu_system_total = cpu_tracker.cpu_system_total
    last_cpu_total = cpu_tracker.cpu_total
    prometheus_updates = 0

    def _shutdown(sig, frame):
        nonlocal _running
        print("\n[omniflow-io] Shutting down …", flush=True)
        _running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while _running:
            time.sleep(BASE_INTERVAL)
            tick += 1
            ticks_since_sample += 1

            current_interval = poller.interval
            should_sample = (not poller.tracker._initialised) or (
                    ticks_since_sample >= current_interval
            )

            value = None
            read_ns = 0
            tracker_ns = 0
            adaptive_ns = 0
            sample_window_ticks = None
            tick_result = None
            sampled = False
            status_name = None
            timing_measured = False
            active_t0 = None

            if should_sample:
                sampled = True
                sampled_ticks += 1
                sample_window_ticks = ticks_since_sample
                timing_measured = (
                    DETAIL_TIMING_EVERY_SAMPLED > 0
                    and sampled_ticks % DETAIL_TIMING_EVERY_SAMPLED == 0
                )
                if timing_measured:
                    active_t0 = time.perf_counter_ns()

                # Read the eBPF map only when the adaptive schedule asks for it.
                # The raw count is normalised by the number of skipped base ticks
                # so the reported value stays in counts per BASE_INTERVAL.
                read_t0 = time.perf_counter_ns() if timing_measured else None
                raw_value = probe.read()
                if timing_measured and read_t0 is not None:
                    read_ns = time.perf_counter_ns() - read_t0
                value = raw_value / max(1, sample_window_ticks)
                last_value = value

                tracker_t0 = time.perf_counter_ns() if timing_measured else None
                tick_result = poller.tracker.update(value)
                if timing_measured and tracker_t0 is not None:
                    tracker_ns = time.perf_counter_ns() - tracker_t0

                poller._steps_since_sample = 0
                poller._recent_statuses.append(tick_result.status)

                adaptive_t0 = time.perf_counter_ns() if timing_measured else None
                poller._update_urgency(tick_result)
                if timing_measured and adaptive_t0 is not None:
                    adaptive_ns = time.perf_counter_ns() - adaptive_t0

                n_sampled += 1
                ticks_since_sample = 0
                status_name = tick_result.status.name

                if tick_result.status == Status.OUTLIER:
                    outlier_counter.inc()
                elif tick_result.status == Status.DRIFT_RESET:
                    drift_counter.inc()
            else:
                poller._steps_since_sample = ticks_since_sample

            cpu_refreshed = cpu_tracker.maybe_refresh(
                force=(tick == 1 or tick % CPU_ACCOUNTING_INTERVAL_TICKS == 0 or sampled)
            )
            if cpu_refreshed:
                last_cpu_pct = cpu_tracker.cpu_pct
                last_cpu_user_total = cpu_tracker.cpu_user_total
                last_cpu_system_total = cpu_tracker.cpu_system_total
                last_cpu_total = cpu_tracker.cpu_total

            publish_metrics = False
            if sampled:
                publish_metrics = (
                    sampled_ticks == 1
                    or sampled_ticks % PROMETHEUS_UPDATE_EVERY_SAMPLED == 0
                )
            elif UPDATE_PROMETHEUS_ON_UNSAMPLED:
                publish_metrics = True
            metrics_ns = 0
            if publish_metrics:
                prometheus_updates += 1
                metrics_t0 = time.perf_counter_ns() if timing_measured else None
                metrics["rate"].set(last_value)
                metrics["mean"].set(poller.tracker.mean)
                metrics["std"].set(poller.tracker.std)
                metrics["urgency"].set(poller._urgency)
                metrics["interval"].set(current_interval)
                metrics["sample_ratio"].set(n_sampled / max(1, tick))
                metrics["read_seconds"].set(read_ns / 1e9)
                metrics["tracker_seconds"].set(tracker_ns / 1e9)
                metrics["adaptive_seconds"].set(adaptive_ns / 1e9)
                metrics["cpu_pct"].set(last_cpu_pct)
                metrics["cpu_user_total"].set(last_cpu_user_total)
                metrics["cpu_system_total"].set(last_cpu_system_total)
                if timing_measured and metrics_t0 is not None:
                    metrics_ns = time.perf_counter_ns() - metrics_t0
                    metrics["metrics_seconds"].set(metrics_ns / 1e9)

            emit_row = sampled or EMIT_UNSAMPLED_LOGS
            if emit_row:
                # Keep the structured trace sparse in adaptive mode so unsampled
                # ticks do not pay full stdout/logging overhead.
                payload = {
                    "t": tick,
                    "value": None if value is None else round(value, 2),
                    "mean": round(poller.tracker.mean, 2),
                    "std": round(poller.tracker.std, 2),
                    "urgency": round(poller._urgency, 4),
                    "interval": current_interval,
                    "sampled": sampled,
                    "status": status_name,
                    "sample_window_ticks": sample_window_ticks,
                    "sample_ratio_live": round(n_sampled / max(1, tick), 6),
                    "timing_measured": timing_measured,
                    "cpu_accounting_source": cpu_tracker.source_name,
                    "cpu_snapshot_interval_ticks": CPU_ACCOUNTING_INTERVAL_TICKS,
                    "prometheus_update_every_sampled": PROMETHEUS_UPDATE_EVERY_SAMPLED,
                    "prometheus_updated": publish_metrics,
                    "prometheus_updates_total": prometheus_updates,
                    "detail_timing_every_sampled": DETAIL_TIMING_EVERY_SAMPLED,
                    "read_overhead_us": None if not timing_measured else round(read_ns / 1_000, 3),
                    "tracker_overhead_us": None if not timing_measured else round(tracker_ns / 1_000, 3),
                    "adaptive_overhead_us": None if not timing_measured else round(adaptive_ns / 1_000, 3),
                    "metrics_overhead_us": None if not timing_measured else round(metrics_ns / 1_000, 3),
                    "emit_overhead_us": None if not timing_measured else round(last_emit_ns / 1_000, 3),
                    "active_loop_us": None if not timing_measured else round(last_active_ns / 1_000, 3),
                    "monitor_cpu_pct": round(last_cpu_pct, 4),
                    "monitor_user_cpu_s_total": round(last_cpu_user_total, 6),
                    "monitor_system_cpu_s_total": round(last_cpu_system_total, 6),
                    "monitor_total_cpu_s_total": round(last_cpu_total, 6),
                }
                emit_t0 = time.perf_counter_ns() if timing_measured else None
                print(json.dumps(payload, separators=(",", ":")), flush=True)
                if timing_measured and emit_t0 is not None and active_t0 is not None:
                    emit_ns = time.perf_counter_ns() - emit_t0
                    active_ns = time.perf_counter_ns() - active_t0
                    last_emit_ns = emit_ns
                    last_active_ns = active_ns
                    if publish_metrics:
                        metrics["emit_seconds"].set(emit_ns / 1e9)
                        metrics["active_loop_seconds"].set(active_ns / 1e9)

    finally:
        probe.stop()
        print("[omniflow-io] Probe detached.  Bye.", flush=True)


if __name__ == "__main__":
    run()
