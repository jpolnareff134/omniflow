"""
Overhead measurement utilities.

Measures end-to-end workload cost across different monitoring states.

This module also provides a phase-based end-to-end overhead runner that
compares the same workload under multiple monitoring states:

- baseline: no probe attached
- attached_idle: probe attached, never read
- attached_fixed: probe attached, polled at a fixed interval
- attached_adaptive: probe attached, polled by the live adaptive loop
"""

from __future__ import annotations

import json
import resource
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from probe.base import Probe
from tracker.pipeline import PipelineConfig

# ------------------------------
# Rate generation helpers
# ------------------------------

# "Nice" tick values we snap to when generating intermediate rates
_NICE = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60]


def nice_rates(
        min_interval: float,
        max_interval: float,
        base_interval: float = 1.0,
        max_steps: int = 8,
) -> list[float]:
    """Generate a list of human-friendly fixed comparison intervals.

    The intervals are in **seconds** (real wall-clock).  Since the adaptive
    poller's ``min_interval`` / ``max_interval`` are in *steps*, they are
    multiplied by *base_interval* to get seconds.

    The resulting list always includes the exact min and max endpoints,
    plus intermediate values snapped to human-friendly numbers. At most
    *max_steps* values are returned.
    """
    lo = min_interval * base_interval
    hi = max_interval * base_interval
    if lo > hi:
        lo, hi = hi, lo

    # Collect candidates from the nice table that fall in [lo, hi]
    candidates = sorted({lo, hi} | {n for n in _NICE if lo < n < hi})

    # If too many, thin out by keeping evenly-spaced picks + endpoints
    if len(candidates) > max_steps:
        # Always keep first and last; pick evenly from the middle
        inner = candidates[1:-1]
        n_pick = max_steps - 2
        step = max(1, len(inner) // n_pick)
        picked = inner[::step][:n_pick]
        candidates = sorted({candidates[0], candidates[-1]} | set(picked))

    return candidates


@dataclass
class WorkloadRunResult:
    """Result returned by a workload runner."""

    exit_code: int
    operations: int | None = None
    wall_s: float | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class EndToEndPhaseReport:
    """End-to-end overhead report for a single monitoring phase."""

    label: str
    monitoring_mode: str
    repeat_index: int
    interval_s: float | None
    wall_s: float
    exit_code: int

    monitor_user_cpu_s: float
    monitor_system_cpu_s: float
    workload_user_cpu_s: float
    workload_system_cpu_s: float

    operations: int | None
    probe_reads: int
    sampled_reads: int
    mean_read_ns: float
    p95_read_ns: float
    max_read_ns: float
    total_read_ns: float

    stdout_tail: str | None = None
    stderr_tail: str | None = None

    @property
    def throughput_hz(self) -> float | None:
        if self.operations is None or self.wall_s <= 0:
            return None
        return self.operations / self.wall_s

    @property
    def sample_ratio(self) -> float:
        if self.probe_reads <= 0:
            return 0.0
        return self.sampled_reads / self.probe_reads

    @property
    def requested_rate_hz(self) -> float | None:
        if self.interval_s is None or self.interval_s <= 0:
            return None
        return 1.0 / self.interval_s

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["throughput_hz"] = self.throughput_hz
        payload["sample_ratio"] = self.sample_ratio
        payload["requested_rate_hz"] = self.requested_rate_hz
        return payload


@dataclass
class EndToEndOverheadReport:
    """Bundle of all phase reports plus deltas versus baseline."""

    phases: list[EndToEndPhaseReport]

    @property
    def baseline(self) -> EndToEndPhaseReport:
        for phase in self.phases:
            if phase.monitoring_mode == "baseline":
                return phase
        raise ValueError("No baseline phase present in report.")

    def deltas_vs_baseline(self) -> dict[str, dict[str, float | None]]:
        baseline = self.baseline
        baseline_throughput = baseline.throughput_hz
        payload: dict[str, dict[str, float | None]] = {}

        for phase in self.phases:
            key = f"{phase.label}#{phase.repeat_index}"
            throughput = phase.throughput_hz
            if baseline_throughput is None or throughput is None or baseline_throughput == 0:
                throughput_delta_pct = None
            else:
                throughput_delta_pct = ((throughput - baseline_throughput)
                                        / baseline_throughput) * 100.0

            payload[key] = {
                "monitor_cpu_s_delta": (
                        phase.monitor_user_cpu_s + phase.monitor_system_cpu_s
                        - baseline.monitor_user_cpu_s - baseline.monitor_system_cpu_s
                ),
                "workload_cpu_s_delta": (
                        phase.workload_user_cpu_s + phase.workload_system_cpu_s
                        - baseline.workload_user_cpu_s - baseline.workload_system_cpu_s
                ),
                "wall_s_delta": phase.wall_s - baseline.wall_s,
                "throughput_delta_pct": throughput_delta_pct,
                "probe_reads_delta": phase.probe_reads - baseline.probe_reads,
                "mean_read_ns_delta": phase.mean_read_ns - baseline.mean_read_ns,
            }
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "phases": [phase.to_dict() for phase in self.phases],
            "deltas_vs_baseline": self.deltas_vs_baseline(),
        }

    def summary_lines(self) -> list[str]:
        lines = ["End-to-end overhead:"]
        for phase in self.phases:
            throughput = phase.throughput_hz
            throughput_str = f"{throughput:.1f} ops/s" if throughput is not None else "n/a"
            lines.append(
                "  "
                f"{phase.label:<24s} "
                f"wall={phase.wall_s:.2f}s "
                f"throughput={throughput_str:<14s} "
                f"reads={phase.probe_reads:<6d} "
                f"mean_read={phase.mean_read_ns / 1_000:.1f}us "
                f"monitor_cpu={(phase.monitor_user_cpu_s + phase.monitor_system_cpu_s) * 1_000:.1f}ms"
            )
        return lines


@dataclass(frozen=True)
class _PhaseSpec:
    label: str
    monitoring_mode: str
    repeat_index: int
    interval_s: float | None = None


def _tail_text(text: str, max_lines: int = 12) -> str | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


def _extract_operations(stdout: str) -> int | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "operations" in payload:
            try:
                return int(payload["operations"])
            except (TypeError, ValueError):
                return None
    return None


class _MonitoringWorker:
    """Background monitor used by end-to-end overhead phases."""

    def __init__(
            self,
            probe_factory: Callable[[], Probe],
            monitoring_mode: str,
            interval_s: float | None,
            pipe_config: PipelineConfig,
    ) -> None:
        self._probe_factory = probe_factory
        self._monitoring_mode = monitoring_mode
        self._interval_s = interval_s
        self._pipe_config = pipe_config
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.read_overheads_ns: list[int] = []
        self.probe_reads = 0
        self.sampled_reads = 0
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        self._thread.join()
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        probe = self._probe_factory()
        try:
            probe.start()
            if self._monitoring_mode == "attached_idle":
                while not self._stop.wait(0.05):
                    pass
                return

            if self._monitoring_mode == "attached_fixed":
                tracker = self._pipe_config._make_tracker()
                interval_s = self._interval_s or 0.0
                while not self._stop.is_set():
                    t0 = time.perf_counter_ns()
                    value = probe.read()
                    elapsed_ns = time.perf_counter_ns() - t0
                    self.read_overheads_ns.append(elapsed_ns)
                    self.probe_reads += 1
                    tracker.update(value)
                    self.sampled_reads += 1
                    if self._stop.wait(max(0.0, interval_s)):
                        break
                return

            if self._monitoring_mode == "attached_adaptive":
                poller = self._pipe_config._make_poller()
                tick = 0
                base_interval = self._interval_s or 0.0
                while not self._stop.is_set():
                    t0 = time.perf_counter_ns()
                    value = probe.read()
                    elapsed_ns = time.perf_counter_ns() - t0
                    self.read_overheads_ns.append(elapsed_ns)
                    self.probe_reads += 1

                    result = poller.feed(value, time_index=tick)
                    tick += 1
                    if result.sampled:
                        self.sampled_reads += 1
                    sleep_s = result.interval * base_interval
                    if self._stop.wait(max(0.0, sleep_s)):
                        break
                return

            raise ValueError(f"Unsupported monitoring mode: {self._monitoring_mode}")
        except Exception as exc:  # pragma: no cover - propagated after join
            self.error = exc
        finally:
            try:
                probe.stop()
            except Exception:
                pass


def make_command_workload_runner(
        command_template: str,
        *,
        extra_format: dict[str, Any] | None = None,
) -> Callable[[float], WorkloadRunResult]:
    """Return a subprocess workload runner.

    ``command_template`` may contain ``{duration}`` and any key from
    *extra_format*. The rendered command is tokenized with ``shlex.split``.
    If the command prints a trailing JSON line with an ``operations`` field,
    the resulting report includes throughput.
    """
    fmt = extra_format or {}

    def _runner(duration: float) -> WorkloadRunResult:
        rendered = command_template.format(duration=duration, **fmt)
        command = shlex.split(rendered)
        t0 = time.monotonic()
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        wall_s = time.monotonic() - t0
        return WorkloadRunResult(
            exit_code=proc.returncode,
            operations=_extract_operations(proc.stdout),
            wall_s=wall_s,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    return _runner


def make_synthetic_syscall_workload_runner(
        syscall: str,
        *,
        block_size: int = 4096,
) -> Callable[[float], WorkloadRunResult]:
    """Return a child-process workload runner for a syscall-heavy loop."""

    script = r'''
import json
import os
import sys
import tempfile
import time

kind = sys.argv[1]
duration = float(sys.argv[2])
block_size = int(sys.argv[3])
payload = b"x" * block_size
fd, path = tempfile.mkstemp(prefix="omniflow-overhead-")
created = True

try:
    if kind == "read":
        os.write(fd, payload * 256)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)

    start = time.monotonic()
    operations = 0
    offset = 0
    limit = block_size * 1024

    while time.monotonic() - start < duration:
        if kind == "write":
            os.write(fd, payload)
        elif kind == "pwrite64":
            os.pwrite(fd, payload, offset)
            offset = (offset + block_size) % limit
        elif kind == "fsync":
            os.write(fd, payload)
            os.fsync(fd)
        elif kind == "read":
            buf = os.read(fd, block_size)
            if not buf:
                os.lseek(fd, 0, os.SEEK_SET)
                continue
        elif kind == "openat":
            os.close(fd)
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        else:
            raise SystemExit(f"unsupported synthetic workload: {kind}")
        operations += 1

    print(json.dumps({
        "operations": operations,
        "kind": kind,
        "duration_s": time.monotonic() - start,
    }))
finally:
    try:
        os.close(fd)
    except OSError:
        pass
    if created and os.path.exists(path):
        os.unlink(path)
'''

    def _runner(duration: float) -> WorkloadRunResult:
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", script, syscall, str(duration), str(block_size)],
            check=False,
            capture_output=True,
            text=True,
        )
        wall_s = time.monotonic() - t0
        return WorkloadRunResult(
            exit_code=proc.returncode,
            operations=_extract_operations(proc.stdout),
            wall_s=wall_s,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    return _runner


def _safe_percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), percentile))


def _run_phase(
        spec: _PhaseSpec,
        probe_factory: Callable[[], Probe],
        workload_runner: Callable[[float], WorkloadRunResult],
        phase_duration: float,
        warmup_s: float,
        pipe_config: PipelineConfig,
) -> EndToEndPhaseReport:
    worker: _MonitoringWorker | None = None
    if spec.monitoring_mode != "baseline":
        worker = _MonitoringWorker(
            probe_factory=probe_factory,
            monitoring_mode=spec.monitoring_mode,
            interval_s=spec.interval_s,
            pipe_config=pipe_config,
        )
        worker.start()
        if warmup_s > 0:
            time.sleep(warmup_s)

    ru_self_before = resource.getrusage(resource.RUSAGE_SELF)
    ru_children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start = time.monotonic()
    result = workload_runner(phase_duration)

    if worker is not None:
        worker.stop()
        worker.join()

    wall_s = result.wall_s if result.wall_s is not None else (time.monotonic() - wall_start)
    ru_self_after = resource.getrusage(resource.RUSAGE_SELF)
    ru_children_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    read_overheads = worker.read_overheads_ns if worker is not None else []

    return EndToEndPhaseReport(
        label=spec.label,
        monitoring_mode=spec.monitoring_mode,
        repeat_index=spec.repeat_index,
        interval_s=spec.interval_s,
        wall_s=wall_s,
        exit_code=result.exit_code,
        monitor_user_cpu_s=ru_self_after.ru_utime - ru_self_before.ru_utime,
        monitor_system_cpu_s=ru_self_after.ru_stime - ru_self_before.ru_stime,
        workload_user_cpu_s=ru_children_after.ru_utime - ru_children_before.ru_utime,
        workload_system_cpu_s=ru_children_after.ru_stime - ru_children_before.ru_stime,
        operations=result.operations,
        probe_reads=worker.probe_reads if worker is not None else 0,
        sampled_reads=worker.sampled_reads if worker is not None else 0,
        mean_read_ns=float(np.mean(read_overheads)) if read_overheads else 0.0,
        p95_read_ns=_safe_percentile(read_overheads, 95),
        max_read_ns=float(max(read_overheads)) if read_overheads else 0.0,
        total_read_ns=float(sum(read_overheads)) if read_overheads else 0.0,
        stdout_tail=_tail_text(result.stdout),
        stderr_tail=_tail_text(result.stderr),
    )


def measure_end_to_end_overhead(
        probe_factory: Callable[[], Probe],
        workload_runner: Callable[[float], WorkloadRunResult],
        *,
        phase_duration: float,
        fixed_intervals: Sequence[float] | None = None,
        adaptive_base_interval: float | None = None,
        warmup_s: float = 0.0,
        repeats: int = 1,
        include_idle: bool = True,
        include_adaptive: bool = True,
        pipe_config: PipelineConfig | None = None,
) -> EndToEndOverheadReport:
    """Compare workload cost across monitoring phases.

    The baseline phase runs the workload with no probe attached.
    ``attached_idle`` isolates kernel hook cost, ``attached_fixed`` keeps a
    fixed read cadence, and ``attached_adaptive`` runs the live adaptive loop.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    fixed_intervals = list(fixed_intervals or [])
    if not fixed_intervals and not include_idle and not include_adaptive:
        raise ValueError("At least one monitored phase must be enabled.")

    cfg = pipe_config or PipelineConfig()
    phases: list[EndToEndPhaseReport] = []

    for repeat_index in range(1, repeats + 1):
        phase_specs = [_PhaseSpec("baseline", "baseline", repeat_index)]
        if include_idle:
            phase_specs.append(_PhaseSpec("attached_idle", "attached_idle", repeat_index))
        for interval in fixed_intervals:
            phase_specs.append(
                _PhaseSpec(
                    label=f"attached_fixed_{interval:.3f}s",
                    monitoring_mode="attached_fixed",
                    repeat_index=repeat_index,
                    interval_s=float(interval),
                )
            )
        if include_adaptive:
            phase_specs.append(
                _PhaseSpec(
                    label="attached_adaptive",
                    monitoring_mode="attached_adaptive",
                    repeat_index=repeat_index,
                    interval_s=adaptive_base_interval,
                )
            )

        for spec in phase_specs:
            phases.append(
                _run_phase(
                    spec=spec,
                    probe_factory=probe_factory,
                    workload_runner=workload_runner,
                    phase_duration=phase_duration,
                    warmup_s=warmup_s,
                    pipe_config=cfg,
                )
            )

    return EndToEndOverheadReport(phases=phases)
