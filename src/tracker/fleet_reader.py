"""
Fleet reader: per-container dual tracking.

Discovers Docker (or any cgroup v2) containers automatically and runs
an independent dense reference tracker + adaptive poller for each one.
This is the "one tracker per container" counterpart of
:class:`~probe.cgroup.CgroupFleetProbe` (which aggregates).

The fleet is re-scanned periodically so containers that appear or
disappear mid-experiment are handled gracefully.
"""

from __future__ import annotations

import json
import logging as log
import os
import resource
import signal
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from probe.cgroup import CgroupCpuProbe, _discover_container_cgroups, _DOCKER_CGROUP_GLOBS
from tracker.pipeline import PipelineConfig
from tracker.windowed import (
    WindowedTracker, AdaptivePoller,
    TickResult, PollResult,
    InfoLossReport, evaluate_info_loss,
)


# ------------------------------
# Per-container state (internal)
# ------------------------------

class _ContainerTrack:
    """Mutable state for one container's dual-tracking."""

    __slots__ = (
        "cgroup_path", "name", "probe",
        "full_tracker", "poller",
        "readings", "full_results", "poll_results",
        "overheads_ns", "tick",
    )

    def __init__(
            self,
            cgroup_path: str,
            config: PipelineConfig,
    ) -> None:
        self.cgroup_path = cgroup_path
        self.name = os.path.basename(cgroup_path)
        self.probe = CgroupCpuProbe(cgroup_path=cgroup_path)
        self.full_tracker: WindowedTracker = config._make_tracker()
        self.poller: AdaptivePoller = config._make_poller()
        self.readings: list[float] = []
        self.full_results: list[TickResult] = []
        self.poll_results: list[PollResult] = []
        self.overheads_ns: list[int] = []
        self.tick: int = 0

    def start(self) -> None:
        self.probe.start()

    def stop(self) -> None:
        self.probe.stop()

    def step(self) -> float:
        """Read, feed both tracks, return the reading."""
        t0 = time.perf_counter_ns()
        value = self.probe.read()
        overhead = time.perf_counter_ns() - t0

        self.readings.append(value)
        self.overheads_ns.append(overhead)

        self.full_results.append(self.full_tracker.update(value))
        self.poll_results.append(self.poller.step(value, time_index=self.tick))

        self.tick += 1
        return value


# ------------------------------
# Per-container result
# ------------------------------

@dataclass
class ContainerResult:
    """Tracking results for a single container."""

    name: str
    cgroup_path: str
    trace: NDArray[np.float64]
    full_results: list[TickResult]
    poll_results: list[PollResult]
    info_loss: InfoLossReport
    read_overheads_ns: list[int]

    @property
    def n_total(self) -> int:
        return len(self.trace)

    @property
    def n_sampled(self) -> int:
        return sum(1 for pr in self.poll_results if pr.sampled)

    @property
    def mean_read_overhead_us(self) -> float:
        if not self.read_overheads_ns:
            return 0.0
        return float(np.mean(self.read_overheads_ns)) / 1_000

    def summary_line(self) -> str:
        il = self.info_loss
        return (
            f"{self.name:>40s}  n={self.n_total:5d}  "
            f"sampled={self.n_sampled:5d} ({il.sample_ratio:5.1%})  "
            f"nrmse={il.nrmse_mean:.4f}  "
            f"overhead={self.mean_read_overhead_us:.0f}us"
        )


# ------------------------------
# Fleet result
# ------------------------------

@dataclass
class FleetReaderResult:
    """All per-container tracking results plus fleet-level metadata."""

    containers: dict[str, ContainerResult]
    config: PipelineConfig
    wall_seconds: float
    n_ticks: int

    # Fleet-level overhead measurements
    sweep_times_ns: list[int]  # wall-clock ns per sweep (read all containers)
    user_cpu_s: float  # total user CPU consumed by monitoring
    system_cpu_s: float  # total system CPU consumed by monitoring

    @property
    def mean_sweep_us(self) -> float:
        if not self.sweep_times_ns:
            return 0.0
        return float(np.mean(self.sweep_times_ns)) / 1_000

    @property
    def p99_sweep_us(self) -> float:
        if not self.sweep_times_ns:
            return 0.0
        return float(np.percentile(self.sweep_times_ns, 99)) / 1_000

    @property
    def monitoring_cpu_pct(self) -> float:
        """Fraction of wall-clock time spent in CPU by this process."""
        if self.wall_seconds <= 0:
            return 0.0
        return (self.user_cpu_s + self.system_cpu_s) / self.wall_seconds * 100.0

    def summary(self) -> str:
        sweep_arr = np.array(self.sweep_times_ns, dtype=np.float64) / 1_000
        lines = [
            f"Fleet summary: {len(self.containers)} containers, "
            f"{self.n_ticks} ticks over {self.wall_seconds:.1f}s",
            f"  Monitoring overhead: "
            f"mean_sweep={self.mean_sweep_us:.0f}us  "
            f"p99_sweep={self.p99_sweep_us:.0f}us  "
            f"user_cpu={self.user_cpu_s * 1000:.1f}ms  "
            f"sys_cpu={self.system_cpu_s * 1000:.1f}ms  "
            f"cpu_load={self.monitoring_cpu_pct:.2f}%",
            "-" * 90,
        ]
        for cr in sorted(self.containers.values(), key=lambda c: c.name):
            lines.append(cr.summary_line())
        lines.append("-" * 90)
        return "\n".join(lines)

    def plot(self, save_dir: str | None = None) -> None:
        """Generate per-container dense-reference and adaptive figures."""
        from plot.plot import plot_tracking, plot_adaptive_polling

        if not self.containers:
            log.warning("Fleet: nothing to plot (0 containers)")
            return

        for name, cr in sorted(self.containers.items()):
            tag = name[:12]  # short container id
            tr_path = os.path.join(save_dir, f"dense_reference_{tag}.png") if save_dir else None
            ap_path = os.path.join(save_dir, f"adaptive_replay_{tag}.png") if save_dir else None

            if cr.full_results:
                plot_tracking(
                    cr.full_results,
                    title=f"Dense Reference - {tag}",
                    save_path=tr_path,
                )
            if cr.poll_results and len(cr.trace) > 0:
                plot_adaptive_polling(
                    cr.trace,
                    cr.poll_results,
                    full_results=cr.full_results or None,
                    title=f"Adaptive Replay - {tag}",
                    save_path=ap_path,
                )

    def save_json(
            self,
            save_dir: str,
            generator: dict[str, Any] | None = None,
            filename: str = "fleet.json",
    ) -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        manifest: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": self.config.to_dict(),
            "generator": generator,
            "wall_seconds": self.wall_seconds,
            "n_ticks": self.n_ticks,
            "n_containers": len(self.containers),
            "overhead": {
                "mean_sweep_us": self.mean_sweep_us,
                "p99_sweep_us": self.p99_sweep_us,
                "user_cpu_s": self.user_cpu_s,
                "system_cpu_s": self.system_cpu_s,
                "monitoring_cpu_pct": self.monitoring_cpu_pct,
            },
            "containers": {},
        }

        for name, cr in sorted(self.containers.items()):
            il = cr.info_loss
            manifest["containers"][name] = {
                "cgroup_path": cr.cgroup_path,
                "n_total": cr.n_total,
                "n_sampled": cr.n_sampled,
                "sample_ratio": il.sample_ratio,
                "nrmse_mean": il.nrmse_mean,
                "rmse_mean": il.rmse_mean,
                "correlation": il.correlation,
                "mean_read_overhead_us": cr.mean_read_overhead_us,
                "info_loss": asdict(il),
            }

        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        log.info("Fleet results saved to %s", path)
        return path


# ------------------------------
# Fleet reader
# ------------------------------

class FleetReader:
    """Run per-container dual tracking across a fleet of cgroups.

    Parameters
    ----------
    config : PipelineConfig, optional
        Tracker / poller tunables (same config used for every container).
    base_interval : float
        Seconds between consecutive read sweeps.
    cgroup_patterns : list[str], optional
        Glob patterns to discover container cgroup dirs.
    refresh_interval : float
        Seconds between re-scans for new / removed containers.
    """

    def __init__(
            self,
            config: PipelineConfig | None = None,
            base_interval: float = 1.0,
            cgroup_patterns: list[str] | None = None,
            refresh_interval: float = 10.0,
    ) -> None:
        self.config = config or PipelineConfig()
        self.base_interval = base_interval
        self._patterns = cgroup_patterns or _DOCKER_CGROUP_GLOBS
        self._refresh_interval = refresh_interval
        self._tracks: dict[str, _ContainerTrack] = {}
        self._finished_tracks: dict[str, _ContainerTrack] = {}
        self._last_refresh: float = 0.0

    # -- discovery -----------------------------------------------------------

    def _refresh(self) -> None:
        now = time.monotonic()
        current = set(_discover_container_cgroups(self._patterns))

        # Start tracking new containers
        new_paths = current - set(self._tracks)
        for path in sorted(new_paths):
            track = _ContainerTrack(path, self.config)
            try:
                track.start()
                self._tracks[path] = track
                log.info("Fleet: tracking new container %s", track.name)
            except Exception as exc:
                log.warning("Fleet: failed to start probe for %s: %s", path, exc)

        # Prune gone containers (preserve their data)
        gone = set(self._tracks) - current
        for path in gone:
            track = self._tracks.pop(path)
            track.stop()
            if track.readings:
                self._finished_tracks[path] = track
            log.info("Fleet: stopped tracking %s (kept %d readings)",
                     track.name, len(track.readings))

        self._last_refresh = now
        log.debug("Fleet: tracking %d containers", len(self._tracks))

    # -- main loop -----------------------------------------------------------

    def run(self, duration: float) -> FleetReaderResult:
        """Execute fleet monitoring for *duration* seconds."""
        self._refresh()

        if not self._tracks:
            log.warning("Fleet: no containers found matching %s", self._patterns)

        stop_flag = False

        def _sig(_s: int, _f: Any) -> None:
            nonlocal stop_flag
            stop_flag = True

        prev_handler = signal.signal(signal.SIGINT, _sig)
        progress_every = max(1, int(10 / self.base_interval))

        sweep_times_ns: list[int] = []
        ru_before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.monotonic()
        tick = 0

        try:
            while not stop_flag:
                # Re-scan periodically
                now = time.monotonic()
                if now - self._last_refresh >= self._refresh_interval:
                    self._refresh()

                # Read all containers and time the sweep
                sweep_t0 = time.perf_counter_ns()
                dead: list[str] = []
                for path, track in self._tracks.items():
                    try:
                        track.step()
                    except (FileNotFoundError, RuntimeError, AssertionError):
                        dead.append(path)
                sweep_ns = time.perf_counter_ns() - sweep_t0
                sweep_times_ns.append(sweep_ns)

                for path in dead:
                    track = self._tracks.pop(path)
                    track.stop()
                    if track.readings:
                        self._finished_tracks[path] = track
                    log.debug("Fleet: pruned dead container %s (kept %d readings)",
                              track.name, len(track.readings))

                tick += 1

                # Progress reporting
                if tick % progress_every == 0:
                    elapsed = time.monotonic() - start
                    log.info(
                        "  [fleet] tick=%d  %.0fs elapsed  "
                        "containers=%d  sweep=%.0fus",
                        tick, elapsed, len(self._tracks),
                        sweep_ns / 1_000,
                    )

                # Duration check
                elapsed = time.monotonic() - start
                if 0 < duration <= elapsed:
                    break

                # Sleep until next slot
                next_time = tick * self.base_interval
                remaining = next_time - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            signal.signal(signal.SIGINT, prev_handler)
            # Stop all probes
            for track in self._tracks.values():
                track.stop()

        wall = time.monotonic() - start
        ru_after = resource.getrusage(resource.RUSAGE_SELF)
        user_cpu = ru_after.ru_utime - ru_before.ru_utime
        system_cpu = ru_after.ru_stime - ru_before.ru_stime

        # Build per-container results from both active and finished tracks
        all_tracks = {**self._finished_tracks, **self._tracks}
        results: dict[str, ContainerResult] = {}
        for path, track in all_tracks.items():
            if not track.readings:
                continue
            trace = np.array(track.readings, dtype=np.float64)
            info_loss = evaluate_info_loss(
                trace,
                full_tracker=self.config._make_tracker(),
                poller=self.config._make_poller(),
            )
            results[track.name] = ContainerResult(
                name=track.name,
                cgroup_path=path,
                trace=trace,
                full_results=track.full_results,
                poll_results=track.poll_results,
                info_loss=info_loss,
                read_overheads_ns=track.overheads_ns,
            )

        log.info(
            "Fleet: %d containers tracked, %d ticks over %.1fs",
            len(results), tick, wall,
        )

        return FleetReaderResult(
            containers=results,
            config=self.config,
            wall_seconds=wall,
            n_ticks=tick,
            sweep_times_ns=sweep_times_ns,
            user_cpu_s=user_cpu,
            system_cpu_s=system_cpu,
        )
