"""
cgroup v2 probes for CPU and memory metrics.

These probes read from the cgroup2 filesystem (typically mounted at
``/sys/fs/cgroup``).  They are scoped to a specific cgroup slice,
making them ideal for monitoring containers, systemd services, or
any isolated workload.

If no *cgroup_path* is provided the probe auto-detects the cgroup of
the current process via ``/proc/self/cgroup``.
"""

from __future__ import annotations

import glob
import logging as _log
import os
import time
from typing import Any

from probe.base import Probe

_CGROUPFS = "/sys/fs/cgroup"


def _self_cgroup() -> str:
    """Return the cgroup v2 path of the current process."""
    with open("/proc/self/cgroup") as f:
        for line in f:
            # cgroup v2: "0::<path>"
            parts = line.strip().split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                return os.path.join(_CGROUPFS, parts[2].lstrip("/"))
    raise RuntimeError(
        "Could not detect cgroup v2 path from /proc/self/cgroup. "
        "Is cgroup v2 enabled?"
    )


# ------------------------------
# CPU usage (from cpu.stat -> usage_usec)
# ------------------------------

class CgroupCpuProbe(Probe):
    """CPU utilisation (%) for a cgroup, derived from ``cpu.stat``.

    Each :meth:`read` returns the CPU busy fraction since the previous
    read, expressed as a percentage of wall-clock time (can exceed 100 %
    on multi-core systems).  The first read after :meth:`start` returns
    0.0 because no delta is available yet.

    Parameters
    ----------
    cgroup_path : str, optional
        Absolute path to the cgroup directory
        (e.g. ``/sys/fs/cgroup/system.slice/myservice.service``).
        If *None*, the current process's cgroup is used.
    """

    def __init__(self, cgroup_path: str | None = None) -> None:
        self._cgroup_path = cgroup_path
        self._stat_path: str = ""
        self._prev_usage_us: int = 0
        self._prev_time: float = 0.0
        self._started: bool = False

    def start(self) -> None:
        cg = self._cgroup_path or _self_cgroup()
        self._stat_path = os.path.join(cg, "cpu.stat")
        if not os.path.exists(self._stat_path):
            raise RuntimeError(
                f"CgroupCpuProbe: {self._stat_path} not found. "
                "Ensure cgroup v2 is mounted and cpu controller is enabled."
            )
        self._prev_usage_us = self._read_usage_usec()
        self._prev_time = time.monotonic()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> float:
        assert self._started, "Call start() first"
        now = time.monotonic()
        usage = self._read_usage_usec()
        dt = now - self._prev_time
        delta_us = usage - self._prev_usage_us
        self._prev_usage_us = usage
        self._prev_time = now
        if dt <= 0:
            return 0.0
        # delta_us is in microseconds; dt is in seconds
        # Return a pure CPU fraction (0.0 = idle, 1.0 = one full core busy).
        # Multi-core pods can exceed 1.0.  The prometheus-adapter converts this
        # to milli-units (e.g. 0.015 -> 15m) matching Kubernetes CPU quantities.
        return (delta_us / 1e6) / dt

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cgroup_cpu",
            "source": self._stat_path,
        }

    def _read_usage_usec(self) -> int:
        """Parse ``usage_usec`` from ``cpu.stat``."""
        with open(self._stat_path) as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        raise RuntimeError(
            f"'usage_usec' not found in {self._stat_path}"
        )


# ------------------------------
# Memory usage (from memory.current / memory.max)
# ------------------------------

class CgroupMemProbe(Probe):
    """Memory utilisation (%) for a cgroup.

    Reads ``memory.current`` and ``memory.max`` to compute the fraction
    of the cgroup's memory limit currently in use.  If the cgroup has no
    limit (``max``), the value is reported as a raw byte count instead.

    Parameters
    ----------
    cgroup_path : str, optional
        Absolute path to the cgroup directory.
        If *None*, the current process's cgroup is used.
    """

    def __init__(self, cgroup_path: str | None = None) -> None:
        self._cgroup_path = cgroup_path
        self._current_path: str = ""
        self._max_path: str = ""
        self._has_limit: bool = False
        self._max_bytes: int = 0
        self._started: bool = False

    def start(self) -> None:
        cg = self._cgroup_path or _self_cgroup()
        self._current_path = os.path.join(cg, "memory.current")
        self._max_path = os.path.join(cg, "memory.max")
        if not os.path.exists(self._current_path):
            raise RuntimeError(
                f"CgroupMemProbe: {self._current_path} not found. "
                "Ensure cgroup v2 is mounted and memory controller is enabled."
            )
        # Determine if a memory limit is set
        if os.path.exists(self._max_path):
            raw = self._read_file(self._max_path)
            if raw != "max":
                self._has_limit = True
                self._max_bytes = int(raw)
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> float:
        assert self._started, "Call start() first"
        current = int(self._read_file(self._current_path))
        if self._has_limit and self._max_bytes > 0:
            return (current / self._max_bytes) * 100.0
        # No limit - return raw megabytes as a gauge
        return current / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cgroup_mem",
            "source": self._current_path,
            "has_limit": self._has_limit,
            "max_bytes": self._max_bytes if self._has_limit else None,
        }

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path) as f:
            return f.read().strip()


# ------------------------------
# Fleet probe: aggregate CPU across all Docker containers
# ------------------------------

# Common cgroup v2 patterns where Docker places container scopes.
_DOCKER_CGROUP_GLOBS = [
    "/sys/fs/cgroup/system.slice/docker-*.scope",
    "/sys/fs/cgroup/docker/*/",
]

# Kubernetes cgroup v2 patterns (systemd driver, the k8s default).
_K8S_CGROUP_GLOBS = [
    # systemd cgroup driver (kubeadm default, MicroK8s)
    "/sys/fs/cgroup/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod*.slice/",
    "/sys/fs/cgroup/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod*.slice/",
    "/sys/fs/cgroup/kubepods.slice/kubepods-pod*.slice/",
    # cgroupfs cgroup driver (minikube kvm2/docker default)
    "/sys/fs/cgroup/kubepods/burstable/pod*/",
    "/sys/fs/cgroup/kubepods/besteffort/pod*/",
    "/sys/fs/cgroup/kubepods/pod*/",
]

# Combined: try both Docker and K8s patterns by default.
_ALL_CGROUP_GLOBS = _DOCKER_CGROUP_GLOBS + _K8S_CGROUP_GLOBS


def _discover_container_cgroups(
        patterns: list[str] | None = None,
) -> list[str]:
    """Return sorted list of cgroup directories matching *patterns*."""
    patterns = patterns or _ALL_CGROUP_GLOBS
    found: set[str] = set()
    for pat in patterns:
        for path in glob.glob(pat):
            cpu_stat = os.path.join(path, "cpu.stat")
            if os.path.isfile(cpu_stat):
                found.add(path)
    return sorted(found)


class CgroupFleetProbe(Probe):
    """Aggregate CPU utilisation across many cgroups (containers).

    Behaves like a node-exporter: each :meth:`read` returns the **sum**
    of per-container CPU % since the previous read.  Containers that
    appear or disappear between reads are handled gracefully; new ones
    start contributing from the next cycle, removed ones are pruned.

    Parameters
    ----------
    cgroup_patterns : list[str], optional
        Glob patterns to discover container cgroup directories.
        Defaults to common Docker cgroup v2 layouts.
    refresh_interval : float
        Seconds between re-scanning for new / removed containers.
        Set to 0 to discover only once at :meth:`start`.
    """

    def __init__(
            self,
            cgroup_patterns: list[str] | None = None,
            refresh_interval: float = 10.0,
    ) -> None:
        self._patterns = cgroup_patterns or _DOCKER_CGROUP_GLOBS
        self._refresh_interval = refresh_interval

        # per-cgroup tracking: path -> (prev_usage_us, prev_time)
        self._state: dict[str, tuple[int, float]] = {}
        self._last_per_container: dict[str, float] = {}
        self._last_refresh: float = 0.0
        self._started: bool = False

    def start(self) -> None:
        self._refresh()
        if not self._state:
            _log.warning(
                "CgroupFleetProbe: no containers found matching %s",
                self._patterns,
            )
        self._started = True

    def stop(self) -> None:
        self._started = False
        self._state.clear()
        self._last_per_container.clear()

    def read(self) -> float:
        """Return **total CPU %** across all tracked containers."""
        assert self._started, "Call start() first"

        # Periodically re-scan for new / removed containers
        now = time.monotonic()
        if (
                self._refresh_interval > 0
                and now - self._last_refresh >= self._refresh_interval
        ):
            self._refresh()

        total_cpu = 0.0
        dead: list[str] = []
        per_container: dict[str, float] = {}

        for path, (prev_us, prev_t) in self._state.items():
            try:
                usage_us = self._read_usage_usec(path)
            except (FileNotFoundError, RuntimeError):
                dead.append(path)
                continue

            dt = now - prev_t
            delta_us = usage_us - prev_us
            self._state[path] = (usage_us, now)

            if dt > 0 and delta_us >= 0:
                cpu_pct = (delta_us / 1e6) / dt * 100.0
                total_cpu += cpu_pct
                per_container[path] = cpu_pct

        # Prune dead containers
        for p in dead:
            del self._state[p]
            _log.debug("CgroupFleetProbe: pruned dead cgroup %s", p)

        self._last_per_container = per_container
        return total_cpu

    def read_all(self) -> dict[str, float]:
        """Return per-container CPU % from the most recent :meth:`read`."""
        return dict(self._last_per_container)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cgroup_fleet",
            "patterns": self._patterns,
            "n_containers": len(self._state),
            "refresh_interval": self._refresh_interval,
        }

    # -- internals -----------------------------------------------------------

    def _refresh(self) -> None:
        """Discover containers and initialise state for new ones."""
        now = time.monotonic()
        current_paths = set(_discover_container_cgroups(self._patterns))
        # Add new containers
        for path in current_paths - set(self._state):
            try:
                usage_us = self._read_usage_usec(path)
                self._state[path] = (usage_us, now)
                _log.debug("CgroupFleetProbe: tracking new cgroup %s", path)
            except (FileNotFoundError, RuntimeError):
                pass
        self._last_refresh = now
        _log.debug(
            "CgroupFleetProbe: tracking %d containers", len(self._state),
        )

    @staticmethod
    def _read_usage_usec(cgroup_path: str) -> int:
        stat_path = os.path.join(cgroup_path, "cpu.stat")
        with open(stat_path) as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        raise RuntimeError(f"'usage_usec' not found in {stat_path}")
