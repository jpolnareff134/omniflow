"""
``/proc`` filesystem probes for CPU, memory, disk, and network metrics.
"""

from __future__ import annotations

import os
import time
from typing import Any

from probe.base import Probe


# ------------------------------
# CPU utilisation (from /proc/stat)
# ------------------------------

class ProcCpuProbe(Probe):
    """Overall CPU utilisation (%) computed from ``/proc/stat`` deltas.

    Each :meth:`read` returns the CPU busy fraction since the previous
    read.  The very first read after :meth:`start` returns 0.0 because
    no delta is available yet.
    """

    def __init__(self) -> None:
        self._prev_total: int = 0
        self._prev_idle: int = 0
        self._started: bool = False

    # -- Probe interface -----------------------------------------------------

    def start(self) -> None:
        if not os.path.exists("/proc/stat"):
            raise RuntimeError(
                "ProcCpuProbe requires Linux (/proc/stat not found)."
            )
        self._prev_total, self._prev_idle = self._read_raw()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> float:
        assert self._started, "Call start() first"
        total, idle = self._read_raw()
        dt = total - self._prev_total
        di = idle - self._prev_idle
        self._prev_total = total
        self._prev_idle = idle
        if dt == 0:
            return 0.0
        return (1.0 - di / dt) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {"type": "proc_cpu", "source": "/proc/stat"}

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _read_raw() -> tuple[int, int]:
        """Parse the aggregate CPU line and return (total, idle) jiffies."""
        with open("/proc/stat") as f:
            line = f.readline()
        # Fields: cpu  user nice system idle iowait irq softirq steal ...
        values = [int(p) for p in line.split()[1:]]
        total = sum(values)
        idle = values[3] + values[4]  # idle + iowait
        return total, idle


# ------------------------------
# Memory utilisation (from /proc/meminfo)
# ------------------------------

class ProcMemProbe(Probe):
    """Memory utilisation (%) from ``/proc/meminfo``.

    Memory is an instant gauge - no delta is needed.
    """

    def __init__(self) -> None:
        self._started: bool = False

    def start(self) -> None:
        if not os.path.exists("/proc/meminfo"):
            raise RuntimeError(
                "ProcMemProbe requires Linux (/proc/meminfo not found)."
            )
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> float:
        assert self._started, "Call start() first"
        info = self._read_raw()
        total = info.get("MemTotal", 1)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        if total == 0:
            return 0.0
        return (1.0 - available / total) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {"type": "proc_mem", "source": "/proc/meminfo"}

    @staticmethod
    def _read_raw() -> dict[str, int]:
        """Parse ``/proc/meminfo`` into ``{key: kB_value}``."""
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        return info


# ------------------------------
# Disk I/O (from /proc/diskstats)
# ------------------------------

class ProcDiskProbe(Probe):
    """Disk I/O operations per second from ``/proc/diskstats`` (delta).

    Sums reads_completed + writes_completed across all block devices
    (or a single device if *device* is given).
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._prev_ops: int = 0
        self._prev_time: float = 0.0
        self._started: bool = False

    def start(self) -> None:
        if not os.path.exists("/proc/diskstats"):
            raise RuntimeError(
                "ProcDiskProbe requires Linux (/proc/diskstats not found)."
            )
        self._prev_ops = self._read_raw()
        self._prev_time = time.monotonic()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> float:
        assert self._started, "Call start() first"
        now = time.monotonic()
        ops = self._read_raw()
        dt = now - self._prev_time
        delta = ops - self._prev_ops
        self._prev_ops = ops
        self._prev_time = now
        return delta / dt if dt > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "proc_disk",
            "source": "/proc/diskstats",
            "device": self._device,
        }

    def _read_raw(self) -> int:
        total = 0
        with open("/proc/diskstats") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 14:
                    continue
                dev = parts[2]
                if self._device and dev != self._device:
                    continue
                if not self._device and (
                        dev.startswith("loop") or dev.startswith("ram")
                ):
                    continue
                total += int(parts[3]) + int(parts[7])  # reads + writes
        return total


# ------------------------------
# Network throughput (from /proc/net/dev)
# ------------------------------

class ProcNetProbe(Probe):
    """Network bytes/s (RX+TX) from ``/proc/net/dev`` (delta).

    Sums across all non-loopback interfaces, or a single interface if
    *interface* is given.
    """

    def __init__(self, interface: str | None = None) -> None:
        self._interface = interface
        self._prev_bytes: int = 0
        self._prev_time: float = 0.0
        self._started: bool = False

    def start(self) -> None:
        if not os.path.exists("/proc/net/dev"):
            raise RuntimeError(
                "ProcNetProbe requires Linux (/proc/net/dev not found)."
            )
        self._prev_bytes = self._read_raw()
        self._prev_time = time.monotonic()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> float:
        assert self._started, "Call start() first"
        now = time.monotonic()
        byt = self._read_raw()
        dt = now - self._prev_time
        delta = byt - self._prev_bytes
        self._prev_bytes = byt
        self._prev_time = now
        return delta / dt if dt > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "proc_net",
            "source": "/proc/net/dev",
            "interface": self._interface,
        }

    def _read_raw(self) -> int:
        total = 0
        with open("/proc/net/dev") as f:
            for i, line in enumerate(f):
                if i < 2:  # skip header lines
                    continue
                parts = line.split()
                if not parts:
                    continue
                iface = parts[0].rstrip(":")
                if not self._interface and iface == "lo":
                    continue
                if self._interface and iface != self._interface:
                    continue
                total += int(parts[1]) + int(parts[9])  # rx + tx bytes
        return total
