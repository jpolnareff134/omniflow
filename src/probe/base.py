"""
Abstract probe interface.

All monitoring probes (eBPF syscall counter, /proc readers, stack
samplers, ...) implement this interface so the adaptive pipeline can
drive them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Probe(ABC):
    """Base class for all monitoring probes.

    A probe encapsulates a single monitoring data source that produces
    one scalar metric per read.  Probes follow a start/stop lifecycle
    and are usable as context managers.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialise or attach the probe (e.g. load BPF, open files)."""

    @abstractmethod
    def stop(self) -> None:
        """Release resources (e.g. detach BPF, close handles)."""

    @abstractmethod
    def read(self) -> float:
        """Take a single measurement and return the metric value.

        Semantics depend on the probe:

        * **Counter probes** return the count since the last read (delta).
        * **Gauge probes** return the current value or utilisation %.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable description for experiment manifests."""

    # -- context manager convenience -----------------------------------------

    def __enter__(self) -> "Probe":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
