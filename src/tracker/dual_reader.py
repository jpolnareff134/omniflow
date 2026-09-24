"""
Dual-reader experiment runner.

Reads a probe at a fixed base rate while simultaneously maintaining both
a dense reference tracker and an adaptive poller. This is the mainline
fidelity-evaluation path for live data: collection stays dense while the
adaptive schedule is replayed offline against the same signal.
"""

from __future__ import annotations

import logging as log
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from probe.base import Probe
from tracker.pipeline import PipelineConfig, PipelineResult
from tracker.windowed import (
    TickResult, PollResult,
    InfoLossReport, evaluate_info_loss,
)


# ------------------------------
# Result bundle
# ------------------------------

@dataclass
class DualReaderResult:
    """All outputs from a dense collection plus offline replay run."""

    trace: NDArray[np.float64]
    timestamps: list[float]

    # Dense reference tracker results
    full_results: list[TickResult]

    # Adaptive poller results (some points skipped)
    poll_results: list[PollResult]

    # Information-loss metrics
    info_loss: InfoLossReport

    # Per-read overhead measurements (nanoseconds)
    read_overheads_ns: list[int]

    # Config used
    config: PipelineConfig

    # -- convenience --

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

    def to_pipeline_result(self) -> PipelineResult:
        """Convert to a :class:`PipelineResult` for plotting compatibility."""
        return PipelineResult(
            trace=self.trace,
            config=self.config,
            full_results=self.full_results,
            poll_results=self.poll_results,
            info_loss=self.info_loss,
        )


# ------------------------------
# Dual reader
# ------------------------------

class DualReader:
    """Run a probe at fixed rate with two parallel processing tracks.

    The probe is read every *base_interval* seconds.  Each reading is:

    1. Fed to a **dense reference tracker** which processes every point.
    2. Passed to an **AdaptivePoller** which decides whether to process
       or skip it.

    After the run the adaptive poller's decisions are compared against
    the dense reference path to quantify information loss. This mirrors
    the synthetic replay workflow, but on live probe data.

    Parameters
    ----------
    probe : Probe
        A monitoring probe (must already be started, or started
        externally via a ``with`` block).
    config : PipelineConfig, optional
        Pipeline tunables.  Defaults are loaded from ``config.py``.
    base_interval : float
        Seconds between consecutive dense reference reads.
    """

    def __init__(
            self,
            probe: Probe,
            config: PipelineConfig | None = None,
            base_interval: float = 1.0,
    ) -> None:
        self.probe = probe
        self.config = config or PipelineConfig()
        self.base_interval = base_interval

    # --------------------------------------------------------------------- #

    def run(
        self,
        duration: float | None = None,
        stop_predicate: Callable[[], bool] | None = None,
    ) -> DualReaderResult:
        """Execute the dual-reader run.

        The run ends when any of the following occurs:

        * *duration* seconds have elapsed (if *duration* is a positive number);
        * the optional *stop_predicate* returns ``True``;
        * ``SIGINT`` is received.

        The probe **must** already be started (e.g. via a surrounding
        ``with probe:`` block).

        Returns a :class:`DualReaderResult` with both processing tracks
        and dense-vs-adaptive information-loss metrics.
        """
        full_tracker = self.config._make_tracker()
        poller = self.config._make_poller()

        readings: list[float] = []
        timestamps: list[float] = []
        full_results: list[TickResult] = []
        poll_results: list[PollResult] = []
        overheads: list[int] = []

        stop_flag = False

        def _sig(_s: int, _f: Any) -> None:
            nonlocal stop_flag
            stop_flag = True

        prev_handler = signal.signal(signal.SIGINT, _sig)

        progress_every = max(1, int(10 / self.base_interval))  # ~every 10s

        try:
            start = time.monotonic()
            t = 0

            while not stop_flag and not (stop_predicate is not None and stop_predicate()):
                # -- read --
                t0 = time.perf_counter_ns()
                value = self.probe.read()
                overhead = time.perf_counter_ns() - t0

                readings.append(value)
                timestamps.append(time.monotonic() - start)
                overheads.append(overhead)

                # -- dense reference path: always process --
                full_tick = full_tracker.update(value)
                full_results.append(full_tick)

                # -- adaptive: let poller decide --
                poll_result = poller.step(value, time_index=t)
                poll_results.append(poll_result)

                t += 1

                # Periodic progress (every ~10 seconds of wall-clock)
                if t % progress_every == 0:
                    elapsed = time.monotonic() - start
                    n_sampled_so_far = sum(
                        1 for pr in poll_results if pr.sampled
                    )
                    log.info(
                        "  [dual] %4d readings in %.0fs  "
                        "sampled=%d (%.0f%%)  value=%.1f  "
                        "urgency=%.3f  overhead=%d us",
                        t, elapsed,
                        n_sampled_so_far,
                        100.0 * n_sampled_so_far / t,
                        value,
                        poll_result.urgency,
                        overhead // 1_000,
                    )

                log.debug(
                    "  t=%4d  value=%10.1f  sampled=%s  "
                    "interval=%d  urgency=%.3f  overhead=%d us",
                    t, value, poll_result.sampled,
                    poll_result.interval, poll_result.urgency,
                    overhead // 1_000,
                )

                # -- check duration --
                if duration is not None and duration > 0:
                    elapsed = time.monotonic() - start
                    if elapsed >= duration:
                        break

                # -- sleep until next slot --
                next_time = t * self.base_interval
                remaining = next_time - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            signal.signal(signal.SIGINT, prev_handler)

        if not readings:
            raise RuntimeError("No readings collected.")

        trace = np.array(readings, dtype=np.float64)

        # Compute info-loss metrics (fresh instances for determinism)
        info_loss = evaluate_info_loss(
            trace,
            full_tracker=self.config._make_tracker(),
            poller=self.config._make_poller(),
        )

        n_sampled = sum(1 for pr in poll_results if pr.sampled)
        log.info(
            "Dual reader: %d readings over %.1fs, "
            "adaptive sampled %d/%d (%.1f%%)",
            len(readings), timestamps[-1],
            n_sampled, len(readings),
            100.0 * n_sampled / len(readings),
        )

        return DualReaderResult(
            trace=trace,
            timestamps=timestamps,
            full_results=full_results,
            poll_results=poll_results,
            info_loss=info_loss,
            read_overheads_ns=overheads,
            config=self.config,
        )
