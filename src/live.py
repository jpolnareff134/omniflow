"""
Supported live entry point.

Mainline live modes:

- **fixed**: regular dense collection at a constant interval.
- **adaptive**: true live adaptive collection with skipped reads.
- **dual**: dense collection plus offline adaptive replay for fidelity analysis.
- **overhead**: end-to-end workload comparison across baseline and monitored phases.

Probes
------
- ``syscall``  - eBPF-based syscall counter (requires root + Linux).
- ``cpu``      - Overall CPU utilisation from ``/proc/stat``.
- ``mem``      - Memory utilisation from ``/proc/meminfo``.
- ``disk``     - Disk I/O ops/s from ``/proc/diskstats``.
- ``net``      - Network bytes/s from ``/proc/net/dev``.

- ``cgroup_cpu``   - CPU utilisation from a single cgroup.
- ``cgroup_mem``   - Memory utilisation from a single cgroup.

Usage
-----
    # Regular dense syscall collection.
    sudo ./entrypoint.sh live --probe syscall --mode fixed --duration 60

    # True adaptive CPU monitoring.
    ./entrypoint.sh live --probe cpu --mode adaptive --duration 300

    # Dense collection plus offline adaptive replay.
    ./entrypoint.sh live --probe cpu --mode dual --duration 300

    # End-to-end overhead comparison.
    sudo ./entrypoint.sh live --probe syscall --mode overhead --syscall write --duration 15

"""

from __future__ import annotations

import argparse
import json
import logging as log
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config
from probe.base import Probe
from support.log import initialize_log
from tracker.dual_reader import DualReader, DualReaderResult
from tracker.pipeline import Pipeline, PipelineConfig, PipelineResult
from tracker.windowed import PollResult

# ------------------------------
# Probe factory
# ------------------------------

_PROBE_TYPES: dict[str, type] = {}

# BCC/eBPF is Linux-only
try:
    from probe.syscall_counter import SyscallCounter

    _PROBE_TYPES["syscall"] = SyscallCounter
except Exception:
    pass

# /proc probes are only importable on Linux
try:
    from probe.proc_stat import (
        ProcCpuProbe, ProcMemProbe, ProcDiskProbe, ProcNetProbe,
    )

    _PROBE_TYPES.update({
        "cpu": ProcCpuProbe,
        "mem": ProcMemProbe,
        "disk": ProcDiskProbe,
        "net": ProcNetProbe,
    })
except Exception:
    pass

# cgroup v2 probes (Linux only)
try:
    from probe.cgroup import CgroupCpuProbe, CgroupMemProbe

    _PROBE_TYPES.update({
        "cgroup_cpu": CgroupCpuProbe,
        "cgroup_mem": CgroupMemProbe,
    })
except Exception:
    pass


def _make_probe(name: str, **kwargs: Any) -> Probe:
    """Instantiate a probe by name."""
    if name not in _PROBE_TYPES:
        raise ValueError(
            f"Unknown probe '{name}'. "
            f"Available: {list(_PROBE_TYPES)}"
        )
    cls = _PROBE_TYPES[name]
    # Filter kwargs to what the constructor actually accepts
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid)


def _normalize_live_value(
        probe: Probe,
        raw_value: float,
        sample_window_ticks: int,
) -> float:
    """Normalize counter-style probes back to the base tick unit."""
    probe_type = probe.to_dict().get("type")
    if probe_type == "ebpf_syscall_counter":
        return float(raw_value) / max(1, sample_window_ticks)
    return float(raw_value)


def _summarize_live_sampling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_points = len(rows)
    sampled_indices = np.array([
        int(row["t"])
        for row in rows
        if row.get("sampled") and row.get("value") is not None
    ], dtype=np.int64)
    sampled_values = np.array([
        float(row["value"])
        for row in rows
        if row.get("sampled") and row.get("value") is not None
    ], dtype=np.float64)
    interval_series = np.array([float(row["interval"]) for row in rows], dtype=np.float64)
    urgency_series = np.array([float(row["urgency"]) for row in rows], dtype=np.float64)
    ratio_series = np.array([float(row["sample_ratio_live"]) for row in rows], dtype=np.float64)

    if sampled_indices.size > 1:
        gaps = np.diff(sampled_indices).astype(np.float64)
        max_gap = int(np.max(gaps))
        mean_gap = float(np.mean(gaps))
        p95_gap = float(np.percentile(gaps, 95))
    elif sampled_indices.size == 1:
        max_gap = 1
        mean_gap = 1.0
        p95_gap = 1.0
    else:
        max_gap = 0
        mean_gap = 0.0
        p95_gap = 0.0

    return {
        "n_total": int(total_points),
        "n_sampled": int(sampled_indices.size),
        "sample_ratio": float(sampled_indices.size / max(1, total_points)),
        "max_gap": max_gap,
        "mean_gap": mean_gap,
        "p95_gap": p95_gap,
        "sampled_indices": sampled_indices,
        "sampled_values": sampled_values,
        "interval_series": interval_series,
        "urgency_series": urgency_series,
        "ratio_series": ratio_series,
    }


def _plot_live_sampling(
        sampling: dict[str, Any],
        poll_interval_s: float,
        save_path: str | None = None,
) -> None:
    from matplotlib import pyplot as plt

    total_points = int(sampling["n_total"])
    sampled_indices = np.asarray(sampling["sampled_indices"])
    sampled_values = np.asarray(sampling["sampled_values"])
    interval_series = np.asarray(sampling["interval_series"])
    urgency_series = np.asarray(sampling["urgency_series"])
    ratio_series = np.asarray(sampling["ratio_series"])

    t_sec = np.arange(total_points) * poll_interval_s
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)

    ax = axes[0]
    if sampled_indices.size:
        ax.plot(sampled_indices * poll_interval_s, sampled_values,
                linewidth=0.8, color="steelblue", alpha=0.8)
        ax.scatter(sampled_indices * poll_interval_s, sampled_values,
                   s=12, color="midnightblue",
                   label=f"sampled ({sampled_indices.size}/{total_points})")
    ax.set_ylabel("Observed value")
    ax.set_title("Live Adaptive Collection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.step(t_sec, interval_series * poll_interval_s, where="post",
            color="darkorange", linewidth=0.9)
    ax.fill_between(t_sec, interval_series * poll_interval_s,
                    alpha=0.25, color="darkorange", step="post")
    ax.set_ylabel("Poll interval (s)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.step(t_sec, urgency_series, where="post", color="crimson", linewidth=0.9)
    ax.fill_between(t_sec, 0, urgency_series, alpha=0.30, color="crimson", step="post")
    ax.set_ylabel("Urgency")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.plot(t_sec, ratio_series, color="teal", linewidth=1.0)
    ax.fill_between(t_sec, ratio_series, alpha=0.30, color="teal")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
    ax.set_ylabel("Cum. sample ratio")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        log.info("Figure saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def _save_live_sampling_result(
        save_dir: str,
        probe: Probe,
        pipe_config: PipelineConfig,
        base_interval: float,
        comment: str | None,
        rows: list[dict[str, Any]],
        sampling: dict[str, Any],
        status_counts: dict[str, int],
) -> str:
    sampled_values = np.asarray(sampling["sampled_values"], dtype=np.float64)
    if sampled_values.size:
        trace_stats = {
            "n_points": int(sampled_values.size),
            "mean": float(np.mean(sampled_values)),
            "std": float(np.std(sampled_values)),
            "min": float(np.min(sampled_values)),
            "max": float(np.max(sampled_values)),
        }
    else:
        trace_stats = {
            "n_points": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_config": pipe_config.to_dict(),
        "generator": {
            **probe.to_dict(),
            "mode": "adaptive",
            "base_interval": base_interval,
            "live_sparse": True,
            "comment": comment,
        },
        "trace_stats": trace_stats,
        "result_summary": {
            "info_loss": None,
            "sampling_summary": {
                "n_total": int(sampling["n_total"]),
                "n_sampled": int(sampling["n_sampled"]),
                "sample_ratio": float(sampling["sample_ratio"]),
                "max_gap": int(sampling["max_gap"]),
                "mean_gap": float(sampling["mean_gap"]),
                "p95_gap": float(sampling["p95_gap"]),
            },
            "status_counts": {
                "adaptive": status_counts,
            },
        },
    }

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "experiment.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    trace_path = os.path.join(save_dir, "trace.json")
    with open(trace_path, "w") as handle:
        json.dump(rows, handle, indent=2, default=str)

    return path


# ------------------------------
# Mode: fixed (original behaviour)
# ------------------------------

def run_fixed(
        probe: Probe,
        poll_interval: float,
        duration: float,
        log_folder: str | None,
        comment: str | None = None,
) -> PipelineResult | None:
    """Read at a constant interval, then run the pipeline post-hoc."""
    samples: list[float] = []
    stop = False

    def _sig(_s: int, _f: Any) -> None:
        nonlocal stop
        stop = True

    prev = signal.signal(signal.SIGINT, _sig)

    try:
        with probe:
            start = time.monotonic()
            while not stop:
                time.sleep(poll_interval)
                value = probe.read()
                samples.append(value)
                log.info(f"  t={len(samples):>4d}  value={value:.1f}")
                if 0 < duration <= (time.monotonic() - start):
                    break
    finally:
        signal.signal(signal.SIGINT, prev)

    if not samples:
        log.warning("No samples collected.")
        return None

    trace = np.array(samples, dtype=np.float64)
    log.info(f"Collected {len(trace)} samples. Running pipeline ...")

    pipe = Pipeline()
    result = pipe.run(trace)
    [log.info(summary) for summary in result.summary().split("\n")]
    result.plot(save_dir=log_folder)

    if log_folder:
        result.save_json(
            save_dir=log_folder,
            generator={**probe.to_dict(), "mode": "fixed",
                       "poll_interval": poll_interval,
                       "comment": comment},
        )
        log.info(f"Experiment saved to {log_folder}")

    return result


# ------------------------------
# Mode: adaptive (real skipped-read loop)
# ------------------------------

def run_adaptive(
        probe: Probe,
        pipe_config: PipelineConfig,
        base_interval: float,
        duration: float,
        log_folder: str | None,
        comment: str | None = None,
) -> dict[str, Any] | None:
    """Adaptive live collection: urgency controls real sleep duration.

    The adaptive interval is evaluated on every base tick, but probe reads
    only happen when the current schedule says to sample. This means live
    adaptive runs produce sparse traces and should be evaluated with live
    sampling summaries rather than dense information-loss metrics.
    """
    poller = pipe_config._make_poller()
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    stop = False

    def _sig(_s: int, _f: Any) -> None:
        nonlocal stop
        stop = True

    prev = signal.signal(signal.SIGINT, _sig)

    try:
        with probe:
            start = time.monotonic()
            tick = 0
            n_sampled = 0
            ticks_since_sample = 0

            while not stop:
                time.sleep(base_interval)
                tick += 1
                tick_index = tick - 1
                ticks_since_sample += 1

                current_interval = poller.interval
                should_sample = (not poller.tracker._initialised) or (
                        ticks_since_sample >= current_interval
                )

                value = None
                sample_window_ticks = None
                tick_result = None

                if should_sample:
                    sample_window_ticks = ticks_since_sample
                    raw_value = probe.read()
                    value = _normalize_live_value(probe, raw_value, sample_window_ticks)
                    tick_result = poller.tracker.update(value)
                    poller._steps_since_sample = 0
                    poller._recent_statuses.append(tick_result.status)
                    poller._update_urgency(tick_result)
                    n_sampled += 1
                    ticks_since_sample = 0
                    status_name = tick_result.status.name
                    status_counts[status_name] = status_counts.get(status_name, 0) + 1
                    pr = PollResult(
                        time_index=tick_index,
                        sampled=True,
                        tick=tick_result,
                        interval=poller.interval,
                        urgency=poller._urgency,
                    )
                else:
                    poller._steps_since_sample = ticks_since_sample
                    pr = PollResult(
                        time_index=tick_index,
                        sampled=False,
                        tick=None,
                        interval=current_interval,
                        urgency=poller._urgency,
                    )
                    status_name = None

                rows.append({
                    "t": tick_index,
                    "value": None if value is None else float(value),
                    "urgency": float(pr.urgency),
                    "interval": int(pr.interval),
                    "sampled": bool(pr.sampled),
                    "status": status_name,
                    "sample_window_ticks": sample_window_ticks,
                    "sample_ratio_live": float(n_sampled / max(1, tick)),
                })

                if pr.sampled:
                    log.info(
                        "  t=%4d  elapsed_s=%.1f  value=%10.1f  urgency=%.3f  "
                        "next_interval=%d  sampled=yes  window=%d  status=%s",
                        tick_index,
                        time.monotonic() - start,
                        value,
                        pr.urgency,
                        pr.interval,
                        sample_window_ticks,
                        status_name,
                    )
                else:
                    log.info(
                        "  t=%4d  elapsed_s=%.1f  urgency=%.3f  next_interval=%d  sampled=no",
                        tick_index,
                        time.monotonic() - start,
                        pr.urgency,
                        pr.interval,
                    )

                if 0 < duration <= (time.monotonic() - start):
                    break

    finally:
        signal.signal(signal.SIGINT, prev)

    if not rows or not any(row["sampled"] for row in rows):
        log.warning("No samples collected.")
        return None

    sampling = _summarize_live_sampling(rows)
    log.info(
        "Adaptive live sampling: %d sampled ticks over %d total (%.1f%%)",
        sampling["n_sampled"],
        sampling["n_total"],
        sampling["sample_ratio"] * 100.0,
    )
    log.info(
        "Sampling gaps: max=%d  mean=%.1f  p95=%.1f",
        sampling["max_gap"],
        sampling["mean_gap"],
        sampling["p95_gap"],
    )
    if status_counts:
        log.info("Adaptive sampled-status counts: %s", status_counts)

    if log_folder:
        _plot_live_sampling(
            sampling,
            poll_interval_s=base_interval,
            save_path=os.path.join(log_folder, "live_adaptive_collection.png"),
        )
        _save_live_sampling_result(
            save_dir=log_folder,
            probe=probe,
            pipe_config=pipe_config,
            base_interval=base_interval,
            comment=comment,
            rows=rows,
            sampling=sampling,
            status_counts=status_counts,
        )
        log.info(f"Experiment saved to {log_folder}")

    return {
        "sampling_summary": {
            "n_total": int(sampling["n_total"]),
            "n_sampled": int(sampling["n_sampled"]),
            "sample_ratio": float(sampling["sample_ratio"]),
            "max_gap": int(sampling["max_gap"]),
            "mean_gap": float(sampling["mean_gap"]),
            "p95_gap": float(sampling["p95_gap"]),
        },
        "status_counts": status_counts,
    }


# ------------------------------
# Mode: dual (dense-reference evaluation)
# ------------------------------

def run_dual(
        probe: Probe,
        pipe_config: PipelineConfig,
        base_interval: float,
        duration: float,
        log_folder: str | None,
        comment: str | None = None,
) -> DualReaderResult | None:
    """Fixed-rate reading with dual processing tracks.

    Best for accuracy evaluation: every reading is collected (no data
    loss from the collection itself), while the adaptive poller's
    skip/sample decisions reveal what *would* have been missed.
    """
    with probe:
        reader = DualReader(
            probe=probe,
            config=pipe_config,
            base_interval=base_interval,
        )
        dr = reader.run(duration=duration)

    [log.info(summary) for summary in str(dr.info_loss).split("\n")]
    log.info(f"Mean read overhead: {dr.mean_read_overhead_us:.1f} us")

    # Convert for plotting
    pr = dr.to_pipeline_result()
    pr.plot(save_dir=log_folder)

    if log_folder:
        pr.save_json(
            save_dir=log_folder,
            generator={
                **probe.to_dict(),
                "mode": "dual",
                "base_interval": base_interval,
                "mean_read_overhead_us": dr.mean_read_overhead_us,
                "comment": comment,
            },
        )
        log.info(f"Experiment saved to {log_folder}")

    return dr


# ------------------------------
# Mode: overhead (end-to-end, phase-based)
# ------------------------------

def run_overhead(
        probe_factory: Any,
        pipe_config: PipelineConfig,
        base_interval: float,
        duration: float,
        log_folder: str | None,
        comment: str | None = None,
        fixed_rates: list[float] | None = None,
        fixed_steps: int = 8,
        repeats: int = 1,
        warmup: float = 0.0,
        include_idle: bool = True,
        include_adaptive: bool = True,
        command_template: str = "",
        synthetic_workload: str = "write",
        synthetic_block_size: int = 4096,
        syscall_name: str | None = None,
) -> None:
    """Run phase-based end-to-end overhead evaluation."""
    from eval.overhead import (
        make_command_workload_runner,
        make_synthetic_syscall_workload_runner,
        measure_end_to_end_overhead,
        nice_rates,
    )

    if fixed_rates is None:
        fixed_rates = nice_rates(
            min_interval=pipe_config.min_interval,
            max_interval=pipe_config.max_interval,
            base_interval=base_interval,
            max_steps=fixed_steps,
        )

    if command_template:
        workload_runner = make_command_workload_runner(
            command_template,
            extra_format={
                "syscall": syscall_name or "",
                "block_size": synthetic_block_size,
            },
        )
    else:
        workload_kind = synthetic_workload
        if workload_kind == "auto":
            workload_kind = syscall_name or "write"
        workload_runner = make_synthetic_syscall_workload_runner(
            workload_kind,
            block_size=synthetic_block_size,
        )

    report = measure_end_to_end_overhead(
        probe_factory=probe_factory,
        workload_runner=workload_runner,
        phase_duration=duration,
        fixed_intervals=fixed_rates,
        adaptive_base_interval=base_interval,
        warmup_s=warmup,
        repeats=repeats,
        include_idle=include_idle,
        include_adaptive=include_adaptive,
        pipe_config=pipe_config,
    )

    for line in report.summary_lines():
        log.info(line)

    if log_folder:
        payload = {
            "comment": comment,
            "mode": "overhead",
            "duration_s": duration,
            "warmup_s": warmup,
            "repeats": repeats,
            "fixed_rates_s": fixed_rates,
            "include_idle": include_idle,
            "include_adaptive": include_adaptive,
            "probe": probe_factory().to_dict(),
            "report": report.to_dict(),
        }
        path = os.path.join(log_folder, "overhead.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info("Overhead results saved to %s", path)


# ------------------------------
# CLI
# ------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Live probe collection: fixed, adaptive, dual, or overhead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--probe", default="syscall",
        choices=list(_PROBE_TYPES),
        help="Monitoring probe to use.",
    )
    p.add_argument(
        "--mode", default="fixed",
        choices=["fixed", "adaptive", "dual", "overhead"],
        help="Collection mode: dense fixed, sparse adaptive, dense+offline dual, or end-to-end overhead.",
    )
    p.add_argument(
        "--interval", type=float, default=config.EBPF_POLL_INTERVAL,
        help="Base collection interval in seconds.",
    )
    p.add_argument(
        "--duration", type=float, default=300,
        help="Total capture duration in seconds.",
    )
    p.add_argument(
        "--comment", type=str, default="",
        help="Free-text note describing this experiment.",
    )
    p.add_argument(
        "--log-level", "-l", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # End-to-end overhead options
    p.add_argument(
        "--overhead-rates", type=str, default="",
        help="Comma-separated fixed collection intervals for overhead mode. "
             "Overrides auto-derived fixed-rate comparison legs.",
    )
    p.add_argument(
        "--overhead-steps", type=int, default=8,
        help="Max number of auto-derived fixed-rate comparison legs in overhead mode.",
    )
    p.add_argument(
        "--overhead-repeats", type=int, default=1,
        help="How many times to run the full phase sequence in overhead mode.",
    )
    p.add_argument(
        "--overhead-warmup", type=float, default=0.0,
        help="Warm-up time after attaching the monitor and before starting the workload.",
    )
    p.add_argument(
        "--overhead-command", type=str, default="",
        help="External workload command template for overhead mode. Supports "
             "{duration}, {syscall}, and {block_size} placeholders.",
    )
    p.add_argument(
        "--overhead-workload", type=str,
        default="auto",
        choices=["auto", "write", "pwrite64", "fsync", "read", "openat"],
        help="Synthetic workload used by overhead mode when no external command is provided.",
    )
    p.add_argument(
        "--overhead-block-size", type=int, default=4096,
        help="Block size for the synthetic overhead workload.",
    )
    p.add_argument(
        "--overhead-no-idle", action="store_true",
        help="Skip the attached-idle phase in overhead mode.",
    )
    p.add_argument(
        "--overhead-no-adaptive", action="store_true",
        help="Skip the attached-adaptive phase in overhead mode.",
    )

    # Probe-specific
    p.add_argument(
        "--syscall", default=config.EBPF_SYSCALL,
        help="Syscall to trace (syscall probe only).",
    )
    p.add_argument(
        "--cgroup-path", type=str, default=None,
        help="Cgroup v2 directory path (cgroup_cpu / cgroup_mem probes). "
             "Auto-detected from /proc/self/cgroup if omitted.",
    )

    args = p.parse_args()

    log_folder = initialize_log(
        log_level=args.log_level,
        name="live",
        console_only=False,
        application_type=f"live-{args.probe}-{args.mode}",
        create_out_subfolders=True,
    )

    probe_kwargs = {
        "syscall": args.syscall,
        "poll_interval": args.interval,
        "cgroup_path": args.cgroup_path,
    }

    def _probe_factory() -> Probe:
        return _make_probe(args.probe, **probe_kwargs)

    probe = _probe_factory()

    pipe_config = PipelineConfig()

    # Write comment to log and to a file inside the output folder
    if args.comment:
        log.info("Comment: %s", args.comment)
        if log_folder:
            with open(os.path.join(log_folder, "COMMENT.txt"), "w") as f:
                f.write(args.comment + "\n")

    log.info(
        "Probe: %s | Mode: %s | Interval: %.2fs | Duration: %.0fs",
        args.probe, args.mode, args.interval, args.duration,
    )

    comment = args.comment or None

    if args.mode == "fixed":
        result = run_fixed(
            probe, args.interval, args.duration, log_folder,
            comment=comment,
        )
        sys.exit(0 if result is not None else 1)

    elif args.mode == "adaptive":
        result = run_adaptive(
            probe, pipe_config, args.interval, args.duration, log_folder,
            comment=comment,
        )
        sys.exit(0 if result is not None else 1)

    elif args.mode == "dual":
        result = run_dual(
            probe, pipe_config, args.interval, args.duration, log_folder,
            comment=comment,
        )
        sys.exit(0 if result is not None else 1)

    elif args.mode == "overhead":
        overhead_rates = None
        if args.overhead_rates:
            overhead_rates = [float(r) for r in args.overhead_rates.split(",")]
        run_overhead(
            probe_factory=_probe_factory,
            pipe_config=pipe_config,
            base_interval=args.interval,
            duration=args.duration,
            log_folder=log_folder,
            comment=comment,
            fixed_rates=overhead_rates,
            fixed_steps=args.overhead_steps,
            repeats=args.overhead_repeats,
            warmup=args.overhead_warmup,
            include_idle=not args.overhead_no_idle,
            include_adaptive=not args.overhead_no_adaptive,
            command_template=args.overhead_command,
            synthetic_workload=args.overhead_workload,
            synthetic_block_size=args.overhead_block_size,
            syscall_name=args.syscall if isinstance(args.syscall, str) else None,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
