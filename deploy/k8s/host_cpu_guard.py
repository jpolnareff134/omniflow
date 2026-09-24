#!/usr/bin/env python3
"""Pin Minikube QEMU processes and reject host CPU contamination."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            if first > last:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(part))
    if not cpus or min(cpus) < 0:
        raise ValueError("CPU list must contain non-negative CPU numbers")
    return cpus


def format_cpu_list(cpus: set[int]) -> str:
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _read_argv(path: Path) -> list[str]:
    try:
        return [
            value.decode(errors="replace")
            for value in path.read_bytes().split(b"\0")
            if value
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def find_qemu_pids(profile: str, proc_root: Path = Path("/proc")) -> list[int]:
    pids = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        argv = _read_argv(entry / "cmdline")
        if (
            argv
            and os.path.basename(argv[0]).startswith("qemu-system-")
            and profile in " ".join(argv[1:])
        ):
            pids.append(int(entry.name))
    return sorted(pids)


def _task_ids(pid: int, proc_root: Path = Path("/proc")) -> list[int]:
    try:
        return sorted(int(path.name) for path in (proc_root / str(pid) / "task").iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def pin_qemu_processes(
    profile: str, cpus: set[int], expected_processes: int = 2
) -> dict:
    pids = find_qemu_pids(profile)
    if len(pids) != expected_processes:
        raise RuntimeError(
            f"expected {expected_processes} QEMU processes for {profile!r}, found {pids}"
        )
    pinned_tasks: dict[str, list[int]] = {}
    for _ in range(3):
        for pid in pids:
            tids = _task_ids(pid)
            if not tids:
                raise RuntimeError(f"QEMU process {pid} disappeared while pinning")
            for tid in tids:
                os.sched_setaffinity(tid, cpus)
            pinned_tasks[str(pid)] = tids
        time.sleep(0.1)
        current_tasks = {pid: _task_ids(pid) for pid in pids}
        if all(current_tasks.values()) and all(
            os.sched_getaffinity(tid) == cpus
            for tids in current_tasks.values()
            for tid in tids
        ):
            return {
                "profile": profile,
                "cpus": format_cpu_list(cpus),
                "qemu_pids": pids,
                "pinned_tasks": pinned_tasks,
                "wall_ns": time.time_ns(),
            }
    raise RuntimeError("QEMU threads did not retain the requested CPU affinity")


def _read_cpu_ticks(cpus: set[int], proc_root: Path = Path("/proc")) -> dict[int, tuple[int, int]]:
    result = {}
    with (proc_root / "stat").open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
                continue
            cpu = int(fields[0][3:])
            if cpu not in cpus:
                continue
            values = [int(value) for value in fields[1:]]
            # guest and guest_nice are already included in user and nice.
            total = sum(values[:8])
            # User, nice, and system account process work. IRQ/softirq time is
            # excluded because host networking for the pinned VMs executes in
            # those contexts and cannot be attributed to an unrelated process.
            process_busy = sum(values[:3])
            result[cpu] = (total, process_busy)
    if set(result) != cpus:
        raise RuntimeError(f"missing CPU counters for {sorted(cpus - set(result))}")
    return result


def _read_process_ticks(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    text = (proc_root / str(pid) / "stat").read_text()
    suffix = text[text.rfind(")") + 2 :].split()
    return int(suffix[11]) + int(suffix[12]), int(suffix[19])


def take_snapshot(
    profile: str,
    cpus: set[int],
    expected_processes: int = 2,
    proc_root: Path = Path("/proc"),
) -> dict:
    pids = find_qemu_pids(profile, proc_root)
    if len(pids) != expected_processes:
        raise RuntimeError(
            f"expected {expected_processes} QEMU processes for {profile!r}, found {pids}"
        )
    qemu = {}
    for pid in pids:
        ticks, start_time = _read_process_ticks(pid, proc_root)
        if proc_root == Path("/proc"):
            affinities = {frozenset(os.sched_getaffinity(tid)) for tid in _task_ids(pid)}
            if affinities != {frozenset(cpus)}:
                rendered = sorted(format_cpu_list(set(item)) for item in affinities)
                raise RuntimeError(f"QEMU process {pid} has unexpected affinities {rendered}")
        qemu[str(pid)] = {"ticks": ticks, "start_time": start_time}
    return {
        "monotonic_ns": time.monotonic_ns(),
        "wall_ns": time.time_ns(),
        "cpu": _read_cpu_ticks(cpus, proc_root),
        "qemu": qemu,
    }


def calculate_interval(previous: dict, current: dict) -> dict:
    if {
        pid: item["start_time"] for pid, item in previous["qemu"].items()
    } != {
        pid: item["start_time"] for pid, item in current["qemu"].items()
    }:
        raise RuntimeError("QEMU process identity changed during monitoring")
    total_ticks = sum(
        current["cpu"][cpu][0] - previous["cpu"][cpu][0]
        for cpu in previous["cpu"]
    )
    busy_ticks = sum(
        current["cpu"][cpu][1] - previous["cpu"][cpu][1]
        for cpu in previous["cpu"]
    )
    qemu_ticks = sum(
        current["qemu"][pid]["ticks"] - previous["qemu"][pid]["ticks"]
        for pid in previous["qemu"]
    )
    if total_ticks <= 0 or min(busy_ticks, qemu_ticks) < 0:
        raise RuntimeError("non-monotonic host CPU counters")
    foreign_ticks = max(0, busy_ticks - qemu_ticks)
    return {
        "total_ticks": total_ticks,
        "busy_ticks": busy_ticks,
        "qemu_ticks": qemu_ticks,
        "foreign_busy_ticks": foreign_ticks,
        "host_busy_fraction": busy_ticks / total_ticks,
        "qemu_fraction": qemu_ticks / total_ticks,
        "foreign_busy_fraction": foreign_ticks / total_ticks,
    }


def record_failure(args: argparse.Namespace, reason: str, last_sample: dict | None = None) -> None:
    failure = {
        "reason": reason,
        "phase": args.phase,
        "cpus": args.cpus,
        "threshold": args.threshold,
        "required_consecutive": args.consecutive,
        "last_sample": last_sample,
        "wall_ns": time.time_ns(),
    }
    if args.failure_file:
        Path(args.failure_file).write_text(json.dumps(failure, indent=2) + "\n")
    if args.signal_pid:
        try:
            os.kill(args.signal_pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass


def monitor(args: argparse.Namespace) -> int:
    cpus = parse_cpu_list(args.cpus)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    consecutive = 0
    previous = take_snapshot(args.profile, cpus, args.expected_processes)
    while True:
        stopping = bool(args.stop_file and Path(args.stop_file).exists())
        remaining = args.duration - (time.monotonic() - started) if args.duration else None
        if remaining is not None and remaining <= 0:
            return 0
        if stopping:
            elapsed = (time.monotonic_ns() - previous["monotonic_ns"]) / 1e9
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
        else:
            time.sleep(min(args.interval, remaining) if remaining is not None else args.interval)
        current = take_snapshot(args.profile, cpus, args.expected_processes)
        sample = calculate_interval(previous, current)
        consecutive = consecutive + 1 if sample["foreign_busy_fraction"] > args.threshold else 0
        record = {
            "phase": args.phase,
            "cpus": format_cpu_list(cpus),
            "threshold": args.threshold,
            "consecutive_excess": consecutive,
            "wall_ns": current["wall_ns"],
            "interval_s": (current["monotonic_ns"] - previous["monotonic_ns"]) / 1e9,
            **sample,
        }
        with output.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        if consecutive >= args.consecutive:
            record_failure(
                args, "foreign CPU activity exceeded threshold", record
            )
            return 120
        if stopping:
            return 0
        previous = current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pin = subparsers.add_parser("pin")
    pin.add_argument("--profile", required=True)
    pin.add_argument("--cpus", required=True)
    pin.add_argument("--expected-processes", type=int, default=2)

    watch = subparsers.add_parser("monitor")
    watch.add_argument("--profile", required=True)
    watch.add_argument("--cpus", required=True)
    watch.add_argument("--expected-processes", type=int, default=2)
    watch.add_argument("--phase", required=True)
    watch.add_argument("--output", required=True)
    watch.add_argument("--failure-file")
    watch.add_argument("--stop-file")
    watch.add_argument("--signal-pid", type=int)
    watch.add_argument("--duration", type=float, default=0)
    watch.add_argument("--interval", type=float, default=1.0)
    watch.add_argument("--threshold", type=float, default=0.05)
    watch.add_argument("--consecutive", type=int, default=3)

    args = parser.parse_args()
    try:
        if args.command == "pin":
            result = pin_qemu_processes(
                args.profile, parse_cpu_list(args.cpus), args.expected_processes
            )
            print(json.dumps(result, indent=2))
            return 0
        return monitor(args)
    except (OSError, RuntimeError, ValueError) as exc:
        if args.command == "monitor":
            record_failure(args, str(exc))
        print(json.dumps({"error": str(exc), "wall_ns": time.time_ns()}))
        return 120


if __name__ == "__main__":
    raise SystemExit(main())
