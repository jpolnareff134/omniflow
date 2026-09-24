"""
Minimal eBPF syscall counter.

Attaches to ``raw_syscalls:sys_enter``, filters by syscall number, and
increments a per-CPU array.  Userspace reads the map at a configurable
polling interval and yields the aggregate count-per-interval.

Supports monitoring a **single** syscall or an **aggregate** of multiple
syscalls (e.g. ``write`` + ``pwrite64`` + ``fsync``) via the ``syscalls``
constructor parameter.

Requires:
    - Linux kernel >= 4.15
    - ``bcc`` Python package (``python3-bcc`` / ``pip install bcc``)
    - root or ``CAP_BPF`` + ``CAP_PERFMON``
"""

from __future__ import annotations

import ctypes
import signal
import time
from collections.abc import Generator
from typing import Any, Optional

# noinspection PyUnresolvedReferences
from bcc import BPF

import config
from probe.base import Probe

# ------------------------------
# BPF C source templates
# ------------------------------

# Single-syscall variant – backward-compatible placeholder approach.
_BPF_SRC_SINGLE = r"""
#include <uapi/linux/ptrace.h>

BPF_ARRAY(counter, u64, 1);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    if (args->id != SYSCALL_NR)
        return 0;
    int zero = 0;
    u64 *val = counter.lookup(&zero);
    if (val)
        __sync_fetch_and_add(val, 1);
    return 0;
}
"""

# Multi-syscall variant – SYSCALL_FILTER is replaced by a generated block.
_BPF_SRC_MULTI = r"""
#include <uapi/linux/ptrace.h>

BPF_ARRAY(counter, u64, 1);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    SYSCALL_FILTER
    int zero = 0;
    u64 *val = counter.lookup(&zero);
    if (val)
        __sync_fetch_and_add(val, 1);
    return 0;
}
"""

# Keep the old name as an alias for callers that imported it directly.
_BPF_SRC = _BPF_SRC_SINGLE


def _make_multi_filter(nrs: list[int]) -> str:
    """Return a C snippet that passes only syscalls whose NR is in *nrs*."""
    checks = " || ".join(f"args->id == {nr}" for nr in nrs)
    return f"if (!({checks})) return 0;"


# ------------------------------
# Helpers
# ------------------------------

def _resolve_syscall_nr(name: str) -> int:
    """Look up the syscall number for *name* via ``ausyscall`` or the
    kernel's ``unistd_64.h`` constants exposed through ``/usr/include``.

    Falls back to a small built-in table for the most common calls.
    """
    import subprocess
    import shutil

    # Try ausyscall first (audit package)
    if shutil.which("ausyscall"):
        try:
            out = subprocess.check_output(
                ["ausyscall", "--exact", name], text=True,
            ).strip()
            return int(out)
        except (subprocess.CalledProcessError, ValueError):
            pass

    # Try python-based lookup via syscall tables
    try:
        import os, re
        for header in (
                "/usr/include/asm/unistd_64.h",
                "/usr/include/x86_64-linux-gnu/asm/unistd_64.h",
                "/usr/include/asm-generic/unistd.h",
        ):
            if not os.path.isfile(header):
                continue
            with open(header) as f:
                for line in f:
                    m = re.match(
                        rf"#define\s+__NR_(?:3264_)?{re.escape(name)}\s+(\d+)",
                        line,
                    )
                    if m:
                        return int(m.group(1))
    except Exception:
        pass

    # Built-in fallback for common syscalls (x86-64)
    _COMMON: dict[str, int] = {
        "read": 0, "write": 1, "open": 2, "close": 3,
        "stat": 4, "fstat": 5, "lseek": 8, "mmap": 9,
        "poll": 7, "mprotect": 10, "ioctl": 16,
        "pread64": 17, "pwrite64": 18,
        "access": 21, "pipe": 22, "select": 23, "sched_yield": 24,
        "dup": 32, "dup2": 33, "getpid": 39,
        "socket": 41, "connect": 42, "accept": 43,
        "sendto": 44, "recvfrom": 45, "bind": 49, "listen": 50,
        "clone": 56, "fork": 57, "execve": 59,
        "exit": 60, "kill": 62,
        "fsync": 74, "fdatasync": 75, "ftruncate": 77,
        "getdents": 78, "getcwd": 79, "rename": 82,
        "mkdir": 83, "unlink": 87, "symlink": 88,
        "futex": 202, "epoll_wait": 232, "openat": 257,
        "preadv": 295, "pwritev": 296,
        "preadv2": 327, "pwritev2": 328,
    }
    if name in _COMMON:
        return _COMMON[name]

    raise ValueError(
        f"Cannot resolve syscall '{name}'. "
        "Install the 'audit' / 'auditd' package for ausyscall, "
        "or pass a numeric syscall number."
    )


# ------------------------------
# Main counter class
# ------------------------------

class SyscallCounter(Probe):
    """eBPF-backed syscall rate counter.

    Can monitor a **single** syscall or aggregate the counts for a
    **list** of syscalls (e.g. ``["write", "pwrite64", "fsync"]``).

    Parameters
    ----------
    syscall : str | int | None
        Single syscall name or number.  Mutually exclusive with *syscalls*.
        Falls back to ``config.EBPF_SYSCALL`` when both are ``None``.
    syscalls : list[str | int] | None
        List of syscall names / numbers to aggregate.  When provided,
        *syscall* must be ``None``.
    poll_interval : float
        Seconds between userspace reads.
    """

    def __init__(
            self,
            syscall: str | int | None = None,
            poll_interval: float | None = None,
            *,
            syscalls: list[str | int] | None = None,
    ) -> None:
        if syscalls is not None and syscall is not None:
            raise ValueError("Specify either 'syscall' or 'syscalls', not both.")

        self.poll_interval = (
            poll_interval if poll_interval is not None
            else config.EBPF_POLL_INTERVAL
        )

        if syscalls is not None:
            # Multi-syscall mode
            resolved = []
            names = []
            for sc in syscalls:
                if isinstance(sc, int):
                    resolved.append(sc)
                    names.append(str(sc))
                else:
                    resolved.append(_resolve_syscall_nr(sc))
                    names.append(sc)
            self._syscall_nrs: list[int] = resolved
            self.syscall_name: str = "+".join(names)
            # Keep singular attr for backward compat (first NR)
            self.syscall_nr: int = resolved[0]
        else:
            sc = syscall if syscall is not None else config.EBPF_SYSCALL
            if isinstance(sc, int):
                self.syscall_nr = sc
                self.syscall_name = str(sc)
            else:
                self.syscall_name = sc
                self.syscall_nr = _resolve_syscall_nr(sc)
            self._syscall_nrs = [self.syscall_nr]

        self._bpf: Optional[BPF] = None

    # -- lifecycle -----------------------------------------------------------

    def attach(self) -> None:
        """Compile the BPF program and attach to the tracepoint."""
        if len(self._syscall_nrs) == 1:
            src = _BPF_SRC_SINGLE.replace("SYSCALL_NR", str(self._syscall_nrs[0]))
        else:
            filt = _make_multi_filter(self._syscall_nrs)
            src = _BPF_SRC_MULTI.replace("SYSCALL_FILTER", filt)
        self._bpf = BPF(text=src)
        # Tracepoint is auto-attached by TRACEPOINT_PROBE macro.

    def detach(self) -> None:
        """Clean up BPF resources."""
        if self._bpf is not None:
            self._bpf.cleanup()
            self._bpf = None

    # -- Probe interface (delegates to attach/detach/read_and_reset) ---------

    def start(self) -> None:  # noqa: D102
        self.attach()

    def stop(self) -> None:  # noqa: D102
        self.detach()

    def read(self) -> float:  # noqa: D102
        return float(self.read_and_reset())

    def __enter__(self) -> "SyscallCounter":
        self.attach()
        return self

    def __exit__(self, *_: Any) -> None:
        self.detach()

    # -- reading -------------------------------------------------------------

    def read_and_reset(self) -> int:
        """Read the current aggregate count and reset the map to zero."""
        assert self._bpf is not None, "Call attach() first"
        table = self._bpf["counter"]
        total = 0
        for cpu_val in table.values():
            total += cpu_val.value
        # Reset
        table[ctypes.c_int(0)] = ctypes.c_ulong(0)
        return total

    def stream(self, duration: float = 0) -> Generator[float, None, None]:
        """Yield syscall counts per interval.

        Parameters
        ----------
        duration : float
            Total seconds to stream (0 = indefinite, stop with Ctrl-C).

        Yields
        ------
        float
            Aggregate syscall count observed during the last interval.
        """
        assert self._bpf is not None, "Call attach() first"
        stop = False

        def _sig(s: int, f: Any) -> None:
            nonlocal stop
            stop = True

        prev = signal.signal(signal.SIGINT, _sig)
        try:
            start = time.monotonic()
            while not stop:
                time.sleep(self.poll_interval)
                yield float(self.read_and_reset())
                if 0 < duration <= (time.monotonic() - start):
                    break
        finally:
            signal.signal(signal.SIGINT, prev)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description for experiment manifests."""
        return {
            "type": "ebpf_syscall_counter",
            "syscall": self.syscall_name,
            "syscall_nr": self.syscall_nr,
            "poll_interval": self.poll_interval,
        }
