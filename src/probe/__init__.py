from probe.base import Probe

# BCC (eBPF) is Linux-only; /proc probes are Linux-only too.
# Guard both so macOS dev machines can still import the package.
try:
    from probe.syscall_counter import SyscallCounter
except Exception:  # pragma: no cover
    pass

try:
    from probe.proc_stat import (
        ProcCpuProbe, ProcMemProbe, ProcDiskProbe, ProcNetProbe,
    )
except Exception:  # pragma: no cover
    pass

try:
    from probe.cgroup import CgroupCpuProbe, CgroupMemProbe
except Exception:  # pragma: no cover
    pass

__all__ = [
    "Probe",
    "SyscallCounter",
    "ProcCpuProbe",
    "ProcMemProbe",
    "ProcDiskProbe",
    "ProcNetProbe",
    "CgroupCpuProbe",
    "CgroupMemProbe",
]
