#!/usr/bin/env python3
"""Canonical metadata for the original Mertz--Nunes DaCapo workloads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Benchmark:
    name: str
    jar_name: str
    size: str
    smoke_size: str
    threads: int
    seconds: int
    heap: str
    type_rule: str
    type_count: int
    extra_jvm_args: tuple[str, ...] = ()


BENCHMARKS: dict[str, Benchmark] = {
    "cassandra": Benchmark(
        "cassandra", "original-cassandra.jar", "default", "small", 200, 1660, "4g", "min", 100
    ),
    "h2": Benchmark(
        "h2", "original-benchmarks.jar", "huge", "small", 20, 1780, "4g", "exact", 8
    ),
    "lusearch": Benchmark(
        "lusearch", "original-benchmarks.jar", "large", "small", 6, 1640, "4g", "min", 100
    ),
    "tradebeans": Benchmark(
        "tradebeans", "original-benchmarks.jar", "huge", "small", 24, 1780, "4g", "exact", 8,
        ("-Djboss.modules.system.pkgs=org.dacapo.omniflow.agent",),
    ),
    "xalan": Benchmark(
        "xalan", "original-benchmarks.jar", "large", "small", 6, 1640, "4g", "exact", 17
    ),
}

ORDER = ("cassandra", "h2", "lusearch", "tradebeans", "xalan")
POLICY_ORDER = ("adp", "uni", "full", "inv", "omni", "nom")
MONITORED_POLICIES = frozenset(("adp", "uni", "full", "inv", "omni"))


def parse_benchmarks(value: str) -> list[Benchmark]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names or names == ["all"]:
        names = list(ORDER)
    unknown = sorted(set(names) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
    seen: set[str] = set()
    result: list[Benchmark] = []
    for name in ORDER:
        if name in names and name not in seen:
            result.append(BENCHMARKS[name])
            seen.add(name)
    return result


def runtime_jar(root: Path, benchmark: Benchmark) -> Path:
    return root / "runtime" / benchmark.jar_name
