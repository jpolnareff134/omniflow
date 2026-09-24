#!/usr/bin/env python3
"""Classify the Tradebeans workload contained in the released companion JAR.

The supplied release launches a benchmark named ``tradebeans`` but its active
harness is backed by the H2/TPCC implementation and does not contain the
DayTrader request class described in the paper.  The suite still executes this
exact released arm, clearly labelled, to avoid cherry-picking it out of the
comparison.  Exit status 66 means "released TPCC substitute", not "unusable";
callers decide whether they need strict DayTrader semantics or release-artifact
reproduction.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DAYTRADER_SUFFIX = "org/apache/geronimo/daytrader/javaee6/dacapo/DaCapoTrader.class"
WORKLOAD_SOURCE = "harness/src/org/dacapo/harness/TradebeansWorkload.java"
CONFIG_SOURCE = "cnf/tradebeans.cnf"


def read_member(zf: zipfile.ZipFile, name: str) -> str:
    try:
        return zf.read(name).decode("utf-8", errors="replace")
    except KeyError:
        return ""


def classify(path: Path) -> tuple[str, list[str]]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        daytrader = [name for name in names if name.endswith(DAYTRADER_SUFFIX)]
        workload = read_member(zf, WORKLOAD_SOURCE)
        config = read_member(zf, CONFIG_SOURCE)

    reasons: list[str] = []
    if daytrader:
        reasons.append(f"DayTrader request class present: {daytrader[0]}")
    else:
        reasons.append("DayTrader request class absent")
    if "TPCCSubmitter" in workload or "org.dacapo.h2" in workload:
        reasons.append("TradebeansWorkload source references the H2/TPCC implementation")
    if "dacapo-h2.jar" in config:
        reasons.append("tradebeans.cnf stages dacapo-h2.jar")

    if daytrader and "TPCCSubmitter" not in workload and "org.dacapo.h2" not in workload:
        return "daytrader-candidate", reasons
    return "tpcc-substitute", reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jar", type=Path)
    parser.add_argument(
        "--allow-substitute",
        action="store_true",
        help="return success for the exact released TPCC-backed arm",
    )
    args = parser.parse_args()
    if not args.jar.is_file():
        raise SystemExit(f"missing Tradebeans artifact: {args.jar}")
    try:
        status, reasons = classify(args.jar)
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"invalid JAR/ZIP: {args.jar}: {exc}") from exc
    print(f"status={status}")
    print(f"artifact={args.jar.resolve()}")
    for reason in reasons:
        print(f"evidence={reason}")
    if status == "tpcc-substitute":
        print("released_artifact_arm=enabled_when_explicitly_labelled")
        print("daytrader_semantic_reproduction=false")
    if status != "daytrader-candidate" and not args.allow_substitute:
        print(
            "classification_exit=66 (released TPCC-backed arm; strict DayTrader semantics unavailable)",
            file=sys.stderr,
        )
        raise SystemExit(66)


if __name__ == "__main__":
    main()
