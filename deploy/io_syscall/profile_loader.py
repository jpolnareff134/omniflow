#!/usr/bin/env python3
"""Shared adaptive profile loader for the I/O experiment harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_PROFILE_PATH = Path(__file__).with_name("adaptive_profiles.json")
_ENV_KEY_MAP = {
    "alpha_base": "OMNIFLOW_ALPHA_BASE",
    "beta": "OMNIFLOW_BETA",
    "outlier_threshold": "OMNIFLOW_OUTLIER_THRESHOLD",
    "drift_tolerance": "OMNIFLOW_DRIFT_TOLERANCE",
    "initial_variance_floor": "OMNIFLOW_TRACKER_VARIANCE_FLOOR",
    "warmup": "OMNIFLOW_WARMUP",
    "warmup_positive_std_only": "OMNIFLOW_WARMUP_POSITIVE_STD_ONLY",
    "variance_sensitivity": "OMNIFLOW_VARIANCE_SENSITIVITY",
    "sigma_ref": "OMNIFLOW_SIGMA_REF",
    "sigma_ref_adapt": "OMNIFLOW_SIGMA_REF_ADAPT",
    "instability_window": "OMNIFLOW_INSTABILITY_WINDOW",
    "instability_weight": "OMNIFLOW_INSTABILITY_WEIGHT",
    "drift_boost": "OMNIFLOW_DRIFT_BOOST",
    "drift_decay": "OMNIFLOW_DRIFT_DECAY",
    "cooldown_rate": "OMNIFLOW_COOLDOWN_RATE",
    "urgency_smoothing": "OMNIFLOW_URGENCY_SMOOTHING",
    "cooldown_threshold": "OMNIFLOW_COOLDOWN_THRESHOLD",
    "outlier_decay": "OMNIFLOW_OUTLIER_DECAY",
    "interval_mapping": "OMNIFLOW_INTERVAL_MAPPING",
    "interval_power": "OMNIFLOW_INTERVAL_POWER",
    "interval_logistic_midpoint": "OMNIFLOW_INTERVAL_LOGISTIC_MIDPOINT",
    "interval_logistic_steepness": "OMNIFLOW_INTERVAL_LOGISTIC_STEEPNESS",
    "interval_exponential_steepness": "OMNIFLOW_INTERVAL_EXPONENTIAL_STEEPNESS",
    "sigma_band": "OMNIFLOW_SIGMA_BAND",
}


def load_profiles() -> dict[str, dict[str, Any]]:
    payload = json.loads(_PROFILE_PATH.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected profile map in {_PROFILE_PATH}")
    return {str(name).lower(): values for name, values in payload.items()}


def available_profiles() -> list[str]:
    return sorted(load_profiles())


def load_profile(name: str) -> dict[str, Any]:
    profile_name = name.strip().lower()
    profiles = load_profiles()
    if profile_name not in profiles:
        supported = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown adaptive profile '{name}'. Supported profiles: {supported}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"Profile '{name}' must map to an object")
    return dict(profile)


def profile_to_env_lines(name: str) -> list[str]:
    profile = load_profile(name)
    lines: list[str] = []
    for key, env_key in _ENV_KEY_MAP.items():
        if key in profile:
            lines.append(f"{env_key}={profile[key]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared adaptive profile helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Print OMNIFLOW_* env lines for a profile")
    env_parser.add_argument("profile")

    list_parser = subparsers.add_parser("list", help="List profile names")
    list_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "env":
        for line in profile_to_env_lines(args.profile):
            print(line)
        return

    profiles = available_profiles()
    if args.json:
        print(json.dumps(profiles))
    else:
        for profile in profiles:
            print(profile)


if __name__ == "__main__":
    main()
