"""Helpers for selecting adaptive-policy operating points by sample budget."""

from __future__ import annotations


def select_by_sample_ratio(candidates: list[dict], target: float, slack: float) -> dict:
    """Select the closest candidate at or above ``target`` within ``slack``.

    Candidate dictionaries must contain ``report.sample_ratio`` and
    ``sweep_index``.  No fidelity field is inspected, so the same choice is made
    regardless of reconstruction error, correlation, peak recall, or event
    metrics attached to a candidate.
    """
    if slack < 0.0:
        raise ValueError("slack must be non-negative")

    upper = target + slack
    eps = 1e-12
    in_band = [
        candidate
        for candidate in candidates
        if target - eps <= candidate["report"]["sample_ratio"] <= upper + eps
    ]
    selected = min(
        in_band,
        key=lambda candidate: (
            candidate["report"]["sample_ratio"] - target,
            candidate["sweep_index"],
        ),
        default=None,
    )

    below = [
        candidate
        for candidate in candidates
        if candidate["report"]["sample_ratio"] < target - eps
    ]
    above = [
        candidate
        for candidate in candidates
        if candidate["report"]["sample_ratio"] >= target - eps
    ]
    nearest_below = max(
        below,
        key=lambda candidate: candidate["report"]["sample_ratio"],
        default=None,
    )
    nearest_above = min(
        above,
        key=lambda candidate: (
            candidate["report"]["sample_ratio"] - target,
            candidate["sweep_index"],
        ),
        default=None,
    )

    return {
        "target_sample_ratio": target,
        "budget_slack": slack,
        "upper_sample_ratio": upper,
        "selection_rule": (
            "minimum realized sample ratio in "
            "[r_omniflow, r_omniflow + budget_slack]; no fidelity metric used"
        ),
        "selected": selected,
        "nearest_below": nearest_below,
        "nearest_above": nearest_above,
        "n_candidates": len(candidates),
        "n_candidates_in_band": len(in_band),
    }
