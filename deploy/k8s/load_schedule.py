"""Shared load schedules for Locust, orchestration, and result analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    duration_s: int
    users: int
    spawn_rate: int
    direction: str


SHAPES: dict[str, tuple[Stage, ...]] = {
    "phased": (
        Stage("warmup", 60, 5, 5, "stable"),
        Stage("ramp", 120, 50, 1, "up"),
        Stage("sustained", 60, 50, 5, "stable"),
        Stage("fast_ramp", 60, 150, 3, "up"),
        Stage("spike", 30, 200, 5, "up"),
        Stage("cooldown", 30, 10, 10, "down"),
        Stage("pulse", 60, 75, 5, "up"),
        Stage("cooldown2", 30, 10, 10, "down"),
        Stage("tail", 60, 10, 5, "stable"),
    ),
    "oscillating": (
        Stage("warmup", 60, 10, 10, "stable"),
        Stage("spike1", 30, 180, 20, "up"),
        Stage("valley1", 30, 5, 20, "down"),
        Stage("spike2", 30, 180, 20, "up"),
        Stage("valley2", 30, 5, 20, "down"),
        Stage("spike3", 30, 180, 20, "up"),
        Stage("valley3", 30, 5, 20, "down"),
        Stage("spike4", 30, 180, 20, "up"),
        Stage("valley4", 30, 5, 20, "down"),
        Stage("tail", 60, 5, 5, "stable"),
    ),
    "staircase": (
        Stage("baseline", 60, 5, 5, "stable"),
        Stage("step1", 60, 30, 5, "up"),
        Stage("step2", 60, 60, 5, "up"),
        Stage("step3", 60, 100, 5, "up"),
        Stage("step4", 60, 150, 5, "up"),
        Stage("step5", 60, 200, 5, "up"),
        Stage("sustained_peak", 90, 200, 5, "stable"),
        Stage("hard_drop", 60, 5, 30, "down"),
    ),
    "flash_crowd": (
        Stage("warmup", 30, 5, 5, "stable"),
        Stage("flash1", 120, 200, 50, "up"),
        Stage("drop1", 60, 5, 50, "down"),
        Stage("rest", 30, 5, 5, "stable"),
        Stage("flash2", 120, 200, 50, "up"),
        Stage("drop2", 60, 5, 50, "down"),
        Stage("tail", 30, 5, 5, "stable"),
    ),
}


def get_stages(shape: str) -> tuple[Stage, ...]:
    try:
        return SHAPES[shape]
    except KeyError as exc:
        raise ValueError(f"Unknown load shape: {shape}") from exc


def total_duration(shape: str) -> int:
    return sum(stage.duration_s for stage in get_stages(shape))


def phase_boundaries(shape: str) -> list[dict]:
    result = []
    start = 0
    for stage in get_stages(shape):
        end = start + stage.duration_s
        result.append({
            **asdict(stage),
            "start_s": start,
            "end_s": end,
        })
        start = end
    return result


def crossed_events(shape: str, previous_s: float, current_s: float) -> list[dict]:
    """Return all phase events crossed in ``(previous_s, current_s]``."""
    events = []
    boundaries = phase_boundaries(shape)
    if previous_s < 0 <= current_s:
        events.append({"event": "load_start", "scheduled_offset_s": 0})
    for phase in boundaries:
        if previous_s < phase["start_s"] <= current_s:
            events.append({
                "event": "phase_start",
                "phase": phase["name"],
                "direction": phase["direction"],
                "users": phase["users"],
                "spawn_rate": phase["spawn_rate"],
                "scheduled_offset_s": phase["start_s"],
            })
        if previous_s < phase["end_s"] <= current_s:
            events.append({
                "event": "phase_end",
                "phase": phase["name"],
                "scheduled_offset_s": phase["end_s"],
            })
    duration = total_duration(shape)
    if previous_s < duration <= current_s:
        events.append({"event": "load_end", "scheduled_offset_s": duration})
    return events


def stage_at(shape: str, elapsed_s: float) -> Stage | None:
    if elapsed_s < 0:
        return None
    elapsed = 0
    for stage in get_stages(shape):
        elapsed += stage.duration_s
        if elapsed_s < elapsed:
            return stage
    return None
