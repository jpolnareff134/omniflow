"""
Phased HTTP load generator for the HPA autoscaling experiment.

Env:
  TARGET_HOST   base URL of the workload service (default: http://nginx)
  LOAD_SHAPE    which load shape to run (default: phased)
                  phased       - original: warm-up / ramp / spike / pulse
                  oscillating  - rapid high/low toggles (tests anti-thrashing)
                  staircase    - monotonic step ramp then hard drop (tests proactive detection)
                  flash_crowd  - two instant max-load spikes (tests fastest scale-up)
"""

import os
import json
import random
import time

from locust import HttpUser, LoadTestShape, task, constant

from load_schedule import crossed_events, get_stages, stage_at, total_duration

TARGET_HOST = os.environ.get("TARGET_HOST", "http://nginx")
LOAD_SHAPE = os.environ.get("LOAD_SHAPE", "phased")
EXPERIMENT_RELEASE_FILE = os.environ.get(
    "EXPERIMENT_RELEASE_FILE", "/tmp/locust-release"
)
EXPERIMENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "0"))
WAIT_SECONDS = float(os.environ.get("LOCUST_WAIT_SECONDS", "1.0"))
random.seed(EXPERIMENT_SEED)


class WebUser(HttpUser):
    host = TARGET_HOST
    wait_time = constant(WAIT_SECONDS)

    @task
    def get_index(self):
        # Bound request time so Locust can stop users at a phase boundary even
        # when the deliberately saturated workload stops responding promptly.
        self.client.get("/", timeout=5)


def _emit(event: dict, t0_ns: int) -> None:
    now_ns = time.time_ns()
    print(json.dumps({
        **event,
        "scheduled_epoch_ns": t0_ns + int(event.get("scheduled_offset_s", 0) * 1e9),
        "observed_epoch_ns": now_ns,
        "observed_offset_s": (now_ns - t0_ns) / 1e9,
        "shape": LOAD_SHAPE,
        "seed": EXPERIMENT_SEED,
    }), flush=True)


class ActiveLoad(LoadTestShape):
    """Runs the shape selected by LOAD_SHAPE env var."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        get_stages(LOAD_SHAPE)
        self._t0_ns = None
        self._release_mono_ns = None
        self._released = False
        self._previous_elapsed = -float("inf")
        open("/tmp/locust-armed", "w").close()
        _emit({"event": "load_armed", "scheduled_offset_s": 0}, time.time_ns())

    def tick(self):
        if not self._released:
            if not os.path.exists(EXPERIMENT_RELEASE_FILE):
                return 0, 1
            self._released = True
            self._t0_ns = time.time_ns()
            self._release_mono_ns = time.monotonic_ns()
            with open("/tmp/locust-started", "w") as started:
                started.write(f"{self._t0_ns}\n")
            self._previous_elapsed = -1e-9

        elapsed = (time.monotonic_ns() - self._release_mono_ns) / 1e9
        for event in crossed_events(LOAD_SHAPE, self._previous_elapsed, elapsed):
            _emit(event, self._t0_ns)
        self._previous_elapsed = elapsed
        stage = stage_at(LOAD_SHAPE, elapsed)
        if stage is None:
            if elapsed < 0:
                return 0, 1
            if elapsed >= total_duration(LOAD_SHAPE):
                return None
            return 0, 1
        return stage.users, stage.spawn_rate
