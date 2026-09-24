"""
Configuration module for OmniFlow pipeline.

Provides default values for the tracker and adaptive poller parameters,
with support for environment variable overrides.
"""

import os


def _get_float(key: str, default: float) -> float:
    """Get float value from environment or return default."""
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _get_int(key: str, default: int) -> int:
    """Get int value from environment or return default."""
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _get_str(key: str, default: str) -> str:
    """Get string value from environment or return default."""
    return os.environ.get(key, default)


def _get_bool(key: str, default: bool) -> bool:
    """Get a boolean value from environment or return default."""
    value = os.environ.get(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean value, got {value!r}")


# ------------------------------
# Tracker Configuration
# ------------------------------

ALPHA_BASE = _get_float("OMNIFLOW_ALPHA_BASE", 0.05)
"""Base smoothing factor for the tracker.
Must be in the range (0, 1). Higher values make the tracker more responsive to recent changes."""

BETA = _get_float("OMNIFLOW_BETA", 0.5)
"""Adaptation factor for the tracker.
Given p(t) = (1/sqrt(2 * pi)) * exp(-0.5 * z(t)^2), where z(t) is the z-score of the current point, the effective smoothing factor is:
    alpha_eff = alpha_base * (1 - beta * p(t))
Higher beta means that points with high p(t) (i.e., close to the mean) will significantly reduce the effective learning rate, while points with low p(t) (i.e., potential anomalies) will have less reduction, allowing the tracker to adapt more quickly to changes."""

OUTLIER_THRESHOLD = _get_float("OMNIFLOW_OUTLIER_THRESHOLD", 1.0)
"""Z-score threshold for outlier detection.
Below this threshold, points are considered normal; above it, they are flagged as outliers.
Lower values will make the tracker more sensitive to deviations, while higher values will make it less sensitive."""

DRIFT_TOLERANCE = _get_int("OMNIFLOW_DRIFT_TOLERANCE", 3)
"""Number of consecutive outliers before drift is detected."""

TRACKER_VARIANCE_FLOOR = _get_float("OMNIFLOW_TRACKER_VARIANCE_FLOOR", 0.0)
"""Lower bound for the tracker variance seed and drift-reset variance."""

WARMUP_POSITIVE_STD_ONLY = _get_bool("OMNIFLOW_WARMUP_POSITIVE_STD_ONLY", False)
"""If true, auto-calibration counts only NORMAL warm-up samples with std > 0."""

# ------------------------------
# Adaptive Poller Configuration
# ------------------------------

MIN_INTERVAL = _get_int("OMNIFLOW_MIN_INTERVAL", 1)
"""Minimum polling interval. Cannot be less than 1,
cannot be greater than MAX_INTERVAL. The internal
code clips the interval anyway, but setting this too high may limit the responsiveness of the adaptive poller."""

MAX_INTERVAL = _get_int("OMNIFLOW_MAX_INTERVAL", 20)
"""Maximum polling interval. No upper bound, but should be greater than MIN_INTERVAL to allow for adaptive behavior. Setting this too high may cause the poller to become unresponsive during stable periods."""

if MAX_INTERVAL < MIN_INTERVAL:
    raise ValueError(f"MAX_INTERVAL ({MAX_INTERVAL}) must be greater than or equal to MIN_INTERVAL ({MIN_INTERVAL}).")

# Component 1) of the urgency score is the variance component, which is based on the ratio of the tracker's current variance to a reference variance. This parameter controls how sensitive the polling interval is to changes in the tracker's variance.

VARIANCE_SENSITIVITY = _get_float("OMNIFLOW_VARIANCE_SENSITIVITY", 1.0)
"""Variance sensitivity for adaptive interval control."""

INTERVAL_MAPPING = _get_str("OMNIFLOW_INTERVAL_MAPPING", "linear_rate")
"""Urgency-to-interval mapping strategy.

Supported values:
    - ``original_interval``: legacy linear interpolation in interval space.
    - ``linear_rate``: linear interpolation in interval space.
    - ``smoothstep_interval``: smoothstep easing in interval space.
    - ``geometric_rate``: geometric interpolation in interval space.
    - ``power_rate``: interval-space interpolation using ``urgency ** interval_power``.
    - ``logistic_rate``: interval-space interpolation after a normalized logistic transform.
    - ``exponential_ease_out``: early-vigilant normalized exponential easing.
"""

INTERVAL_POWER = _get_float("OMNIFLOW_INTERVAL_POWER", 2.0)
"""Shape parameter used by the ``power_rate`` urgency-to-interval mapping.

Must be strictly positive. Values greater than 1 make the mapping flatter
at low urgency and steeper near urgency = 1.
"""

INTERVAL_LOGISTIC_MIDPOINT = _get_float("OMNIFLOW_INTERVAL_LOGISTIC_MIDPOINT", 0.5)
"""Midpoint of the normalized logistic curve used by ``logistic_rate``.

Must be in the open interval (0, 1). Lower values move the urgency knee earlier.
"""

INTERVAL_LOGISTIC_STEEPNESS = _get_float("OMNIFLOW_INTERVAL_LOGISTIC_STEEPNESS", 8.0)
"""Steepness of the normalized logistic curve used by ``logistic_rate``.

Must be strictly positive. Higher values make the curve more threshold-like.
"""

INTERVAL_EXPONENTIAL_STEEPNESS = _get_float("OMNIFLOW_INTERVAL_EXPONENTIAL_STEEPNESS", 4.0)
"""Steepness of the normalized exponential ease-out urgency mapping."""

SIGMA_REF = _get_float("OMNIFLOW_SIGMA_REF", 0.0)
"""Reference standard deviation for the variance component (0 = auto-calibrate from signal)."""

WARMUP = _get_int("OMNIFLOW_WARMUP", 5)
"""Number of NORMAL points to observe before locking auto-calibrated sigma_ref.
Needed because the initial variance seed is inflated (proportional to signal
magnitude) and takes several observations to converge to the true level."""

SIGMA_REF_ADAPT = _get_float("OMNIFLOW_SIGMA_REF_ADAPT", 0.01)
"""EMA rate at which sigma_ref tracks the signal's evolving variability.
After warmup, sigma_ref is updated as:
    sigma_ref = (1 - sigma_ref_adapt) * sigma_ref + sigma_ref_adapt * tracker.std
This prevents the variance component from staying pegged at 1.0 after a regime change.
Set to 0.0 to lock sigma_ref after warmup (old behaviour)."""

# Component 2) of the urgency score is the instability component, which is based on the percentage of outliers in the recent window of samples. This parameter controls how many recent samples are considered when calculating this percentage.

INSTABILITY_WINDOW = _get_int("OMNIFLOW_INSTABILITY_WINDOW", 20)
"""Number of recently sampled points to consider when calculating the instability influence on the polling interval."""

INSTABILITY_WEIGHT = _get_float("OMNIFLOW_INSTABILITY_WEIGHT", 1.0)
"""At each point, the number of outliers in the past *instability_window* is computed, the percentage of outliers is calculated, and this is multiplied by *instability_weight* to get the instability component of the urgency score.
Values greater than 1 are allowed, but the resulting component is clipped to the interval [0, 1]."""

# Component 3) of the urgency score is the drift component, which receives an impulse when a drift is detected (i.e., when the number of consecutive anomalies exceeds the drift tolerance). This parameter controls how much the polling interval should be reduced when a drift is detected.

DRIFT_BOOST = _get_float("OMNIFLOW_DRIFT_BOOST", 1.0)
"""Impulse applied to the drift component when drift is detected.
Component 3) of the urgency calculation, 0 means no impulse, 1 means full impulse. Larger values are clipped to the interval [0, 1]."""

DRIFT_DECAY = _get_float("OMNIFLOW_DRIFT_DECAY", 0.90)
"""Decay rate for the drift component. When a drift is not detected, the current drift component decays by multiplying it with DRIFT_DECAY each sample step. This allows the system to gradually return to normal polling intervals after a drift has been detected."""

COOLDOWN_RATE = _get_float("OMNIFLOW_COOLDOWN_RATE", 0.10)
"""When the raw current urgency is below the cooldown threshold, the effective urgency is reduced by multiplying it with (1 - cooldown_rate). This allows the poller to return to more normal intervals more quickly after a spike in urgency, while still allowing for some responsiveness to new anomalies."""

URGENCY_SMOOTHING = _get_float("OMNIFLOW_URGENCY_SMOOTHING", 0.60)
"""Exponential smoothing factor for urgency updates.
The new urgency is computed as:
    urgency = urgency_smoothing * urgency_prev + (1 - urgency_smoothing) * raw_urgency
Values closer to 1 keep more memory of the previous urgency, while lower values react more quickly to the current raw urgency."""

COOLDOWN_THRESHOLD = _get_float("OMNIFLOW_COOLDOWN_THRESHOLD", 0.20)
"""Raw urgency threshold below which cooldown is applied.
When raw urgency falls below this value, the smoothed urgency is additionally
multiplied by (1 - cooldown_rate) to accelerate the return to calm sampling intervals."""

# Component 4) of the urgency score is the outlier component, which is an impulse applied when a new outlier is observed (i.e., when a point exceeds the z-score threshold). This parameter controls how much the polling interval should be reduced in response to new outliers.

OUTLIER_DECAY = _get_float("OMNIFLOW_OUTLIER_DECAY", 0.85)
"""Exponential decay rate for the outlier component.
A single outlier injects urgency proportional to how far the z-score
exceeds the threshold; this decays by *outlier_decay* each sample step.
Set to 0.0 to disable outlier kicks entirely."""

# ------------------------------
# eBPF Probe Configuration
# ------------------------------

EBPF_SYSCALL = _get_str("OMNIFLOW_EBPF_SYSCALL", "read")
"""Syscall to trace (e.g. read, write, openat, close)."""

EBPF_POLL_INTERVAL = _get_float("OMNIFLOW_EBPF_POLL_INTERVAL", 0.2)
"""Seconds between userspace reads of the eBPF map."""

# ------------------------------
# Plot Configuration
# ------------------------------

SIGMA_BAND = _get_float("OMNIFLOW_SIGMA_BAND", 3.0)
"""Number of standard deviations for confidence bands in plots."""

# ------------------------------
# Other Constants
# ------------------------------

# SEED = _get_int("OMNIFLOW_SEED", 0)
# if SEED == 0:
#     SEED = None  # Use None for true randomness if seed is set to 0
# """Random seed for reproducibility."""
