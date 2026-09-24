"""
Generic trace builder & composer.

Extends the primitives in :mod:`src.data.synth` with:

* **Segment** - a declarative description of one contiguous region of a trace
  (distribution, modifiers, bounds, length).
* **build_trace** - turn a list of Segments into a single NumPy array.
* **TraceComposer** - fluent API for incrementally constructing complex traces
  by appending segments, with optional cosine bridging to smooth
  discontinuities at the joins.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data.synth import (
    generate_uniform,
    generate_normal,
    generate_beta,
    generate_lognormal,
    generate_mixture,
    add_trend,
    add_seasonality,
    add_noise,
    inject_anomalies,
)


# ------------------------------
# Segment descriptor
# ------------------------------

def _load_file(n: int = 0, path: str = "", **_kwargs) -> NDArray[np.float64]:
    """Load data points from a JSON file.

    The file may contain:

    * A flat JSON array of numbers: ``[1.0, 2.5, 3.1, ...]``
    * An array of objects with a ``value`` key:
      ``[{"t": 0, "value": 1.0}, ...]``  (as exported by ``save_json``).

    The *n* parameter from the Segment is **ignored** - the length is
    determined by the file content.
    """
    import json as _json

    with open(path) as fh:
        raw = _json.load(fh)

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Expected a non-empty JSON array in '{path}'")

    # [{"t": .., "value": ..}, ...] or plain [float, ...]
    if isinstance(raw[0], dict):
        values = [float(item["value"]) for item in raw]
    else:
        values = [float(v) for v in raw]

    return np.array(values, dtype=np.float64)


GENERATORS = {
    "uniform": generate_uniform,
    "normal": generate_normal,
    "beta": generate_beta,
    "lognormal": generate_lognormal,
    "mixture": generate_mixture,
    "constant": lambda n, value=0.0, **kwargs: np.full(n, value, dtype=np.float64),
    "file": _load_file,
}


@dataclass
class Segment:
    """Declarative description of one piece of a trace.

    Parameters
    ----------
    n : int
        Number of data points in this segment.
    distribution : str
        Name of the base distribution (key in ``GENERATORS``).
    dist_kwargs : dict
        Keyword arguments forwarded to the distribution generator
        (e.g. ``mean``, ``std``, ``a``, ``b``, ``components``, ...).
    trend_slope : float
        Linear drift per sample (0 = flat).
    season_period : int
        Sinusoidal period in samples (0 = no seasonality).
    season_amplitude : float
        Peak deviation of the seasonal component.
    noise_std : float
        Additive Gaussian noise std (0 = none).
    anomaly_fraction : float
        Fraction of points to perturb as anomalies.
    anomaly_magnitude : float
        Maximum anomaly deviation.
    clip_low : float or None
        Hard lower bound applied after all modifiers.
    clip_high : float or None
        Hard upper bound applied after all modifiers.
    label : str
        Human-readable label (for legends / debugging).
    seed : int or None
        Per-segment reproducibility seed.
    """
    n: int = 500
    distribution: str = "normal"
    dist_kwargs: dict[str, Any] = field(default_factory=dict)
    trend_slope: float = 0.0
    season_period: int = 0
    season_amplitude: float = 10.0
    noise_std: float = 0.0
    anomaly_fraction: float = 0.0
    anomaly_magnitude: float = 30.0
    clip_low: float | None = None
    clip_high: float | None = None
    label: str = ""
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict describing this segment."""
        return asdict(self)


def _build_one(seg: Segment) -> NDArray[np.float64]:
    """Materialise a single Segment into an array."""
    if seg.distribution not in GENERATORS:
        raise ValueError(
            f"Unknown distribution '{seg.distribution}'. "
            f"Choose from {list(GENERATORS)}"
        )
    data = GENERATORS[seg.distribution](seg.n, seed=seg.seed, **seg.dist_kwargs)

    # For file-sourced segments the length comes from the file itself;
    # override seg.n so downstream helpers (e.g. segment_boundaries) stay
    # consistent.
    if seg.distribution == "file" and len(data) != seg.n:
        object.__setattr__(seg, "n", len(data))

    if seg.trend_slope:
        data = add_trend(data, slope=seg.trend_slope)
    if seg.season_period > 0:
        data = add_seasonality(data, period=seg.season_period,
                               amplitude=seg.season_amplitude)
    if seg.noise_std > 0:
        data = add_noise(data, std=seg.noise_std, seed=seg.seed)
    if seg.anomaly_fraction > 0:
        data = inject_anomalies(data, fraction=seg.anomaly_fraction,
                                magnitude=seg.anomaly_magnitude,
                                seed=seg.seed)
    if seg.clip_low is not None or seg.clip_high is not None:
        data = np.clip(data, seg.clip_low, seg.clip_high)
    return data


# ------------------------------
# Bridging helper
# ------------------------------

def _cosine_bridge(left_val: float, right_val: float,
                   n: int) -> NDArray[np.float64]:
    """Smooth cosine interpolation from *left_val* to *right_val* over *n*
    samples.  The result starts at ``left_val`` and ends at ``right_val``
    without discontinuities in the first derivative at the endpoints.
    """
    t = np.linspace(0, np.pi, n)
    return left_val + (right_val - left_val) * (1 - np.cos(t)) / 2


def _linear_bridge(left_val: float, right_val: float,
                   n: int) -> NDArray[np.float64]:
    """Simple linear ramp from *left_val* to *right_val*."""
    return np.linspace(left_val, right_val, n)


BRIDGE_METHODS = {
    "cosine": _cosine_bridge,
    "linear": _linear_bridge,
}


def build_trace(
        segments: list[Segment],
        bridge_length: int = 0,
        bridge_method: str = "cosine",
) -> NDArray[np.float64]:
    """Concatenate several :class:`Segment` objects into one trace.

    Parameters
    ----------
    segments : list[Segment]
        Ordered list of segments.
    bridge_length : int
        Number of interpolation samples inserted between consecutive
        segments (0 = hard cut, no bridging).
    bridge_method : str
        ``"cosine"`` (smooth S-curve) or ``"linear"`` (ramp).

    Returns
    -------
    NDArray
        The composed trace.
    """
    if bridge_method not in BRIDGE_METHODS:
        raise ValueError(
            f"Unknown bridge method '{bridge_method}'. "
            f"Choose from {list(BRIDGE_METHODS)}"
        )
    bridge_fn = BRIDGE_METHODS[bridge_method]

    parts: list[NDArray[np.float64]] = []
    arrays = [_build_one(seg) for seg in segments]

    for i, arr in enumerate(arrays):
        if i > 0 and bridge_length > 0:
            left_val = float(arrays[i - 1][-1])
            right_val = float(arr[0])
            parts.append(bridge_fn(left_val, right_val, bridge_length))
        parts.append(arr)

    return np.concatenate(parts)


class TraceComposer:
    """Fluent builder for multi-segment traces.

    Example
    -------
    >>> trace = (
    ...     TraceComposer(bridge_length=30)
    ...     .add("normal", 400, mean=20, std=2, label="idle")
    ...     .add("normal", 200, mean=70, std=5, noise_std=3,
    ...          anomaly_fraction=0.05, label="busy")
    ...     .add("normal", 300, mean=25, std=2, label="recovery")
    ...     .build()
    ... )
    """

    def __init__(
            self,
            bridge_length: int = 0,
            bridge_method: str = "cosine",
    ) -> None:
        self.bridge_length = bridge_length
        self.bridge_method = bridge_method
        self._segments: list[Segment] = []

    # -- fluent append --

    def add(
            self,
            distribution: str = "normal",
            n: int = 500,
            *,
            trend_slope: float = 0.0,
            season_period: int = 0,
            season_amplitude: float = 10.0,
            noise_std: float = 0.0,
            anomaly_fraction: float = 0.0,
            anomaly_magnitude: float = 30.0,
            clip_low: float | None = None,
            clip_high: float | None = None,
            label: str = "",
            seed: int | None = None,
            **dist_kwargs,
    ) -> "TraceComposer":
        """Append a segment.  Returns *self* for chaining."""
        self._segments.append(Segment(
            n=n,
            distribution=distribution,
            dist_kwargs=dist_kwargs,
            trend_slope=trend_slope,
            season_period=season_period,
            season_amplitude=season_amplitude,
            noise_std=noise_std,
            anomaly_fraction=anomaly_fraction,
            anomaly_magnitude=anomaly_magnitude,
            clip_low=clip_low,
            clip_high=clip_high,
            label=label,
            seed=seed,
        ))
        return self

    def add_segment(self, segment: Segment) -> "TraceComposer":
        """Append a pre-built :class:`Segment`.  Returns *self*."""
        self._segments.append(segment)
        return self

    def add_many(self, segments: list[Segment]) -> "TraceComposer":
        """Append multiple pre-built :class:`Segment`s.  Returns *self*."""
        self._segments.extend(segments)
        return self

    # -- build --

    @property
    def segments(self) -> list[Segment]:
        """Read-only view of the current segment list."""
        return list(self._segments)

    def build(self) -> NDArray[np.float64]:
        """Materialise the full trace."""
        if not self._segments:
            return np.array([], dtype=np.float64)
        return build_trace(self._segments,
                           bridge_length=self.bridge_length,
                           bridge_method=self.bridge_method)

    # -- convenience: segment boundaries for plotting --

    def segment_boundaries(self) -> list[tuple[int, int, str]]:
        """Return ``(start, end, label)`` for each segment in the built trace,
        accounting for bridge insertions."""
        boundaries: list[tuple[int, int, str]] = []
        pos = 0
        for i, seg in enumerate(self._segments):
            if i > 0 and self.bridge_length > 0:
                pos += self.bridge_length  # skip bridge
            start = pos
            pos += seg.n
            boundaries.append((start, pos, seg.label or f"seg-{i}"))
        return boundaries

    def __str__(self) -> str:
        return (
                f"TraceComposer(bridge_length={self.bridge_length}, "
                f"bridge_method='{self.bridge_method}', "
                f"segments=[\n  " +
                ",\n  ".join(str(s) for s in self._segments) +
                "\n])"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict describing the full composer state."""
        return {
            "bridge_length": self.bridge_length,
            "bridge_method": self.bridge_method,
            "segments": [s.to_dict() for s in self._segments],
        }

    # -- YAML / dict loading -------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceComposer":
        """Build a :class:`TraceComposer` from a plain dict.

        The expected schema matches :meth:`to_dict` output::

            bridge_length: 30
            bridge_method: cosine
            segments:
              - distribution: normal
                n: 500
                label: idle
                mean: 20.0      # < distribution kwargs live at top level
                std: 2.0
        """
        # Segment-level keys that are NOT distribution kwargs
        _SEG_KEYS = {f.name for f in Segment.__dataclass_fields__.values()}

        composer = cls(
            bridge_length=d.get("bridge_length", 0),
            bridge_method=d.get("bridge_method", "cosine"),
        )
        for raw in d.get("segments", []):
            seg_kwargs: dict[str, Any] = {}
            dist_kwargs: dict[str, Any] = {}
            for k, v in raw.items():
                if k in _SEG_KEYS:
                    seg_kwargs[k] = v
                else:
                    dist_kwargs[k] = v
            if dist_kwargs:
                seg_kwargs.setdefault("dist_kwargs", {}).update(dist_kwargs)
            composer.add_segment(Segment(**seg_kwargs))
        return composer

    @classmethod
    def from_yaml(cls, path: str) -> "TraceComposer":
        """Load a :class:`TraceComposer` from a YAML file.

        Parameters
        ----------
        path : str
            Path to a ``.yaml`` / ``.yml`` file whose top-level structure
            matches :meth:`to_dict`.
        """
        import yaml  # deferred so the dep is optional for non-YAML users
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
