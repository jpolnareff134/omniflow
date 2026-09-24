import numpy as np
from numpy.typing import NDArray


def generate_uniform(n: int, low: float = 0.0, high: float = 100.0,
                     seed: int | None = None) -> NDArray[np.float64]:
    """Draw *n* samples from a uniform distribution U(low, high)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=n)


def generate_normal(n: int, mean: float = 50.0, std: float = 10.0,
                    clip_low: float | None = None,
                    clip_high: float | None = None,
                    seed: int | None = None) -> NDArray[np.float64]:
    """Draw *n* samples from a (optionally clipped) normal distribution N(mean, variance)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(mean, std, size=n)
    if clip_low is not None or clip_high is not None:
        data = np.clip(data, clip_low, clip_high)
    return data


def generate_beta(n: int, a: float = 2.0, b: float = 5.0,
                  scale: float = 100.0,
                  seed: int | None = None) -> NDArray[np.float64]:
    """Draw *n* samples from Beta(a, b) scaled to [0, scale].

    Beta distributions are excellent for modelling quantities that are
    naturally bounded (e.g. CPU-usage percentage).
    """
    rng = np.random.default_rng(seed)
    return rng.beta(a, b, size=n) * scale


def generate_lognormal(n: int, mean: float = 3.0, sigma: float = 0.5,
                       clip_high: float | None = None,
                       seed: int | None = None) -> NDArray[np.float64]:
    """Draw *n* samples from a log-normal distribution, optionally clipped."""
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean, sigma, size=n)
    if clip_high is not None:
        data = np.clip(data, 0.0, clip_high)
    return data


def generate_mixture(n: int,
                     components: list[dict] | None = None,
                     seed: int | None = None) -> NDArray[np.float64]:
    """Draw *n* samples from a Gaussian-mixture model.

    Parameters
    ----------
    components : list of dict, optional
        Each dict must contain ``mean``, ``std``, and ``weight`` keys.
        Weights are normalised automatically.  Defaults to a bimodal mix
        that mimics a machine alternating between idle (~20 %) and busy
        (~75 %) states.
    """
    if components is None:
        components = [
            {"mean": 20.0, "std": 5.0, "weight": 0.6},
            {"mean": 75.0, "std": 8.0, "weight": 0.4},
        ]

    rng = np.random.default_rng(seed)
    weights = np.array([c["weight"] for c in components])
    weights /= weights.sum()

    indices = rng.choice(len(components), size=n, p=weights)
    data = np.empty(n, dtype=np.float64)
    for i, comp in enumerate(components):
        mask = indices == i
        data[mask] = rng.normal(comp["mean"], comp["std"], size=mask.sum())

    return data


def add_trend(data: NDArray[np.float64],
              slope: float = 0.05) -> NDArray[np.float64]:
    """Add a linear trend of *slope* per sample."""
    n = len(data)
    return data + slope * np.arange(n)


def add_seasonality(data: NDArray[np.float64],
                    period: int = 100,
                    amplitude: float = 10.0) -> NDArray[np.float64]:
    """Overlay a sinusoidal seasonal component."""
    n = len(data)
    season = amplitude * np.sin(2 * np.pi * np.arange(n) / period)
    return data + season


def add_noise(data: NDArray[np.float64],
              std: float = 2.0,
              seed: int | None = None) -> NDArray[np.float64]:
    """Add zero-mean Gaussian noise."""
    rng = np.random.default_rng(seed)
    return data + rng.normal(0, std, size=len(data))


def inject_anomalies(data: NDArray[np.float64],
                     fraction: float = 0.02,
                     magnitude: float = 30.0,
                     seed: int | None = None) -> NDArray[np.float64]:
    """Randomly inject point anomalies (spikes / dips).

    Parameters
    ----------
    fraction : float
        Proportion of points to perturb (0-1).
    magnitude : float
        Maximum absolute deviation added to anomalous points.
    """
    rng = np.random.default_rng(seed)
    data = data.copy()
    n_anomalies = max(1, int(len(data) * fraction))
    idx = rng.choice(len(data), size=n_anomalies, replace=False)
    data[idx] += rng.uniform(-magnitude, magnitude, size=n_anomalies)
    return data
