from __future__ import annotations

import argparse
import itertools
import json
import logging as log
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

# Restrict BLAS / OpenMP thread count *before* numpy is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
import numpy as np
from numpy.typing import NDArray

from support.log import initialize_log
from tracker.pipeline import PipelineConfig
from tracker.windowed import InfoLossReport, IntervalMapping, evaluate_info_loss
from tracker.windowed import compute_fixed_urgency_trace, evaluate_info_loss_fixed_urgency, track_fixed_urgency

# ------------------------------
# Worker helpers (module-level for pickling)
# ------------------------------

_WORKER_DATA: Optional[NDArray] = None
_WORKER_COUNTER: Any = None
_CPU_ASSIGNMENTS: list[list[int]] | None = None


def _init_worker(
        data: NDArray,
        counter: Any,
        cpu_assignments: list[list[int]],
) -> None:
    """Initialise worker: pin to assigned CPUs and store shared data."""
    global _WORKER_DATA, _WORKER_COUNTER, _CPU_ASSIGNMENTS
    _WORKER_DATA = data
    _WORKER_COUNTER = counter
    _CPU_ASSIGNMENTS = cpu_assignments

    with counter.get_lock():
        wid = counter.value
        counter.value += 1

    set_affinity = getattr(os, "sched_setaffinity", None)
    if cpu_assignments and wid < len(cpu_assignments) and callable(set_affinity):
        set_affinity(0, cpu_assignments[wid])
        log.debug("Worker %d pinned to CPUs %s", wid, cpu_assignments[wid])


def _eval_worker(cfg: PipelineConfig) -> "EvaluationResult":
    """Worker entry-point: uses module-level data set by _init_worker."""
    assert _WORKER_DATA is not None, "Worker data not initialised"
    return EvaluationResult.from_run(cfg, _WORKER_DATA)


# Parameter names accepted in each YAML section, derived from PipelineConfig.
_TRACKER_PARAMS = {"alpha_base", "beta", "outlier_threshold", "drift_tolerance"}
_POLLER_PARAMS = set(PipelineConfig.__dataclass_fields__) - _TRACKER_PARAMS - {"sigma_band"}


def _jsonable(v: Any) -> Any:
    """Coerce numpy scalars to native Python types for JSON."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _active_label_params(
        config: PipelineConfig,
        param_names: Sequence[str],
) -> list[str]:
    """Return the subset of parameters worth showing for this config."""
    active: list[str] = []
    mapping = config.interval_mapping
    for name in param_names:
        if name == "interval_power" and mapping != IntervalMapping.POWER_RATE.value:
            continue
        if name in {"interval_logistic_midpoint", "interval_logistic_steepness"}:
            if mapping != IntervalMapping.LOGISTIC_RATE.value:
                continue
        if name == "interval_exponential_steepness":
            if mapping != IntervalMapping.EXPONENTIAL_EASE_OUT.value:
                continue
        active.append(name)
    return active


def _config_label(
        config: PipelineConfig,
        param_names: Sequence[str],
) -> str:
    """Return a compact human-readable label for a configuration."""
    values = config.to_dict()
    parts: list[str] = []
    for name in _active_label_params(config, param_names):
        value = values[name]
        if isinstance(value, float):
            parts.append(f"{name}={value:g}")
        else:
            parts.append(f"{name}={value}")
    return ", ".join(parts) if parts else "default"


def _downsample_profile_indices(n_points: int, max_points: int) -> NDArray[np.int64]:
    """Return evenly spaced indices for compact profile storage."""
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    return np.unique(np.linspace(0, n_points - 1, num=max_points, dtype=np.int64))


def _profile_keep_indices(
        n_points: int,
        max_points: int | None,
) -> NDArray[np.int64]:
    """Return kept indices for profile storage, optionally without downsampling."""
    if max_points is None:
        return np.arange(n_points, dtype=np.int64)
    return _downsample_profile_indices(n_points, max_points)


def _build_profile_record(
        config: PipelineConfig,
        data: Sequence[float] | NDArray,
        urgency_trace: Sequence[float] | NDArray,
        config_id: int,
        label_params: Sequence[str],
        max_points: int | None = None,
) -> dict[str, Any]:
    """Build a compact interval/sample-ratio profile for a configuration."""
    poll_results = track_fixed_urgency(
        np.asarray(data, dtype=np.float64),
        config._make_poller(),
        np.asarray(urgency_trace, dtype=np.float64),
    )
    n = len(poll_results)
    time_index = np.arange(n, dtype=np.int64)
    intervals = np.array([pr.interval for pr in poll_results], dtype=np.int64)
    sample_flags = np.array([1.0 if pr.sampled else 0.0 for pr in poll_results], dtype=np.float64)
    sample_ratio = np.cumsum(sample_flags) / (time_index + 1)
    keep = _profile_keep_indices(n, max_points=max_points)

    return {
        "config_id": config_id,
        "config_label": _config_label(config, label_params),
        "time_index": [int(v) for v in time_index[keep]],
        "interval": [int(v) for v in intervals[keep]],
        "sample_ratio": [float(v) for v in sample_ratio[keep]],
    }


def _build_shared_urgency_profile(
        urgency_trace: Sequence[float] | NDArray,
        max_points: int | None = None,
) -> dict[str, Any]:
    """Build a compact shared urgency profile for plotting."""
    urgency_trace = np.asarray(urgency_trace, dtype=np.float64)
    time_index = np.arange(len(urgency_trace), dtype=np.int64)
    keep = _profile_keep_indices(len(urgency_trace), max_points=max_points)
    return {
        "time_index": [int(v) for v in time_index[keep]],
        "urgency": [float(v) for v in urgency_trace[keep]],
    }


def _build_urgency_profiles(
        data: Sequence[float] | NDArray,
        urgency_trace: Sequence[float] | NDArray,
        configs: Sequence[PipelineConfig],
        label_params: Sequence[str],
        max_points: int | None = None,
) -> list[dict[str, Any]]:
    """Build compact interval traces for urgency sweep plotting."""
    return [
        _build_profile_record(
            cfg,
            data,
            urgency_trace,
            i,
            label_params,
            max_points=max_points,
        )
        for i, cfg in enumerate(configs, 1)
    ]


# ------------------------------
# Evaluation result
# ------------------------------


@dataclass
class EvaluationResult:
    """Result from a single parameter configuration evaluation."""

    config: PipelineConfig
    info_loss: InfoLossReport

    @classmethod
    def from_run(
            cls,
            config: PipelineConfig,
            data: Sequence[float] | NDArray,
    ) -> "EvaluationResult":
        """Run evaluation with given config."""
        log.debug("Evaluating config: %s", config.to_dict())

        info_loss = evaluate_info_loss(
            data,
            full_tracker=config._make_tracker(),
            poller=config._make_poller(),
        )

        log.debug("Evaluation complete: sample_ratio=%.3f", info_loss.sample_ratio)
        return cls(config=config, info_loss=info_loss)

    @classmethod
    def from_fixed_urgency_run(
            cls,
            config: PipelineConfig,
            data: Sequence[float] | NDArray,
            urgency_trace: Sequence[float] | NDArray,
    ) -> "EvaluationResult":
        """Run evaluation with a fixed deterministic urgency trace."""
        log.debug("Evaluating fixed-urgency config: %s", config.to_dict())

        info_loss = evaluate_info_loss_fixed_urgency(
            data,
            full_tracker=config._make_tracker(),
            poller=config._make_poller(),
            urgency_trace=urgency_trace,
        )

        log.debug("Fixed-urgency evaluation complete: sample_ratio=%.3f", info_loss.sample_ratio)
        return cls(config=config, info_loss=info_loss)

    def to_record(self) -> dict[str, Any]:
        """Flatten config + all info-loss metrics into one dict."""
        return {
            **self.config.to_dict(),
            **asdict(self.info_loss),
        }


# ------------------------------
# Sweep result container
# ------------------------------


@dataclass
class SweepResult:
    """Container for a completed sweep."""

    results: list[EvaluationResult]
    sweep_params: dict[str, list[Any]]
    n_data_points: int

    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per configuration."""
        label_params = list(self.sweep_params.keys())
        return [
            {
                "config_id": i,
                "config_label": _config_label(result.config, label_params),
                **result.to_record(),
            }
            for i, result in enumerate(self.results, 1)
        ]

    def save_json(
            self,
            save_dir: str,
            filename: str = "sweep.json",
            extra_manifest: Mapping[str, Any] | None = None,
    ) -> str:
        """Write the full sweep manifest to *save_dir*."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        label_params = list(self.sweep_params.keys())

        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sweep_params": {k: [_jsonable(v) for v in vs] for k, vs in self.sweep_params.items()},
            "n_data_points": self.n_data_points,
            "n_configurations": len(self.results),
            "config_legend": [
                {
                    "config_id": i,
                    "config_label": _config_label(result.config, label_params),
                }
                for i, result in enumerate(self.results, 1)
            ],
            "results": self.to_records(),
        }
        if extra_manifest:
            manifest.update(extra_manifest)

        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        log.info("Sweep results saved to %s (%d configs)", path, len(self.results))
        return path

    def summary(self, sort_by: str = "sample_ratio") -> str:
        """Print a table of the key tradeoff axes, sorted by *sort_by*."""
        records = sorted(self.to_records(), key=lambda record: record.get(sort_by, 0))
        header = (
            f"{'cfg':>3s}  {'sample_%':>8s}  {'nrmse':>8s}  {'recon_n':>8s}  "
            f"{'peak_r5':>7s}  {'corr':>7s}  {'cov_3sigma':>10s}  "
            f"{'max_gap':>7s}  {'mean_gap':>8s}  {'p95_gap':>7s}"
        )
        lines = [
            f"Sweep summary  ({len(records)} configurations, {self.n_data_points} data points)",
            "-" * len(header),
            header,
            "-" * len(header),
        ]
        for record in records:
            lines.append(
                f"{record['config_id']:>3d}  {record['sample_ratio']:8.3f}  {record['nrmse_mean']:8.4f}  "
                f"{record['raw_recon_nrmse']:8.4f}  {record['peak_recall_top5']:7.3f}  "
                f"{record['correlation']:7.4f}  {record['coverage_3sigma']:10.3f}  "
                f"{record['max_gap']:7d}  {record['mean_gap']:8.1f}  {record['p95_gap']:7.0f}"
            )
        lines.append("-" * len(header))
        lines.append("Config legend:")
        for record in records:
            lines.append(f"  [{record['config_id']}] {record['config_label']}")
        return "\n".join(lines)


# ------------------------------
# Parameter sweep utilities
# ------------------------------


class ParameterSweep:
    """Systematic exploration of parameter space."""

    def __init__(
            self,
            data: Sequence[float] | NDArray,
            base_config: PipelineConfig | None = None,
            n_jobs: int = 1,
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.base_config = base_config or PipelineConfig()
        self.n_jobs = n_jobs

    def grid_search(
            self,
            tracker_params: Mapping[str, Sequence[Any]] | None = None,
            poller_params: Mapping[str, Sequence[Any]] | None = None,
    ) -> SweepResult:
        """Exhaustive grid search over multiple parameters."""
        tracker_params = dict(tracker_params or {})
        poller_params = dict(poller_params or {})

        self._validate_param_keys(tracker_params, poller_params)

        log.info(
            "Starting grid search: tracker_params=%s, poller_params=%s",
            list(tracker_params.keys()), list(poller_params.keys()),
        )

        tracker_keys = list(tracker_params.keys())
        tracker_vals = [tracker_params[k] for k in tracker_keys]
        poller_keys = list(poller_params.keys())
        poller_vals = [poller_params[k] for k in poller_keys]

        base = asdict(self.base_config)
        configs: list[PipelineConfig] = []
        for t_combo in itertools.product(*tracker_vals) if tracker_vals else [()]:
            for p_combo in itertools.product(*poller_vals) if poller_vals else [()]:
                overrides = {
                    **dict(zip(tracker_keys, t_combo)),
                    **dict(zip(poller_keys, p_combo)),
                }
                configs.append(PipelineConfig(**{**base, **overrides}))

        per_param_counts = {
            **{k: len(v) for k, v in tracker_params.items()},
            **{k: len(v) for k, v in poller_params.items()},
        }
        log.info("Grid search cardinality per parameter: %s", per_param_counts)

        valid_configs: list[PipelineConfig] = []
        for cfg in configs:
            if not (0 < cfg.alpha_base < 1):
                log.debug("Skipping config with invalid alpha_base: %s", cfg.alpha_base)
                continue
            if not (cfg.beta > 0):
                log.debug("Skipping config with invalid beta: %s", cfg.beta)
                continue
            if not (cfg.outlier_threshold > 0):
                log.debug("Skipping config with invalid outlier_threshold: %s", cfg.outlier_threshold)
                continue
            if not (cfg.drift_tolerance >= 0):
                log.debug("Skipping config with invalid drift_tolerance: %s", cfg.drift_tolerance)
                continue
            if not (cfg.max_interval > 0):
                log.debug("Skipping config with invalid max_interval: %s", cfg.max_interval)
                continue
            if not (cfg.max_interval <= len(self.data)):
                log.debug(
                    "Skipping config with max_interval > data length: %d > %d",
                    cfg.max_interval, len(self.data),
                )
                continue
            if not (cfg.max_interval >= cfg.min_interval):
                log.debug(
                    "Skipping config with max_interval < min_interval: %d < %d",
                    cfg.max_interval, cfg.min_interval,
                )
                continue
            valid_configs.append(cfg)

        label_params = tracker_keys + poller_keys
        log.info("Grid search: evaluating %d configurations", len(valid_configs))
        results = self._evaluate_configs(valid_configs, label_params=label_params)

        ratios = [result.info_loss.sample_ratio for result in results]
        rmses = [result.info_loss.rmse_mean for result in results]
        log.info(
            "Grid search complete. sample_ratio range: [%.3f, %.3f], rmse_mean range: [%.4f, %.4f]",
            min(ratios), max(ratios), min(rmses), max(rmses),
        )

        swept_params = {**tracker_params, **poller_params}
        return SweepResult(
            results=results,
            sweep_params={k: list(v) for k, v in swept_params.items()},
            n_data_points=len(self.data),
        )

    def _evaluate_configs(
            self,
            configs: list[PipelineConfig],
            label_params: Sequence[str] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate a list of configurations (possibly in parallel)."""
        if not configs:
            log.warning("No configurations to evaluate.")
            return []

        total = len(configs)
        log_every = max(1, total // 200)
        log.info("Information will be logged every %d configurations", log_every)

        if self.n_jobs == 1:
            log.info("Evaluating %d configurations serially", total)
            results: list[EvaluationResult] = []
            for i, cfg in enumerate(configs, 1):
                results.append(EvaluationResult.from_run(cfg, self.data))
                if i % log_every == 0 or i == total:
                    if label_params:
                        log.info(
                            "  Progress: %d / %d  (%.0f %%)  [%d] %s",
                            i, total, 100 * i / total, i, _config_label(cfg, label_params),
                        )
                    else:
                        log.info("  Progress: %d / %d  (%.0f %%)", i, total, 100 * i / total)
            return results

        max_workers = (os.cpu_count() or 4) if self.n_jobs <= 0 else self.n_jobs
        chunksize = max(1, total // (max_workers * 4))
        log.info(
            "Evaluating %d configurations in parallel (workers=%d, chunksize=%d)",
            total, max_workers, chunksize,
        )

        n_cpus = os.cpu_count() or max_workers
        cpu_assignments: list[list[int]] = []
        for w in range(max_workers):
            lo = w * n_cpus // max_workers
            hi = (w + 1) * n_cpus // max_workers
            cpu_assignments.append(list(range(lo, hi)) or [w % n_cpus])

        counter = multiprocessing.Value("i", 0)
        results: list[EvaluationResult] = []
        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(self.data, counter, cpu_assignments),
        ) as executor:
            for i, result in enumerate(executor.map(_eval_worker, configs, chunksize=chunksize), 1):
                results.append(result)
                if i % log_every == 0 or i == total:
                    if label_params:
                        log.info(
                            "  Progress: %d / %d  (%.0f %%)  [%d] %s",
                            i, total, 100 * i / total, i, _config_label(configs[i - 1], label_params),
                        )
                    else:
                        log.info("  Progress: %d / %d  (%.0f %%)", i, total, 100 * i / total)

        return results

    @staticmethod
    def _validate_param_keys(
            tracker_params: Mapping[str, Sequence[Any]],
            poller_params: Mapping[str, Sequence[Any]],
    ) -> None:
        """Validate grid-search parameter names before creating combinations."""
        invalid_tracker = sorted(set(tracker_params.keys()) - _TRACKER_PARAMS)
        invalid_poller = sorted(set(poller_params.keys()) - _POLLER_PARAMS)

        if invalid_tracker or invalid_poller:
            parts: list[str] = []
            if invalid_tracker:
                parts.append(
                    f"invalid tracker params: {invalid_tracker} (allowed: {sorted(_TRACKER_PARAMS)})"
                )
            if invalid_poller:
                parts.append(
                    f"invalid poller params: {invalid_poller} (allowed: {sorted(_POLLER_PARAMS)})"
                )
            raise ValueError("; ".join(parts))

    def evaluate_configs(
            self,
            configs: Sequence[PipelineConfig],
            sweep_params: Mapping[str, Sequence[Any]] | None = None,
    ) -> SweepResult:
        """Evaluate an explicit list of configurations."""
        label_params = list((sweep_params or {}).keys())
        results = self._evaluate_configs(list(configs), label_params=label_params)
        return SweepResult(
            results=results,
            sweep_params={k: list(v) for k, v in (sweep_params or {}).items()},
            n_data_points=len(self.data),
        )


def _load_trace(path: str) -> list[float]:
    """Load trace values from a JSON file (list of {value: ...} objects)."""
    with open(path, "r") as f:
        jdata = json.load(f)
    return [item["value"] for item in jdata]


def _load_grid_yaml(path: str) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Load tracker/poller grid definitions from a YAML file."""
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Grid YAML must be a mapping, got {type(raw).__name__}")

    tracker_grid = raw.get("tracker", {})
    poller_grid = raw.get("poller", {})

    for section, grid in [("tracker", tracker_grid), ("poller", poller_grid)]:
        for key, vals in grid.items():
            if not isinstance(vals, list):
                raise ValueError(
                    f"Grid YAML: {section}.{key} must be a list, got {type(vals).__name__}"
                )

    return tracker_grid, poller_grid


def _build_urgency_configs(
        base_config: PipelineConfig | None = None,
) -> tuple[list[PipelineConfig], dict[str, list[Any]]]:
    """Build the fixed urgency-ablation search space."""
    base = asdict(base_config or PipelineConfig())
    configs = [
        PipelineConfig(**{**base, "interval_mapping": IntervalMapping.ORIGINAL_INTERVAL.value}),
        PipelineConfig(**{**base, "interval_mapping": IntervalMapping.SMOOTHSTEP_INTERVAL.value}),
        PipelineConfig(**{**base, "interval_mapping": IntervalMapping.GEOMETRIC_RATE.value}),
        PipelineConfig(**{
            **base,
            "interval_mapping": IntervalMapping.EXPONENTIAL_EASE_OUT.value,
            "interval_exponential_steepness": 4.0,
        }),
    ]

    power_values = [0.5, 2.0]
    for interval_power in power_values:
        configs.append(PipelineConfig(**{
            **base,
            "interval_mapping": IntervalMapping.POWER_RATE.value,
            "interval_power": interval_power,
        }))

    logistic_midpoints = [0.4, 0.6]
    logistic_steepness = [12.0]
    for midpoint in logistic_midpoints:
        for steepness in logistic_steepness:
            configs.append(PipelineConfig(**{
                **base,
                "interval_mapping": IntervalMapping.LOGISTIC_RATE.value,
                "interval_logistic_midpoint": midpoint,
                "interval_logistic_steepness": steepness,
            }))

    sweep_params = {
        "interval_mapping": [
            IntervalMapping.ORIGINAL_INTERVAL.value,
            IntervalMapping.SMOOTHSTEP_INTERVAL.value,
            IntervalMapping.GEOMETRIC_RATE.value,
            IntervalMapping.POWER_RATE.value,
            IntervalMapping.LOGISTIC_RATE.value,
            IntervalMapping.EXPONENTIAL_EASE_OUT.value,
        ],
        "interval_power": power_values,
        "interval_logistic_midpoint": logistic_midpoints,
        "interval_logistic_steepness": logistic_steepness,
        "interval_exponential_steepness": [4.0],
    }
    return configs, sweep_params


def _load_data_file(path: str) -> list[float] | None:
    if not os.path.isfile(path):
        log.error("Data file not found: %s", path)
        return None
    log.info("Loading trace data from %s", path)
    data = _load_trace(path)
    log.info("Loaded %d data points", len(data))
    return data


def _run_sweep_and_save(
        sweep: ParameterSweep,
        *,
        tracker_grid: Mapping[str, Sequence[Any]] | None = None,
        poller_grid: Mapping[str, Sequence[Any]] | None = None,
        configs: Sequence[PipelineConfig] | None = None,
        sweep_params: Mapping[str, Sequence[Any]] | None = None,
        filename: str = "sweep.json",
        log_level: str = "INFO",
        extra_manifest: Mapping[str, Any] | None = None,
) -> None:
    log_folder = initialize_log(
        log_level=log_level,
        name="sweep",
        application_type="parameter_sweep",
    )

    if configs is not None:
        log.info("Total grid size: %d configurations", len(configs))
        sweep_result = sweep.evaluate_configs(configs, sweep_params=sweep_params)
    else:
        tracker_grid = dict(tracker_grid or {})
        poller_grid = dict(poller_grid or {})
        n_configs = 1
        for vals in tracker_grid.values():
            n_configs *= len(vals)
        for vals in poller_grid.values():
            n_configs *= len(vals)
        log.info("Total grid size: %d configurations", n_configs)
        sweep_result = sweep.grid_search(
            tracker_params=tracker_grid,
            poller_params=poller_grid,
        )

    log.info("\n%s", sweep_result.summary())

    save_dir = log_folder or "out"
    path = sweep_result.save_json(save_dir, filename=filename, extra_manifest=extra_manifest)
    log.info("Results ready for analysis: %s", path)


# ------------------------------
# CLI: evaluate
# ------------------------------


def cmd_evaluate_tracker(args: argparse.Namespace) -> None:
    """Run a tracker-parameter grid search defined by YAML."""
    data = _load_data_file(args.data_file)
    if data is None:
        return
    if not os.path.isfile(args.grid):
        log.error("Grid YAML not found: %s", args.grid)
        return

    log.info("Loading grid definition from %s", args.grid)
    tracker_grid, poller_grid = _load_grid_yaml(args.grid)
    sweep = ParameterSweep(data, n_jobs=args.n_jobs)
    _run_sweep_and_save(
        sweep,
        tracker_grid=tracker_grid,
        poller_grid=poller_grid,
        filename="tracker_sweep.json",
        log_level=args.log_level,
    )


def cmd_evaluate_urgency(args: argparse.Namespace) -> None:
    """Run the built-in urgency-mapping ablation sweep."""
    data = _load_data_file(args.data_file)
    if data is None:
        return

    configs, sweep_params = _build_urgency_configs()
    label_params = list(sweep_params.keys())
    base_config = PipelineConfig()
    urgency_trace = compute_fixed_urgency_trace(np.asarray(data, dtype=np.float64), base_config._make_poller())
    shared_urgency_profile = _build_shared_urgency_profile(urgency_trace)
    interval_profiles = _build_urgency_profiles(data, urgency_trace, configs, label_params)

    log_folder = initialize_log(
        log_level=args.log_level,
        name="sweep",
        application_type="parameter_sweep",
    )
    if not log_folder:
        log.error("Failed to initialize log folder for urgency sweep.")
        return

    total = len(configs)
    log.info("Total grid size: %d configurations", total)
    log.info("Information will be logged every %d configurations", 1)
    log.info("Fixed urgency trace computed once for %d points", len(urgency_trace))
    log.info("Evaluating %d fixed-urgency configurations serially", total)

    results: list[EvaluationResult] = []
    for i, cfg in enumerate(configs, 1):
        results.append(EvaluationResult.from_fixed_urgency_run(cfg, data, urgency_trace))
        log.info(
            "  Progress: %d / %d  (%.0f %%)  [%d] %s",
            i, total, 100 * i / total, i, _config_label(cfg, label_params),
        )

    sweep_result = SweepResult(
        results=results,
        sweep_params={k: list(v) for k, v in sweep_params.items()},
        n_data_points=len(data),
    )
    log.info("\n%s", sweep_result.summary())
    path = sweep_result.save_json(
        log_folder,
        filename="urgency_sweep.json",
        extra_manifest={
            "plot_kind": "urgency",
            "shared_urgency_profile": shared_urgency_profile,
            "config_profiles": interval_profiles,
        },
    )
    log.info("Results ready for analysis: %s", path)


# ------------------------------
# CLI: plot
# ------------------------------


def cmd_plot(args: argparse.Namespace) -> None:
    """Load a sweep JSON and generate plots."""
    from plot.sweep import (
        load_sweep,
        plot_configuration_metric,
        plot_importance,
        plot_marginal,
        plot_pareto,
        plot_urgency_profiles,
    )

    log_folder = initialize_log(
        log_level=args.log_level,
        name="sweep_plot",
        console_only=False,
        application_type="sweep_plot",
    )

    if not log_folder:
        log.error("Failed to initialize log folder for plots.")
        return

    columns, meta = load_sweep(args.sweep_json)
    sweep_params = meta.get("sweep_params", {})

    pareto_y = args.pareto_y or args.metric
    metric = args.metric

    if args.plot_target == "urgency":
        plot_configuration_metric(
            columns,
            metric=metric,
            save_path=os.path.join(log_folder, f"configuration_{metric}.png"),
        )
        plot_urgency_profiles(
            meta,
            combined_save_path=os.path.join(log_folder, "urgency_interval_profiles.png"),
            zoom_save_path=os.path.join(log_folder, "urgency_interval_profiles_zoom.png"),
        )
    else:
        plot_marginal(
            columns,
            sweep_params,
            metric=metric,
            save_path=os.path.join(log_folder, f"marginal_{metric}.png"),
        )
        plot_importance(
            columns,
            sweep_params,
            metric=metric,
            save_path=os.path.join(log_folder, f"importance_{metric}.png"),
        )
    plot_pareto(
        columns,
        x_metric=args.pareto_x,
        y_metric=pareto_y,
        save_path=os.path.join(log_folder, f"pareto_{args.pareto_x}_vs_{pareto_y}.png"),
    )

    log.info("All plots generated.")


# ------------------------------
# CLI dispatcher
# ------------------------------


def _add_eval_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to JSON file containing trace data.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for evaluation (default: 1). Use -1 to use all available CPUs.",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )


def _add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "sweep_json",
        help="Path to sweep JSON produced by the corresponding evaluate command.",
    )
    parser.add_argument(
        "--metric",
        default="rmse_mean",
        help="Metric to analyse (default: rmse_mean).",
    )
    parser.add_argument("--pareto-x", default="sample_ratio")
    parser.add_argument(
        "--pareto-y",
        default=None,
        help="Y-axis for Pareto plot (default: same as --metric).",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep: evaluate tracker configs or plot results.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Run a tracker or urgency sweep.")
    eval_sub = p_eval.add_subparsers(dest="evaluate_target", required=True)

    p_eval_tracker = eval_sub.add_parser(
        "tracker",
        help="Run a tracker/poller sweep defined by a YAML grid.",
    )
    _add_eval_common_args(p_eval_tracker)
    p_eval_tracker.add_argument(
        "--grid",
        type=str,
        required=True,
        help="Path to YAML file defining the parameter grid (tracker/poller sections with lists of values).",
    )

    p_eval_urgency = eval_sub.add_parser(
        "urgency",
        help="Run the built-in urgency mapping ablation sweep.",
    )
    _add_eval_common_args(p_eval_urgency)

    p_plot = sub.add_parser(
        "plot",
        help="Generate plots from tracker or urgency sweep results.",
    )
    plot_sub = p_plot.add_subparsers(dest="plot_target", required=True)
    p_plot_tracker = plot_sub.add_parser("tracker", help="Plot tracker sweep results.")
    _add_plot_args(p_plot_tracker)
    p_plot_urgency = plot_sub.add_parser("urgency", help="Plot urgency sweep results.")
    _add_plot_args(p_plot_urgency)

    args = parser.parse_args()

    if args.command == "evaluate":
        if args.evaluate_target == "tracker":
            cmd_evaluate_tracker(args)
        elif args.evaluate_target == "urgency":
            cmd_evaluate_urgency(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()
