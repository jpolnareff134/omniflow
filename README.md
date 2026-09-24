# OmniFlow

OmniFlow is a research project exploring adaptive monitoring techniques to reduce overhead while preserving critical information. It
investigates whether polling rates, usually driven by external factors like throughput/latency/user input, can be adapted based on the signal itself without sacrificing the ability to capture important changes.

## Idea

OmniFlow uses a moving average estimator to track the signal and an urgency metric to decide when to sample. The goal is to sample more frequently when the signal is changing rapidly (high urgency) and less frequently when it's stable (low urgency). This way, we can capture important changes without overwhelming the system with data.

## Repository Layout

```text
deploy/
  cfp_multitest/  # CloudPerfTrace paper experiments
  k8s/            # Kubernetes HPA experiment
  io_syscall/     # I/O syscall monitoring experiment
  dacapo/         # DaCapo Java-agent experiments
src/
  config.py       # runtime defaults and environment overrides
  synthetic.py    # offline trace experiments
  live.py         # supported live entrypoint
  sweep.py        # parameter sweeps and Pareto analysis
  data/           # dataset loaders and trace composition helpers
  probe/          # live data sources (/proc, cgroup, syscall counter)
  tracker/        # tracker, poller, dual reader, pipeline
  eval/           # end-to-end overhead helpers
  plot/           # plotting helpers
  support/        # shared logging/utilities
tests/            # offline and HPA regression tests
paper_results/    # compact recorded results and provenance
datasets/         # external datasets, not tracked by git
out/              # experiment outputs, not tracked by git
extras/           # supplementary helpers (full checkout only; omitted from paper export)
```

Notes:

- `datasets/cloudperftrace` should contain [CloudPerfTrace](https://huggingface.co/datasets/AmirShahbaz/CloudPerfTrace).
- `out/` is created by experiments as needed and is gitignored.
- All new run folders follow the canonical format defined in `src/support/log.py`: `out/<YYYYMMDD_HHMMSS.ffffff>__<application_type>_<hostname>/`.

## Features

The repository is organized around four main experiment types:

- offline trace experiments in `src/synthetic.py`
- live probe collection in `src/live.py`
- parameter sweeps in `src/sweep.py`
- deploy experiments in `deploy/cfp_multitest`, `deploy/k8s`, `deploy/io_syscall`, and `deploy/dacapo`

Synthetic trace experiments support both regular dense-reference vs. adaptive-replay comparisons and a batch mode that produces an aggregated overview plot with auto-zoomed panels for very long traces. It supports arbitrary numeric columns from JSON or CloudPerfTrace Parquet traces, and has modules for generating synthetic traces with configurable signal characteristics.

The live probe collection supports multiple modes:

- `fixed`: regular dense collection. Every scheduled read happens. Use this as the live no-sampling baseline.
- `dual`: dense collection plus offline adaptive replay. This is the evaluation mode when you want information-loss metrics, because full-resolution data is still collected.
- `adaptive`: true live adaptive collection. Reads are actually skipped. Output is therefore a sparse live trace with sampling summaries, not same-run dense information-loss metrics.
- `overhead`: end-to-end workload comparison across baseline, attached-idle, fixed-rate comparison legs, and adaptive monitoring.

## Quick Start

Use `./entrypoint.sh` when possible. It handles the Linux and root requirements for live modes.

```bash
# Offline trace experiment on CloudPerfTrace.
python3 src/synthetic.py --cloudperftrace datasets/cloudperftrace --task 4 --column tr_self --metric 2 --max-rows 100 --batch --comment "quick-test"

# Live adaptive CPU monitoring.
./entrypoint.sh live --probe cpu --mode adaptive --duration 120 --interval 0.5

# Dense collection plus offline adaptive replay.
./entrypoint.sh live --probe cpu --mode dual --duration 120 --interval 0.5

# End-to-end overhead comparison.
./entrypoint.sh live --probe cpu --mode overhead --duration 30 --interval 0.5
```

## Supported Commands

### Synthetic

```bash
# CloudPerfTrace slice.
python3 src/synthetic.py --cloudperftrace datasets/cloudperftrace --task 4 --column tr_self --metric 2 --max-rows 100 --batch

# JSON trace using a specific field.
python3 src/synthetic.py --json /path/to/trace.json --field cpus --batch
```

### Live

```bash
# Regular dense live collection.
./entrypoint.sh live --probe cpu --mode fixed --duration 120 --interval 0.5

# True adaptive live collection with skipped reads.
./entrypoint.sh live --probe syscall --syscall read --mode adaptive --duration 120 --interval 0.2

# Dense collection plus offline replay for fidelity analysis.
./entrypoint.sh live --probe cpu --mode dual --duration 120 --interval 0.5

# Workload overhead comparison.
./entrypoint.sh live --probe cpu --mode overhead --duration 30 --interval 0.5
```

### Sweep

```bash
# Evaluate tracker parameter grid.
python3 src/sweep.py evaluate tracker --data-file path/to/trace.json --grid path/to/grid.yaml

# Evaluate urgency-mapping variants.
python3 src/sweep.py evaluate urgency --data-file path/to/trace.json

# Plot saved sweep results.
python3 src/sweep.py plot tracker path/to/tracker_sweep.json --metric rmse_mean
```

## Deploy Experiments

### CloudPerfTrace paper experiments (`deploy/cfp_multitest/`)

Entrypoint: `deploy/cfp_multitest/run_paper_experiments.py`

Reproduces the five CloudPerfTrace paper cases plus comparison policies (budget-matched N-of-M, Huang-inspired wavelet-rate, and Magalhaes-inspired truncated-exponential). For each case it runs OmniFlow adaptive, derives the reciprocal fixed interval, and reports information-loss metrics through the same dense-reference code used everywhere else.

```bash
python3 deploy/cfp_multitest/run_paper_experiments.py \
  --cloudperftrace datasets/cloudperftrace \
  --max-rows 0 --max-points 0
```

Post-processing and optional tools in this directory:

- `render_paper_results.py RUN.json --out OUT` - renders a Markdown/LaTeX table from a `run.json` produced by the entrypoint.
- `run_interval_mapping_experiment.py` - urgency-to-interval mapping ablation (optional; can also be done via `src/sweep.py`).

### Kubernetes HPA (`deploy/k8s/`)

Entrypoints:

- `deploy/k8s/setup-cluster.sh` - one-time cluster setup.
- `deploy/k8s/run-experiment.sh` - runs the HPA autoscaling experiment.
- `deploy/k8s/run-all-shapes.sh` - orchestrates `run-experiment.sh` across workload shapes.
- `deploy/k8s/teardown-cluster.sh` - optional teardown.

```bash
cd deploy/k8s

# First, set up the cluster and deploy the baseline and OmniFlow HPA controllers.
./setup-cluster.sh

# Run a single shape (default: phased, 5 repeats).
./run-experiment.sh --kubeconfig ./kubeconfig --repeats 5

# Run all shapes, waiting 30 seconds to stabilize the HPA before each run and 45 seconds for the workload to ramp up before the HPA is armed.
./run-all-shapes.sh --kubeconfig ./kubeconfig --max-attempts 20 --arm-lead-seconds 45 --hpa-stable-seconds 30

# Optional teardown after inspection.
./teardown-cluster.sh
```

Key helper files:

- `omniflow_daemon.py` - container entrypoint for the OmniFlow daemon (used by `Dockerfile.daemon`).
- `locustfile.py` + `load_schedule.py` - load generator logic and shape definitions (used by `Dockerfile.locust`).
- `experiment_sync.py` - synchronization primitives used by the experiment runner.
- `collect-results.py` - aggregates daemon logs, HPA events, and replica timelines into `analysis.json`.
- `gen_load_shapes.py` - standalone helper for generating workload-shape definitions.
- `offline_test.py` - standalone offline replay / sanity-check tool.
- `plot_comparison.py` - standalone plotting helper for existing `analysis.json` files.

### I/O Syscall (`deploy/io_syscall/`)

Entrypoint: `deploy/io_syscall/setup.sh`

Note: the experiment requires the same Kubernetes cluster setup as the HPA experiment. If you haven't set up the cluster yet, run `deploy/k8s/setup-cluster.sh` first.

```bash
cd deploy/io_syscall

# Full Kubernetes-based syscall experiment.
./setup.sh --profile omniflow

# Optional teardown.
./teardown.sh
```

Key helper files:

- `omniflow_io_daemon.py` - container entrypoint for the I/O daemon (used by `Dockerfile.daemon_io`).
- `evaluate.py` - offline evaluation of a captured JSON trace.
- `profile_loader.py` + `adaptive_profiles.json` - shared adaptive-profile loader.
- `summarize_workload_overhead.py` and `aggregate_io_results.py` - workload-overhead aggregation helpers called by `setup.sh`.
- `build_io_report.py` - standalone LaTeX report generator for existing results.
- `plot_io_workloads_pgf.py` - standalone plotting helper for the load shapes used in the experiment.

### DaCapo (`deploy/dacapo/`)

Given the complexity, requirements of external source files, and the need for Java,
the DaCapo experiment has its own README in `deploy/dacapo/README.md`. View the README for instructions on how to run the DaCapo experiments and interpret the results.

## Configuration

OmniFlow has several tunable parameters that affect its behavior. Most of them are available inside `src/config.py` and can be overridden via `OMNIFLOW_*` environment variables.

## Requirements

- Python 3.12+
- Everything from `requirements.txt`
  - While not technically everything is required for every experiment, the full list is small and it's easier to maintain a single set of dependencies.
- Linux for `/proc`, cgroup, and eBPF-backed live probes
- root plus BCC for syscall monitoring
- Docker, kubectl, and minikube for the Kubernetes-based deploy experiments
