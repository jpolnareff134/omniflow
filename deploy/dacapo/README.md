# Reproducing the Mertz–Nunes DaCapo sampling experiment with OmniFlow

This repository contains the Java agent, runners, validators, and analysis tools
used to compare OmniFlow with the sampling policies evaluated by Mertz and
Nunes in *Software runtime monitoring with adaptive sampling rate to collect
representative samples of execution traces* (Journal of Systems and Software,
2023).

The package runs the workloads distributed with the companion artifact,
reproduces its full-monitoring, adaptive, uniform, and inverse-throughput
policies, and adds OmniFlow under the same workload schedules and monitoring-cycle boundaries.

## Compared policies

| Name | Role in this package |
|---|---|
| FUM | Full monitoring. It is the paper-compatible reference execution. |
| ADP | The adaptive policy from the released implementation. |
| UNI | Independent Bernoulli sampling with probability 0.5. |
| INV | Sampling rate inversely proportional to observed throughput. |
| OmniFlow | Model-free adaptive sampling driven by the monitored heap-delta stream. |
| NOM | Optional agent-free diagnostic. It is not part of the paper-facing comparison. |

ADP always runs first in a repetition. Its monitoring-cycle markers are then
reused by FUM, UNI, INV, and OmniFlow. This keeps the time windows identical
across policies, as in the original evaluation procedure.

## Workload matrix

All publication runs use Java 8 and a 4 GiB maximum heap.

| Reported arm | Artifact | Size | Threads | Fixed schedule | Request-type check |
|---|---|---:|---:|---:|---:|
| Cassandra | `cassandra.jar` | `default` | 200 | 1,660 s | at least 100 |
| H2 | `benchmarks.jar` | `huge` | 20 | 1,780 s | exactly 8 |
| Lusearch | `benchmarks.jar` | `large` | 6 | 1,640 s | at least 100 |
| tradebeans-release* | `benchmarks.jar` | `huge` | 24 | 1,780 s | exactly 8 |
| xalan-release* | `benchmarks.jar` | `large` | 6 | 1,640 s | exactly 17 |

The asterisks identify differences between the paper’s description and the
released companion artifact. They are explained below.

## Differences from the published description

### Tradebeans

The paper describes a DayTrader stock-trading workload with twelve request
types. The active `tradebeans` implementation in the companion material and binary
uses the H2/TPCC request path and exposes eight transaction types, with nothing that matches the paper’s DayTrader description.

To maintain comparability, we also run this experiment using the exact released
command and report it as `tradebeans-release*`.

### Xalan

The paper reports sixteen Xalan request types. The released source queues
seventeen XML input documents. The original aspect also used a JVM-specific
`StreamResult` identity, which is not stable across separate policy executions.
This package uses the XML input filename as the request key and reports all
seventeen released inputs as `xalan-release*`.

### Cassandra

The companion launcher does not select an explicit Cassandra size, so the
publication configuration is `default`.

### External benchmark data

Lusearch, Xalan, and Cassandra require external data available in the original artifact:

- the Lusearch index and query files from `benchmarks.jar`;
- the Xalan stylesheet and XML inputs from `benchmarks.jar`;
- Cassandra configuration and data from the supplied `data.zip`.

## RMSE definition and interpretation

For each request type, the evaluator computes the mean of positive heap deltas.
Negative deltas are discarded because they generally indicate that garbage
collection occurred during the measured interval. RMSE compares the policy’s
per-request means with those from the separate FUM execution in the same
repetition block.

This follows the paper, but it means the public RMSE contains both sampling error
and variation between independent JVM executions. The artifact also records two
internal diagnostics:

- paired same-run selected-versus-dense RMSE;
- pairwise RMSE among independent FUM executions.

## Requirements

- Linux/x86_64 for final runs;
- Java 8;
- Python 3;
- Bash;
- `jar`, `unzip`, and SHA-256 tooling;
- the released `benchmarks.jar`;
- the released Cassandra `cassandra.jar` and `data.zip`.

Known runtime hashes:

```text
benchmarks.jar      03dae4d926aba665cfdf3b22d0cd71221d2fe0303b1113c0c0ba2e53dfe54598
cassandra.jar       e92e780b25003169a419c98bea8cb2138e24675b90bd6a137ba892a7435ff4e4
Cassandra data.zip  ebf8a6b94f1d5640ee8630f82711ce016a318cfceca15c7e31acd8d714693a8f
Cassandra data.zip  9b259a066c14f8abadbdf6c8110af175cb601d10642946b47334a664050b3b26
```

Both known Cassandra data archives are accepted only after a structural check
for `data/dat/cassandra/conf/cassandra.yaml`.

## Install and stage the runtime

```bash
JAVA8=/path/to/java8/bin/java
ORIGINAL_DIR=/path/to/JSS-2022/experiment

./scripts/stage_paper_runtime.sh \
  --java "$JAVA8" \
  --benchmarks-jar "$ORIGINAL_DIR/benchmarks.jar" \
  --cassandra-jar "$ORIGINAL_DIR/cassandra/cassandra.jar" \
  --cassandra-data-zip "$ORIGINAL_DIR/cassandra/data.zip"
```

The artifact contains the Java agent sources and build script, not a prebuilt
agent JAR or ASM binary libraries. Staging rebuilds the agent when it is missing
or stale. `agent/build.sh` uses the ASM classes in the staged Cassandra JAR as a
fallback when the external JAR contains them, and fails loudly otherwise. This
keeps third-party benchmark and ASM binaries outside the artifact.

A manual build can be requested with:

```bash
JAVAC="$(dirname "$JAVA8")/javac" ./agent/build.sh
```

## Check everything is ready

```bash
./scripts/smoke_paper_suite.sh \
  --java "$JAVA8" \
  --benchmarks all \
  --seconds 30 \
  --startup-timeout 600 \
  --smoke-cycle-length-ms 10000 \
  --uniform-rate 0.5 \
  --seed 1
```

## Run the full experiment

```bash
./scripts/run_paper_suite.sh \
  --java "$JAVA8" \
  --benchmarks all \
  --reps 10 \
  --uniform-rate 0.5 \
  --seed 1 \
  --cycle-length-ms 180000 \
  --prune-workdirs \
  --output ./paper-suite-runs-v25-final
```

The runner is resumable. Re-running the same command with `--resume` reuses only
arms whose metadata and strict validation match the requested experiment.
Existing measurement files are never overwritten.

## Outputs

The main output directory contains:

```text
paper-summary.md       paper-facing sampling ratio and RMSE
paper-summary.json     machine-readable results and diagnostics
paper-details.csv      per-benchmark, per-repetition policy values
diagnostics.md         throughput, overhead, paired RMSE, markers, validity checks
```

Each repetition directory retains the policy logs, summaries, telemetry, cycle
aggregates, marker files, and run metadata. Controller telemetry is optional and
is not required to calculate the paper metrics.

## Publication scripts

The paper execution path consists of:

- `stage_paper_runtime.sh`: validates the three external benchmark archives and
  stages them without copying them into the source artifact.
- `smoke_paper_suite.sh`: runs shortened integration checks for the selected
  benchmarks and policies.
- `run_paper_suite.sh`: runs ADP, UNI, FUM, INV, and OmniFlow, validates each arm,
  supports safe resume, and emits the paper-facing summaries.
- `validate_paper.py`: validates one benchmark-policy run's schedule, marker,
  request-count, and workload outputs.
- `evaluate_paper_suite.py`: computes the per-repetition and paper-facing
  sampling-ratio and RMSE results.
- `paper_benchmarks.py`: defines the canonical workload sizes, thread counts,
  schedules, request-type checks, and artifact labels.
- `extract_cassandra_data.sh`, `extract_lusearch_data.sh`, and
  `extract_xalan_data.sh`: stage workload inputs from the external archives.
- `inspect_tradebeans_artifact.py`: verifies and labels the released Tradebeans
  workload variant.

`--prune-workdirs` on `run_paper_suite.sh` removes a policy work directory only
after that policy has passed strict validation. Logs and measurement files are
preserved.

## References

- J. Mertz and I. Nunes, “Software runtime monitoring with adaptive sampling
  rate to collect representative samples of execution traces,” *Journal of
  Systems and Software*, vol. 202, 111708, 2023.
- Companion artifact: https://www.inf.ufrgs.br/prosoft/resources/2022/jss-adaptive-sampling/
