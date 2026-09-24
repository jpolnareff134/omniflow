#!/usr/bin/env bash
# setup.sh – Adaptive I/O Syscall Rate Monitoring
#
# Deploys the OmniFlow I/O daemon on a single cluster node, then runs
# workloads sequentially (controlled by --experiment):
#   1. PostgreSQL + pgbench    for N seconds  -> teardown     (--experiment postgres)
#   2. Redis + redis-benchmark for N seconds  -> teardown     (--experiment redis)
#   3. Latency p99 benchmark   two arms       -> teardown     (--experiment latency)
#        Arm a (1/1)  : sample every point – true dense-reference baseline
#        Arm c (${ADAPTIVE_MIN_INTERVAL}/${ADAPTIVE_MAX_INTERVAL}) : OmniFlow adaptive collection
#   4. Idle-node daemon floor  two arms       -> no workload  (--experiment floor or --with-floor)
#        Arm a (1/1)  : dense-reference floor
#        Arm c (${ADAPTIVE_MIN_INTERVAL}/${ADAPTIVE_MAX_INTERVAL}) : adaptive floor
#      All arms use fio (Flexible I/O Tester, github.com/axboe/fio) as the
#      workload: CALM->RAMP->SPIKE cycle, 2 asynchronous replicas, --fsync=1 to
#      generate the same write+fsync syscall mix as PostgreSQL/Redis.
#      evaluate.py is called with matching --min-interval/--max-interval so
#      the offline replay uses the same poller config as the live daemon.
# Optionally evaluates each captured trace, including daemon overhead
# summaries and overhead plots when the trace contains telemetry fields.
#
# Prerequisites
# -------------
#   - kubectl available in PATH and pointed at the target cluster
#     (set KUBECONFIG env var or use --kubeconfig flag)
#   - Docker available (for building the OmniFlow daemon image)
#   - minikube available (optional; used with --profile to load images)
#
# Usage
# -----
#   # Auto-detect first worker node, 120 s per workload
#   ./setup.sh --profile omniflow
#
#   # Explicit node + 5-minute windows
#   ./setup.sh --profile omniflow --node <worker-node> --duration 300
#
#   # Save output + skip evaluation
#   ./setup.sh --profile omniflow --duration 120 --out out/run_001 --no-eval
#
#   # Also run baseline/full/adaptive workload overhead collection
#   ./setup.sh --profile omniflow --duration 120 --with-overhead
#
#   # Add idle-node floor profiling and repeat each experiment 5 times
#   ./setup.sh --profile omniflow --duration 120 --with-overhead --with-floor --repeats 5
#
#   # Without minikube (image must already be in the cluster)
#   ./setup.sh --kubeconfig ~/.kube/config --no-minikube --node my-node
#
#   # Run only the latency p99 detection experiment
#   ./setup.sh --profile omniflow --experiment latency

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,54p' "$0"
  exit 0
fi

cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

KUBECONFIG_PATH="${KUBECONFIG:-}"
PROFILE="omniflow"
USE_MINIKUBE=true
NS="omniflow-io"
TARGET_NODE=""
DURATION=120
WARMUP=20        # seconds after workload starts before collecting (init time)
OUT_DIR=""
RUN_EVAL=true
EXPERIMENT="all"  # all | postgres | redis | latency | floor
WITH_OVERHEAD=false
REPEATS=1
ADAPTIVE_PROFILE="default"
SKIP_BUILD=0
FULL_MIN_INTERVAL=1
FULL_MAX_INTERVAL=1
# FIXED_INTERVAL=3
ADAPTIVE_MIN_INTERVAL=1
ADAPTIVE_MAX_INTERVAL=10
LATENCY_CALM_DURATION=60
LATENCY_RAMP_DURATION=20
LATENCY_SPIKE_DURATION=15
LATENCY_CYCLE_DURATION=$((LATENCY_CALM_DURATION + LATENCY_RAMP_DURATION + LATENCY_SPIKE_DURATION))
LATENCY_CAPTURE_SLACK=30

while [[ $# -gt 0 ]]; do
  case $1 in
    --kubeconfig)  KUBECONFIG_PATH="$2"; shift 2 ;;
    --profile)     PROFILE="$2";         shift 2 ;;
    --no-minikube) USE_MINIKUBE=false;   shift   ;;
    --skip-build)  SKIP_BUILD=1;         shift   ;;
    --node)        TARGET_NODE="$2";     shift 2 ;;
    --duration)    DURATION="$2";        shift 2 ;;
    --warmup)      WARMUP="$2";          shift 2 ;;
    --out)         OUT_DIR="$2";         shift 2 ;;
    --no-eval)     RUN_EVAL=false;       shift   ;;
    --with-overhead) WITH_OVERHEAD=true; shift   ;;
    --repeats)     REPEATS="$2";        shift 2 ;;
    --adaptive-profile) ADAPTIVE_PROFILE="$2"; shift 2 ;;
    --experiment)  EXPERIMENT="$2";      shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

case "$EXPERIMENT" in
  all|postgres|redis|latency|floor) ;;
  *) echo "ERROR: --experiment must be one of: all postgres redis latency floor" >&2; exit 1 ;;
esac

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "ERROR: --repeats must be an integer >= 1" >&2
  exit 1
fi

if [[ -n "$KUBECONFIG_PATH" ]]; then
  export KUBECONFIG="$KUBECONFIG_PATH"
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(python3 "$REPO_ROOT/src/support/log.py" --base out --application-type io)"
fi
mkdir -p "$OUT_DIR"

# Duplicate stdout and stderr of this script to a log file in the output directory, so we have a record of what happened during setup and can debug if needed.
exec > >(tee -a "$OUT_DIR/setup.log") 2>&1

# ── Helper ───────────────────────────────────────────────────────────────────
_kubectl() { kubectl "$@"; }

_repeat_dir() {
  local base_dir="$1" repeat_index="$2"
  if (( REPEATS == 1 )); then
    printf '%s\n' "$base_dir"
  else
    printf '%s/repeat_%02d\n' "$base_dir" "$repeat_index"
  fi
}

_daemon_env_for_profile() {
  local profile="$1"
  python3 profile_loader.py env "$profile"
}

_daemon_eval_args() {
  local profile="$1"
  printf -- '--config-profile %s' "$profile"
}

_wait_deployment_replicas() {
  local deployment="$1" expected="$2" timeout_s="${3:-120}"
  local start now current
  start=$(date +%s)
  while true; do
    current=$(_kubectl get deployment "$deployment" -n "$NS" \
      -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
    [[ -z "$current" ]] && current=0
    if [[ "$current" == "$expected" ]]; then
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout_s )); then
      echo "ERROR: timed out waiting for deployment/${deployment} readyReplicas=${expected} (current=${current})" >&2
      return 1
    fi
    sleep 2
  done
}

_daemon_scale() {
  local replicas="$1"
  echo "  Scaling omniflow daemon to ${replicas} replica(s) ..."
  _kubectl scale deployment/omniflow-io-daemon -n "$NS" --replicas="$replicas" >/dev/null
  if [[ "$replicas" -gt 0 ]]; then
    _kubectl rollout status deployment/omniflow-io-daemon -n "$NS" --timeout=120s
  fi
  _wait_deployment_replicas omniflow-io-daemon "$replicas" 120
}

_find_pod_by_selector() {
  local selector="$1"
  _kubectl get pods -n "$NS" -l "$selector" \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true
}

_wait_latest_pod_ready() {
  local selector="$1" timeout_s="${2:-60}"
  local start now pod
  start=$(date +%s)

  while true; do
    pod=$(_find_pod_by_selector "$selector")
    if [[ -n "$pod" ]]; then
      if _kubectl wait pod -n "$NS" "$pod" --for=condition=Ready --timeout=10s >/dev/null 2>&1; then
        printf '%s\n' "$pod"
        return 0
      fi
    fi

    now=$(date +%s)
    if (( now - start >= timeout_s )); then
      echo "ERROR: timed out waiting for a ready pod matching selector '$selector'" >&2
      return 1
    fi
    sleep 2
  done
}

START_CAPTURE_PID=""

_start_capture() {
  local pod="$1" outfile="$2" mode="$3" since="${4:-2s}"
  mkdir -p "$(dirname "$outfile")"
  if [[ "$mode" == "json" ]]; then
    (_kubectl logs -f -n "$NS" "$pod" --since="$since" 2>/dev/null \
      | grep --line-buffered '^{' > "$outfile") &
  else
    (_kubectl logs -f -n "$NS" "$pod" --since="$since" 2>/dev/null > "$outfile") &
  fi
  START_CAPTURE_PID=$!
}

_stop_capture() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

_count_latency_phase_samples() {
  local logfile="$1"
  [[ -f "$logfile" ]] || {
    echo 0
    return 0
  }
  python3 - "$logfile" <<'PY'
import re
import sys
from pathlib import Path

line_re = re.compile(r"99(?:\.0+)?th=\[\s*([0-9.]+)\s*([kKmMgG]?)\]", re.IGNORECASE)
count = 0
in_latency_percentiles = False
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    if re.search(r"^\s*lat percentiles \((nsec|usec|msec|sec)\):", line, re.IGNORECASE):
        in_latency_percentiles = True
        continue
    if in_latency_percentiles and "percentiles (" in line:
        in_latency_percentiles = False
    if in_latency_percentiles and line_re.search(line):
        count += 1
        in_latency_percentiles = False
print(count)
PY
}

_latency_target_phase_samples() {
  python3 - "$WARMUP" "$DURATION" "$LATENCY_CYCLE_DURATION" "$LATENCY_CALM_DURATION" "$LATENCY_RAMP_DURATION" "$LATENCY_SPIKE_DURATION" <<'PY'
import sys

warmup = int(sys.argv[1])
duration = int(sys.argv[2])
cycle = int(sys.argv[3])
offsets = [int(sys.argv[4]), int(sys.argv[4]) + int(sys.argv[5]), int(sys.argv[4]) + int(sys.argv[5]) + int(sys.argv[6])]
stop_at = ((warmup + duration + cycle - 1) // cycle) * cycle
count = 0
cycle_start = 0
while cycle_start < stop_at:
    for offset in offsets:
        t = cycle_start + offset
        if warmup < t <= stop_at:
            count += 1
    cycle_start += cycle
print(count)
PY
}

_wait_for_latency_phase_completion() {
  local logfile="$1" target_count="$2"
  local start now elapsed count timeout_s
  start=$(date +%s)
  timeout_s=$((LATENCY_CYCLE_DURATION + LATENCY_CAPTURE_SLACK))
  while true; do
    count=$(_count_latency_phase_samples "$logfile")
    if (( count >= target_count )); then
      return 0
    fi
    now=$(date +%s)
    elapsed=$((now - start))
    if (( elapsed >= timeout_s )); then
      echo "WARNING: latency workload log did not reach ${target_count} completed phase samples within ${timeout_s}s (saw ${count})." >&2
      return 1
    fi
    sleep 1
  done
}

_collect_workload_logs() {
  local label="$1" selector="$2" logfile="$3" phase_target="${4:-}"
  local pid pod
  pod=$(_find_pod_by_selector "$selector")
  [[ -n "$pod" ]] || { echo "ERROR: workload pod not running for selector '$selector'" >&2; return 1; }

  echo "  Streaming ${label} workload logs from ${pod} for ${DURATION}s ..."
  _start_capture "$pod" "$logfile" raw
  pid="$START_CAPTURE_PID"
  sleep "$DURATION"
  if [[ -n "$phase_target" ]]; then
    echo "  Waiting for latency workload to finish the current cycle boundary (${phase_target} completed phase samples) ..."
    if ! _wait_for_latency_phase_completion "$logfile" "$phase_target"; then
      _stop_capture "$pid"
      return 1
    fi
  fi
  _stop_capture "$pid"
  echo "  Collected workload logs -> ${logfile}"
}

# ── Auto-detect target node ──────────────────────────────────────────────────
if [[ -z "$TARGET_NODE" ]]; then
  TARGET_NODE=$(kubectl get nodes --no-headers \
    --selector='!node-role.kubernetes.io/control-plane' \
    -o custom-columns='NAME:.metadata.name' 2>/dev/null \
    | grep -v '^NAME' | head -1 || true)
  [[ -n "$TARGET_NODE" ]] || {
    echo "ERROR: Cannot auto-detect a worker node. Use --node <name>." >&2
    exit 1
  }
fi

echo "=== I/O Syscall Rate Monitoring ==="
echo "  node=${TARGET_NODE}  ns=${NS}"
echo "  duration=${DURATION}s  warmup=${WARMUP}s  eval=${RUN_EVAL}  overhead=${WITH_OVERHEAD}"
echo "  experiment=${EXPERIMENT}  repeats=${REPEATS}  adaptive_profile=${ADAPTIVE_PROFILE}"
echo "  out=${OUT_DIR}"
echo ""

# ── Phase 1: Download kernel headers for Docker image ───────────────────────
# Headers for the minikube ISO kernel (6.6.95) are downloaded from the Ubuntu
# mainline PPA onto the host and COPYed into the image at build time. BCC uses
# them via BCC_KERNEL_SOURCE (set in the DaemonSet manifest).
echo "--- Phase 1: Downloading kernel headers for Docker image ---"
KERNEL_HEADERS_DIR="$REPO_ROOT/deploy/io_syscall/.build-headers"
if [[ $SKIP_BUILD -eq 1 ]]; then
  echo "  Skipping kernel header download."
elif [[ (-d "$KERNEL_HEADERS_DIR" && -f "$KERNEL_HEADERS_DIR/linux-headers-all.deb" && -f "$KERNEL_HEADERS_DIR/linux-headers-amd64.deb")  ]]; then
  echo "  Kernel headers already downloaded in $KERNEL_HEADERS_DIR, skipping download."
else
  echo "  Downloading kernel headers for version 6.6.95..."
  mkdir -p "$KERNEL_HEADERS_DIR"
  BASE="https://kernel.ubuntu.com/mainline/v6.6.95/amd64"
  PKG="6.6.95-060695"; TS="202506271118"
  curl -fsSL -o "$KERNEL_HEADERS_DIR/linux-headers-all.deb" \
    "${BASE}/linux-headers-${PKG}_${PKG}.${TS}_all.deb"
  curl -fsSL -o "$KERNEL_HEADERS_DIR/linux-headers-amd64.deb" \
    "${BASE}/linux-headers-${PKG}-generic_${PKG}.${TS}_amd64.deb"
  echo "  Downloaded kernel headers to $KERNEL_HEADERS_DIR"
fi

# ── Phase 2: Build OmniFlow I/O daemon image ─────────────────────────────────
echo ""
echo "--- Phase 2: Building OmniFlow I/O daemon image ---"

if [[ $SKIP_BUILD -eq 1 ]]; then
  echo "  Skipping Docker image build."
else
  echo "  Building Docker image 'omniflow-io-daemon:latest' from $REPO_ROOT ..."

  # Need to supply KERNEL_HEADERS_DIR as a relative path to REPO_ROOT
  docker build \
    -t omniflow-io-daemon:latest \
    -f Dockerfile.daemon_io \
    --build-arg KERNEL_HEADERS_DIR="$(realpath --relative-to="$REPO_ROOT" "$KERNEL_HEADERS_DIR")" \
    "$REPO_ROOT"

  if $USE_MINIKUBE; then
    echo "Loading image into minikube profile '$PROFILE' ..."
    minikube image load omniflow-io-daemon:latest -p "$PROFILE"
  else
    # Image registry note: push the image to your registry and update
    # manifests/daemonset.yaml (image: field + imagePullPolicy: Always).
    echo "NOTE: --no-minikube set.  Push omniflow-io-daemon:latest to your"
    echo "      registry and update manifests/daemonset.yaml accordingly."
  fi
fi

# Clean up header debs now that the image is built
# rm -rf "$KERNEL_HEADERS_DIR"

# ── Phase 3: Namespace ───────────────────────────────────────────────────────
echo ""
echo "--- Phase 3: Creating namespace ---"
_kubectl apply -f manifests/namespace.yaml

# ── Phase 4: OmniFlow daemon (stays up for both workloads) ──────────────────
echo ""
echo "--- Phase 4: Deploying OmniFlow I/O daemon on '$TARGET_NODE' ---"
sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/daemonset.yaml \
  | _kubectl apply -f -
_kubectl rollout status deployment/omniflow-io-daemon \
  -n "$NS" --timeout=120s

# ── Collect helper ────────────────────────────────────────────────────────────
# _collect <label> <trace-file> [eval-extra-args]
# Streams daemon logs for DURATION seconds, builds trace.json, calls evaluate.py.
# The resulting trace now also carries daemon overhead telemetry, which is
# turned into a dedicated overhead figure by evaluate.py.
# eval-extra-args is passed verbatim to evaluate.py (e.g. "--min-interval 1 --max-interval 25")
_collect() {
  local label="$1" trace_file="$2" eval_extra="${3:-}"
  mkdir -p "$(dirname "$trace_file")"

  local pod
  pod=$(_kubectl get pods -n "$NS" -l app=omniflow-io-daemon \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)
  [[ -n "$pod" ]] || { echo "ERROR: daemon pod not running" >&2; return 1; }

  local tmpfile
  tmpfile=$(mktemp)

  echo "  Streaming ${label} from ${pod} for ${DURATION}s ..."
  _kubectl logs -f -n "$NS" "$pod" --since=2s 2>/dev/null \
    | grep --line-buffered '^{' > "$tmpfile" &
  local bgpid=$!
  sleep "$DURATION"
  kill "$bgpid" 2>/dev/null || true
  wait "$bgpid" 2>/dev/null || true

  local count=0
  {
    printf '[\n'
    local first=true
    while IFS= read -r line; do
      [[ "$line" == \{* ]] || continue
      $first && first=false || printf ',\n'
      printf '%s' "$line"
      (( count++ )) || true
    done < "$tmpfile"
    printf '\n]\n'
  } > "$trace_file"
  rm -f "$tmpfile"

  if [[ "$count" -eq 0 ]]; then
    echo "  WARNING: No records collected for '${label}'." >&2
    return 1
  fi
  echo "  Collected ${count} records -> ${trace_file}"

  if $RUN_EVAL; then
    # shellcheck disable=SC2086
    python3 evaluate.py --json "$trace_file" --out "$(dirname "$trace_file")" $eval_extra
  else
    echo "  (--no-eval)  python3 evaluate.py --json ${trace_file} ${eval_extra}"
  fi
}

_run_replay_eval() {
  local label="$1" source_trace="$2" out_dir="$3" eval_extra="${4:-}"
  mkdir -p "$out_dir"

  if $RUN_EVAL; then
    echo "  Replaying ${label} from ${source_trace} ..."
    # shellcheck disable=SC2086
    python3 evaluate.py --json "$source_trace" --out "$out_dir" $eval_extra
  else
    echo "  (--no-eval) replay ${label} from ${source_trace} into ${out_dir}"
  fi
}

# ── Daemon interval patcher ───────────────────────────────────────────────────
# _patch_daemon_policy <min> <max> <profile>
# Sets the daemon's adaptive profile and interval bounds on the Deployment,
# then waits for the new pod to become ready.
_patch_daemon_policy() {
  local min_int="$1" max_int="$2" profile="$3"
  local -a env_args
  local line pod

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    env_args+=("$line")
  done < <(_daemon_env_for_profile "$profile")
  env_args+=("OMNIFLOW_MIN_INTERVAL=${min_int}" "OMNIFLOW_MAX_INTERVAL=${max_int}")

  echo "  Setting adaptive profile '${profile}' with MIN=${min_int} MAX=${max_int} ..."
  _kubectl set env deployment/omniflow-io-daemon -n "$NS" "${env_args[@]}"
  _kubectl rollout status deployment/omniflow-io-daemon \
    -n "$NS" --timeout=120s

  pod=$(_wait_latest_pod_ready 'app=omniflow-io-daemon' 60)
  if [[ -n "$pod" ]]; then
    echo "  Ready daemon pod: ${pod}"
  else
    echo "  WARNING: could not find running daemon pod to verify env vars" >&2
  fi
}

_aggregate_workload_summaries() {
  local kind="$1" out_dir="$2"
  shift 2
  (( $# > 0 )) || return 0
  python3 aggregate_io_results.py workload \
    --kind "$kind" \
    --out "$out_dir" \
    --summary "$@"
}

_workload_summary_is_bad() {
  local summary_path="$1"
  [[ -f "$summary_path" ]] || return 1
  python3 - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
warnings = ((payload.get("comparisons") or {}).get("warnings") or [])
raise SystemExit(0 if warnings else 1)
PY
}

_record_attempt_failure() {
  local out_dir="$1" reason="$2"
  mkdir -p "$out_dir"
  printf '%s\n' "$reason" > "$out_dir/attempt_failure.txt"
}

_collect_trace_and_workload() {
  local label="$1" trace_file="$2" workload_selector="$3" workload_log="$4" eval_extra="${5:-}" phase_target="${6:-}"
  mkdir -p "$(dirname "$trace_file")"
  mkdir -p "$(dirname "$workload_log")"

  local daemon_pod workload_pod daemon_tmp daemon_pid workload_pid count
  daemon_pod=$(_find_pod_by_selector 'app=omniflow-io-daemon')
  workload_pod=$(_find_pod_by_selector "$workload_selector")
  [[ -n "$daemon_pod" ]] || { echo "ERROR: daemon pod not running" >&2; return 1; }
  [[ -n "$workload_pod" ]] || { echo "ERROR: workload pod not running for selector '$workload_selector'" >&2; return 1; }

  daemon_tmp=$(mktemp)
  echo "  Streaming ${label} daemon logs from ${daemon_pod} and workload logs from ${workload_pod} for ${DURATION}s ..."
  _start_capture "$daemon_pod" "$daemon_tmp" json
  daemon_pid="$START_CAPTURE_PID"
  _start_capture "$workload_pod" "$workload_log" raw
  workload_pid="$START_CAPTURE_PID"
  sleep "$DURATION"
  if [[ -n "$phase_target" ]]; then
    echo "  Waiting for latency workload to finish the current cycle boundary (${phase_target} completed phase samples) ..."
    if ! _wait_for_latency_phase_completion "$workload_log" "$phase_target"; then
      _stop_capture "$daemon_pid"
      _stop_capture "$workload_pid"
      rm -f "$daemon_tmp"
      return 1
    fi
  fi
  _stop_capture "$daemon_pid"
  _stop_capture "$workload_pid"

  count=0
  {
    printf '[\n'
    local first=true
    while IFS= read -r line; do
      [[ "$line" == \{* ]] || continue
      $first && first=false || printf ',\n'
      printf '%s' "$line"
      (( count++ )) || true
    done < "$daemon_tmp"
    printf '\n]\n'
  } > "$trace_file"
  rm -f "$daemon_tmp"

  if [[ "$count" -eq 0 ]]; then
    echo "  WARNING: No daemon records collected for '${label}'." >&2
    return 1
  fi

  echo "  Collected ${count} daemon records -> ${trace_file}"
  echo "  Collected workload logs           -> ${workload_log}"

  if $RUN_EVAL; then
    # shellcheck disable=SC2086
    python3 evaluate.py --json "$trace_file" --out "$(dirname "$trace_file")" $eval_extra
  else
    echo "  (--no-eval)  python3 evaluate.py --json ${trace_file} ${eval_extra}"
  fi
}

_summarize_workload_overhead() {
  local kind="$1" baseline_log="$2" full_log="$3" adaptive_log="$4" full_eval_dir="$5" adaptive_eval_dir="$6" replay_eval_dir="$7" replay_urgency_eval_dir="$8" out_dir="$9"
  if ! $RUN_EVAL; then
    return 0
  fi
  python3 summarize_workload_overhead.py \
    --kind "$kind" \
    --baseline "$baseline_log" \
    --full "$full_log" \
    --adaptive "$adaptive_log" \
    --full-eval-dir "$full_eval_dir" \
    --adaptive-eval-dir "$adaptive_eval_dir" \
    --adaptive-replay-eval-dir "$replay_eval_dir" \
    --replay-urgency-eval-dir "$replay_urgency_eval_dir" \
    --out "$out_dir"
}

_run_three_arm_workload() {
  local kind="$1" label="$2" workload_selector="$3" out_dir="$4" warmup_msg="$5" deploy_fn="$6" teardown_fn="$7"
  local full_eval_args adaptive_eval_args profile_eval_args

  mkdir -p "$out_dir"
  profile_eval_args=$(_daemon_eval_args "$ADAPTIVE_PROFILE")
  full_eval_args="--min-interval ${FULL_MIN_INTERVAL} --max-interval ${FULL_MAX_INTERVAL} ${profile_eval_args}"
  adaptive_eval_args="--min-interval ${ADAPTIVE_MIN_INTERVAL} --max-interval ${ADAPTIVE_MAX_INTERVAL} ${profile_eval_args}"

  echo "  [baseline] scaling daemon down for workload-only measurement ..."
  _daemon_scale 0
  "$deploy_fn"
  echo "  Waiting ${WARMUP}s for ${warmup_msg} ..."
  sleep "$WARMUP"
  _collect_workload_logs "${kind}-baseline" "$workload_selector" "$out_dir/workload_baseline.log"
  echo "  Tearing down ${label} baseline run ..."
  "$teardown_fn"

  echo "  [full] restoring daemon with MIN=MAX=${FULL_MIN_INTERVAL} ..."
  _daemon_scale 1
  _patch_daemon_policy "$FULL_MIN_INTERVAL" "$FULL_MAX_INTERVAL" "$ADAPTIVE_PROFILE"
  "$deploy_fn"
  echo "  Waiting ${WARMUP}s for ${warmup_msg} ..."
  sleep "$WARMUP"
  _collect_trace_and_workload \
    "${kind}-full" \
    "$out_dir/full/trace.json" \
    "$workload_selector" \
    "$out_dir/full/workload.log" \
    "$full_eval_args"
  echo "  Tearing down ${label} full-monitoring run ..."
  "$teardown_fn"

  _run_replay_eval \
    "${kind}-adaptive-replay" \
    "$out_dir/full/trace.json" \
    "$out_dir/adaptive_replay" \
    "$adaptive_eval_args --include-replay-urgency-values"

  echo "  [adaptive] configuring daemon MIN=${ADAPTIVE_MIN_INTERVAL} MAX=${ADAPTIVE_MAX_INTERVAL} ..."
  _daemon_scale 1
  _patch_daemon_policy "$ADAPTIVE_MIN_INTERVAL" "$ADAPTIVE_MAX_INTERVAL" "$ADAPTIVE_PROFILE"
  "$deploy_fn"
  echo "  Waiting ${WARMUP}s for ${warmup_msg} ..."
  sleep "$WARMUP"
  _collect_trace_and_workload \
    "${kind}-adaptive" \
    "$out_dir/adaptive/trace.json" \
    "$workload_selector" \
    "$out_dir/adaptive/workload.log" \
    "$adaptive_eval_args"
  echo "  Tearing down ${label} adaptive run ..."
  "$teardown_fn"

  _summarize_workload_overhead \
    "$kind" \
    "$out_dir/workload_baseline.log" \
    "$out_dir/full/workload.log" \
    "$out_dir/adaptive/workload.log" \
    "$out_dir/full" \
    "$out_dir/adaptive" \
    "" \
    "$out_dir/adaptive_replay" \
    "$out_dir"
}

_run_floor_capture() {
  local out_dir="$1"
  local full_eval_args adaptive_eval_args profile_eval_args

  mkdir -p "$out_dir"
  profile_eval_args=$(_daemon_eval_args "$ADAPTIVE_PROFILE")
  full_eval_args="--min-interval ${FULL_MIN_INTERVAL} --max-interval ${FULL_MAX_INTERVAL} ${profile_eval_args}"
  adaptive_eval_args="--min-interval ${ADAPTIVE_MIN_INTERVAL} --max-interval ${ADAPTIVE_MAX_INTERVAL} ${profile_eval_args}"

  echo "  [full] idle-node daemon floor with MIN=MAX=${FULL_MIN_INTERVAL} ..."
  _daemon_scale 1
  _patch_daemon_policy "$FULL_MIN_INTERVAL" "$FULL_MAX_INTERVAL" "$ADAPTIVE_PROFILE"
  echo "  Waiting ${WARMUP}s for idle daemon to stabilise ..."
  sleep "$WARMUP"
  _collect "floor-full" "$out_dir/full/trace.json" "$full_eval_args"

  echo "  [adaptive] idle-node daemon floor with MIN=${ADAPTIVE_MIN_INTERVAL} MAX=${ADAPTIVE_MAX_INTERVAL} ..."
  _daemon_scale 1
  _patch_daemon_policy "$ADAPTIVE_MIN_INTERVAL" "$ADAPTIVE_MAX_INTERVAL" "$ADAPTIVE_PROFILE"
  echo "  Waiting ${WARMUP}s for adaptive daemon to re-initialise ..."
  sleep "$WARMUP"
  _collect "floor-adaptive" "$out_dir/adaptive/trace.json" "$adaptive_eval_args"
}

_run_latency_benchmark() {
  local out_dir="$1"
  local full_eval_args adaptive_eval_args profile_eval_args latency_phase_target

  mkdir -p "$out_dir"
  profile_eval_args=$(_daemon_eval_args "$ADAPTIVE_PROFILE")
  full_eval_args="--min-interval ${FULL_MIN_INTERVAL} --max-interval ${FULL_MAX_INTERVAL} ${profile_eval_args}"
  adaptive_eval_args="--min-interval ${ADAPTIVE_MIN_INTERVAL} --max-interval ${ADAPTIVE_MAX_INTERVAL} ${profile_eval_args}"
  latency_phase_target=$(_latency_target_phase_samples)

  echo "  [baseline] scaling daemon down for workload-only latency measurement ..."
  if ! _daemon_scale 0; then
    _record_attempt_failure "$out_dir" "baseline: failed to scale daemon down"
    return 1
  fi
  if ! _deploy_latency; then
    _record_attempt_failure "$out_dir" "baseline: failed to deploy fio-latency workload"
    _teardown_latency || true
    return 1
  fi
  echo "  Waiting ${WARMUP}s from deterministic fio cycle start ..."
  sleep "$WARMUP"
  if ! _collect_workload_logs "latency-baseline" "app=fio-latency" "$out_dir/workload_baseline.log" "$latency_phase_target"; then
    _record_attempt_failure "$out_dir" "baseline: failed while collecting fio baseline logs"
    _teardown_latency || true
    return 1
  fi
  echo "  Tearing down fio-latency baseline run ..."
  _teardown_latency

  echo ""
  echo "  -- Arm b: dense-reference full monitoring (MIN=MAX=1) --"
  if ! _daemon_scale 1; then
    _record_attempt_failure "$out_dir" "full: failed to scale daemon up"
    return 1
  fi
  if ! _patch_daemon_policy "$FULL_MIN_INTERVAL" "$FULL_MAX_INTERVAL" "$ADAPTIVE_PROFILE"; then
    _record_attempt_failure "$out_dir" "full: failed to patch daemon policy"
    return 1
  fi
  if ! _deploy_latency; then
    _record_attempt_failure "$out_dir" "full: failed to deploy fio-latency workload"
    _teardown_latency || true
    return 1
  fi
  echo "  Waiting ${WARMUP}s from deterministic fio cycle start ..."
  sleep "$WARMUP"
  if ! _collect_trace_and_workload \
    "latency-full" \
    "$out_dir/full/trace.json" \
    "app=fio-latency" \
    "$out_dir/full/workload.log" \
    "$full_eval_args" \
    "$latency_phase_target"; then
    _record_attempt_failure "$out_dir" "full: failed while collecting daemon or workload logs"
    _teardown_latency || true
    return 1
  fi
  echo "  Tearing down fio-latency full-monitoring run ..."
  _teardown_latency

  if ! _run_replay_eval \
    "latency-adaptive-replay" \
    "$out_dir/full/trace.json" \
    "$out_dir/adaptive_replay" \
    "$adaptive_eval_args --include-replay-urgency-values"; then
    _record_attempt_failure "$out_dir" "adaptive-replay: failed to evaluate replay from full trace"
    return 1
  fi

  echo ""
  echo "  -- Arm c: OmniFlow adaptive (MIN=${ADAPTIVE_MIN_INTERVAL}, MAX=${ADAPTIVE_MAX_INTERVAL}) --"
  if ! _daemon_scale 1; then
    _record_attempt_failure "$out_dir" "adaptive: failed to scale daemon up"
    return 1
  fi
  if ! _patch_daemon_policy "$ADAPTIVE_MIN_INTERVAL" "$ADAPTIVE_MAX_INTERVAL" "$ADAPTIVE_PROFILE"; then
    _record_attempt_failure "$out_dir" "adaptive: failed to patch daemon policy"
    return 1
  fi
  if ! _deploy_latency; then
    _record_attempt_failure "$out_dir" "adaptive: failed to deploy fio-latency workload"
    _teardown_latency || true
    return 1
  fi
  echo "  Waiting ${WARMUP}s from deterministic fio cycle start ..."
  sleep "$WARMUP"
  if ! _collect_trace_and_workload \
    "latency-adaptive" \
    "$out_dir/adaptive/trace.json" \
    "app=fio-latency" \
    "$out_dir/adaptive/workload.log" \
    "$adaptive_eval_args" \
    "$latency_phase_target"; then
    _record_attempt_failure "$out_dir" "adaptive: failed while collecting daemon or workload logs"
    _teardown_latency || true
    return 1
  fi

  echo ""
  echo "  Tearing down fio-latency workload ..."
  _teardown_latency

  if ! _summarize_workload_overhead \
    latency \
    "$out_dir/workload_baseline.log" \
    "$out_dir/full/workload.log" \
    "$out_dir/adaptive/workload.log" \
    "$out_dir/full" \
    "$out_dir/adaptive" \
    "$out_dir/adaptive_replay" \
    "$out_dir/adaptive_replay" \
    "$out_dir"; then
    _record_attempt_failure "$out_dir" "summary: failed to build latency workload summary"
    return 1
  fi
}

_deploy_latency() {
  echo "  Deploying fio-latency workload on '${TARGET_NODE}' ..."
  sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/fio_latency.yaml | _kubectl apply -f -
  _kubectl rollout status deployment/fio-latency -n "$NS" --timeout=240s
  _wait_latest_pod_ready 'app=fio-latency' 120 >/dev/null
}

_teardown_latency() {
  sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/fio_latency.yaml \
    | _kubectl delete -f - --ignore-not-found --timeout=90s
  echo "  fio-latency workload removed."
}

_deploy_postgres() {
  sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/postgres.yaml | _kubectl apply -f -
  _kubectl rollout status statefulset/postgres -n "$NS" --timeout=180s
}

_teardown_postgres() {
  sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/postgres.yaml \
    | _kubectl delete -f - --ignore-not-found --timeout=90s
}

_deploy_redis() {
  sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/redis.yaml | _kubectl apply -f -
  _kubectl rollout status statefulset/redis -n "$NS" --timeout=120s
}

_teardown_redis() {
  sed "s/NODE_PLACEHOLDER/${TARGET_NODE}/g" manifests/redis.yaml \
    | _kubectl delete -f - --ignore-not-found --timeout=90s
}

# ── Phase 5: PostgreSQL workload ─────────────────────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "postgres" ]]; then
  echo ""
  echo "--- Phase 5: PostgreSQL workload (warmup ${WARMUP}s + collect ${DURATION}s) ---"
  pg_summaries=()
  pg_full_dirs=()
  pg_adaptive_dirs=()
  pg_replay_dirs=()
  repeat=0
  attempt=0
  max_attempts=$((REPEATS + 5))
  while (( repeat < REPEATS )); do
    attempt=$((attempt + 1))
    if (( attempt > max_attempts )); then
      echo "ERROR: exceeded ${max_attempts} postgres attempts while trying to collect ${REPEATS} healthy repeats" >&2
      exit 1
    fi
    run_dir=$(_repeat_dir "$OUT_DIR/postgres" "$attempt")
    if (( REPEATS > 1 )); then
      echo "  [attempt ${attempt}; accepted ${repeat}/${REPEATS}] output -> ${run_dir}"
    fi
    if $WITH_OVERHEAD; then
      _run_three_arm_workload \
        postgres \
        PostgreSQL \
        "job-name=pgbench" \
        "$run_dir" \
        "pgbench to initialise" \
        _deploy_postgres \
        _teardown_postgres
      if $RUN_EVAL; then
        pg_summaries+=("$run_dir/workload_overhead_summary.json")
        if _workload_summary_is_bad "$run_dir/workload_overhead_summary.json"; then
          echo "  PostgreSQL attempt ${attempt} marked unhealthy; collecting one more repeat."
          continue
        fi
        pg_full_dirs+=("$run_dir/full")
        pg_adaptive_dirs+=("$run_dir/adaptive")
        pg_replay_dirs+=("$run_dir/adaptive_replay")
      fi
    else
      _deploy_postgres
      echo "  Waiting ${WARMUP}s for pgbench to initialise ..."
      sleep "$WARMUP"
      _collect "postgres" "$run_dir/trace.json"
      echo "  Tearing down PostgreSQL ..."
      _teardown_postgres
    fi
    repeat=$((repeat + 1))
  done
  if $WITH_OVERHEAD && $RUN_EVAL && (( REPEATS > 1 )); then
    _aggregate_workload_summaries postgres "$OUT_DIR/postgres" "${pg_summaries[@]}"
  fi
  if $WITH_OVERHEAD && $RUN_EVAL; then
    python3 aggregate_io_results.py eval \
      --label postgres \
      --out "$OUT_DIR/postgres" \
      --full "${pg_full_dirs[@]}" \
      --adaptive "${pg_adaptive_dirs[@]}" \
      --adaptive-replay "${pg_replay_dirs[@]}"
  fi
  echo "  PostgreSQL removed."
fi

# ── Phase 6: Redis workload ──────────────────────────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "redis" ]]; then
  echo ""
  echo "--- Phase 6: Redis workload (warmup ${WARMUP}s + collect ${DURATION}s) ---"
  redis_summaries=()
  redis_full_dirs=()
  redis_adaptive_dirs=()
  redis_replay_dirs=()
  repeat=0
  attempt=0
  max_attempts=$((REPEATS + 5))
  while (( repeat < REPEATS )); do
    attempt=$((attempt + 1))
    if (( attempt > max_attempts )); then
      echo "ERROR: exceeded ${max_attempts} redis attempts while trying to collect ${REPEATS} healthy repeats" >&2
      exit 1
    fi
    run_dir=$(_repeat_dir "$OUT_DIR/redis" "$attempt")
    if (( REPEATS > 1 )); then
      echo "  [attempt ${attempt}; accepted ${repeat}/${REPEATS}] output -> ${run_dir}"
    fi
    if $WITH_OVERHEAD; then
      _run_three_arm_workload \
        redis \
        Redis \
        "app=redis-benchmark" \
        "$run_dir" \
        "redis-benchmark to stabilise" \
        _deploy_redis \
        _teardown_redis
      if $RUN_EVAL; then
        redis_summaries+=("$run_dir/workload_overhead_summary.json")
        if _workload_summary_is_bad "$run_dir/workload_overhead_summary.json"; then
          echo "  Redis attempt ${attempt} marked unhealthy; collecting one more repeat."
          continue
        fi
        redis_full_dirs+=("$run_dir/full")
        redis_adaptive_dirs+=("$run_dir/adaptive")
        redis_replay_dirs+=("$run_dir/adaptive_replay")
      fi
    else
      _deploy_redis
      echo "  Waiting ${WARMUP}s for redis-benchmark to stabilise ..."
      sleep "$WARMUP"
      _collect "redis" "$run_dir/trace.json"
      echo "  Tearing down Redis ..."
      _teardown_redis
    fi
    repeat=$((repeat + 1))
  done
  if $WITH_OVERHEAD && $RUN_EVAL && (( REPEATS > 1 )); then
    _aggregate_workload_summaries redis "$OUT_DIR/redis" "${redis_summaries[@]}"
  fi
  if $WITH_OVERHEAD && $RUN_EVAL; then
    python3 aggregate_io_results.py eval \
      --label redis \
      --out "$OUT_DIR/redis" \
      --full "${redis_full_dirs[@]}" \
      --adaptive "${redis_adaptive_dirs[@]}" \
      --adaptive-replay "${redis_replay_dirs[@]}"
  fi
  echo "  Redis removed."
fi

# ── Phase 7: Daemon floor profiling ──────────────────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "floor" ]]; then
  echo ""
  echo "--- Phase 7: Idle-node daemon floor profiling ---"
  floor_full_dirs=()
  floor_adaptive_dirs=()
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    run_dir=$(_repeat_dir "$OUT_DIR/floor" "$repeat")
    if (( REPEATS > 1 )); then
      echo "  [repeat ${repeat}/${REPEATS}] output -> ${run_dir}"
    fi
    _run_floor_capture "$run_dir"
    if $RUN_EVAL; then
      floor_full_dirs+=("$run_dir/full")
      floor_adaptive_dirs+=("$run_dir/adaptive")
    fi
  done
  if $RUN_EVAL; then
    python3 aggregate_io_results.py eval \
      --label floor \
      --out "$OUT_DIR/floor" \
      --full "${floor_full_dirs[@]}" \
      --adaptive "${floor_adaptive_dirs[@]}"
  fi
fi

# ── Phase 8: Latency p99 detection benchmark ─────────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "latency" ]]; then
  echo ""
  echo "--- Phase 8: Latency p99 detection benchmark (three arms) ---"
  echo "  Arm a (daemon off) – workload-only latency baseline"
  echo "  Arm b (1/1)        – dense-reference full monitoring"
  echo "  Arm c (${ADAPTIVE_MIN_INTERVAL}/${ADAPTIVE_MAX_INTERVAL}) – OmniFlow adaptive"
  latency_full_dirs=()
  latency_adaptive_dirs=()
  latency_replay_dirs=()
  latency_summaries=()
  repeat=0
  attempt=0
  max_attempts=$((REPEATS + 5))
  while (( repeat < REPEATS )); do
    attempt=$((attempt + 1))
    if (( attempt > max_attempts )); then
      echo "ERROR: exceeded ${max_attempts} latency attempts while trying to collect ${REPEATS} healthy repeats" >&2
      exit 1
    fi
    run_dir=$(_repeat_dir "$OUT_DIR/latency" "$attempt")
    if (( REPEATS > 1 )); then
      echo "  [attempt ${attempt}; accepted ${repeat}/${REPEATS}] output -> ${run_dir}"
    fi
    if ! _run_latency_benchmark "$run_dir"; then
      echo "  Latency attempt ${attempt} failed before producing a usable summary; collecting one more repeat."
      continue
    fi
    if $RUN_EVAL; then
      latency_full_dirs+=("$run_dir/full")
      latency_adaptive_dirs+=("$run_dir/adaptive")
      latency_replay_dirs+=("$run_dir/adaptive_replay")
      latency_summaries+=("$run_dir/workload_overhead_summary.json")
      if _workload_summary_is_bad "$run_dir/workload_overhead_summary.json"; then
        echo "  Latency attempt ${attempt} marked unhealthy; collecting one more repeat."
        continue
      fi
    fi
    repeat=$((repeat + 1))
  done
  if $RUN_EVAL; then
    python3 aggregate_io_results.py eval \
      --label latency \
      --out "$OUT_DIR/latency" \
      --full "${latency_full_dirs[@]}" \
      --adaptive "${latency_adaptive_dirs[@]}" \
      --adaptive-replay "${latency_replay_dirs[@]}"
    if (( REPEATS > 1 )); then
      _aggregate_workload_summaries latency "$OUT_DIR/latency" "${latency_summaries[@]}"
    fi
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "Final step: compressing large logs and summarizing overhead results ..."
# workload.log, workload_baseline.log
find "$OUT_DIR" -type f \( -name 'workload.log' -o -name 'workload_baseline.log' \) \
  -exec gzip -f {} \;
  
echo ""
echo "=== Done ==="
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "postgres" ]]; then
  if $WITH_OVERHEAD; then
    if (( REPEATS > 1 )); then
      echo "  postgres repeats  : $OUT_DIR/postgres/"
      $RUN_EVAL && echo "  postgres aggregate: $OUT_DIR/postgres/aggregate_workload_overhead_summary.json"
    else
      echo "  postgres baseline : $OUT_DIR/postgres/workload_baseline.log.gz"
      echo "  postgres full     : $OUT_DIR/postgres/full/trace.json"
      echo "  postgres adaptive : $OUT_DIR/postgres/adaptive/trace.json"
    fi
  else
    if (( REPEATS > 1 )); then
      echo "  postgres repeats  : $OUT_DIR/postgres/"
    else
      echo "  postgres          : $OUT_DIR/postgres/trace.json"
    fi
  fi
fi
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "redis" ]]; then
  if $WITH_OVERHEAD; then
    if (( REPEATS > 1 )); then
      echo "  redis repeats     : $OUT_DIR/redis/"
      $RUN_EVAL && echo "  redis aggregate   : $OUT_DIR/redis/aggregate_workload_overhead_summary.json"
    else
      echo "  redis baseline    : $OUT_DIR/redis/workload_baseline.log.gz"
      echo "  redis full        : $OUT_DIR/redis/full/trace.json"
      echo "  redis adaptive    : $OUT_DIR/redis/adaptive/trace.json"
    fi
  else
    if (( REPEATS > 1 )); then
      echo "  redis repeats     : $OUT_DIR/redis/"
    else
      echo "  redis             : $OUT_DIR/redis/trace.json"
    fi
  fi
fi
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "floor" ]]; then
  if (( REPEATS > 1 )); then
    echo "  floor repeats     : $OUT_DIR/floor/"
  else
    echo "  floor full        : $OUT_DIR/floor/full/trace.json"
    echo "  floor adaptive    : $OUT_DIR/floor/adaptive/trace.json"
  fi
  $RUN_EVAL && echo "  floor summary     : $OUT_DIR/floor/aggregate_eval_summary.json"
fi
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "latency" ]]; then
  if (( REPEATS > 1 )); then
    echo "  latency repeats   : $OUT_DIR/latency/"
  else
    echo "  latency baseline  : $OUT_DIR/latency/workload_baseline.log.gz"
    echo "  latency full      : $OUT_DIR/latency/full/trace.json"
    echo "  latency replay    : $OUT_DIR/latency/adaptive_replay/"
    echo "  latency adaptive  : $OUT_DIR/latency/adaptive/trace.json"
  fi
  $RUN_EVAL && echo "  latency summary   : $OUT_DIR/latency/aggregate_eval_summary.json"
  $RUN_EVAL && (( REPEATS > 1 )) && echo "  latency workload  : $OUT_DIR/latency/aggregate_workload_overhead_summary.json"
fi
echo ""
echo "  Daemon still running on '${TARGET_NODE}'.  Teardown: ./teardown.sh"
