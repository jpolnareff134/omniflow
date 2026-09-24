#!/usr/bin/env bash
# run-experiment.sh - HPA autoscaling experiment orchestration.
#
# All operations are via kubectl (except image builds).
# Assumes KUBECONFIG is set and the cluster is ready.
#
# Phases:
#   1. Build & load container images
#   2. Deploy baseline infrastructure (namespace, workload, prometheus, adapter)
#   3. Run Metrics Server with 15s HPA sync / 60s metric resolution
#   4. Run Metrics Server with 15s HPA sync / 15s metric resolution
#   5. Run native OmniFlow with 15s Prometheus scrape / 15s HPA sync
#   5. Collect results
#
# Usage:
#   ./run-experiment.sh [--kubeconfig PATH] [--duration SECS] [--profile NAME]
#                       [--shape NAME] [--repeats N] [--arm NAME]
#                       --host-cpus LIST
#                       [--omniflow-min-interval N] [--omniflow-max-interval N]
#
# Defaults:
#   kubeconfig=./kubeconfig  duration=<shape duration + buffer>
#   profile=omniflow  shape=phased  repeats=5
# Output is written to out/{RUN_ID}/ automatically.

set -Eeuo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
SCRIPT_DIR="$(pwd)"

KUBECONFIG_PATH="${KUBECONFIG:-./kubeconfig}"
DURATION=""      # set per-shape below unless overridden by --duration
NS="omniflow-hpa-default"
PROFILE="omniflow"
SHAPE="phased"
REPEATS=5
METRICS_SERVER_RESOLUTION="60s"
SETTLE_SECONDS=75
HPA_STABLE_SECONDS=30
ARM_LEAD_SECONDS=45
POST_LOAD_SECONDS=90
OBSERVE_INTERVAL=1
RESUME=false
REQUESTED_RUN_DIR=""
SMOKE_TEST=false
REQUESTED_ARM=""
OMNIFLOW_MIN_INTERVAL=15
OMNIFLOW_MAX_INTERVAL=30
SKIP_INFRA=false
KEEP_RESOURCES=false
MIN_FREE_GB=20
TIMELINE_PID=""
LOCUST_LOG_PID=""
DISK_MONITOR_PID=""
RUNNER_PID="$$"
CURRENT_REP_DIR=""
CURRENT_LABEL=""
CURRENT_HPA_NAME=""
CURRENT_STEP="initializing"
DAEMON_LOG_SINCE=""  # retained in metadata for backwards compatibility
DAEMON_CONTROL_PORT=9101
DAEMON_TAR=""
LOCUST_TAR=""
HOST_CPUS=""
HOST_CPU_THRESHOLD="0.05"
HOST_CPU_INTERVAL="1"
HOST_CPU_CONSECUTIVE=3
HOST_CPU_PRE_SECONDS=15
HOST_CPU_POST_SECONDS=10
HOST_QEMU_PROCESSES=2
PROMETHEUS_SCRAPE_SECONDS=15
HOST_GUARD_PID=""
HOST_GUARD_STOP_FILE=""
HOST_CONTAMINATION_FILE=""

usage() {
  cat <<'USAGE'
Usage: ./run-experiment.sh [options]

Run one HPA shape experiment, or use --smoke-test for the offline harness
tests. The full experiment requires --host-cpus LIST and a configured cluster.

Options include --kubeconfig PATH, --shape NAME, --repeats N, --run-dir DIR,
--arm NAME, --host-cpus LIST, --omniflow-min-interval N,
--omniflow-max-interval N, --resume, --skip-infra, and --smoke-test.
USAGE
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --kubeconfig) KUBECONFIG_PATH="$2"; shift 2 ;;
    --duration)   DURATION="$2";        shift 2 ;;
    --profile)    PROFILE="$2";         shift 2 ;;
    --shape)      SHAPE="$2";           shift 2 ;;
    --repeats)    REPEATS="$2";         shift 2 ;;
    --metrics-server-resolution) METRICS_SERVER_RESOLUTION="$2"; shift 2 ;;
    --settle-seconds) SETTLE_SECONDS="$2"; shift 2 ;;
    --hpa-stable-seconds) HPA_STABLE_SECONDS="$2"; shift 2 ;;
    --arm-lead-seconds) ARM_LEAD_SECONDS="$2"; shift 2 ;;
    --post-load-seconds) POST_LOAD_SECONDS="$2"; shift 2 ;;
    --observe-interval) OBSERVE_INTERVAL="$2"; shift 2 ;;
    --run-dir) REQUESTED_RUN_DIR="$2"; shift 2 ;;
    --arm) REQUESTED_ARM="$2"; shift 2 ;;
    --omniflow-min-interval) OMNIFLOW_MIN_INTERVAL="$2"; shift 2 ;;
    --omniflow-max-interval) OMNIFLOW_MAX_INTERVAL="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --smoke-test) SMOKE_TEST=true; shift ;;
    --namespace) NS="$2"; shift 2 ;;
    --skip-infra) SKIP_INFRA=true; shift ;;
    --keep-resources) KEEP_RESOURCES=true; shift ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --host-cpus) HOST_CPUS="$2"; shift 2 ;;
    --host-cpu-threshold) HOST_CPU_THRESHOLD="$2"; shift 2 ;;
    --host-cpu-interval) HOST_CPU_INTERVAL="$2"; shift 2 ;;
    --host-cpu-consecutive) HOST_CPU_CONSECUTIVE="$2"; shift 2 ;;
    --host-cpu-pre-seconds) HOST_CPU_PRE_SECONDS="$2"; shift 2 ;;
    --host-cpu-post-seconds) HOST_CPU_POST_SECONDS="$2"; shift 2 ;;
    --host-qemu-processes) HOST_QEMU_PROCESSES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -n "$REQUESTED_ARM" && ! "$REQUESTED_ARM" =~ ^(metrics_60s|metrics_15s|omniflow)$ ]]; then
  echo "--arm must be metrics_60s, metrics_15s, or omniflow" >&2
  exit 1
fi

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repeats must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  echo "--min-free-gb must be a non-negative integer" >&2
  exit 1
fi
if [[ "$SMOKE_TEST" != true && -z "$HOST_CPUS" ]]; then
  echo "--host-cpus is required; run the harness under taskset on disjoint CPUs" >&2
  exit 1
fi
if ! [[ "$HOST_CPU_CONSECUTIVE" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$HOST_QEMU_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Host CPU consecutive count and QEMU process count must be positive integers" >&2
  exit 1
fi
python3 - "$HOST_CPU_THRESHOLD" "$HOST_CPU_INTERVAL" \
  "$HOST_CPU_PRE_SECONDS" "$HOST_CPU_POST_SECONDS" <<'PYEOF'
import sys
values = [float(value) for value in sys.argv[1:]]
if not (0 <= values[0] <= 1) or any(value <= 0 for value in values[1:]):
    raise SystemExit("Host CPU threshold must be in [0,1] and intervals/durations positive")
PYEOF
if ! [[ "$OMNIFLOW_MIN_INTERVAL" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$OMNIFLOW_MAX_INTERVAL" =~ ^[1-9][0-9]*$ ]] ||
   (( OMNIFLOW_MAX_INTERVAL < OMNIFLOW_MIN_INTERVAL )); then
  echo "OmniFlow intervals must be positive integers with max >= min" >&2
  exit 1
fi

# Default duration = locust total + 60s buffer, unless --duration was passed
if [[ -z "$DURATION" ]]; then
  DURATION=$(PYTHONPATH="$SCRIPT_DIR" python3 -c \
    'from load_schedule import total_duration; import sys; print(total_duration(sys.argv[1]) + 70)' "$SHAPE")
fi

if [[ "$SMOKE_TEST" == true ]]; then
  PYTHONPATH="$SCRIPT_DIR" python3 -m unittest discover -s "$SCRIPT_DIR/tests" -v
  exit 0
fi

export KUBECONFIG="$KUBECONFIG_PATH"

# -- Render namespace-scoped manifests ----------------------------
# All shape-local resources are parameterised by ${OMNIFLOW_NAMESPACE} so that
# multiple shapes can run concurrently in isolated namespaces.
MANIFEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/omniflow-manifests.XXXXXX")"
export OMNIFLOW_NAMESPACE="$NS"

_substitute_namespace() {
  python3 - "$1" "$2" "$NS" <<'PYEOF'
import sys
src, dst, ns = sys.argv[1:]
text = open(src).read()
open(dst, "w").write(text.replace("${OMNIFLOW_NAMESPACE}", ns))
PYEOF
}

_render_manifests() {
  local src
  for src in manifests/namespace.yaml manifests/workload.yaml \
             manifests/hpa-baseline.yaml manifests/hpa-fixed-15s.yaml \
             manifests/hpa-omniflow.yaml manifests/locust-job.yaml; do
    local dst
    dst="${MANIFEST_DIR}/$(basename "$src")"
    _substitute_namespace "$src" "$dst"
  done

  python3 - "manifests/daemonset.yaml" "${MANIFEST_DIR}/daemonset.yaml" \
    "$OMNIFLOW_MIN_INTERVAL" "$OMNIFLOW_MAX_INTERVAL" <<'PYEOF'
import sys

src, dst, minimum, maximum = sys.argv[1:]
values = {
    "OMNIFLOW_MIN_INTERVAL": minimum,
    "OMNIFLOW_MAX_INTERVAL": maximum,
}
current = None
changed = set()
output = []
for line in open(src):
    stripped = line.strip()
    if stripped.startswith("- name:"):
        current = stripped.split(":", 1)[1].strip()
    elif current in values and stripped.startswith("value:"):
        indent = line[:len(line) - len(line.lstrip())]
        line = f'{indent}value: "{values[current]}"\n'
        changed.add(current)
        current = None
    output.append(line)
if changed != values.keys():
    raise SystemExit(f"Failed to render daemon intervals: changed={sorted(changed)}")
open(dst, "w").writelines(output)
PYEOF
}
_render_manifests

# Ensure the shared infrastructure namespace exists; it is deployed below unless
# the caller is handling it centrally (e.g. run-all-shapes.sh).
if [[ "$SKIP_INFRA" != true ]]; then
  kubectl apply -f manifests/namespace-infra.yaml
fi

cleanup_manifests() {
  rm -rf "$MANIFEST_DIR"
}

# -- Run directory (all output goes here) -------------------------
if [[ -n "$REQUESTED_RUN_DIR" ]]; then
  RUN_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$REQUESTED_RUN_DIR")"
  RUN_ID="$(basename "$RUN_DIR")"
else
  RUN_DIR="$(python3 "$REPO_ROOT/src/support/log.py" --base "${SCRIPT_DIR}/out" --application-type "k8s")"
  RUN_ID="$(basename "$RUN_DIR")"
fi
OUT_DIR="${RUN_DIR}/results"
mkdir -p "$OUT_DIR"
RUN_LOCK_FILE="$RUN_DIR/.run-experiment.lock"
exec 201>"$RUN_LOCK_FILE"
if ! flock -n 201; then
  echo "Another run-experiment.sh process currently holds $RUN_LOCK_FILE" >&2
  exit 1
fi
echo "$$" >&201

echo "=== HPA Autoscaling Experiment ==="
echo "  kubeconfig=$KUBECONFIG_PATH"
echo "  duration=${DURATION}s"
echo "  repeats=$REPEATS"
echo "  metrics_server_resolution=$METRICS_SERVER_RESOLUTION"
echo "  settle_seconds=$SETTLE_SECONDS"
echo "  hpa_stable_seconds=$HPA_STABLE_SECONDS"
echo "  arm_lead_seconds=$ARM_LEAD_SECONDS"
echo "  post_load_seconds=$POST_LOAD_SECONDS"
echo "  observe_interval=${OBSERVE_INTERVAL}s"
echo "  omniflow_intervals=${OMNIFLOW_MIN_INTERVAL}-${OMNIFLOW_MAX_INTERVAL}s"
echo "  host_cpus=$HOST_CPUS"
echo "  host_cpu_foreign_threshold=$HOST_CPU_THRESHOLD"
echo "  run_dir=$RUN_DIR"

# -- Save experiment parameters ------------------------------------
if [[ "$RESUME" != true || ! -f "$RUN_DIR/params.json" ]]; then
python3 - <<PYEOF
import json
# Read container env vars from the daemonset manifest (they are never
# present in the host shell environment).
env = {}
try:
    import yaml
    with open("${MANIFEST_DIR}/daemonset.yaml") as _f:
        docs = list(yaml.safe_load_all(_f))
    ds = next((d for d in docs if d and d.get("kind") == "DaemonSet"), None)
    if ds:
        env_list = (ds["spec"]["template"]["spec"]["containers"][0]
                    .get("env", []))
        env = {e["name"]: e["value"] for e in env_list if "value" in e}
except Exception as _exc:
    env = {"_error": str(_exc)}
p = dict(
    run_id="${RUN_ID}",
    kubeconfig="${KUBECONFIG_PATH}",
    duration=${DURATION},
    repeats=${REPEATS},
    metrics_server_resolution="${METRICS_SERVER_RESOLUTION}",
    settle_seconds=${SETTLE_SECONDS},
    hpa_stable_seconds=${HPA_STABLE_SECONDS},
    arm_lead_seconds=${ARM_LEAD_SECONDS},
    post_load_seconds=${POST_LOAD_SECONDS},
    observe_interval=${OBSERVE_INTERVAL},
    namespace="${NS}",
    shape="${SHAPE}",
    execution_mode="isolated_arm_major",
    host_cpus="${HOST_CPUS}",
    host_cpu_threshold=float("${HOST_CPU_THRESHOLD}"),
    host_cpu_interval=float("${HOST_CPU_INTERVAL}"),
    host_cpu_consecutive=${HOST_CPU_CONSECUTIVE},
    host_cpu_pre_seconds=float("${HOST_CPU_PRE_SECONDS}"),
    host_cpu_post_seconds=float("${HOST_CPU_POST_SECONDS}"),
    host_qemu_processes=${HOST_QEMU_PROCESSES},
    env=env,
)
open("${RUN_DIR}/params.json", "w").write(json.dumps(p, indent=2) + "\n")
PYEOF
fi

if [[ "$RESUME" == true ]]; then
  python3 - "$RUN_DIR/params.json" "$SHAPE" "$NS" "$REPEATS" "$DURATION" \
    "$SETTLE_SECONDS" "$HPA_STABLE_SECONDS" "$ARM_LEAD_SECONDS" \
    "$POST_LOAD_SECONDS" "$OBSERVE_INTERVAL" "$HOST_CPUS" \
    "$HOST_CPU_THRESHOLD" "$HOST_CPU_INTERVAL" "$HOST_CPU_CONSECUTIVE" \
    "$HOST_CPU_PRE_SECONDS" "$HOST_CPU_POST_SECONDS" "$HOST_QEMU_PROCESSES" \
    "${MANIFEST_DIR}/daemonset.yaml" <<'PYEOF'
import json
import sys

path = sys.argv[1]
actual = json.load(open(path))
keys = (
    "shape", "namespace", "repeats", "duration", "settle_seconds",
    "hpa_stable_seconds", "arm_lead_seconds", "post_load_seconds",
    "observe_interval", "host_cpus", "host_cpu_threshold",
    "host_cpu_interval", "host_cpu_consecutive", "host_cpu_pre_seconds",
    "host_cpu_post_seconds", "host_qemu_processes",
)
expected = dict(zip(keys, [
    sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]),
    float(sys.argv[10]), sys.argv[11], float(sys.argv[12]),
    float(sys.argv[13]), int(sys.argv[14]), float(sys.argv[15]),
    float(sys.argv[16]), int(sys.argv[17]),
]))
mismatches = {
    key: (actual.get(key), value)
    for key, value in expected.items()
    if actual.get(key) != value
}
if mismatches:
    print(f"Refusing incompatible resume for {path}: {mismatches}", file=sys.stderr)
    raise SystemExit(1)
try:
    import yaml
    docs = list(yaml.safe_load_all(open(sys.argv[18])))
    daemonset = next(doc for doc in docs if doc and doc.get("kind") == "DaemonSet")
    env = daemonset["spec"]["template"]["spec"]["containers"][0].get("env", [])
    expected_env = {entry["name"]: entry["value"] for entry in env if "value" in entry}
except Exception as exc:
    print(f"Cannot validate daemon configuration for resume: {exc}", file=sys.stderr)
    raise SystemExit(1)
if actual.get("env") != expected_env:
    print("Refusing resume because daemon environment differs from params.json", file=sys.stderr)
    raise SystemExit(1)
PYEOF
fi

# The harness and all children must inherit a housekeeping affinity that does
# not overlap the CPUs reserved for QEMU. Pinning QEMU requires root, while
# monitoring /proc does not.
PYTHONPATH="$SCRIPT_DIR" python3 - "$HOST_CPUS" <<'PYEOF'
import json
import os
import sys
from host_cpu_guard import format_cpu_list, parse_cpu_list

reserved = parse_cpu_list(sys.argv[1])
affinity = set(os.sched_getaffinity(0))
overlap = reserved & affinity
if overlap:
    raise SystemExit(
        f"runner affinity overlaps reserved CPUs {format_cpu_list(overlap)}; "
        "launch run-all-shapes.sh with taskset on housekeeping CPUs"
    )
print(json.dumps({"runner_cpus": format_cpu_list(affinity),
                  "reserved_cpus": format_cpu_list(reserved)}, indent=2))
PYEOF
PINNING_JSON=$(sudo -n PYTHONPATH="$SCRIPT_DIR" python3 \
  "$SCRIPT_DIR/host_cpu_guard.py" pin --profile "$PROFILE" --cpus "$HOST_CPUS" \
  --expected-processes "$HOST_QEMU_PROCESSES")
printf '%s\n' "$PINNING_JSON" > "$RUN_DIR/host-cpu-pinning.json"

# -- Phase 1: Build images ---------------------------------------
echo ""
echo "--- Phase 1: Building container images ---"

docker build -t omniflow-daemon:latest -f Dockerfile.daemon "$REPO_ROOT" # --no-cache
docker build -t omniflow-locust:latest -f Dockerfile.locust "$REPO_ROOT" # --no-cache

# Load images into every minikube node via direct SSH (minikube image load
# only reliably loads to the control-plane node with kvm2 driver).
echo "Loading images into all minikube nodes (profile=$PROFILE) ..."
DAEMON_TAR="/tmp/omniflow-daemon-${NS}.tar"
LOCUST_TAR="/tmp/omniflow-locust-${NS}.tar"
docker save omniflow-daemon:latest > "$DAEMON_TAR"
docker save omniflow-locust:latest > "$LOCUST_TAR"

_load_images_to_nodes() {
  local minikube_home="${MINIKUBE_HOME:-$HOME/.minikube}"
  python3 - <<PYEOF
import json, os, subprocess, sys
base = os.path.expanduser("${minikube_home}/machines")
nodes = [d for d in os.listdir(base) if os.path.isfile(f"{base}/{d}/config.json")]
failed = False
for node in sorted(nodes):
    cfg = json.load(open(f"{base}/{node}/config.json"))
    ip  = cfg["Driver"]["IPAddress"]
    key = cfg["Driver"]["SSHKeyPath"]
    for img, tar in [("omniflow-daemon:latest", "${DAEMON_TAR}"),
                     ("omniflow-locust:latest",  "${LOCUST_TAR}")]:
        print(f"  {node} ({ip}): loading {img}", flush=True)
        with open(tar, "rb") as f:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        "-i", key, f"docker@{ip}", "docker load"],
                stdin=f, capture_output=True)
        if r.returncode != 0:
            failed = True
            print(f"    ERROR: {r.stderr.decode().strip()}", file=sys.stderr)
        else:
            print(f"    OK: {r.stdout.decode().strip()}", flush=True)
if not nodes or failed:
    raise SystemExit(1)
PYEOF
}
_load_images_to_nodes

DAEMON_IMAGE="omniflow-daemon:latest"

delete_workload_namespace() {
  if ! kubectl get ns "$NS" >/dev/null 2>&1; then
    return 0
  fi
  echo "Deleting workload namespace $NS ..."
  kubectl delete ns "$NS" --wait=true --timeout=120s
  while kubectl get ns "$NS" >/dev/null 2>&1; do
    sleep 2
  done
}

create_workload_namespace() {
  kubectl apply -f "${MANIFEST_DIR}/namespace.yaml"
  kubectl apply -f "${MANIFEST_DIR}/workload.yaml"
  kubectl -n "$NS" rollout status deployment/nginx --timeout=120s
  kubectl -n "$NS" wait --for=condition=ready pod -l app=nginx --timeout=120s
}

# -- Phase 2: Deploy base infrastructure -------------------------
echo ""
echo "--- Phase 2: Deploying base infrastructure ---"

delete_workload_namespace
create_workload_namespace

if [[ "$SKIP_INFRA" != true ]]; then
  echo "Deploying shared monitoring infrastructure ..."
  kubectl apply -f manifests/prometheus.yaml
  kubectl apply -f manifests/prometheus-adapter.yaml
fi

kubectl apply -f "${MANIFEST_DIR}/daemonset.yaml"
kubectl -n omniflow-infra set image daemonset/omniflow-daemon daemon="$DAEMON_IMAGE"
# The image tag is intentionally stable and imagePullPolicy is Never. Loading a
# rebuilt image into the nodes does not change the DaemonSet pod template, so an
# explicit restart is required or existing pods continue running stale code.
kubectl -n omniflow-infra rollout restart daemonset/omniflow-daemon

echo "Waiting for pods to be ready ..."
kubectl -n "$NS" wait --for=condition=ready pod -l app=nginx --timeout=120s
kubectl -n omniflow-infra wait --for=condition=ready pod -l app=prometheus --timeout=120s || true
kubectl -n omniflow-infra rollout status daemonset/omniflow-daemon --timeout=120s
kubectl -n omniflow-infra rollout status deployment/prometheus-adapter --timeout=120s || true

# Restart only the shape-local workload. Shared daemon logs are isolated during
# analysis by namespace and each repetition's observed load window.
echo "Restarting workload for clean state ..."
kubectl -n "$NS" rollout restart deployment/nginx
kubectl -n "$NS" wait --for=condition=ready pod -l app=nginx --timeout=120s

# Scale nginx back to 1 to ensure a known starting point
kubectl -n "$NS" scale deployment/nginx --replicas=1
sleep 5
kubectl -n "$NS" wait --for=condition=ready pod -l app=nginx --timeout=60s

# Wait for rollout to fully reconcile and the surviving pod to be Ready
kubectl -n "$NS" rollout status deployment/nginx --timeout=120s

# Preserve the actual cluster cadence. Kubernetes defaults are version and
# deployment dependent; do not infer them later from a manuscript claim.
kubectl -n kube-system get deployment metrics-server -o json > "$RUN_DIR/metrics-server.json" 2>/dev/null || true
kubectl -n kube-system get pods -l component=kube-controller-manager -o json > "$RUN_DIR/controller-manager-pods.json" 2>/dev/null || true
# The previous shape can legitimately leave Metrics Server at either cadence.
# Each resource-metric arm sets and verifies its own resolution below.

# Wait for the raw metrics API to report CPU for the nginx pod.
# This is the same endpoint the HPA queries; kubectl-top can succeed
# while the HPA pipeline still returns empty.
echo "Waiting for metrics API to have CPU data for nginx ..."
METRICS_API_READY=false
for i in $(seq 1 60); do
  if kubectl get --raw "/apis/metrics.k8s.io/v1beta1/namespaces/$NS/pods" 2>/dev/null | \
     python3 -c "
import json, sys
data = json.load(sys.stdin)
pods = [p for p in data.get('items', []) if 'nginx' in p['metadata']['name']]
if pods:
    cpu = pods[0]['containers'][0]['usage'].get('cpu', '0')
    print(f'  nginx pod metrics: cpu={cpu}', flush=True)
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
    echo "  metrics API ready (attempt $i)"
    METRICS_API_READY=true
    break
  fi
  if (( i % 6 == 0 )); then echo "  still waiting (${i}x5s) ..."; fi
  sleep 5
done
if [[ "$METRICS_API_READY" != true ]]; then
  echo "Metrics API did not report nginx CPU after 300 seconds" >&2
  exit 1
fi

echo "Base infrastructure ready."

# -- Helper functions ---------------------------------------------
capture_metric_payload() {
  local MODE="$1"
  local OUTPUT="$2"
  if [[ "$MODE" == "omniflow" ]]; then
    kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/$NS/pods/%2A/omniflow_cpu" > "$OUTPUT"
  else
    kubectl get --raw "/apis/metrics.k8s.io/v1beta1/namespaces/$NS/pods" > "$OUTPUT"
  fi
}

capture_daemon_track_state() {
  local OUTPUT="$1"
  python3 - "$OUTPUT" "$DAEMON_CONTROL_PORT" <<'PYEOF'
import json
import os
import subprocess
import sys
import tempfile

output, port = sys.argv[1:]
pods = json.loads(subprocess.check_output([
    "kubectl", "-n", "omniflow-infra", "get", "pods",
    "-l", "app=omniflow-daemon", "-o", "json",
]))
nodes = []
code = (
    "import sys,urllib.request; "
    f"sys.stdout.buffer.write(urllib.request.urlopen('http://127.0.0.1:{port}/tracks', "
    "timeout=30).read())"
)
for item in pods.get("items", []):
    name = item["metadata"]["name"]
    raw = subprocess.check_output([
        "kubectl", "-n", "omniflow-infra", "exec", f"pod/{name}",
        "--", "python3", "-c", code,
    ])
    payload = json.loads(raw)
    nodes.append({"daemon_pod": name, "tracks": payload.get("tracks", [])})
if not nodes:
    raise SystemExit("No OmniFlow daemon pods found")
fd, temporary = tempfile.mkstemp(prefix=".daemon-state.", dir=os.path.dirname(output))
try:
    with os.fdopen(fd, "w") as stream:
        json.dump({"nodes": nodes}, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PYEOF
}

wait_for_namespace_tracks_gone() {
  local OUTPUT="$1"
  echo "Waiting for daemon state from the previous namespace instance to disappear ..."
  for i in $(seq 1 60); do
    if capture_daemon_track_state "$OUTPUT" && \
       PYTHONPATH="$SCRIPT_DIR" python3 - "$OUTPUT" "$NS" <<'PYEOF'
import json
import sys
from experiment_sync import daemon_sample_versions

payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if not daemon_sample_versions(payload, sys.argv[2]) else 1)
PYEOF
    then
      echo "  old daemon tracks removed (attempt $i)"
      return 0
    fi
    sleep 2
  done
  echo "Daemon retained tracks for deleted namespace $NS" >&2
  return 1
}

capture_daemon_sample_baseline() {
  local PREFIX="$1"
  echo "Capturing OmniFlow scheduler counters for every Ready nginx pod ..."
  for i in $(seq 1 60); do
    if capture_daemon_track_state "${PREFIX}_daemon_samples.json" && \
       PYTHONPATH="$SCRIPT_DIR" python3 - \
         "${PREFIX}_pods.json" "${PREFIX}_daemon_samples.json" "$NS" <<'PYEOF'
import json
import sys
from experiment_sync import daemon_sample_versions, ready_pod_names

pods = ready_pod_names(json.load(open(sys.argv[1])))
versions = daemon_sample_versions(
    json.load(open(sys.argv[2])), sys.argv[3], set(pods)
)
raise SystemExit(0 if pods and set(versions) == set(pods) else 1)
PYEOF
    then
      echo "  OmniFlow scheduler baseline captured (attempt $i)"
      return 0
    fi
    sleep 2
  done
  echo "Could not capture OmniFlow counters for every Ready pod" >&2
  return 1
}

daemon_samples_are_fresh() {
  local PREFIX="$1"
  if capture_daemon_track_state "${PREFIX}_daemon_samples_current.json" && \
     PYTHONPATH="$SCRIPT_DIR" python3 - \
      "${PREFIX}_pods_current.json" "${PREFIX}_daemon_samples.json" \
      "${PREFIX}_daemon_samples_current.json" "$NS" <<'PYEOF'
import json
import sys
from experiment_sync import (
    daemon_sample_versions,
    fresh_daemon_samples,
    ready_pod_names,
)

pods = ready_pod_names(json.load(open(sys.argv[1])))
expected = set(pods)
baseline = daemon_sample_versions(json.load(open(sys.argv[2])), sys.argv[4], expected)
current = daemon_sample_versions(json.load(open(sys.argv[3])), sys.argv[4], expected)
ok, reason = fresh_daemon_samples(pods, baseline, current)
print(reason)
raise SystemExit(0 if ok else 1)
PYEOF
  then
    cp "${PREFIX}_daemon_samples_current.json" "${PREFIX}_daemon_sample_target.json"
    return 0
  fi
  return 1
}

metric_mode_name() {
  [[ "$1" == "omniflow" ]] && echo custom || echo resource
}

capture_metric_baseline() {
  local MODE="$1"
  local PREFIX="$2"
  local METRIC_MODE
  METRIC_MODE=$(metric_mode_name "$MODE")
  echo "Capturing complete metric baseline for current Ready nginx pods ..."
  for i in $(seq 1 60); do
    kubectl -n "$NS" get pods -l app=nginx -o json > "${PREFIX}_pods.json"
    if capture_metric_payload "$MODE" "${PREFIX}_metrics.json" 2>/dev/null && \
       PYTHONPATH="$SCRIPT_DIR" python3 - "${PREFIX}_pods.json" "${PREFIX}_metrics.json" "$METRIC_MODE" <<'PYEOF'
import json, sys
from experiment_sync import metric_versions, ready_pod_names
pods = ready_pod_names(json.load(open(sys.argv[1])))
metrics = metric_versions(json.load(open(sys.argv[2])), sys.argv[3], set(pods))
raise SystemExit(0 if pods and set(pods) == set(metrics) else 1)
PYEOF
    then
      echo "  metric baseline captured (attempt $i)"
      return 0
    fi
    sleep 5
  done
  echo "Could not capture a complete metric baseline after 300 seconds" >&2
  return 1
}

wait_for_fresh_metrics() {
  local MODE="$1"
  local PREFIX="$2"
  local METRIC_MODE
  METRIC_MODE=$(metric_mode_name "$MODE")
  if [[ "$MODE" == "omniflow" ]]; then
    capture_daemon_sample_baseline "$PREFIX"
  fi
  echo "Waiting for every metric timestamp to advance beyond the baseline ..."
  for i in $(seq 1 90); do
    kubectl -n "$NS" get pods -l app=nginx -o json > "${PREFIX}_pods_current.json"
    local DAEMON_FRESH=true
    if [[ "$MODE" == "omniflow" && ! -f "${PREFIX}_daemon_sample_target.json" ]]; then
      if ! daemon_samples_are_fresh "$PREFIX"; then
        DAEMON_FRESH=false
      fi
    fi
    if [[ "$DAEMON_FRESH" == true ]] && \
       capture_metric_payload "$MODE" "${PREFIX}_metrics_current.json" 2>/dev/null && \
       PYTHONPATH="$SCRIPT_DIR" python3 - \
         "${PREFIX}_pods_current.json" "${PREFIX}_metrics.json" \
          "${PREFIX}_metrics_current.json" "$METRIC_MODE" \
          "${PREFIX}_daemon_sample_target.json" "$NS" \
          "$((PROMETHEUS_SCRAPE_SECONDS + 2))" <<'PYEOF'
import json, sys
from experiment_sync import (
    daemon_sample_versions,
    fresh_metrics,
    metric_versions,
    metrics_published_after_samples,
    ready_pod_names,
)
pods = ready_pod_names(json.load(open(sys.argv[1])))
expected = set(pods)
baseline = metric_versions(json.load(open(sys.argv[2])), sys.argv[4], expected)
current = metric_versions(json.load(open(sys.argv[3])), sys.argv[4], expected)
ok, reason = fresh_metrics(pods, baseline, current)
if ok and sys.argv[4] == "custom":
    samples = daemon_sample_versions(
        json.load(open(sys.argv[5])), sys.argv[6], expected
    )
    ok, reason = metrics_published_after_samples(
        pods, current, samples, float(sys.argv[7])
    )
print(reason)
raise SystemExit(0 if ok else 1)
PYEOF
    then
      cp "${PREFIX}_metrics_current.json" "${PREFIX}_metrics_fresh.json"
      cp "${PREFIX}_pods_current.json" "${PREFIX}_pods_fresh.json"
      if [[ "$MODE" == "omniflow" ]]; then
        cp "${PREFIX}_daemon_sample_target.json" "${PREFIX}_daemon_samples_fresh.json"
      fi
      echo "  fresh metrics accepted (attempt $i)"
      return 0
    fi
    sleep 5
  done
  echo "Metrics did not advance for every Ready pod after 450 seconds" >&2
  return 1
}

configure_metrics_server_resolution() {
  local RESOLUTION="$1"
  if kubectl -n kube-system get deployment metrics-server -o json | \
       python3 -c 'import json, sys; args=json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0].get("args", []); sys.exit(0 if sys.argv[1] in args else 1)' \
         "--metric-resolution=$RESOLUTION"; then
    return 0
  fi

  python3 - "$RESOLUTION" <<'PYEOF'
import json
import subprocess
import sys

resolution = sys.argv[1]
deployment = json.loads(subprocess.check_output([
    "kubectl", "-n", "kube-system", "get", "deployment", "metrics-server", "-o", "json"
]))
container = deployment["spec"]["template"]["spec"]["containers"][0]
args = [arg for arg in container.get("args", []) if not arg.startswith("--metric-resolution=")]
args.append(f"--metric-resolution={resolution}")
patch = [{
    "op": "replace" if "args" in container else "add",
    "path": "/spec/template/spec/containers/0/args",
    "value": args,
}]
subprocess.run([
    "kubectl", "-n", "kube-system", "patch", "deployment", "metrics-server",
    "--type=json", "-p", json.dumps(patch),
], check=True)
PYEOF
  kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s
  kubectl -n kube-system get deployment metrics-server -o json | \
    python3 -c 'import json, sys; args=json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0].get("args", []); sys.exit(0 if sys.argv[1] in args else 1)' \
      "--metric-resolution=$RESOLUTION"
}

wait_for_stable_hpa() {
  local HPA_NAME="$1"
  local PREFIX="$2"
  local STABLE_FOR=0
  echo "Waiting for $HPA_NAME and nginx to remain stable at one replica ..."
  for i in $(seq 1 120); do
    kubectl -n "$NS" get hpa "$HPA_NAME" -o json > "${PREFIX}_hpa.json" 2>/dev/null || true
    kubectl -n "$NS" get deployment nginx -o json > "${PREFIX}_deployment.json" 2>/dev/null || true
    if PYTHONPATH="$SCRIPT_DIR" python3 - "${PREFIX}_hpa.json" "${PREFIX}_deployment.json" <<'PYEOF'
import json, sys
from experiment_sync import stable_hpa
try:
    ok, reason = stable_hpa(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
except (OSError, json.JSONDecodeError):
    ok, reason = False, "incomplete status"
print(reason)
raise SystemExit(0 if ok else 1)
PYEOF
    then
      STABLE_FOR=$((STABLE_FOR + 5))
    else
      STABLE_FOR=0
    fi
    if (( STABLE_FOR >= HPA_STABLE_SECONDS )); then
      echo "  HPA stable for ${STABLE_FOR}s"
      return 0
    fi
    sleep 5
  done
  echo "$HPA_NAME did not stabilize after 600 seconds" >&2
  return 1
}

wait_for_locust_job() {
  local TIMEOUT_SECONDS="$1"
  local ELAPSED=0
  while (( ELAPSED < TIMEOUT_SECONDS )); do
    local COMPLETE FAILED
    COMPLETE=$(kubectl -n "$NS" get job locust-load \
      -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)
    FAILED=$(kubectl -n "$NS" get job locust-load \
      -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
    if [[ "$COMPLETE" == "True" ]]; then
      return 0
    fi
    if [[ "$FAILED" == "True" ]]; then
      echo "Locust job failed" >&2
      return 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
  done
  echo "Locust job timed out after ${TIMEOUT_SECONDS}s" >&2
  return 1
}

stop_observers() {
  if [[ -n "$TIMELINE_PID" ]]; then
    kill "$TIMELINE_PID" 2>/dev/null || true
    wait "$TIMELINE_PID" 2>/dev/null || true
    TIMELINE_PID=""
  fi
  if [[ -n "$LOCUST_LOG_PID" ]]; then
    kill "$LOCUST_LOG_PID" 2>/dev/null || true
    wait "$LOCUST_LOG_PID" 2>/dev/null || true
    LOCUST_LOG_PID=""
  fi
  if [[ -n "$DISK_MONITOR_PID" ]]; then
    kill "$DISK_MONITOR_PID" 2>/dev/null || true
    wait "$DISK_MONITOR_PID" 2>/dev/null || true
    DISK_MONITOR_PID=""
  fi
}

host_cpu_monitor_args() {
  printf '%s\0' \
    --profile "$PROFILE" \
    --cpus "$HOST_CPUS" \
    --expected-processes "$HOST_QEMU_PROCESSES" \
    --interval "$HOST_CPU_INTERVAL" \
    --threshold "$HOST_CPU_THRESHOLD" \
    --consecutive "$HOST_CPU_CONSECUTIVE"
}

run_host_cpu_window() {
  local PHASE="$1" DURATION_SECONDS="$2" OUTPUT="$3" FAILURE_FILE="$4"
  local -a COMMON_ARGS=()
  while IFS= read -r -d '' ARG; do COMMON_ARGS+=("$ARG"); done < <(host_cpu_monitor_args)
  python3 "$SCRIPT_DIR/host_cpu_guard.py" monitor "${COMMON_ARGS[@]}" \
    --phase "$PHASE" --duration "$DURATION_SECONDS" --output "$OUTPUT" \
    --failure-file "$FAILURE_FILE"
}

start_host_cpu_guard() {
  local OUTPUT="$1" FAILURE_FILE="$2"
  local -a COMMON_ARGS=()
  HOST_GUARD_STOP_FILE="${CURRENT_REP_DIR}/.stop-host-cpu-guard"
  rm -f "$HOST_GUARD_STOP_FILE" "$FAILURE_FILE"
  while IFS= read -r -d '' ARG; do COMMON_ARGS+=("$ARG"); done < <(host_cpu_monitor_args)
  python3 "$SCRIPT_DIR/host_cpu_guard.py" monitor "${COMMON_ARGS[@]}" \
    --phase during --output "$OUTPUT" --failure-file "$FAILURE_FILE" \
    --stop-file "$HOST_GUARD_STOP_FILE" --signal-pid "$RUNNER_PID" &
  HOST_GUARD_PID=$!
}

stop_host_cpu_guard() {
  local ENFORCE="${1:-true}" CODE=0
  [[ -n "$HOST_GUARD_PID" ]] || return 0
  touch "$HOST_GUARD_STOP_FILE"
  set +e
  wait "$HOST_GUARD_PID"
  CODE=$?
  set -e
  HOST_GUARD_PID=""
  if [[ "$ENFORCE" == true && "$CODE" -ne 0 ]]; then
    return "$CODE"
  fi
}

require_free_disk() {
  local available_kb required_kb
  available_kb=$(df -Pk "$RUN_DIR" | python3 -c \
    'import sys; rows=[line.split() for line in sys.stdin if line.strip()]; print(rows[-1][3])')
  required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if (( available_kb < required_kb )); then
    echo "Only $((available_kb / 1024 / 1024)) GiB free at $RUN_DIR; " \
         "at least ${MIN_FREE_GB} GiB is required." >&2
    return 1
  fi
}

start_disk_monitor() {
  (
    local available_kb required_kb
    required_kb=$((MIN_FREE_GB * 1024 * 1024))
    while true; do
      available_kb=$(df -Pk "$RUN_DIR" | python3 -c \
        'import sys; rows=[line.split() for line in sys.stdin if line.strip()]; print(rows[-1][3])')
      if (( available_kb < required_kb )); then
        echo "Disk-space guard tripped: only $((available_kb / 1024 / 1024)) GiB free." >&2
        kill -TERM "$RUNNER_PID"
        exit 0
      fi
      sleep 10
    done
  ) &
  DISK_MONITOR_PID=$!
}

reset_daemon_traces() {
  local found=0
  local pod
  for pod in $(kubectl -n omniflow-infra get pods -l app=omniflow-daemon -o name); do
    found=$((found + 1))
    if ! kubectl -n omniflow-infra exec "$pod" -- python3 -c \
      "import urllib.request; request=urllib.request.Request('http://127.0.0.1:${DAEMON_CONTROL_PORT}/trace/reset', data=b'', method='POST'); urllib.request.urlopen(request, timeout=30).read()"; then
      echo "Failed to reset trace file in $pod" >&2
      return 1
    fi
  done
  if (( found == 0 )); then
    echo "No OmniFlow daemon pods found while resetting traces" >&2
    return 1
  fi
}

collect_daemon_traces() {
  local log_prefix="$1"
  local found=0
  local total_readings=0
  local current_namespace_readings=0
  local pod pod_name daemon_out raw_tmp reading_count namespace_count

  for pod in $(kubectl -n omniflow-infra get pods -l app=omniflow-daemon -o name); do
    found=$((found + 1))
    pod_name=$(basename "$pod")
    daemon_out="${log_prefix}_daemon_${pod_name}.jsonl.gz"
    raw_tmp=$(mktemp "${TMPDIR:-/tmp}/omniflow-trace.XXXXXX")

    if ! kubectl -n omniflow-infra exec "$pod" -- python3 -c \
      "import shutil,sys,urllib.request; response=urllib.request.urlopen('http://127.0.0.1:${DAEMON_CONTROL_PORT}/trace/snapshot', timeout=60); shutil.copyfileobj(response, sys.stdout.buffer, length=1048576)" \
      > "$raw_tmp"; then
      echo "Failed to collect trace file from $pod" >&2
      rm -f "$raw_tmp"
      return 1
    fi
    if ! grep -q '"event":"daemon_start"' "$raw_tmp"; then
      echo "Trace from $pod lacks daemon_start metadata" >&2
      rm -f "$raw_tmp"
      return 1
    fi
    reading_count=$(grep -c '"value":' "$raw_tmp" || true)
    namespace_count=$(grep -F '"namespace":"'"$NS"'"' "$raw_tmp"       | grep -c '"value":' || true)
    total_readings=$((total_readings + reading_count))
    current_namespace_readings=$((current_namespace_readings + namespace_count))
    gzip -1 -c "$raw_tmp" > "$daemon_out"
    rm -f "$raw_tmp"
  done

  if (( found == 0 )); then
    echo "No OmniFlow daemon pods found while collecting traces" >&2
    return 1
  fi
  if (( total_readings == 0 )); then
    echo "Daemon trace collection produced zero CPU readings; refusing to mark repetition complete" >&2
    return 1
  fi
  if (( current_namespace_readings == 0 )); then
    echo "Daemon traces contain no CPU readings for namespace $NS; refusing to mark repetition complete" >&2
    return 1
  fi
  echo "Collected $current_namespace_readings readings for $NS ($total_readings total) across $found node(s)."
}

repeat_is_complete() {
  local rep_dir="$1" label="$2"
  python3 - "$rep_dir" "$label" "$NS" <<'PYEOF'
import glob
import gzip
import json
import os
import sys

rep_dir, label, namespace = sys.argv[1:]
if not os.path.isfile(os.path.join(rep_dir, "complete.json")):
    raise SystemExit(1)
if os.path.isfile(os.path.join(rep_dir, "host-cpu-contamination.json")):
    raise SystemExit(1)
paths = sorted(glob.glob(os.path.join(rep_dir, f"{label}_daemon_*.jsonl.gz")))
if not paths:
    raise SystemExit(1)
found_start = False
found_reading = False
for path in paths:
    try:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "daemon_start":
                    found_start = True
                if record.get("namespace") == namespace and "value" in record:
                    found_reading = True
                    break
    except (OSError, EOFError):
        raise SystemExit(1)
    if found_start and found_reading:
        break
raise SystemExit(0 if found_start and found_reading else 1)
PYEOF
}

_write_attempt_status() {
  local state="$1" step="$2" message="${3:-}" line="${4:-}" command="${5:-}"
  [[ -n "$CURRENT_REP_DIR" ]] || return 0
  mkdir -p "$CURRENT_REP_DIR"
  python3 - "$CURRENT_REP_DIR/attempt-status.json" "$state" "$step" "$message" "$line" "$command" <<'PYEOF'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

path, state, step, message, line, command = sys.argv[1:]
payload = {
    "state": state,
    "step": step,
    "message": message,
    "line": int(line) if line.isdigit() else None,
    "command": command,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
fd, tmp = tempfile.mkstemp(prefix=".attempt-status.", dir=os.path.dirname(path))
try:
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PYEOF
}

_mark_step() {
  CURRENT_STEP="$1"
  _write_attempt_status running "$CURRENT_STEP" "${2:-}"
  echo "  step: $CURRENT_STEP"
}

_cleanup_k8s_current() {
  set +e
  kubectl -n "$NS" delete job/locust-load --ignore-not-found >/dev/null 2>&1
  if [[ -n "$CURRENT_HPA_NAME" ]]; then
    kubectl -n "$NS" delete hpa "$CURRENT_HPA_NAME" --ignore-not-found >/dev/null 2>&1
  fi
  set -e
}

_cleanup_run() {
  set +e
  stop_host_cpu_guard false
  stop_observers
  cleanup_manifests
  [[ -n "$DAEMON_TAR" ]] && rm -f "$DAEMON_TAR"
  [[ -n "$LOCUST_TAR" ]] && rm -f "$LOCUST_TAR"
}

_on_signal() {
  local signal="$1" code="$2"
  trap - ERR INT TERM
  set +e
  _write_attempt_status interrupted "$CURRENT_STEP" "received $signal"
  echo >&2
  echo "Received $signal during ${CURRENT_LABEL:-experiment}/$CURRENT_STEP; preserving completed repetitions." >&2
  stop_host_cpu_guard false
  stop_observers
  _cleanup_k8s_current
  cleanup_manifests
  exit "$code"
}

_on_error() {
  local code="$1" line="$2" command="$3"
  trap - ERR INT TERM
  set +e
  if [[ -n "$HOST_CONTAMINATION_FILE" && -f "$HOST_CONTAMINATION_FILE" ]]; then
    _write_attempt_status invalidated "$CURRENT_STEP" \
      "reserved host CPUs were contaminated; attempt excluded" "$line" "$command"
  else
    _write_attempt_status failed "$CURRENT_STEP" "command failed with exit code $code" "$line" "$command"
  fi
  echo "Experiment failed at step '$CURRENT_STEP', line $line: $command" >&2
  stop_host_cpu_guard false
  stop_observers
  _cleanup_k8s_current
  cleanup_manifests
  exit "$code"
}

_on_contamination() {
  trap - ERR INT TERM USR1
  set +e
  _write_attempt_status invalidated "$CURRENT_STEP" \
    "reserved host CPUs were contaminated; attempt excluded"
  echo "Reserved-CPU contamination detected during ${CURRENT_LABEL:-experiment}/$CURRENT_STEP; invalidating attempt." >&2
  stop_host_cpu_guard false
  stop_observers
  _cleanup_k8s_current
  cleanup_manifests
  exit 120
}

trap _cleanup_run EXIT
trap '_on_signal INT 130' INT
trap '_on_signal TERM 143' TERM
trap _on_contamination USR1
trap '_on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

run_experiment() {
  local HPA_FILE="$1"
  local LABEL="$2"
  local MODE="$3"
  local REPETITION="$4"
  local REP_DIR
  REP_DIR="$OUT_DIR/$LABEL/repeat_$(printf '%02d' "$REPETITION")"
  local LOG_PREFIX="$REP_DIR/$LABEL"
  local SYNC_PREFIX="$REP_DIR/sync"
  local HPA_NAME
  HPA_NAME=$(kubectl create --dry-run=client -f "$HPA_FILE" -o jsonpath='{.metadata.name}')
  mkdir -p "$REP_DIR"

  CURRENT_REP_DIR="$REP_DIR"
  CURRENT_LABEL="$LABEL"
  CURRENT_HPA_NAME="$HPA_NAME"
  CURRENT_STEP="resume-check"

  if [[ "$RESUME" == true ]] && repeat_is_complete "$REP_DIR" "$LABEL"; then
    echo "--- Skipping validated experiment: $LABEL (repeat $REPETITION/$REPEATS) ---"
    CURRENT_REP_DIR=""
    CURRENT_LABEL=""
    CURRENT_HPA_NAME=""
    CURRENT_STEP="between-repetitions"
    return 0
  fi
  if [[ "$RESUME" == true && -f "$REP_DIR/complete.json" ]]; then
    echo "Completed marker exists but daemon telemetry is missing or invalid; rerunning $LABEL repeat $REPETITION."
  fi
  if compgen -G "$REP_DIR/*" >/dev/null; then
    local FAILED_DIR
    FAILED_DIR="$REP_DIR/failed_attempts/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$FAILED_DIR"
    for ARTIFACT in "$REP_DIR"/*; do
      [[ "$(basename "$ARTIFACT")" == "failed_attempts" ]] && continue
      mv "$ARTIFACT" "$FAILED_DIR/"
    done
    echo "Preserved incomplete attempt in $FAILED_DIR"
  fi

  echo ""
  echo "--- Running experiment: $LABEL (repeat $REPETITION/$REPEATS) ---"
  local HOST_CPU_TELEMETRY="$REP_DIR/host-cpu-telemetry.jsonl"
  HOST_CONTAMINATION_FILE="$REP_DIR/host-cpu-contamination.json"
  : > "$HOST_CPU_TELEMETRY"
  rm -f "$HOST_CONTAMINATION_FILE"
  _write_attempt_status running start "starting repeat $REPETITION of $REPEATS"
  _mark_step pin-qemu
  PINNING_JSON=$(sudo -n PYTHONPATH="$SCRIPT_DIR" python3 \
    "$SCRIPT_DIR/host_cpu_guard.py" pin --profile "$PROFILE" --cpus "$HOST_CPUS" \
    --expected-processes "$HOST_QEMU_PROCESSES")
  printf '%s\n' "$PINNING_JSON" > "$REP_DIR/host-cpu-pinning.json"
  _mark_step host-cpu-precheck
  run_host_cpu_window pre "$HOST_CPU_PRE_SECONDS" \
    "$HOST_CPU_TELEMETRY" "$HOST_CONTAMINATION_FILE"
  start_host_cpu_guard "$HOST_CPU_TELEMETRY" "$HOST_CONTAMINATION_FILE"
  require_free_disk
  start_disk_monitor

  _mark_step configure-metric-source
  case "$MODE" in
    metrics_60s) configure_metrics_server_resolution "60s" ;;
    metrics_15s) configure_metrics_server_resolution "15s" ;;
    omniflow) ;;
    *) echo "Unknown experiment mode: $MODE" >&2; return 1 ;;
  esac
  kubectl -n kube-system get deployment metrics-server -o json \
    > "$REP_DIR/${LABEL}_metrics-server.json"

  # Recreate every namespace-scoped object and wait until the daemon has pruned
  # the old cgroups. This resets workload, HPA, events, metric identities, and
  # adaptive trackers rather than attempting an in-place cleanup.
  _mark_step delete-workload-namespace
  delete_workload_namespace
  wait_for_namespace_tracks_gone "$REP_DIR/daemon-tracks-after-delete.json"

  _mark_step reset-telemetry
  DAEMON_LOG_SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  reset_daemon_traces

  _mark_step create-workload-namespace
  create_workload_namespace
  _mark_step settle-workload
  echo "Settling for ${SETTLE_SECONDS}s before the next load ..."
  sleep "$SETTLE_SECONDS"

  _mark_step synchronize-metrics
  capture_metric_baseline "$MODE" "$SYNC_PREFIX"
  wait_for_fresh_metrics "$MODE" "$SYNC_PREFIX"

  # Apply HPA
  _mark_step apply-hpa
  kubectl apply -f "$HPA_FILE"

  _mark_step wait-stable-hpa
  wait_for_stable_hpa "$HPA_NAME" "$SYNC_PREFIX"

  local T0_NS
  local EXPERIMENT_SEED=$((100000 + REPETITION))
  _mark_step arm-load-generator
  kubectl delete job/locust-load -n "$NS" --ignore-not-found
  local TMP_JOB
  TMP_JOB=$(mktemp "${TMPDIR:-/tmp}/locust-job.XXXXXX")
  PYTHONPATH="$SCRIPT_DIR" python3 experiment_sync.py render-job \
    "${MANIFEST_DIR}/locust-job.yaml" "$TMP_JOB" "$SHAPE" "$EXPERIMENT_SEED"
  kubectl apply -f "$TMP_JOB"
  rm -f "$TMP_JOB"
  kubectl -n "$NS" wait --for=condition=ready pod -l job-name=locust-load --timeout=120s

  T0_NS=$(python3 -c "import time; print(time.time_ns() + int(${ARM_LEAD_SECONDS} * 1e9))")

  python3 - <<PYEOF
import json
payload = {
    "label": "${LABEL}",
    "repeat": ${REPETITION},
    "shape": "${SHAPE}",
    "duration_s": ${DURATION},
    "hpa_manifest": "${HPA_FILE}",
    "hpa_name": "${HPA_NAME}",
    "mode": "${MODE}",
    "execution_mode": "isolated_arm_major",
    "t0_ns": ${T0_NS},
    "release_file": "/tmp/locust-release",
    "arm_position": None,
    "arm_order": ["${LABEL}"],
    "experiment_seed": ${EXPERIMENT_SEED},
    "metric_baseline": "${SYNC_PREFIX}_metrics.json",
    "fresh_metrics": "${SYNC_PREFIX}_metrics_fresh.json",
    "hpa_stable_seconds": ${HPA_STABLE_SECONDS},
    "post_load_seconds": ${POST_LOAD_SECONDS},
    "daemon_log_since": "${DAEMON_LOG_SINCE}",
}
with open("${REP_DIR}/metadata.json", "w") as f:
    json.dump(payload, f, indent=2)
PYEOF

  # -- Background observers start before t0 -----------------------
  _mark_step start-observers
  local TIMELINE_FILE="${LOG_PREFIX}_replica_timeline.jsonl"
  : > "$TIMELINE_FILE"
  (
    while true; do
      REPLICAS=$(kubectl -n "$NS" get deployment/nginx -o jsonpath='{.status.replicas}' 2>/dev/null || true)
      READY=$(kubectl -n "$NS" get deployment/nginx -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
      HPA_DESIRED=$(kubectl -n "$NS" get hpa -o jsonpath='{.items[0].status.desiredReplicas}' 2>/dev/null || true)
      read -r NOW_NS WALL ELAPSED <<< "$(python3 -c 'import sys,time; n=time.time_ns(); t=int(sys.argv[1]); print(n, n/1e9, (n-t)/1e9)' "$T0_NS")"
      printf '{"wall_ns":%s,"wall":%s,"elapsed_s":%s,"replicas":%s,"ready":%s,"hpa_desired":%s}\n' \
        "$NOW_NS" "$WALL" "$ELAPSED" "${REPLICAS:-null}" "${READY:-null}" "${HPA_DESIRED:-null}" >> "$TIMELINE_FILE"
      sleep "$OBSERVE_INTERVAL"
    done
  ) &
  TIMELINE_PID=$!
  kubectl -n "$NS" logs -f job/locust-load > "${LOG_PREFIX}_locust.log" 2>&1 &
  LOCUST_LOG_PID=$!

  OBSERVERS_READY=false
  for _ in $(seq 1 10); do
    if kill -0 "$TIMELINE_PID" 2>/dev/null && kill -0 "$LOCUST_LOG_PID" 2>/dev/null && \
       [[ -s "$TIMELINE_FILE" ]]; then
      OBSERVERS_READY=true
      break
    fi
    sleep 1
  done
  if [[ "$OBSERVERS_READY" != true ]]; then
    echo "Observers failed to start before t0" >&2
    stop_observers
    return 1
  fi
  local LEAD_REMAINING
  LEAD_REMAINING=$(python3 -c 'import sys,time; print((int(sys.argv[1])-time.time_ns())/1e9)' "$T0_NS")
  if ! python3 -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) >= 10 else 1)' "$LEAD_REMAINING"; then
    echo "Insufficient lead time before t0 (${LEAD_REMAINING}s)" >&2
    stop_observers
    return 1
  fi
  echo "Observers active ${LEAD_REMAINING}s before synchronized t0"

  while python3 -c 'import sys,time; raise SystemExit(0 if time.time_ns() < int(sys.argv[1]) else 1)' "$T0_NS"; do
    sleep 1
  done
  _mark_step release-load
  LOCUST_POD=$(kubectl -n "$NS" get pods -l job-name=locust-load \
    -o jsonpath='{.items[0].metadata.name}')
  RELEASE_COMMAND_NS=$(python3 -c 'import time; print(time.time_ns())')
  # The loop variables are intentionally expanded by the pod's shell.
  # shellcheck disable=SC2016
  if ! LOCUST_START_GUEST_NS=$(kubectl -n "$NS" exec "$LOCUST_POD" -- sh -c '
    touch /tmp/locust-release
    i=0
    while [ ! -s /tmp/locust-started ]; do
      i=$((i + 1))
      [ "$i" -lt 1200 ] || exit 1
      sleep 0.1
    done
    cat /tmp/locust-started
  '); then
    echo "Failed to release Locust or receive its start acknowledgment" >&2
    stop_observers
    return 1
  fi
  RELEASE_ACK_NS=$(python3 -c 'import time; print(time.time_ns())')
  if ! [[ "$LOCUST_START_GUEST_NS" =~ ^[0-9]+$ ]]; then
    echo "Invalid Locust start timestamp: $LOCUST_START_GUEST_NS" >&2
    return 1
  fi
  python3 - "$REP_DIR/metadata.json" "$RELEASE_COMMAND_NS" "$RELEASE_ACK_NS" \
    "$LOCUST_START_GUEST_NS" <<'PYEOF'
import json
import os
import sys
import tempfile

path, started, acknowledged, guest_start = sys.argv[1:]
payload = json.load(open(path))
payload["release_command_ns"] = int(started)
payload["release_ack_ns"] = int(acknowledged)
payload["release_midpoint_ns"] = (int(started) + int(acknowledged)) // 2
payload["locust_start_host_lower_ns"] = int(started)
payload["locust_start_host_upper_ns"] = int(acknowledged)
payload["locust_start_host_midpoint_ns"] = (int(started) + int(acknowledged)) // 2
payload["locust_start_guest_ns"] = int(guest_start)
fd, temporary = tempfile.mkstemp(prefix=".metadata.", dir=os.path.dirname(path))
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PYEOF
  echo "Locust acknowledged start between ${RELEASE_COMMAND_NS} and ${RELEASE_ACK_NS}"

  _mark_step run-load
  echo "Waiting for load test completion (up to $((DURATION + ARM_LEAD_SECONDS + 120))s) ..."
  if ! wait_for_locust_job "$((DURATION + ARM_LEAD_SECONDS + 120))"; then
    stop_observers
    kubectl -n "$NS" logs job/locust-load > "${LOG_PREFIX}_locust_failed.log" 2>/dev/null || true
    kubectl -n "$NS" describe job/locust-load >&2 || true
    exit 1
  fi

  _mark_step post-load-observation
  echo "Observing HPA for ${POST_LOAD_SECONDS}s after load completion ..."
  sleep "$POST_LOAD_SECONDS"

  # Stop replica polling after the post-load observation tail.
  stop_observers

  # -- Collect data ----------------------------------------------
  _mark_step collect-artifacts
  echo "Collecting $LABEL data ..."

  # HPA events
  kubectl -n "$NS" describe hpa > "${LOG_PREFIX}_hpa_describe.txt" 2>/dev/null || true

  # HPA status over time (snapshot)
  kubectl -n "$NS" get hpa -o json > "${LOG_PREFIX}_hpa_status.json" 2>/dev/null || true

  # Deployment replica history
  kubectl -n "$NS" get events \
    --field-selector reason=ScalingReplicaSet \
    --sort-by='.lastTimestamp' \
    -o json > "${LOG_PREFIX}_scale_events.json" 2>/dev/null || true

  collect_daemon_traces "$LOG_PREFIX"

  # The follow stream is only a live observer. Refetch the complete canonical
  # Job log after completion so a broken stream cannot silently lose events.
  local LOCUST_CANONICAL
  LOCUST_CANONICAL=$(mktemp "${TMPDIR:-/tmp}/omniflow-locust-log.XXXXXX")
  kubectl -n "$NS" logs job/locust-load > "$LOCUST_CANONICAL"
  python3 - "$LOCUST_CANONICAL" <<'PYEOF'
import json
import sys

events = []
for line in open(sys.argv[1]):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("event") in {"load_start", "load_end"}:
        events.append(record["event"])
if events.count("load_start") != 1 or events.count("load_end") != 1:
    print(f"Incomplete Locust lifecycle events: {events}", file=sys.stderr)
    raise SystemExit(1)
PYEOF
  mv "$LOCUST_CANONICAL" "${LOG_PREFIX}_locust.log"

  # Clean up HPA and locust job
  kubectl delete -f "$HPA_FILE" --ignore-not-found
  kubectl delete job/locust-load -n "$NS" --ignore-not-found

  _mark_step host-cpu-postcheck
  stop_host_cpu_guard
  run_host_cpu_window post "$HOST_CPU_POST_SECONDS" \
    "$HOST_CPU_TELEMETRY" "$HOST_CONTAMINATION_FILE"

  _mark_step finalize
  python3 - <<PYEOF
import json
import os
import tempfile
payload = {"label": "${LABEL}", "repeat": ${REPETITION}}
fd, tmp = tempfile.mkstemp(prefix=".complete.", dir="${REP_DIR}")
with os.fdopen(fd, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
os.replace(tmp, "${REP_DIR}/complete.json")
PYEOF
  _write_attempt_status complete complete "artifacts collected"
  stop_observers

  echo "$LABEL repeat $REPETITION complete."
  CURRENT_REP_DIR=""
  CURRENT_LABEL=""
  CURRENT_HPA_NAME=""
  HOST_CONTAMINATION_FILE=""
  CURRENT_STEP="between-repetitions"
}

# -- Phases 3-5: isolated arm-major repeated experiments -----------
if [[ -n "$REQUESTED_ARM" ]]; then
  ARMS=("$REQUESTED_ARM")
else
  ARMS=(metrics_60s metrics_15s omniflow)
fi
for ARM in "${ARMS[@]}"; do
  for REPETITION in $(seq 1 "$REPEATS"); do
    case "$ARM" in
      metrics_60s) HPA_FILE="${MANIFEST_DIR}/hpa-baseline.yaml" ;;
      metrics_15s) HPA_FILE="${MANIFEST_DIR}/hpa-fixed-15s.yaml" ;;
      omniflow) HPA_FILE="${MANIFEST_DIR}/hpa-omniflow.yaml" ;;
      *) echo "Unknown arm: $ARM" >&2; exit 1 ;;
    esac
    run_experiment "$HPA_FILE" "$ARM" "$ARM" "$REPETITION"
  done
done

# -- Phase 5: Post-processing ------------------------------------
echo ""
echo "--- Phase 5: Post-processing ---"

ANALYSIS_PYTHON="$REPO_ROOT/.direnv/python-3.12.3/bin/python3"
if [[ ! -x "$ANALYSIS_PYTHON" ]]; then
  ANALYSIS_PYTHON=python3
fi
"$ANALYSIS_PYTHON" collect-results.py --run-dir "$RUN_DIR" --out "$RUN_DIR/analysis.json"

if [[ "$KEEP_RESOURCES" != true ]]; then
  echo "Removing completed workload namespace $NS ..."
  kubectl delete namespace "$NS" --wait=true --timeout=180s
fi

# -- Cleanup (optional: leave infrastructure for inspection) ------
echo ""
echo "=== Experiment complete ==="
echo "Run output: $RUN_DIR"
echo "Results:    $RUN_DIR/analysis.json"
echo ""
echo "To tear down K8s resources:"
echo "  kubectl delete namespace $NS"
echo "  kubectl delete namespace omniflow-infra"
echo "  kubectl delete apiservice v1beta1.custom.metrics.k8s.io"
echo ""
echo "To tear down minikube cluster:"
echo "  ./teardown-cluster.sh"
