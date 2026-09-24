#!/usr/bin/env bash
# Run each HPA case as an isolated, arm-major series for every workload shape.
set -Eeuo pipefail
cd "$(dirname "$0")"

REPEATS=5
MAX_ATTEMPTS=20
OUTPUT_ROOT="$(pwd)/out"
SHAPES=(phased oscillating staircase flash_crowd)
ARGS=()
SEQUENTIAL=true
RESUME=false
STATUS_ONLY=false
REQUESTED_ARM=""

usage() {
  cat <<'USAGE'
Usage: ./run-all-shapes.sh [options] [run-experiment options]

Options:
  --repeats N             Repetitions per arm (default: 5)
  --max-attempts N        Attempts per shape before stopping (default: 20)
  --output-root DIR       Parent containing shape run directories
  --shapes A,B,...        Shapes, in the requested execution order
  --arm NAME              Run only metrics_60s, metrics_15s, or omniflow
  --resume                Reuse the latest matching run directory per shape
  --status                Print completion status and exit (implies --resume)
  --sequential            One shape at a time (default; paper-quality)
  --parallel              Rejected: shared daemon lifecycle is not isolated

Safe interruption:
  Press Ctrl-C at any point.  The active repetition is marked interrupted.
  Run the same command again with --resume; complete.json repetitions are
  skipped and the interrupted repetition is rerun.
USAGE
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --repeats) REPEATS="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --shapes) IFS=',' read -r -a SHAPES <<< "$2"; shift 2 ;;
    --arm) REQUESTED_ARM="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --status) STATUS_ONLY=true; RESUME=true; shift ;;
    --sequential) SEQUENTIAL=true; shift ;;
    --parallel) SEQUENTIAL=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -n "$REQUESTED_ARM" && ! "$REQUESTED_ARM" =~ ^(metrics_60s|metrics_15s|omniflow)$ ]]; then
  echo "--arm must be metrics_60s, metrics_15s, or omniflow" >&2
  exit 1
fi

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS < 1 )); then
  echo "--repeats must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || (( MAX_ATTEMPTS < 1 )); then
  echo "--max-attempts must be a positive integer" >&2
  exit 1
fi
for SHAPE in "${SHAPES[@]}"; do
  case "$SHAPE" in
    phased|oscillating|staircase|flash_crowd) ;;
    *) echo "Unknown shape in --shapes: $SHAPE" >&2; exit 1 ;;
  esac
done
if [[ "$SEQUENTIAL" != true ]]; then
  echo "--parallel is unsafe: shape workers share and reset one daemon trace plane." >&2
  echo "Run sequentially until per-run daemon lifecycle isolation is implemented." >&2
  exit 1
fi

REPO_ROOT="$(cd ../.. && pwd)"
INFRA_NS="omniflow-infra"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")"

# A lock file may remain on disk after Ctrl-C; flock state does not.  Therefore
# it is safe to rerun without deleting the file.
LOCK_FILE="$OUTPUT_ROOT/.run-all-shapes.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "Another run-all-shapes.sh process currently holds $LOCK_FILE." >&2
  exit 1
fi
echo "$$" >&200

_on_signal() {
  local name="$1" code="$2"
  echo >&2
  echo "Received $name; stopping safely. Re-run with --resume." >&2
  exit "$code"
}
trap '_on_signal INT 130' INT
trap '_on_signal TERM 143' TERM

_find_latest_run_dir() {
  local shape="$1"
  python3 - "$OUTPUT_ROOT" "$shape" <<'PYEOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
shape = sys.argv[2]
candidates = []
for path in root.iterdir() if root.exists() else []:
    if not path.is_dir():
        continue
    matched = f"___{shape}_k8s" in path.name
    params = path / "params.json"
    if params.is_file():
        try:
            matched = json.loads(params.read_text()).get("shape") == shape
        except Exception:
            pass
    if matched:
        candidates.append(path)
if candidates:
    # Run names begin with a sortable timestamp; mtime breaks ties and supports
    # hand-copied directories.
    candidates.sort(key=lambda p: (p.name, p.stat().st_mtime))
    print(candidates[-1])
PYEOF
}

_new_run_dir() {
  local shape="$1"
  python3 "$REPO_ROOT/src/support/log.py" \
    --base "$OUTPUT_ROOT" --application-type "${shape}_k8s"
}

# Resolve a stable directory for every shape.  In resume mode, this also adopts
# campaigns started by older versions of the runner.
declare -A SHAPE_RUN_DIR
for SHAPE in "${SHAPES[@]}"; do
  RUN_DIR=""
  if [[ "$RESUME" == true ]]; then
    RUN_DIR="$(_find_latest_run_dir "$SHAPE")"
  fi
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(_new_run_dir "$SHAPE")"
  fi
  SHAPE_RUN_DIR["$SHAPE"]="$RUN_DIR"
  echo "$SHAPE -> $RUN_DIR"
done

_print_status() {
  local shape="$1" run_dir="$2"
  python3 - "$run_dir" "$REPEATS" "$shape" "$REQUESTED_ARM" <<'PYEOF'
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
repeats = int(sys.argv[2])
shape = sys.argv[3]
requested_arm = sys.argv[4]
labels = (requested_arm,) if requested_arm else ("metrics_60s", "metrics_15s", "omniflow")
print(f"[{shape}] {run_dir}")
for label in labels:
    complete = [
        i for i in range(1, repeats + 1)
        if (run_dir / "results" / label / f"repeat_{i:02d}" / "complete.json").is_file()
        and not (run_dir / "results" / label / f"repeat_{i:02d}" / "host-cpu-contamination.json").is_file()
    ]
    missing = [i for i in range(1, repeats + 1) if i not in complete]
    print(f"  {label}: {len(complete)}/{repeats} complete; missing={missing}")
PYEOF
}

if [[ "$STATUS_ONLY" == true ]]; then
  for SHAPE in "${SHAPES[@]}"; do
    _print_status "$SHAPE" "${SHAPE_RUN_DIR[$SHAPE]}"
  done
  exit 0
fi

# Shared monitoring infrastructure is deployed only after status resolution.
echo "=== Deploying shared infrastructure ==="
kubectl apply -f manifests/namespace-infra.yaml
kubectl apply -f manifests/prometheus.yaml
kubectl apply -f manifests/prometheus-adapter.yaml
kubectl -n "$INFRA_NS" rollout status deployment/prometheus --timeout=180s
kubectl -n "$INFRA_NS" rollout status deployment/prometheus-adapter --timeout=180s
echo "Shared infrastructure ready."

_validate_shape() {
  local run_dir="$1" shape="$2"
  python3 - "$run_dir" "$REPEATS" "$shape" "$REQUESTED_ARM" <<'PYEOF'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
repeats = int(sys.argv[2])
shape = sys.argv[3]
requested_arm = sys.argv[4]
labels = (requested_arm,) if requested_arm else ("metrics_60s", "metrics_15s", "omniflow")
missing = {
    label: [
        i for i in range(1, repeats + 1)
        if not (run_dir / "results" / label / f"repeat_{i:02d}" / "complete.json").is_file()
        or (run_dir / "results" / label / f"repeat_{i:02d}" / "host-cpu-contamination.json").is_file()
    ]
    for label in labels
}
missing = {k: v for k, v in missing.items() if v}
analysis_path = run_dir / "analysis.json"
if missing or not analysis_path.is_file():
    print(f"Incomplete {shape}: missing={missing}, analysis={analysis_path.is_file()}", file=sys.stderr)
    raise SystemExit(1)
analysis = json.loads(analysis_path.read_text())
if analysis.get("shape") != shape:
    print(f"Analysis shape mismatch: expected {shape}, got {analysis.get('shape')}", file=sys.stderr)
    raise SystemExit(1)
print(f"Validated {repeats * len(labels)} executions in {run_dir}")
PYEOF
}

_run_shape_all() {
  local shape="$1" run_dir="$2"
  local ns="omniflow-hpa-${shape//_/-}"
  local log="$run_dir/run.log"
  mkdir -p "$run_dir"
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    local cmd=(./run-experiment.sh "${ARGS[@]}" --shape "$shape" --repeats "$REPEATS"
               --namespace "$ns" --run-dir "$run_dir" --skip-infra)
    if [[ "$RESUME" == true || "$attempt" -gt 1 ]]; then
      cmd+=(--resume)
    fi
    echo "=== $shape: attempt $attempt/$MAX_ATTEMPTS ===" | tee -a "$log"
    if "${cmd[@]}" 2>&1 | tee -a "$log"; then
      return 0
    fi
    echo "$shape attempt $attempt failed; the next attempt will resume." | tee -a "$log" >&2
  done
  return 1
}

_run_shape_arm() {
  local shape="$1" run_dir="$2" arm="$3"
  local ns="omniflow-hpa-${shape//_/-}"
  local log="$run_dir/run-${arm}.log"
  mkdir -p "$run_dir"
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    local cmd=(./run-experiment.sh "${ARGS[@]}" --shape "$shape" --repeats "$REPEATS"
               --namespace "$ns" --arm "$arm" --run-dir "$run_dir" --skip-infra)
    if [[ "$RESUME" == true || "$attempt" -gt 1 ]]; then
      cmd+=(--resume)
    fi
    echo "=== $shape $arm: attempt $attempt/$MAX_ATTEMPTS ===" | tee -a "$log"
    if "${cmd[@]}" >>"$log" 2>&1; then
      return 0
    fi
    echo "$shape $arm attempt $attempt failed; the next attempt will resume." | tee -a "$log" >&2
  done
  return 1
}

FAILED_SHAPES=()
if [[ "$SEQUENTIAL" == true ]]; then
  for SHAPE in "${SHAPES[@]}"; do
    RUN_DIR="${SHAPE_RUN_DIR[$SHAPE]}"
    SHAPE_FAILED=false
    if [[ -n "$REQUESTED_ARM" ]]; then
      _run_shape_arm "$SHAPE" "$RUN_DIR" "$REQUESTED_ARM" || SHAPE_FAILED=true
    elif ! _run_shape_all "$SHAPE" "$RUN_DIR"; then
      SHAPE_FAILED=true
    fi
    if [[ "$SHAPE_FAILED" != true ]] && ! _validate_shape "$RUN_DIR" "$SHAPE"; then
      SHAPE_FAILED=true
    fi
    if [[ "$SHAPE_FAILED" == true ]]; then
      FAILED_SHAPES+=("$SHAPE:$RUN_DIR")
    fi
  done
fi

if (( ${#FAILED_SHAPES[@]} > 0 )); then
  mapfile -t FAILED_SHAPES < <(printf '%s\n' "${FAILED_SHAPES[@]}" | sort -u)
  echo "Incomplete shapes:" >&2
  printf '  %s\n' "${FAILED_SHAPES[@]}" >&2
  echo "Resume with the same command plus --resume." >&2
  exit 1
fi

for SHAPE in "${SHAPES[@]}"; do
  _validate_shape "${SHAPE_RUN_DIR[$SHAPE]}" "$SHAPE"
done

echo "=== Combined analysis ==="
ANALYSIS_PYTHON="$REPO_ROOT/.direnv/python-3.12.3/bin/python3"
[[ -x "$ANALYSIS_PYTHON" ]] || ANALYSIS_PYTHON=python3
SELECTED_RUN_DIRS=()
for SHAPE in "${SHAPES[@]}"; do
  SELECTED_RUN_DIRS+=("${SHAPE_RUN_DIR[$SHAPE]}")
done
"$ANALYSIS_PYTHON" collect-results.py --run-dirs "${SELECTED_RUN_DIRS[@]}" \
  --out "$OUTPUT_ROOT/combined-analysis.json"

echo "All requested shapes completed: ${SHAPES[*]}"
echo "Combined analysis: $OUTPUT_ROOT/combined-analysis.json"
