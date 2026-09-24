#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_paper_suite.sh --java PATH [options]

Run the original fixed-duration Mertz--Nunes workloads with FUM, ADP, UNI,
INV, and OmniFlow. The default benchmark set is all five applications.

Options:
  --java PATH                Java 8 executable (required)
  --benchmarks LIST          Comma-separated subset or all
                             (default: all)
  --benchmarks-jar PATH      Original benchmarks.jar
  --cassandra-jar PATH       Original cassandra.jar
  --cassandra-data-zip PATH  Original Cassandra data.zip
  --agent PATH               OmniFlow agent JAR
  --output DIR               Output directory (default: ./paper-suite-runs-v25)
  --reps N                   Repetitions (default: 1; final campaign: 10)
  --policies LIST            Must include adp,uni,full,inv,omni
                             (default: adp,uni,full,inv,omni)
  --include-nom              Add agent-free NOM as an internal diagnostic arm
  --uniform-rate P           UNI and initial ADP/INV probability (must be 0.5)
  --seed N                   Bernoulli seed (default: 1)
  --omni-signal MODE         Compatibility option; publication runs require heap_raw
  --cycle-length-ms N        ADP maximum cycle length (must be 180000)
  --resume                   Reuse only strictly validated complete artifacts
  --prune-workdirs           Remove each policy work directory after validation
  --reuse-adp-from DIR       Import validated ADP artifacts from an earlier
                             output, then resume the remaining arms

Xalan output handling:
  The released worker appends transformed HTML to six unbounded xalan.out.*
  files. Publication runs keep those files sparse by punching completed extents
  while retaining the newest 16 MiB per worker. The files are removed when the
  Java process exits; transformed output is not used by validation or metrics.

Publication invariants:
  * Java 8
  * 4 GiB maximum heap for every benchmark
  * original fixed-duration workload and concurrency settings
  * ADP runs first; its exact marker set is reused by all comparison policies
  * existing artifacts are never deleted or overwritten

The paper-facing summary intentionally omits throughput and overhead. Internal
run diagnostics are written separately to diagnostics.md and paper-summary.json.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
JAVA=""
BENCHMARKS="all"
BENCHMARKS_JAR="$ROOT/runtime/original-benchmarks.jar"
CASSANDRA_JAR="$ROOT/runtime/original-cassandra.jar"
CASSANDRA_DATA_ZIP="$ROOT/runtime/original-cassandra-data.zip"
AGENT="$ROOT/agent/omniflow-original-dacapo-agent.jar"
OUTPUT="$PWD/paper-suite-runs-v25"
REPS=1
POLICIES="adp,uni,full,inv,omni"
INCLUDE_NOM=false
UNIFORM_RATE=0.5
SEED=1
OMNI_SIGNAL=heap_raw
CYCLE_LENGTH_MS=180000
RESUME=false
PRUNE_WORKDIRS=false
REUSE_ADP_FROM=""
XALAN_OUTPUT_RETAIN_BYTES=$((16 * 1024 * 1024))
XALAN_OUTPUT_COMPACT_SECONDS=2
VALIDATOR="$SCRIPT_DIR/validate_paper.py"
EVALUATOR="$SCRIPT_DIR/evaluate_paper_suite.py"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --java) JAVA="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --benchmarks-jar|--dacapo-jar) BENCHMARKS_JAR="$2"; shift 2 ;;
    --cassandra-jar) CASSANDRA_JAR="$2"; shift 2 ;;
    --cassandra-data-zip) CASSANDRA_DATA_ZIP="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --include-nom) INCLUDE_NOM=true; shift ;;
    --uniform-rate) UNIFORM_RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --omni-signal) OMNI_SIGNAL="$2"; shift 2 ;;
    --cycle-length-ms) CYCLE_LENGTH_MS="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --prune-workdirs) PRUNE_WORKDIRS=true; shift ;;
    --reuse-adp-from) REUSE_ADP_FROM="$2"; RESUME=true; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$JAVA" ] || { echo "--java must point to Java 8" >&2; exit 2; }
command -v "$JAVA" >/dev/null 2>&1 || [ -x "$JAVA" ] || { echo "Java not found: $JAVA" >&2; exit 1; }
[ -f "$AGENT" ] || { echo "Missing agent: $AGENT" >&2; exit 1; }
[ -x "$VALIDATOR" ] && [ -x "$EVALUATOR" ] || { echo "Missing publication validator/evaluator" >&2; exit 1; }
JAVA_VERSION="$($JAVA -version 2>&1 | head -1)"
if ! echo "$JAVA_VERSION" | grep -Eq 'version "1\.8\.'; then
  echo "Publication mode requires Java 8; found: $JAVA_VERSION" >&2
  exit 64
fi
case "$REPS" in ''|*[!0-9]*) echo "--reps must be a positive integer" >&2; exit 2 ;; esac
[ "$REPS" -ge 1 ] || { echo "--reps must be at least 1" >&2; exit 2; }
case "$CYCLE_LENGTH_MS" in ''|*[!0-9]*) echo "--cycle-length-ms must be positive" >&2; exit 2 ;; esac
[ "$CYCLE_LENGTH_MS" -eq 180000 ] || { echo "Paper mode requires --cycle-length-ms 180000" >&2; exit 2; }
python3 - "$UNIFORM_RATE" "$SEED" "$OMNI_SIGNAL" <<'PY'
import math, sys
rate=float(sys.argv[1]); int(sys.argv[2]); signal=sys.argv[3]
if not math.isfinite(rate) or rate != 0.5:
    raise SystemExit("Paper mode requires --uniform-rate 0.5")
if signal != "heap_raw":
    raise SystemExit("publication runs require --omni-signal heap_raw; alternative signals are internal diagnostics only")
PY

BENCHMARKS_JAR="$(cd "$(dirname "$BENCHMARKS_JAR")" && pwd -P)/$(basename "$BENCHMARKS_JAR")"
CASSANDRA_JAR="$(cd "$(dirname "$CASSANDRA_JAR")" && pwd -P)/$(basename "$CASSANDRA_JAR")"
CASSANDRA_DATA_ZIP="$(cd "$(dirname "$CASSANDRA_DATA_ZIP")" && pwd -P)/$(basename "$CASSANDRA_DATA_ZIP")"
AGENT="$(cd "$(dirname "$AGENT")" && pwd -P)/$(basename "$AGENT")"
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd -P)"
if [ -n "$REUSE_ADP_FROM" ]; then
  [ -d "$REUSE_ADP_FROM" ] || { echo "Missing --reuse-adp-from directory: $REUSE_ADP_FROM" >&2; exit 1; }
  REUSE_ADP_FROM="$(cd "$REUSE_ADP_FROM" && pwd -P)"
fi

normalize_benchmarks() {
  python3 - "$BENCHMARKS" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent) if '__file__' in globals() else '.')
value=sys.argv[1].strip().lower()
order=['cassandra','h2','lusearch','tradebeans','xalan']
names=order if value in ('','all') else [x.strip() for x in value.split(',') if x.strip()]
unknown=sorted(set(names)-set(order))
if unknown: raise SystemExit('Unknown benchmark(s): '+', '.join(unknown))
print(' '.join(x for x in order if x in names))
PY
}
BENCH_ORDER="$(normalize_benchmarks)"
[ -n "$BENCH_ORDER" ] || { echo "No benchmarks requested" >&2; exit 2; }

verify_xalan_output_compaction() {
  case " $BENCH_ORDER " in *" xalan "*) ;; *) return 0 ;; esac
  [ "$(uname -s 2>/dev/null || true)" = Linux ] || {
    echo "Xalan publication runs require Linux fallocate hole punching to bound xalan.out.* disk use." >&2
    exit 64
  }
  command -v fallocate >/dev/null 2>&1 || {
    echo "Xalan publication runs require the util-linux fallocate command." >&2
    exit 64
  }
  local probe
  probe="$(mktemp "$OUTPUT/.xalan-hole-punch-test.XXXXXX")"
  if ! dd if=/dev/urandom of="$probe" bs=1048576 count=2 conv=fsync status=none 2>/dev/null       || ! fallocate --punch-hole --keep-size --offset 0 --length 1048576 "$probe" 2>/dev/null; then
    rm -f -- "$probe"
    echo "The output filesystem does not support fallocate hole punching required for bounded Xalan output." >&2
    exit 64
  fi
  rm -f -- "$probe"
}

verify_xalan_output_compaction

IFS=',' read -r -a REQUESTED_POLICIES <<<"$POLICIES"
POLICY_ORDER=""
for policy in adp uni full inv omni nom; do
  for requested in "${REQUESTED_POLICIES[@]}"; do
    requested="${requested//[[:space:]]/}"
    [ "$requested" = "$policy" ] && POLICY_ORDER="$POLICY_ORDER $policy"
  done
done
for requested in "${REQUESTED_POLICIES[@]}"; do
  requested="${requested//[[:space:]]/}"
  case "$requested" in adp|uni|full|inv|omni|nom) ;; *) echo "Unknown policy: $requested" >&2; exit 2 ;; esac
done
for required in adp uni full inv omni; do
  case " $POLICY_ORDER " in *" $required "*) ;; *) echo "Publication mode requires policy: $required" >&2; exit 2 ;; esac
done
if [ "$INCLUDE_NOM" = true ]; then
  case " $POLICY_ORDER " in *" nom "*) ;; *) POLICY_ORDER="$POLICY_ORDER nom" ;; esac
fi
POLICY_ORDER="${POLICY_ORDER# }"

COMMON_PROPS=(
  -Dsampling_rate=0.5
  -Dsampling_cycle_time=180000
  -Dsampling_adaptive=false
  -Dsampling_enabled=false
  -Dsampling_inversely=false
  -Dsampling_markers=257
)

benchmark_config() {
  case "$1" in
    cassandra) CFG_JAR="$CASSANDRA_JAR"; CFG_SIZE=default; CFG_THREADS=200; CFG_SECONDS=1660; CFG_HEAP=4g; CFG_JVM_EXTRA="" ;;
    h2) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=huge; CFG_THREADS=20; CFG_SECONDS=1780; CFG_HEAP=4g; CFG_JVM_EXTRA="" ;;
    lusearch) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=large; CFG_THREADS=6; CFG_SECONDS=1640; CFG_HEAP=4g; CFG_JVM_EXTRA="" ;;
    tradebeans) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=huge; CFG_THREADS=24; CFG_SECONDS=1780; CFG_HEAP=4g; CFG_JVM_EXTRA="-Djboss.modules.system.pkgs=org.dacapo.omniflow.agent" ;;
    xalan) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=large; CFG_THREADS=6; CFG_SECONDS=1640; CFG_HEAP=4g; CFG_JVM_EXTRA="" ;;
    *) echo "Unknown benchmark: $1" >&2; exit 2 ;;
  esac
}

verify_runtime() {
  local benchmark="$1"
  benchmark_config "$benchmark"
  [ -f "$CFG_JAR" ] || { echo "Missing original runtime for $benchmark: $CFG_JAR" >&2; exit 1; }
  local help sizes
  help="$($JAVA -jar "$CFG_JAR" --help 2>&1 || true)"
  echo "$help" | grep -q -- '--no-validation' || { echo "$CFG_JAR does not advertise --no-validation" >&2; exit 64; }
  sizes="$($JAVA -jar "$CFG_JAR" "$benchmark" --sizes 2>&1 || true)"
  echo "$sizes" | grep -Eq "(^|[[:space:]])$CFG_SIZE($|[[:space:]])" || {
    echo "$benchmark runtime lacks size $CFG_SIZE" >&2; echo "$sizes" >&2; exit 64;
  }
}

for benchmark in $BENCH_ORDER; do verify_runtime "$benchmark"; done
case " $BENCH_ORDER " in
  *" cassandra "*)
    [ -f "$CASSANDRA_DATA_ZIP" ] || { echo "Missing Cassandra data archive: $CASSANDRA_DATA_ZIP" >&2; exit 1; }
    python3 - "$CASSANDRA_DATA_ZIP" <<'PYZIP'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    if 'data/dat/cassandra/conf/cassandra.yaml' not in zf.namelist():
        raise SystemExit('Cassandra data archive lacks data/dat/cassandra/conf/cassandra.yaml')
PYZIP
    ;;
esac
case " $BENCH_ORDER " in
  *" cassandra "*)
    if [ "$(uname -s 2>/dev/null || true)" = Darwin ] && [ "$(uname -m 2>/dev/null || true)" = arm64 ]; then
      echo "Cassandra publication runs are blocked on Apple Silicon: the archived Cassandra 3.11.6 JNA library is x86-only." >&2
      echo "Use smoke_paper_suite.sh for development validation, and run the publication arm on the final Linux/x86_64 machine." >&2
      exit 64
    fi
    ;;
esac
TRADEBEANS_RELEASE_ARTIFACT=false
case " $BENCH_ORDER " in
  *" tradebeans "*)
    set +e
    "$SCRIPT_DIR/inspect_tradebeans_artifact.py" "$BENCHMARKS_JAR"
    tradebeans_rc=$?
    set -e
    if [ "$tradebeans_rc" -eq 66 ]; then
      TRADEBEANS_RELEASE_ARTIFACT=true
      echo "Tradebeans will run as 'Tradebeans (released artifact)': the exact companion-artifact TPCC substitute." >&2
      echo "This arm is included to avoid cherry-picking and is not described as DayTrader." >&2
    elif [ "$tradebeans_rc" -ne 0 ]; then
      exit "$tradebeans_rc"
    fi
    ;;
esac


sha256_file() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else sha256sum "$1" | awk '{print $1}'
  fi
}

ensure_metadata() {
  local benchmark="$1" repdir="$2"
  benchmark_config "$benchmark"
  mkdir -p "$repdir"
  local expected="$repdir/run-metadata.expected" actual="$repdir/run-metadata.txt"
  {
    printf 'schema=paper-suite-v25\n'
    printf 'benchmark=%s\n' "$benchmark"
    if [ "$benchmark" = tradebeans ]; then
      printf 'artifact_mode=released-tpcc-substitute
'
    elif [ "$benchmark" = xalan ]; then
      printf 'artifact_mode=released-source-17-inputs
'
    else
      printf 'artifact_mode=paper-runtime
'
    fi
    printf 'size=%s\n' "$CFG_SIZE"
    printf 'threads=%s\n' "$CFG_THREADS"
    printf 'nominal_seconds=%s\n' "$CFG_SECONDS"
    printf 'heap=%s\n' "$CFG_HEAP"
    printf 'uniform_rate=%s\n' "$UNIFORM_RATE"
    printf 'seed=%s\n' "$SEED"
    printf 'omni_signal=%s\n' "$OMNI_SIGNAL"
    printf 'cycle_length_millis=%s\n' "$CYCLE_LENGTH_MS"
    if [ "$benchmark" = xalan ]; then
      printf 'xalan_output_storage=bounded-hole-punch\n'
      printf 'xalan_output_retain_bytes=%s\n' "$XALAN_OUTPUT_RETAIN_BYTES"
      printf 'xalan_output_compact_seconds=%s\n' "$XALAN_OUTPUT_COMPACT_SECONDS"
    fi
    printf 'runtime_jar_sha256=%s\n' "$(sha256_file "$CFG_JAR")"
    printf 'agent_jar_sha256=%s\n' "$(sha256_file "$AGENT")"
    if [ "$benchmark" = cassandra ]; then printf 'cassandra_data_zip_sha256=%s\n' "$(sha256_file "$CASSANDRA_DATA_ZIP")"; fi
  } > "$expected"
  if [ -f "$actual" ]; then
    if ! cmp -s "$actual" "$expected"; then
      if [ "$benchmark" != xalan ] && python3 - "$actual" "$expected" <<'PYLEGACY'
import sys
from pathlib import Path

def read(path):
    out={}
    for line in Path(path).read_text().splitlines():
        if '=' in line:
            k,v=line.split('=',1); out[k]=v
    return out
actual=read(sys.argv[1]); expected=read(sys.argv[2])
legacy_agents={
    '763f59d5070926b6d177ad328ee2a6550d67f1c9aa40b0dcc95244ce8fdfb7aa',
    '25f062928f3ec13c7cc2c9e2db4e0b380a7cc779776c07f4be40b2875c2c05f8',
}
if actual.get('schema') not in {'paper-suite-v19','paper-suite-v20'}:
    raise SystemExit(1)
if actual.get('agent_jar_sha256') not in legacy_agents:
    raise SystemExit(1)
for key,value in expected.items():
    if key in {'schema','agent_jar_sha256'}:
        continue
    if actual.get(key) != value:
        raise SystemExit(1)
PYLEGACY
      then
        echo "[$benchmark] accepting strictly validated legacy v19/v20 metadata; the current measurement schema preserves the v21 Xalan identity/data fix"
      else
        echo "Run metadata differs from the requested publication configuration: $repdir" >&2
        diff -u "$actual" "$expected" >&2 || true
        rm -f "$expected"
        exit 65
      fi
    fi
    rm -f "$expected"
  else
    local existing found_existing=false
    for existing in "$repdir"/* "$repdir"/.[!.]* "$repdir"/..?*; do
      [ -e "$existing" ] || [ -L "$existing" ] || continue
      [ "$existing" = "$expected" ] && continue
      if [ "$benchmark" = xalan ]; then
        case "$(basename "$existing")" in
          rejected-xalan-missing-data-v20-*) continue ;;
        esac
      fi
      found_existing=true
      break
    done
    if [ "$found_existing" = true ]; then
      echo "Existing repetition artifacts lack compatible publication run metadata and cannot be resumed safely: $repdir" >&2
      rm -f "$expected"
      exit 65
    fi
    mv "$expected" "$actual"
  fi
}

has_any_artifact() {
  local base="$1" log="$2" work="$3"
  [ -e "$base" ] || [ -e "$base.summary.csv" ] || [ -e "$base.controller.csv" ] || [ -e "$base.telemetry.csv" ] \
    || [ -e "$base.cycles.csv" ] || [ -e "$base.markers.txt" ] \
    || [ -e "$log" ] || [ -e "$work" ]
}

validate_policy() {
  local benchmark="$1" policy="$2" repdir="$3" reference="${4:-}"
  local log="$repdir/$policy.log" base="$repdir/$policy.csv"
  local args=(--benchmark "$benchmark" --policy "$policy" --log "$log" --uniform-rate "$UNIFORM_RATE")
  if [ "$policy" != nom ]; then
    args+=(--summary "$base.summary.csv" --telemetry "$base.telemetry.csv" --cycles "$base.cycles.csv" --markers "$base.markers.txt")
    [ -n "$reference" ] && args+=(--reference-markers "$reference")
  fi
  "$VALIDATOR" "${args[@]}"
}

seed_adp_from_previous_output() {
  local benchmark="$1" rep="$2" target_repdir="$3"
  [ -n "$REUSE_ADP_FROM" ] || return 0
  local source_repdir="$REUSE_ADP_FROM/$benchmark/rep-$(printf '%02d' "$rep")"
  [ -d "$source_repdir" ] || { echo "Missing ADP source repetition: $source_repdir" >&2; exit 1; }
  [ -f "$source_repdir/run-metadata.txt" ] || {
    echo "ADP reuse source lacks compatible publication run metadata and cannot be imported safely: $source_repdir" >&2
    exit 65
  }
  if ! cmp -s "$source_repdir/run-metadata.txt" "$target_repdir/run-metadata.txt"; then
    echo "ADP reuse metadata does not match the target publication configuration." >&2
    diff -u "$source_repdir/run-metadata.txt" "$target_repdir/run-metadata.txt" >&2 || true
    exit 65
  fi
  validate_policy "$benchmark" adp "$source_repdir"
  local target_base="$target_repdir/adp.csv" target_log="$target_repdir/adp.log" target_work="$target_repdir/adp-work"
  if has_any_artifact "$target_base" "$target_log" "$target_work"; then
    echo "ADP target already contains artifacts; not importing: $target_repdir" >&2
    return 0
  fi
  mkdir -p "$target_repdir"
  local name
  for name in adp.log adp.csv.summary.csv adp.csv.telemetry.csv adp.csv.cycles.csv adp.csv.markers.txt; do
    cp -p "$source_repdir/$name" "$target_repdir/$name"
  done
  {
    printf 'source_output=%s\n' "$REUSE_ADP_FROM"
    printf 'source_rep=%s\n' "$source_repdir"
    printf 'imported_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$target_repdir/adp.reused-from.txt"
  echo "[$benchmark][adp] imported validated ADP artifacts from: $source_repdir"
}

xalan_compact_outputs_once() {
  local work="$1" file size punch
  local block=4096
  shopt -s nullglob
  for file in "$work"/xalan.out.*; do
    [ -f "$file" ] || continue
    size="$(stat -c %s -- "$file")" || return 1
    if [ "$size" -gt "$XALAN_OUTPUT_RETAIN_BYTES" ]; then
      punch=$((size - XALAN_OUTPUT_RETAIN_BYTES))
      punch=$((punch / block * block))
      if [ "$punch" -gt 0 ]; then
        fallocate --punch-hole --keep-size --offset 0 --length "$punch" "$file" || return 1
      fi
    fi
  done
  shopt -u nullglob
}

xalan_compactor_loop() {
  local work="$1" failure_marker="$2" java_pid="$3"
  trap 'exit 0' TERM INT
  while sleep "$XALAN_OUTPUT_COMPACT_SECONDS"; do
    if ! xalan_compact_outputs_once "$work"; then
      : > "$failure_marker"
      echo "Xalan output compaction failed; terminating the Java process." >&2
      kill -TERM "$java_pid" 2>/dev/null || true
      exit 1
    fi
  done
}

stop_xalan_compactor() {
  local pid="${1:-}" work="$2"
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f -- "$work"/xalan.out.*
}

prune_workdir_if_requested() {
  local benchmark="$1" policy="$2" work="$3"
  [ "$PRUNE_WORKDIRS" = true ] || return 0
  [ -d "$work" ] || return 0
  local size_kb
  size_kb="$(du -sk "$work" 2>/dev/null | awk '{print $1}')"
  rm -rf -- "$work"
  echo "[$benchmark][$policy] pruned validated work directory (${size_kb:-0} KiB): $work"
}

run_policy() {
  local benchmark="$1" policy="$2" repdir="$3" markers="$4"
  benchmark_config "$benchmark"
  local base="$repdir/$policy.csv"
  local log="$repdir/$policy.log"
  local work="$repdir/$policy-work"
  local scratch="$work/scratch"
  local reference=""
  [ "$policy" != adp ] && [ "$policy" != nom ] && reference="$markers"

  if has_any_artifact "$base" "$log" "$work"; then
    if [ "$RESUME" = true ]; then
      echo "[$benchmark][$policy] validating existing artifacts"
      validate_policy "$benchmark" "$policy" "$repdir" "$reference"
      prune_workdir_if_requested "$benchmark" "$policy" "$work"
      echo "[$benchmark][$policy] reusing verified completed measurement"
      return
    fi
    echo "Refusing to overwrite existing artifacts for $benchmark/$policy under $repdir" >&2
    echo "Use --resume or a new --output directory." >&2
    exit 73
  fi

  mkdir -p "$work"
  if [ "$benchmark" = cassandra ]; then
    "$SCRIPT_DIR/extract_cassandra_data.sh" "$CASSANDRA_DATA_ZIP" "$work"
  elif [ "$benchmark" = lusearch ]; then
    "$SCRIPT_DIR/extract_lusearch_data.sh" "$BENCHMARKS_JAR" "$work"
  elif [ "$benchmark" = xalan ]; then
    "$SCRIPT_DIR/extract_xalan_data.sh" "$BENCHMARKS_JAR" "$work"
  fi
  mkdir -p "$scratch"
  echo "[$benchmark][$policy] starting nominal ${CFG_SECONDS}-second original schedule"
  echo "[$benchmark][$policy] size=$CFG_SIZE threads=$CFG_THREADS heap=$CFG_HEAP"
  echo "[$benchmark][$policy] log: $log"
  local -a command=("$JAVA" "${COMMON_PROPS[@]}" "-Xmx$CFG_HEAP")
  [ -n "$CFG_JVM_EXTRA" ] && command+=("$CFG_JVM_EXTRA")
  if [ "$policy" != nom ]; then
    local agent_args="output=$base,policy=$policy,write_rows=false,disable_original_sampler=true,skip_h2_reset=false,exact_benchmark_seconds=true,final_marker_second=$CFG_SECONDS,uniform_rate=$UNIFORM_RATE,seed=$SEED,omni_signal=$OMNI_SIGNAL,cycle_length_millis=$CYCLE_LENGTH_MS,telemetry_console=false,verbose=true"
    [ -n "$reference" ] && agent_args+=",cycle_markers_file=$reference"
    command+=("-javaagent:$AGENT=$agent_args")
  fi
  command+=(-jar "$CFG_JAR" "$benchmark" -v --no-validation --size "$CFG_SIZE" -t "$CFG_THREADS" --scratch-directory "$scratch" --preserve)

  local xalan_compactor_pid="" xalan_java_pid=""
  local xalan_compactor_failure="$work/.xalan-output-compactor.failed"
  local rc
  : > "$log"
  if [ "$benchmark" = xalan ]; then
    rm -f -- "$xalan_compactor_failure" "$work"/xalan.out.*
    set +e
    (cd "$work" && exec "${command[@]}") >>"$log" 2>&1 &
    xalan_java_pid=$!
    xalan_compactor_loop "$work" "$xalan_compactor_failure" "$xalan_java_pid" >>"$log" 2>&1 &
    xalan_compactor_pid=$!
    trap 'kill -TERM "${xalan_java_pid:-}" 2>/dev/null || true; stop_xalan_compactor "${xalan_compactor_pid:-}" "$work"; wait "${xalan_java_pid:-}" 2>/dev/null || true; exit 130' INT
    trap 'kill -TERM "${xalan_java_pid:-}" 2>/dev/null || true; stop_xalan_compactor "${xalan_compactor_pid:-}" "$work"; wait "${xalan_java_pid:-}" 2>/dev/null || true; exit 143' TERM
    wait "$xalan_java_pid"
    rc=$?
    set -e
    stop_xalan_compactor "$xalan_compactor_pid" "$work"
    xalan_compactor_pid=""
    xalan_java_pid=""
    trap - INT TERM
    if [ -f "$xalan_compactor_failure" ]; then
      echo "[$benchmark][$policy] output compaction failed; artifacts preserved for diagnosis." >&2
      rc=74
    fi
    rm -f -- "$xalan_compactor_failure"
  else
    set +e
    (cd "$work" && "${command[@]}") >>"$log" 2>&1
    rc=$?
    set -e
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[$benchmark][$policy] Java exited with code $rc; artifacts preserved." >&2
    tail -100 "$log" >&2 || true
    exit "$rc"
  fi
  validate_policy "$benchmark" "$policy" "$repdir" "$reference"
  prune_workdir_if_requested "$benchmark" "$policy" "$work"
  echo "[$benchmark][$policy] finished"
}

printf 'Original benchmarks JAR: %s\n' "$BENCHMARKS_JAR"
printf 'Original Cassandra JAR: %s\n' "$CASSANDRA_JAR"
printf 'Cassandra data archive: %s\n' "$CASSANDRA_DATA_ZIP"
printf 'Java 8:                %s\n' "$JAVA_VERSION"
printf 'Output directory:      %s\n' "$OUTPUT"
printf 'Benchmarks:            %s\n' "$BENCH_ORDER"
printf 'Policies:              %s\n' "$POLICY_ORDER"
printf 'Repetitions:           %s\n' "$REPS"
printf 'Publication heap:      4g\n'
printf 'OmniFlow signal:       %s\n' "$OMNI_SIGNAL"

for benchmark in $BENCH_ORDER; do
  benchmark_config "$benchmark"
  for rep in $(seq 1 "$REPS"); do
    repdir="$OUTPUT/$benchmark/rep-$(printf '%02d' "$rep")"
    ensure_metadata "$benchmark" "$repdir"
    seed_adp_from_previous_output "$benchmark" "$rep" "$repdir"
    markers="$repdir/adp.csv.markers.txt"
    for policy in $POLICY_ORDER; do
      if [ "$policy" != adp ] && [ "$policy" != nom ] && [ ! -s "$markers" ]; then
        echo "ADP markers are unavailable before $benchmark/$policy: $markers" >&2; exit 1
      fi
      run_policy "$benchmark" "$policy" "$repdir" "$markers"
    done
  done
done

SUMMARY_BENCHMARKS=""
for benchmark in cassandra h2 lusearch tradebeans xalan; do
  if [ -d "$OUTPUT/$benchmark" ]; then
    [ -n "$SUMMARY_BENCHMARKS" ] && SUMMARY_BENCHMARKS+=","
    SUMMARY_BENCHMARKS+="$benchmark"
  fi
done
"$EVALUATOR" "$OUTPUT" --benchmarks "$SUMMARY_BENCHMARKS" \
  --details "$OUTPUT/paper-details.csv" \
  --json "$OUTPUT/paper-summary.json" \
  --diagnostics-md "$OUTPUT/diagnostics.md" | tee "$OUTPUT/paper-summary.md"
echo "Complete publication-suite results preserved at: $OUTPUT"
