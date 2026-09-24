#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  smoke_paper_suite.sh --java PATH [options]

Options:
  --java PATH                    Java executable; Java 8 is required by default
  --benchmarks LIST              Comma-separated subset or all (default: all)
  --benchmarks-jar PATH          Original benchmarks.jar
  --cassandra-jar PATH           Original cassandra.jar
  --cassandra-data-zip PATH      Original Cassandra data.zip
  --seconds N                    Active-workload seconds per policy (default: 30)
  --startup-timeout N            Maximum startup wait (default: 600)
  --smoke-cycle-length-ms N      Smoke-only ADP timeout (default: 10000)
  --output DIR                   Fresh output directory
  --uniform-rate P               Default: 0.5
  --seed N                       Default: 1
  --include-nom                  Also test agent-free NOM
  --allow-tradebeans-substitute  Retained compatibility flag; the released TPCC-backed
                                 Tradebeans arm runs by default
  --allow-non-java8              Development-only escape hatch

The active timer starts only after `Running per second:`.  Smoke mode uses a
short ADP cycle timeout so Lusearch can exercise marker propagation in a
30-second run.  Publication runs always retain the paper value of 180000 ms.

Xalan's monitored request is one whole transform, keyed by its stable XML
input filename. The released source queues 17 XML inputs, all of which are
required in smoke and publication validation. The paper reports 16 types; this
artifact discrepancy is disclosed in README.md.

Tradebeans smoke uses the exact released companion-artifact arm. It is labelled
as TPCC-backed and is not represented as a DayTrader semantic reproduction.

On Apple Silicon, Cassandra 3.11.6 ships an x86-only Darwin JNA library. Smoke
mode applies narrowly scoped development-only fallbacks. Publication runs remain
unmodified and must run on the final Linux/x86_64 experiment machine.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
JAVA=""
BENCHMARKS=all
BENCHMARKS_JAR="$ROOT/runtime/original-benchmarks.jar"
CASSANDRA_JAR="$ROOT/runtime/original-cassandra.jar"
CASSANDRA_DATA_ZIP="$ROOT/runtime/original-cassandra-data.zip"
AGENT="$ROOT/agent/omniflow-original-dacapo-agent.jar"
SECONDS_LIMIT=30
STARTUP_TIMEOUT=600
SMOKE_CYCLE_MS=10000
UNIFORM_RATE=0.5
SEED=1
INCLUDE_NOM=false
ALLOW_NON_JAVA8=false
ALLOW_TRADEBEANS_SUBSTITUTE=true
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT="$ROOT/smoke-output/paper-suite-v25-$STAMP"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --java) JAVA="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --benchmarks-jar|--dacapo-jar) BENCHMARKS_JAR="$2"; shift 2 ;;
    --cassandra-jar) CASSANDRA_JAR="$2"; shift 2 ;;
    --cassandra-data-zip) CASSANDRA_DATA_ZIP="$2"; shift 2 ;;
    --seconds) SECONDS_LIMIT="$2"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
    --smoke-cycle-length-ms) SMOKE_CYCLE_MS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --uniform-rate) UNIFORM_RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --include-nom) INCLUDE_NOM=true; shift ;;
    --allow-tradebeans-substitute) ALLOW_TRADEBEANS_SUBSTITUTE=true; shift ;;
    --allow-non-java8) ALLOW_NON_JAVA8=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$JAVA" ] || { echo "--java is required" >&2; exit 2; }
command -v "$JAVA" >/dev/null 2>&1 || [ -x "$JAVA" ] || { echo "Java not found: $JAVA" >&2; exit 1; }
JAVA_VERSION="$("$JAVA" -version 2>&1 | head -1)"
if ! echo "$JAVA_VERSION" | grep -Eq 'version "1\.8\.' && [ "$ALLOW_NON_JAVA8" != true ]; then
  echo "Paper-runtime smoke requires Java 8; found: $JAVA_VERSION" >&2; exit 64
fi
[ -f "$AGENT" ] || { echo "Missing agent: $AGENT" >&2; exit 1; }
case "$SECONDS_LIMIT" in ''|*[!0-9]*) echo "--seconds must be an integer" >&2; exit 2 ;; esac
[ "$SECONDS_LIMIT" -ge 5 ] || { echo "--seconds must be at least 5" >&2; exit 2; }
case "$STARTUP_TIMEOUT" in ''|*[!0-9]*) echo "--startup-timeout must be an integer" >&2; exit 2 ;; esac
[ "$STARTUP_TIMEOUT" -ge 30 ] || { echo "--startup-timeout must be at least 30" >&2; exit 2; }
case "$SMOKE_CYCLE_MS" in ''|*[!0-9]*) echo "--smoke-cycle-length-ms must be an integer" >&2; exit 2 ;; esac
[ "$SMOKE_CYCLE_MS" -ge 1000 ] || { echo "--smoke-cycle-length-ms must be at least 1000" >&2; exit 2; }
[ "$SMOKE_CYCLE_MS" -lt $((SECONDS_LIMIT * 1000)) ] || {
  echo "--smoke-cycle-length-ms must be shorter than the active smoke interval" >&2; exit 2;
}
[ ! -e "$OUTPUT" ] || { echo "Refusing existing output: $OUTPUT" >&2; exit 73; }
mkdir -p "$OUTPUT"

CASSANDRA_ARM64_DEV=false
if [ "$(uname -s 2>/dev/null || true)" = Darwin ] && [ "$(uname -m 2>/dev/null || true)" = arm64 ]; then
  CASSANDRA_ARM64_DEV=true
fi

BENCHMARKS_LOWER="$(printf %s "$BENCHMARKS" | tr '[:upper:]' '[:lower:]')"
case "$BENCHMARKS_LOWER" in all|'') BENCH_ORDER="cassandra h2 lusearch tradebeans xalan" ;;
*)
  BENCH_ORDER=""; OLDIFS="$IFS"; IFS=','
  for b in $BENCHMARKS; do
    b="$(echo "$b" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    case "$b" in cassandra|h2|lusearch|tradebeans|xalan) BENCH_ORDER="$BENCH_ORDER $b" ;; *) echo "Unknown benchmark: $b" >&2; exit 2 ;; esac
  done
  IFS="$OLDIFS"; BENCH_ORDER="${BENCH_ORDER# }" ;;
esac

TRADEBEANS_SUBSTITUTE=false
case " $BENCH_ORDER " in
  *" tradebeans "*)
    set +e
    "$SCRIPT_DIR/inspect_tradebeans_artifact.py" "$BENCHMARKS_JAR" >"$OUTPUT/tradebeans-artifact.txt" 2>&1
    tb_rc=$?
    set -e
    if [ "$tb_rc" -eq 66 ] && [ "$ALLOW_TRADEBEANS_SUBSTITUTE" = true ]; then
      TRADEBEANS_SUBSTITUTE=true
      echo "Tradebeans released-artifact mode: TPCC substitute included and labelled."
    elif [ "$tb_rc" -ne 0 ]; then
      cat "$OUTPUT/tradebeans-artifact.txt" >&2
      echo "Tradebeans artifact inspection failed unexpectedly; see README.md." >&2
      exit "$tb_rc"
    fi
    ;;
esac


process_running() {
  local pid="$1" state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$pid" 2>/dev/null | awk 'NR==1 {print $1}')"
  [ -n "$state" ] || return 1
  case "$state" in Z*|*Z*) return 1 ;; esac
  return 0
}

benchmark_config() {
  case "$1" in
    cassandra) CFG_JAR="$CASSANDRA_JAR"; CFG_SIZE=small; CFG_THREADS=200; CFG_CLASS='site/ycsb/workloads/CoreWorkload'; CFG_EXTRA="" ;;
    h2) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=small; CFG_THREADS=20; CFG_CLASS='org/dacapo/h2/TPCCSubmitter'; CFG_EXTRA="" ;;
    lusearch) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=small; CFG_THREADS=6; CFG_CLASS='org/dacapo/lusearch/QueryProcessor'; CFG_EXTRA="" ;;
    tradebeans)
      CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=small; CFG_THREADS=24; CFG_EXTRA='-Djboss.modules.system.pkgs=org.dacapo.omniflow.agent'
      if [ "$TRADEBEANS_SUBSTITUTE" = true ]; then CFG_CLASS='org/dacapo/h2/TPCCSubmitter'; else CFG_CLASS='org/apache/geronimo/daytrader/javaee6/dacapo/DaCapoTrader'; fi
      ;;
    xalan) CFG_JAR="$BENCHMARKS_JAR"; CFG_SIZE=small; CFG_THREADS=6; CFG_CLASS='org/dacapo/xalan/XalanWorker'; CFG_EXTRA="" ;;
  esac
}

run_one() {
  local benchmark="$1" policy="$2" markers="$3"
  benchmark_config "$benchmark"
  local dir="$OUTPUT/$benchmark"
  local work="$dir/$policy-work"
  local scratch="$work/scratch"
  local base="$dir/$policy.csv"
  local log="$dir/$policy.log"
  mkdir -p "$work"
  if [ "$benchmark" = cassandra ]; then
    "$SCRIPT_DIR/extract_cassandra_data.sh" "$CASSANDRA_DATA_ZIP" "$work"
  elif [ "$benchmark" = lusearch ]; then
    "$SCRIPT_DIR/extract_lusearch_data.sh" "$BENCHMARKS_JAR" "$work"
  elif [ "$benchmark" = xalan ]; then
    "$SCRIPT_DIR/extract_xalan_data.sh" "$BENCHMARKS_JAR" "$work"
  fi
  mkdir -p "$scratch"
  local -a command=("$JAVA" -Dsampling_rate=0.5 "-Dsampling_cycle_time=$SMOKE_CYCLE_MS" -Dsampling_adaptive=false -Dsampling_enabled=false -Dsampling_inversely=false -Dsampling_markers=257 -Xmx4g)
  [ -n "$CFG_EXTRA" ] && command+=("$CFG_EXTRA")
  if [ "$policy" != nom ]; then
    local args="output=$base,policy=$policy,write_rows=false,disable_original_sampler=true,skip_h2_reset=false,exact_benchmark_seconds=true,uniform_rate=$UNIFORM_RATE,seed=$SEED,omni_signal=heap_raw,cycle_length_millis=$SMOKE_CYCLE_MS,telemetry_console=false,verbose=true"
    [ -n "$markers" ] && args+=",cycle_markers_file=$markers"
    if [ "$benchmark" = cassandra ] && [ "$CASSANDRA_ARM64_DEV" = true ]; then
      args+=",skip_cassandra_native_check=true"
      echo "cassandra/$policy: enabling Apple-Silicon development-only native startup bypass"
    fi
    command+=("-javaagent:$AGENT=$args")
  fi
  command+=(-jar "$CFG_JAR" "$benchmark" -v --no-validation --size "$CFG_SIZE" -t "$CFG_THREADS" --scratch-directory "$scratch" --preserve)
  if [ "$benchmark" = cassandra ] && [ "$policy" = nom ] && [ "$CASSANDRA_ARM64_DEV" = true ]; then
    echo "Cassandra NOM cannot run on Apple Silicon with the archived x86-only JNA library." >&2
    exit 64
  fi
  local pid startup_started active_started rc elapsed waited
  (cd "$work" && exec "${command[@]}") >"$log" 2>&1 & pid=$!
  startup_started="$(date +%s)"
  while ! grep -q 'Running per second:' "$log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e; wait "$pid"; rc=$?; set -e
      echo "$benchmark/$policy exited before entering the fixed workload (rc=$rc)" >&2
      tail -100 "$log" >&2 || true; exit 1
    fi
    if [ $(( $(date +%s) - startup_started )) -ge "$STARTUP_TIMEOUT" ]; then
      kill -TERM "$pid" 2>/dev/null || true; sleep 2; kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
      echo "$benchmark/$policy did not enter the workload within ${STARTUP_TIMEOUT}s" >&2
      tail -100 "$log" >&2 || true; exit 1
    fi
    sleep 1
  done
  active_started="$(date +%s)"
  while [ $(( $(date +%s) - active_started )) -lt "$SECONDS_LIMIT" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e; wait "$pid"; rc=$?; set -e
      echo "$benchmark/$policy ended before ${SECONDS_LIMIT}s of active workload (rc=$rc)" >&2
      tail -100 "$log" >&2 || true; exit 1
    fi
    sleep 1
  done
  kill -TERM "$pid" 2>/dev/null || true
  waited=0
  while process_running "$pid" && [ "$waited" -lt 15 ]; do sleep 1; waited=$((waited+1)); done
  if process_running "$pid"; then kill -KILL "$pid" 2>/dev/null || true; fi
  set +e; wait "$pid"; rc=$?; set -e
  elapsed=$(( $(date +%s) - active_started ))
  [ "$elapsed" -ge "$SECONDS_LIMIT" ] || { echo "$benchmark/$policy active interval was too short: ${elapsed}s" >&2; exit 1; }
  if [ "$policy" != nom ]; then
    grep -q '\[OmniFlowAgent\] exact benchmark per-second hook enabled' "$log" || { echo "$benchmark/$policy lacks exact hook" >&2; exit 1; }
    grep -q "instrument $CFG_CLASS" "$log" || { echo "$benchmark/$policy lacks request instrumentation for $CFG_CLASS" >&2; exit 1; }
    if [ "$benchmark" = cassandra ] && [ "$CASSANDRA_ARM64_DEV" = true ]; then
      grep -q 'Cassandra native-library startup check bypass enabled (development only)' "$log" || {
        echo "$benchmark/$policy lacks the development-only startup-check marker" >&2; exit 1;
      }
      grep -q 'Cassandra native PID fallback enabled (development only)' "$log" || {
        echo "$benchmark/$policy lacks the development-only native-PID fallback marker" >&2; exit 1;
      }
    fi
    test -s "$base.summary.csv"; test -s "$base.telemetry.csv"; test -s "$base.cycles.csv"; test -e "$base.markers.txt"
    if [ "$benchmark" = xalan ]; then
      if grep -Eq 'FileNotFoundException: .*dist/dat/xalan|XalanWorker.*(NullPointerException|Exception)' "$log"; then
        echo "$benchmark/$policy emitted missing-data or worker errors" >&2
        exit 1
      fi
    fi
  fi
  echo "$benchmark/$policy: active workload sampled for ${SECONDS_LIMIT}s (process rc=$rc)"
}

for benchmark in $BENCH_ORDER; do
  benchmark_config "$benchmark"
  [ -f "$CFG_JAR" ] || { echo "Missing runtime for $benchmark: $CFG_JAR" >&2; exit 1; }
  if [ "$benchmark" = cassandra ]; then
    [ -f "$CASSANDRA_DATA_ZIP" ] || { echo "Missing Cassandra data archive: $CASSANDRA_DATA_ZIP" >&2; exit 1; }
  fi
  mkdir -p "$OUTPUT/$benchmark"
  run_one "$benchmark" adp ""
  MARKERS="$OUTPUT/$benchmark/adp.csv.markers.txt"
  [ -s "$MARKERS" ] || { echo "$benchmark ADP emitted no smoke markers even with ${SMOKE_CYCLE_MS}ms timeout" >&2; exit 1; }
  for policy in full uni inv omni; do run_one "$benchmark" "$policy" "$MARKERS"; done
  [ "$INCLUDE_NOM" = true ] && run_one "$benchmark" nom ""
done

python3 - "$OUTPUT" "$BENCH_ORDER" "$UNIFORM_RATE" "$TRADEBEANS_SUBSTITUTE" <<'PY'
import csv, pathlib, sys
root=pathlib.Path(sys.argv[1]); benchmarks=sys.argv[2].split(); target=float(sys.argv[3]); tb_sub=sys.argv[4]=='true'
for benchmark in benchmarks:
    directory=root/benchmark
    def cycles(policy): return list(csv.DictReader((directory/f'{policy}.csv.cycles.csv').open()))
    adp=cycles('adp'); ref_all=[int(r['marker_second']) for r in adp]
    ref_eval=[int(r['marker_second']) for r in adp if r['final_cycle'].lower()!='true']
    for policy in ('adp','full','uni','inv','omni'):
        rows=list(csv.DictReader((directory/f'{policy}.csv.summary.csv').open()))
        typed=[r for r in rows if r['request_type']!='__overall__']; overall=next(r for r in rows if r['request_type']=='__overall__')
        total=int(overall['total_requests']); selected=int(overall['selected_requests'])
        if total <= 0: raise SystemExit(f'{benchmark}/{policy}: no completed monitored requests')
        if benchmark=='xalan' and len(typed) != 17:
            raise SystemExit(f'{benchmark}/{policy}: expected 17 released-source XML input types, found {len(typed)}')
        telemetry=list(csv.DictReader((directory/f'{policy}.csv.telemetry.csv').open()))
        if len(telemetry)<2: raise SystemExit(f'{benchmark}/{policy}: fewer than two exact buckets')
        secs=[int(r['second']) for r in telemetry]
        if secs[0]!=0: raise SystemExit(f'{benchmark}/{policy}: first exact second is not zero')
        if policy=='full' and selected!=total: raise SystemExit(f'{benchmark}: full is not dense')
        if policy=='uni':
            if total >= 100:
                if abs(selected/total-target)>0.08: raise SystemExit(f'{benchmark}: UNI ratio outside smoke tolerance')
            else:
                print(f'{benchmark}/uni: ratio assertion skipped because only {total} monitored requests completed')
        if policy!='adp':
            markers=[int(r['marker_second']) for r in cycles(policy) if r['final_cycle'].lower()!='true']
            unexpected=[m for m in markers if m not in ref_all]
            if unexpected: raise SystemExit(f'{benchmark}/{policy}: non-ADP markers {unexpected}')
            expected=[m for m in ref_eval if m<=max(secs)]; comparable=[m for m in markers if m in ref_eval]
            if comparable!=expected: raise SystemExit(f'{benchmark}/{policy}: marker prefix differs from ADP')
        print(f'{benchmark}/{policy}: types={len(typed)} total={total} selected={selected} ratio={selected/total:.3%} exact_buckets={len(telemetry)}')
    if benchmark=='tradebeans' and tb_sub:
        print('tradebeans-release: exact companion-artifact TPCC arm; included and explicitly not claimed as DayTrader')
    if benchmark=='xalan':
        print('xalan-release: stable XML filename identity; released source queues 17 inputs while the paper reports 16')
print('Paper-suite smoke test passed.')
PY

echo "Outputs preserved at: $OUTPUT"
