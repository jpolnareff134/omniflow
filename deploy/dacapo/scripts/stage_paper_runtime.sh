#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  stage_paper_runtime.sh \
    --benchmarks-jar /path/to/benchmarks.jar \
    --cassandra-jar /path/to/cassandra.jar \
    --cassandra-data-zip /path/to/data.zip \
    [--java /path/to/java8] [--force]

Stages the original experiment artifacts as symbolic links:
  runtime/original-benchmarks.jar
  runtime/original-cassandra.jar
  runtime/original-cassandra-data.zip

The sources are never changed or deleted. Re-running with the same sources is
idempotent. A different existing target is preserved unless --force is used;
with --force it is moved to a timestamped backup first.

The Tradebeans binary is inspected after staging. The released companion artifact
is a TPCC-backed substitute rather than the paper's described DayTrader workload.
The suite runs it as a clearly labelled "Tradebeans (released artifact)" arm so the
benchmark is not cherry-picked out of the comparison.

If the agent JAR is missing or older than its sources, staging rebuilds it. The
build is self-contained when agent/lib is present and can also recover ASM from
the staged Cassandra JAR after a source-only remote sync.

--dacapo-jar remains accepted as an alias for --benchmarks-jar.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
BENCHMARKS_SOURCE=""
CASSANDRA_SOURCE=""
CASSANDRA_DATA_SOURCE=""
JAVA=java
FORCE=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --benchmarks-jar|--dacapo-jar) BENCHMARKS_SOURCE="$2"; shift 2 ;;
    --cassandra-jar) CASSANDRA_SOURCE="$2"; shift 2 ;;
    --cassandra-data-zip) CASSANDRA_DATA_SOURCE="$2"; shift 2 ;;
    --java) JAVA="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$BENCHMARKS_SOURCE" ] || { echo "--benchmarks-jar is required" >&2; exit 2; }
[ -n "$CASSANDRA_SOURCE" ] || { echo "--cassandra-jar is required" >&2; exit 2; }
[ -n "$CASSANDRA_DATA_SOURCE" ] || { echo "--cassandra-data-zip is required" >&2; exit 2; }
[ -f "$BENCHMARKS_SOURCE" ] || { echo "Missing benchmarks.jar: $BENCHMARKS_SOURCE" >&2; exit 1; }
[ -f "$CASSANDRA_SOURCE" ] || { echo "Missing cassandra.jar: $CASSANDRA_SOURCE" >&2; exit 1; }
[ -f "$CASSANDRA_DATA_SOURCE" ] || { echo "Missing Cassandra data.zip: $CASSANDRA_DATA_SOURCE" >&2; exit 1; }
command -v "$JAVA" >/dev/null 2>&1 || [ -x "$JAVA" ] || { echo "Java executable not found: $JAVA" >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 1; }
BENCHMARKS_SOURCE="$(cd "$(dirname "$BENCHMARKS_SOURCE")" && pwd -P)/$(basename "$BENCHMARKS_SOURCE")"
CASSANDRA_SOURCE="$(cd "$(dirname "$CASSANDRA_SOURCE")" && pwd -P)/$(basename "$CASSANDRA_SOURCE")"
CASSANDRA_DATA_SOURCE="$(cd "$(dirname "$CASSANDRA_DATA_SOURCE")" && pwd -P)/$(basename "$CASSANDRA_DATA_SOURCE")"
mkdir -p "$ROOT/runtime"

verify_size() {
  local jar="$1" benchmark="$2" size="$3"
  local help sizes
  help="$("$JAVA" -jar "$jar" --help 2>&1 || true)"
  echo "$help" | grep -q -- '--no-validation' || { echo "$jar does not advertise --no-validation" >&2; exit 64; }
  sizes="$("$JAVA" -jar "$jar" "$benchmark" --sizes 2>&1 || true)"
  echo "$sizes" | grep -Eq "(^|[[:space:]])$size($|[[:space:]])" || {
    echo "$jar does not expose $benchmark size $size" >&2; echo "$sizes" >&2; exit 64;
  }
}
verify_size "$BENCHMARKS_SOURCE" h2 huge
verify_size "$BENCHMARKS_SOURCE" lusearch large
verify_size "$BENCHMARKS_SOURCE" tradebeans huge
verify_size "$BENCHMARKS_SOURCE" xalan large
python3 - "$BENCHMARKS_SOURCE" <<'PYXALAN'
import sys, zipfile
required = {
    'dist/dat/xalan/xmlspec.xsl', 'dist/dat/xalan/acks.xml',
    'dist/dat/xalan/binding.xml', 'dist/dat/xalan/changes.xml',
    'dist/dat/xalan/concepts.xml', 'dist/dat/xalan/controls.xml',
    'dist/dat/xalan/datatypes.xml', 'dist/dat/xalan/expr.xml',
    'dist/dat/xalan/intro.xml', 'dist/dat/xalan/model.xml',
    'dist/dat/xalan/prod-notes.xml', 'dist/dat/xalan/references.xml',
    'dist/dat/xalan/rpm.xml', 'dist/dat/xalan/schema.xml',
    'dist/dat/xalan/structure.xml', 'dist/dat/xalan/template.xml',
    'dist/dat/xalan/terms.xml', 'dist/dat/xalan/ui.xml',
}
with zipfile.ZipFile(sys.argv[1]) as zf:
    missing = sorted(required - set(zf.namelist()))
if missing:
    raise SystemExit('benchmarks.jar lacks required Xalan data: ' + ', '.join(missing))
PYXALAN
verify_size "$CASSANDRA_SOURCE" cassandra default
python3 - "$CASSANDRA_DATA_SOURCE" <<'PYZIP'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    if 'data/dat/cassandra/conf/cassandra.yaml' not in zf.namelist():
        raise SystemExit('Cassandra data archive lacks data/dat/cassandra/conf/cassandra.yaml')
PYZIP

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else sha256sum "$1" | awk '{print $1}'
  fi
}

stage_one() {
  local source="$1" target="$2" label="$3" expected="$4"
  if [ -L "$target" ]; then
    local current
    current="$(python3 - "$target" <<'PY'
import os,sys
print(os.path.realpath(sys.argv[1]))
PY
)"
    if [ "$current" = "$source" ]; then
      echo "$label already staged from the requested source: $target"
      return
    fi
  fi
  if [ -e "$target" ] || [ -L "$target" ]; then
    if [ "$FORCE" != true ]; then
      echo "Refusing to replace existing staged runtime: $target" >&2
      echo "Use --force only when intentionally switching sources." >&2
      exit 73
    fi
    local backup="$target.backup-$(date +%Y%m%d-%H%M%S)"
    mv "$target" "$backup"
    echo "Previous staged runtime preserved at: $backup"
  fi
  ln -s "$source" "$target"
  local digest
  digest="$(sha256_file "$source")"
  printf 'Staged %-27s %s\n' "$label:" "$target"
  printf 'Source:                    %s\n' "$source"
  printf 'SHA-256:                   %s\n' "$digest"
  printf 'Accepted archived digest(s): %s\n' "$expected"
  case "|$expected|" in
    *"|$digest|"*) ;;
    *) echo "WARNING: $label digest differs from all inspected archived variants." >&2 ;;
  esac
}

stage_one "$BENCHMARKS_SOURCE" "$ROOT/runtime/original-benchmarks.jar" "benchmarks.jar" \
  "03dae4d926aba665cfdf3b22d0cd71221d2fe0303b1113c0c0ba2e53dfe54598"
stage_one "$CASSANDRA_SOURCE" "$ROOT/runtime/original-cassandra.jar" "cassandra.jar" \
  "e92e780b25003169a419c98bea8cb2138e24675b90bd6a137ba892a7435ff4e4"
stage_one "$CASSANDRA_DATA_SOURCE" "$ROOT/runtime/original-cassandra-data.zip" "Cassandra data.zip" \
  "ebf8a6b94f1d5640ee8630f82711ce016a318cfceca15c7e31acd8d714693a8f|9b259a066c14f8abadbdf6c8110af175cb601d10642946b47334a664050b3b26"

echo "Tradebeans artifact classification:"
set +e
"$SCRIPT_DIR/inspect_tradebeans_artifact.py" "$ROOT/runtime/original-benchmarks.jar"
tradebeans_rc=$?
set -e
if [ "$tradebeans_rc" -eq 66 ]; then
  echo "Tradebeans will run as the labelled released-artifact TPCC arm; see README.md." >&2
elif [ "$tradebeans_rc" -ne 0 ]; then
  exit "$tradebeans_rc"
fi

AGENT_JAR="$ROOT/agent/omniflow-original-dacapo-agent.jar"
EXPECTED_AGENT_SOURCE_HASH="$(python3 - "$ROOT/agent" <<'PYFINGERPRINT'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted((root / "src").rglob("*.java")) + [root / "manifest.mf", root / "build.sh"]
h = hashlib.sha256()
for path in files:
    rel = path.relative_to(root).as_posix().encode()
    data = path.read_bytes()
    h.update(len(rel).to_bytes(4, "big")); h.update(rel)
    h.update(len(data).to_bytes(8, "big")); h.update(data)
print(h.hexdigest())
PYFINGERPRINT
)"
ACTUAL_AGENT_SOURCE_HASH=""
if [ -f "$AGENT_JAR" ]; then
  ACTUAL_AGENT_SOURCE_HASH="$(unzip -p "$AGENT_JAR" omniflow-agent-source.sha256 2>/dev/null | tr -d '\r\n' || true)"
fi
needs_build=false
[ -f "$AGENT_JAR" ] || needs_build=true
[ "$ACTUAL_AGENT_SOURCE_HASH" = "$EXPECTED_AGENT_SOURCE_HASH" ] || needs_build=true
if [ "$needs_build" = true ]; then
  JAVA_DIR="$(cd "$(dirname "$JAVA")" 2>/dev/null && pwd -P || true)"
  JAVAC_BIN="$JAVA_DIR/javac"
  [ -x "$JAVAC_BIN" ] || JAVAC_BIN=javac
  echo "Building OmniFlow agent with: $JAVAC_BIN"
  JAVAC="$JAVAC_BIN" "$ROOT/agent/build.sh"
fi
[ -s "$AGENT_JAR" ] || { echo "Agent build did not produce: $AGENT_JAR" >&2; exit 1; }
BUILT_AGENT_SOURCE_HASH="$(unzip -p "$AGENT_JAR" omniflow-agent-source.sha256 2>/dev/null | tr -d '\r\n' || true)"
[ "$BUILT_AGENT_SOURCE_HASH" = "$EXPECTED_AGENT_SOURCE_HASH" ] || {
  echo "Agent source fingerprint does not match the packaged sources." >&2
  exit 1
}
echo "Agent ready: $AGENT_JAR"
