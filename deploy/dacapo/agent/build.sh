#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./build.sh"
  echo "Build the DaCapo agent from the included Java sources and available ASM dependency."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$SCRIPT_DIR"

LIB_DIR="$SCRIPT_DIR/lib"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
JAR="$SCRIPT_DIR/omniflow-original-dacapo-agent.jar"
TMP_JAR="$SCRIPT_DIR/.omniflow-original-dacapo-agent.jar.tmp"

JAVAC_BIN="${JAVAC:-javac}"
JAR_BIN="${JAR_TOOL:-}"
if [ -z "$JAR_BIN" ]; then
  candidate="$(cd "$(dirname "$JAVAC_BIN")" 2>/dev/null && pwd -P)/jar"
  if [ -x "$candidate" ]; then JAR_BIN="$candidate"; else JAR_BIN=jar; fi
fi

command -v "$JAVAC_BIN" >/dev/null 2>&1 || [ -x "$JAVAC_BIN" ] || {
  echo "javac not found: $JAVAC_BIN" >&2; exit 1;
}
command -v "$JAR_BIN" >/dev/null 2>&1 || [ -x "$JAR_BIN" ] || {
  echo "jar tool not found: $JAR_BIN" >&2; exit 1;
}
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 1; }

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
rm -f "$TMP_JAR"

SOURCES_FILE="$BUILD_DIR/sources.list"
find "$SRC_DIR" -name '*.java' -print | sort > "$SOURCES_FILE"
if [ ! -s "$SOURCES_FILE" ]; then
  echo "No Java sources found under $SRC_DIR" >&2
  exit 1
fi

JAVAC_VERSION="$($JAVAC_BIN -version 2>&1)"
case "$JAVAC_VERSION" in
  "javac 1.8."*) RELEASE_FLAGS=(-source 1.8 -target 1.8) ;;
  *) RELEASE_FLAGS=(--release 8) ;;
esac

ASM_LIBS=(
  "$LIB_DIR/asm-9.7.jar"
  "$LIB_DIR/asm-commons-9.7.jar"
  "$LIB_DIR/asm-tree-9.7.jar"
)
CLASSPATH=""
DEPENDENCY_MODE=""

all_vendor_libs=true
for dependency in "${ASM_LIBS[@]}"; do
  [ -f "$dependency" ] || all_vendor_libs=false
done

if [ "$all_vendor_libs" = true ]; then
  CLASSPATH="${ASM_LIBS[0]}:${ASM_LIBS[1]}:${ASM_LIBS[2]}"
  DEPENDENCY_MODE="vendored ASM jars"
elif [ -f "$JAR" ] && "$JAR_BIN" tf "$JAR" | grep -q '^org/objectweb/asm/commons/AdviceAdapter.class$'; then
  # A previously built agent is enough to compile a source-only update.  The
  # final JAR is replaced only after the new build succeeds.
  CLASSPATH="$JAR"
  DEPENDENCY_MODE="existing agent JAR fallback"
elif [ -f "$ROOT/runtime/original-cassandra.jar" ] \
     && "$JAR_BIN" tf "$ROOT/runtime/original-cassandra.jar" | grep -q '^org/objectweb/asm/commons/AdviceAdapter.class$'; then
  # The archived Cassandra artifact contains the complete ASM API used by the
  # transformer.  This makes remote source syncs buildable even when an rsync
  # rule omitted agent/lib and the prebuilt agent JAR.
  CLASSPATH="$ROOT/runtime/original-cassandra.jar"
  DEPENDENCY_MODE="staged Cassandra runtime fallback"
else
  cat >&2 <<MSG
Cannot locate the ASM build dependency.

Use one of these supported layouts:
  1. keep the complete agent/lib directory from the release archive;
  2. keep a previously built agent/omniflow-original-dacapo-agent.jar; or
  3. run scripts/stage_paper_runtime.sh first so
     runtime/original-cassandra.jar is available.

The build does not download dependencies from the network.
MSG
  exit 1
fi

# The @argfile form works with the Bash 3.2 shipped by macOS.
"$JAVAC_BIN" "${RELEASE_FLAGS[@]}" \
  -d "$BUILD_DIR" \
  -cp "$CLASSPATH" \
  "@$SOURCES_FILE"

if [ "$all_vendor_libs" = true ]; then
  for asmjar in "${ASM_LIBS[@]}"; do
    unzip -q -o "$asmjar" -d "$BUILD_DIR"
  done
else
  # Extract only ASM from the fallback JAR.  Do not copy benchmark or agent
  # implementation classes into the rebuilt agent.
  FALLBACK_DIR="$BUILD_DIR/.asm-fallback"
  mkdir -p "$FALLBACK_DIR"
  (cd "$FALLBACK_DIR" && "$JAR_BIN" xf "$CLASSPATH" org/objectweb/asm)
  [ -d "$FALLBACK_DIR/org/objectweb/asm" ] || {
    echo "Fallback dependency does not contain org/objectweb/asm" >&2; exit 1;
  }
  mkdir -p "$BUILD_DIR/org/objectweb"
  cp -R "$FALLBACK_DIR/org/objectweb/asm" "$BUILD_DIR/org/objectweb/"
  rm -rf "$FALLBACK_DIR"
fi
find "$BUILD_DIR" -name 'module-info.class' -delete
rm -f "$BUILD_DIR/sources.list"

python3 - "$SCRIPT_DIR" > "$BUILD_DIR/omniflow-agent-source.sha256" <<'PYFINGERPRINT'
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

"$JAR_BIN" cfm "$TMP_JAR" "$SCRIPT_DIR/manifest.mf" -C "$BUILD_DIR" .
mv "$TMP_JAR" "$JAR"

echo "Built $JAR using $JAVAC_VERSION ($DEPENDENCY_MODE)"
ls -lh "$JAR"
