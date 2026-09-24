#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: extract_xalan_data.sh BENCHMARKS_JAR WORK_DIR" >&2
  exit 2
fi
JAR="$1"
WORK="$2"
[ -f "$JAR" ] || { echo "Missing benchmarks JAR: $JAR" >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 1; }
mkdir -p "$WORK"
TARGET="$WORK/benchmarks/bms/xalan/dist/dat/xalan"
REQUIRED=(
  xmlspec.xsl acks.xml binding.xml changes.xml concepts.xml controls.xml
  datatypes.xml expr.xml intro.xml model.xml prod-notes.xml references.xml
  rpm.xml schema.xml structure.xml template.xml terms.xml ui.xml
)
valid=true
for name in "${REQUIRED[@]}"; do
  [ -f "$TARGET/$name" ] || valid=false
done
if [ -e "$TARGET" ]; then
  if [ "$valid" = true ]; then
    exit 0
  fi
  echo "Refusing incomplete existing Xalan data directory: $TARGET" >&2
  exit 73
fi
TMP="$WORK/.xalan-data.tmp-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"
unzip -q "$JAR" 'dist/dat/xalan/*' -d "$TMP"
for name in "${REQUIRED[@]}"; do
  [ -f "$TMP/dist/dat/xalan/$name" ] || {
    echo "benchmarks.jar lacks dist/dat/xalan/$name" >&2
    exit 64
  }
done
mkdir -p "$(dirname "$TARGET")"
mv "$TMP/dist/dat/xalan" "$TARGET"
