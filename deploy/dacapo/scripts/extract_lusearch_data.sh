#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: extract_lusearch_data.sh BENCHMARKS_JAR WORK_DIR" >&2
  exit 2
fi
JAR="$1"
WORK="$2"
[ -f "$JAR" ] || { echo "Missing benchmarks JAR: $JAR" >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 1; }
mkdir -p "$WORK"
TARGET="$WORK/lusearch"
if [ -e "$TARGET" ]; then
  if [ -f "$TARGET/index-default/segments_1" ] && compgen -G "$TARGET/query*.txt" >/dev/null; then
    exit 0
  fi
  echo "Refusing incomplete existing Lusearch data directory: $TARGET" >&2
  exit 73
fi
TMP="$WORK/.lusearch-data.tmp-$$"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"
unzip -q "$JAR" 'data/lusearch/*' -d "$TMP"
[ -f "$TMP/data/lusearch/index-default/segments_1" ] || {
  echo "benchmarks.jar lacks data/lusearch/index-default/segments_1" >&2
  exit 64
}
compgen -G "$TMP/data/lusearch/query*.txt" >/dev/null || {
  echo "benchmarks.jar lacks Lusearch query files" >&2
  exit 64
}
mv "$TMP/data/lusearch" "$TARGET"
