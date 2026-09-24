#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: extract_cassandra_data.sh /path/to/data.zip /fresh/policy-work" >&2
  exit 2
fi
ZIP="$1"
WORK="$2"
[ -f "$ZIP" ] || { echo "Missing Cassandra data archive: $ZIP" >&2; exit 1; }
mkdir -p "$WORK"
YAML="$WORK/data/dat/cassandra/conf/cassandra.yaml"
if [ -e "$WORK/data" ]; then
  [ -f "$YAML" ] || {
    echo "Refusing existing incomplete Cassandra data tree: $WORK/data" >&2
    exit 73
  }
  exit 0
fi
command -v unzip >/dev/null 2>&1 || { echo "unzip is required to stage Cassandra data" >&2; exit 1; }
unzip -q "$ZIP" -d "$WORK"
[ -f "$YAML" ] || {
  echo "Cassandra data archive did not create: $YAML" >&2
  exit 64
}
