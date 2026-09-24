#!/usr/bin/env bash
# teardown-cluster.sh - Delete the minikube cluster created by setup-cluster.sh.
#
# Usage:
#   ./teardown-cluster.sh [--prefix NAME]
#
# Defaults:  prefix=omniflow

set -euo pipefail
cd "$(dirname "$0")"

PREFIX="omniflow"

usage() {
  echo "Usage: ./teardown-cluster.sh [--prefix NAME]"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PROFILE="$PREFIX"

echo "=== Tearing down minikube cluster ==="
if minikube status -p "$PROFILE" &>/dev/null; then
  minikube delete -p "$PROFILE"
  echo "Deleted profile: $PROFILE"
else
  echo "Profile '$PROFILE' not found - skipping"
fi

rm -f ./kubeconfig
echo "Done."
