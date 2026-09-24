#!/usr/bin/env bash
# teardown.sh – Adaptive I/O Syscall Rate Monitoring
#
# Removes all Kubernetes resources deployed by setup.sh.
# Deleting the namespace cascades to all objects inside it.
#
# Usage
# -----
#   ./teardown.sh
#   ./teardown.sh --kubeconfig ~/.kube/config

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,18p' "$0"
  exit 0
fi

cd "$(dirname "$0")"

KUBECONFIG_PATH="${KUBECONFIG:-}"
NS="omniflow-io"

while [[ $# -gt 0 ]]; do
  case $1 in
    --kubeconfig) KUBECONFIG_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -n "$KUBECONFIG_PATH" ]]; then
  export KUBECONFIG="$KUBECONFIG_PATH"
fi

echo "=== I/O Syscall Rate Monitoring – teardown ==="
echo "  namespace=$NS"
echo ""

# Delete namespace (cascades to all child resources)
if kubectl get namespace "$NS" &>/dev/null; then
  echo "Deleting namespace '$NS' and all resources inside it ..."
  kubectl delete namespace "$NS" --timeout=120s
  echo "Namespace deleted."
else
  echo "Namespace '$NS' not found – nothing to remove."
fi

# Remove cluster-scoped RBAC objects (not deleted with the namespace)
for resource in \
    "clusterrole/omniflow-io-daemon" \
    "clusterrolebinding/omniflow-io-daemon"; do
  if kubectl get "$resource" &>/dev/null; then
    kubectl delete "$resource"
    echo "Deleted $resource"
  fi
done

echo ""
echo "Teardown complete."
