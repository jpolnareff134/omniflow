#!/usr/bin/env bash
# setup-cluster.sh - Create a multi-node minikube cluster using the KVM driver.
#
# Prerequisites:
#   - minikube: https://minikube.sigs.k8s.io/docs/start/
#   - kvm2 driver:
#       sudo apt install qemu-kvm libvirt-daemon-system
#       curl -LO https://storage.googleapis.com/minikube/releases/latest/docker-machine-driver-kvm2
#       sudo install docker-machine-driver-kvm2 /usr/local/bin/
#   - kubectl available in PATH
#
# Usage:
#   ./setup-cluster.sh [--nodes N] [--prefix NAME] [--memory MiB] [--cpus N]
#                      [--metrics-resolution DURATION] [--hpa-sync-period DURATION]
#
# Defaults:  nodes=2  prefix=omniflow  memory=12288  cpus=6
#            metrics-resolution=60s  hpa-sync-period=15s

set -euo pipefail
cd "$(dirname "$0")"

NODES=2
PREFIX="omniflow"
CPUS=6
MEMORY=12288
K8S_VERSION="v1.30.0"
METRICS_RESOLUTION="60s"
HPA_SYNC_PERIOD="15s"

usage() {
  cat <<'USAGE'
Usage: ./setup-cluster.sh [--nodes N] [--prefix NAME] [--memory MiB]
                          [--cpus N] [--metrics-resolution DURATION]
                          [--hpa-sync-period DURATION]
USAGE
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --nodes)   NODES="$2";   shift 2 ;;
    --prefix)  PREFIX="$2";  shift 2 ;;
    --memory)  MEMORY="$2";  shift 2 ;;
    --cpus)    CPUS="$2";    shift 2 ;;
    --metrics-resolution) METRICS_RESOLUTION="$2"; shift 2 ;;
    --hpa-sync-period) HPA_SYNC_PERIOD="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PROFILE="$PREFIX"

echo "=== Setting up minikube cluster ==="
echo "  profile=$PROFILE  nodes=$NODES  memory=${MEMORY}MiB  cpus=$CPUS  k8s=$K8S_VERSION"
echo "  metrics-resolution=$METRICS_RESOLUTION  hpa-sync-period=$HPA_SYNC_PERIOD"

# -- Start minikube cluster ---------------------------------------
minikube start \
  --driver=kvm2 \
  --nodes="$NODES" \
  --profile="$PROFILE" \
  --memory="$MEMORY" \
  --cpus="$CPUS" \
  --extra-config="controller-manager.horizontal-pod-autoscaler-sync-period=$HPA_SYNC_PERIOD" \
  --kubernetes-version="$K8S_VERSION"

minikube addons enable metrics-server -p "$PROFILE"

# -- Dedicate the first node only to control-plane --------------------------------
kubectl taint node "$PROFILE" node-role.kubernetes.io/control-plane:NoSchedule
kubectl label node "$PROFILE" node-role.kubernetes.io/control-plane=true
kubectl label node "$PROFILE" node-role.kubernetes.io/worker=false

# Metrics Server's default collection resolution varies by version. Replace
# only that argument while preserving every addon-provided argument.
python3 - "$METRICS_RESOLUTION" <<'PY'
import json
import subprocess
import sys

resolution = sys.argv[1]
deployment = json.loads(subprocess.check_output([
    "kubectl", "-n", "kube-system", "get", "deployment", "metrics-server", "-o", "json"
]))
container = deployment["spec"]["template"]["spec"]["containers"][0]
args = [arg for arg in container.get("args", []) if not arg.startswith("--metric-resolution=")]
args.append(f"--metric-resolution={resolution}")
patch = [{
    "op": "replace" if "args" in container else "add",
    "path": "/spec/template/spec/containers/0/args",
    "value": args,
}]
subprocess.run([
    "kubectl", "-n", "kube-system", "patch", "deployment", "metrics-server",
    "--type=json", "-p", json.dumps(patch),
], check=True)
PY
kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s

# -- Export kubeconfig --------------------------------------------
KUBECONFIG_PATH="./kubeconfig"
KUBECONFIG="$HOME/.kube/config" kubectl config view --raw --minify --context="$PROFILE" > "$KUBECONFIG_PATH"

echo ""
echo "=== Cluster ready ==="
echo "  Profile:    $PROFILE"
echo "  Nodes:      $NODES"
echo "  Kubeconfig: $KUBECONFIG_PATH"
echo ""
echo "Test with:"
echo "  export KUBECONFIG=$KUBECONFIG_PATH"
echo "  kubectl get nodes"
