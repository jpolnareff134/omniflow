#!/bin/bash

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [live|synthetic] [options]"
    exit 0
fi

if [[ "$1" == "live" ]]; then
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "Live mode requires Linux."
        exit 1
    fi
    shift
    # eBPF probes (syscall) need root; /proc probes don't.
    # Default probe is syscall -> needs root unless overridden.
    if echo " $* " | grep -qE -- "--probe (cpu|mem|disk|net)"; then
        python3 src/live.py "$@"
    else
        sudo env PYTHONPATH="$PYTHONPATH" "$(which python3)" src/live.py "$@"
        sudo chown -R "$(id -u)":"$(id -g)" out/
    fi
elif [[ "$1" == "synthetic" ]]; then
    shift
    python3 src/synthetic.py "$@"
else
    echo "Usage: $0 [live|synthetic] (**kwargs)"
    exit 1
fi
