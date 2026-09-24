#!/usr/bin/env python3
"""
Multi-panel HPA comparison dashboard (PDF).

Panels
------
1. Replica timeline - Baseline vs OmniFlow step curves, scale markers, first-scale rules
2. HPA input signal - What each HPA sees: baseline avg raw CPU vs OmniFlow avg EMA, with
   the 30m threshold.  This is the metric that drives scaling decisions.
3. Per-pod raw CPU  - Raw cgroup CPU for every nginx pod (both phases) on separate y-axes.
   Shows why the average drops when replicas appear.
4. Tracker sigma        - OmniFlow EMA standard deviation over time
5. Urgency          - OmniFlow adaptive urgency over time
6. Adaptive interval - OmniFlow adaptive interval over time
7. Load shape       - Concurrent users from Locust

Usage:
    python3 deploy/k8s/plot_comparison.py <run_dir>

Example:
    python3 deploy/k8s/plot_comparison.py path/to/k8s-run
"""
import glob
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- paths ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", nargs="?", type=Path, default=Path("."))
args = parser.parse_args()
run_dir = args.run_dir
results = run_dir / "results"
out_path = results / "hpa_comparison_dashboard.pdf"

params_path = run_dir / "params.json"
namespace = "omniflow-hpa"
if params_path.exists():
    namespace = json.loads(params_path.read_text()).get("namespace", "omniflow-hpa")


# -- helpers -------------------------------------------------------------------
def load_timeline(path):
    rows = [json.loads(l) for l in open(path)]
    t = [r["elapsed_s"] for r in rows]
    replicas = [r["replicas"] for r in rows]
    desired = [r["hpa_desired"] for r in rows]
    start_wall = rows[0]["wall"]
    return t, replicas, desired, start_wall


def scale_events(times, replicas):
    events = []
    for i in range(1, len(replicas)):
        if replicas[i] != replicas[i - 1]:
            diff = replicas[i] - replicas[i - 1]
            events.append((times[i], replicas[i - 1], replicas[i],
                           "up" if diff > 0 else "down", diff))
    return events


def parse_locust_phases(log_path):
    lines = open(log_path).readlines()
    recorded = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "phase_start":
            recorded.append((float(event["observed_offset_s"]), int(event["users"])))
    if recorded:
        return recorded

    # Backward-compatible parser for runs created before synchronized JSON events.
    start_ts = None
    phases = []
    ts_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]")
    for line in lines:
        m = ts_re.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if start_ts is None:
            start_ts = ts
        elapsed = int((ts - start_ts).total_seconds())
        u_m = re.search(r"updating to (\d+) users", line)
        if u_m:
            phases.append((elapsed, int(u_m.group(1))))
    return phases


def load_daemon_signals(results_dir, phase, start_wall, namespace="omniflow-hpa"):
    """Return per-second arrays (elapsed_s, value, mean, std, urgency, interval)
    aggregated (mean) across all nginx pods on all nodes for the given phase.

    Also returns the raw buckets dict for downstream helpers (pod counts etc.).
    """
    # Pass 1: build cgroup_id -> pod_name for nginx pods
    id_map: dict[str, str] = {}
    for fname in sorted(glob.glob(str(results_dir / f"{phase}_daemon_*.jsonl"))):
        for line in open(fname):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("event") == "track_start"
                    and r.get("namespace") == namespace
                    and "nginx" in r.get("pod_name", "")):
                id_map[r["pod"]] = r["pod_name"]

    if not id_map:
        return None

    # Pass 2: collect readings for nginx pods, bucket by integer elapsed second
    buckets: dict[int, list[dict]] = defaultdict(list)
    for fname in sorted(glob.glob(str(results_dir / f"{phase}_daemon_*.jsonl"))):
        for line in open(fname):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "t" not in r or r.get("pod") not in id_map:
                continue
            es = int(r["wall"] - start_wall)
            if es < 0:
                continue
            buckets[es].append(r)

    if not buckets:
        return None

    keys = sorted(buckets)
    t_arr = np.array(keys, dtype=float)
    val_arr = np.array([np.mean([x["value"] for x in buckets[k]]) for k in keys])
    mean_arr = np.array([np.mean([x["mean"] for x in buckets[k]]) for k in keys])
    std_arr = np.array([np.mean([x["std"] for x in buckets[k]]) for k in keys])
    urg_arr = np.array([np.mean([x["urgency"] for x in buckets[k]]) for k in keys])
    int_arr = np.array([np.mean([x["interval"] for x in buckets[k]]) for k in keys])
    return t_arr, val_arr, mean_arr, std_arr, urg_arr, int_arr, buckets


def load_perpod_signals(results_dir, phase, start_wall, namespace="omniflow-hpa"):
    """Return per-pod raw CPU time series for individual pod traces."""
    id_map: dict[str, str] = {}
    for fname in sorted(glob.glob(str(results_dir / f"{phase}_daemon_*.jsonl"))):
        for line in open(fname):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("event") == "track_start"
                    and r.get("namespace") == namespace
                    and "nginx" in r.get("pod_name", "")):
                id_map[r["pod"]] = r["pod_name"]

    if not id_map:
        return {}

    pod_data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fname in sorted(glob.glob(str(results_dir / f"{phase}_daemon_*.jsonl"))):
        for line in open(fname):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "t" not in r or r.get("pod") not in id_map:
                continue
            es = int(r["wall"] - start_wall)
            if es < 0:
                continue
            pname = id_map[r["pod"]]
            pod_data[pname][es].append(r["value"])

    # Collapse to sorted arrays per pod
    result = {}
    for pname, secs in pod_data.items():
        keys = sorted(secs)
        t_arr = np.array(keys, dtype=float)
        v_arr = np.array([np.mean(secs[k]) for k in keys])
        result[pname] = (t_arr, v_arr)
    return result


def rolling(arr, w=10):
    """Simple centred rolling mean.  Falls back to identity for tiny arrays."""
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def hpa_decision_points(times, desired, sig_t, sig_vals):
    """Return (t, signal_value, new_desired) at each hpa_desired transition.

    *sig_t* / *sig_vals* are the signal arrays; we interpolate to get the
    signal value at the exact decision time.
    """
    pts = []
    prev_d = None
    for i, d in enumerate(desired):
        if d != prev_d and d > 0:
            t = times[i]
            # Interpolate signal at this time
            if sig_t is not None and len(sig_t) > 0:
                v = float(np.interp(t, sig_t, sig_vals))
            else:
                v = None
            pts.append((t, v, d))
            prev_d = d
        elif d != prev_d:
            prev_d = d
    return pts


def pod_count_series(sig_buckets):
    """Return (t_arr, count_arr) - number of distinct pods reporting per second."""
    if not sig_buckets:
        return None, None
    keys = sorted(sig_buckets)
    t_arr = np.array(keys, dtype=float)
    cnt = np.array([len(set(x["pod"] for x in sig_buckets[k])) for k in keys])
    return t_arr, cnt


# -- data ----------------------------------------------------------------------
bt, br, bd, b_wall0 = load_timeline(results / "baseline_replica_timeline.jsonl")
ot, or_, od, o_wall0 = load_timeline(results / "omniflow_replica_timeline.jsonl")

b_events = scale_events(bt, br)
o_events = scale_events(ot, or_)

locust_phases = parse_locust_phases(results / "baseline_locust.log")

b_sig = load_daemon_signals(results, "baseline", b_wall0, namespace)
o_sig = load_daemon_signals(results, "omniflow", o_wall0, namespace)

b_perpod = load_perpod_signals(results, "baseline", b_wall0, namespace)
o_perpod = load_perpod_signals(results, "omniflow", o_wall0, namespace)

# Unpack signal arrays; keep buckets for pod-count series
b_buckets = b_sig[-1] if b_sig else {}
o_buckets = o_sig[-1] if o_sig else {}

HPA_THRESHOLD = 0.030  # 30 millicores expressed as fraction of one core

# -- colours (matches src/plot/ palette) --------------------------------------
_DPI = 300
_GRID_A = 0.3
C_BASE = "steelblue"  # baseline - same as "observed" in plot.py
C_BASE_LT = "lightsteelblue"  # baseline raw signal (faint)
C_OMNI = "tomato"  # OmniFlow EMA - same as "Tracker mean" in plot.py
C_LOAD = "teal"  # load shape - same as cumulative ratio in plot.py
C_STD = "mediumpurple"  # tracker sigma  - analogous to drift marker colour
C_URG = "crimson"  # urgency    - identical to plot_adaptive_polling
C_INT = "darkorange"  # poll interval - identical to plot_adaptive_polling

# -- figure with GridSpec ------------------------------------------------------
fig, axes = plt.subplots(
    6, 1, figsize=(14, 20), sharex=False,
    gridspec_kw={"height_ratios": [3, 2, 2, 1, 1, 1]}
)
ax_r, ax_hpa, ax_raw, ax_urg, ax_int, ax_l = axes
all_axes = list(axes)

for ax in all_axes:
    ax.grid(True, alpha=_GRID_A)

max_t = max(bt[-1], ot[-1]) + 15
for ax in all_axes[:-1]:
    ax.set_xlim(0, max_t)
    ax.set_xticklabels([])
ax_l.set_xlim(0, max_t)

# -- panel 1: replica timeline -------------------------------------------------
ax_r.step(bt, br, where="post", color=C_BASE, linewidth=2.4, label="Baseline", zorder=3)
ax_r.step(ot, or_, where="post", color=C_OMNI, linewidth=2.4, label="OmniFlow", zorder=3)
ax_r.step(bt, bd, where="post", color=C_BASE, linewidth=0.9, linestyle="--", alpha=0.3, zorder=2)
ax_r.step(ot, od, where="post", color=C_OMNI, linewidth=0.9, linestyle="--", alpha=0.3, zorder=2)

for events, color, nudge_y in [(b_events, C_BASE, +0.25), (o_events, C_OMNI, -0.45)]:
    for t, prev, curr, direction, diff in events:
        ax_r.scatter(t, curr, marker="^" if direction == "up" else "v",
                     color=color, s=90, zorder=6, edgecolors="white", linewidths=0.4)
        lbl = f"+{diff}" if direction == "up" else str(diff)
        ax_r.annotate(lbl, xy=(t, curr), xytext=(t + 3, curr + nudge_y),
                      color=color, fontsize=8, fontweight="bold", zorder=7)

b_first = next((e for e in b_events if e[3] == "up"), None)
o_first = next((e for e in o_events if e[3] == "up"), None)

for first, color, ha, dx in [(b_first, C_BASE, "right", -3), (o_first, C_OMNI, "left", +3)]:
    if first:
        ax_r.axvline(first[0], color=color, linewidth=1.1, linestyle=":", alpha=0.7, zorder=2)
        ax_r.text(first[0] + dx, 0.5, f"t={first[0]}s",
                  color=color, fontsize=8.5, ha=ha, va="bottom",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

ax_r.set_ylabel("Replicas", fontsize=10)
ax_r.set_ylim(0, max(max(br), max(or_)) + 1.5)
ax_r.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_r.legend(loc="upper right", fontsize=8)

# -- panel 2: HPA input signal (what drives scaling decisions) -------------
if b_sig is not None:
    bt2, bval, _, _, _, _, _ = b_sig
    ax_hpa.plot(bt2, rolling(bval), color=C_BASE, linewidth=1.8,
                label="Baseline HPA input", alpha=0.9, zorder=3)

if o_sig is not None:
    ot2, oval, _, _, _, _, _ = o_sig
    ax_hpa.plot(ot2, rolling(oval), color=C_OMNI, linewidth=1.8,
                label="OmniFlow HPA input", alpha=0.9, zorder=3)

ax_hpa.axhline(HPA_THRESHOLD, color="red", linewidth=0.9, linestyle="--",
               alpha=0.7, zorder=5, label=f"HPA threshold ({int(HPA_THRESHOLD * 1000)}m)")

# HPA decision-point markers (where hpa_desired changes)
b_decisions = hpa_decision_points(
    bt, bd, bt2 if b_sig else None,
    rolling(bval) if b_sig else None)
o_decisions = hpa_decision_points(
    ot, od, ot2 if o_sig else None,
    rolling(oval) if o_sig else None)

for dpts, color, label_prefix in [
    (b_decisions, C_BASE, "B"), (o_decisions, C_OMNI, "O")
]:
    for i, (t, v, des) in enumerate(dpts):
        if v is None:
            continue
        marker = "^" if (i == 0 or des > dpts[i - 1][2]) else "v"
        ax_hpa.scatter(t, v, marker=marker, s=100, color=color,
                       edgecolors="white", linewidths=0.6, zorder=8)
        ax_hpa.annotate(f"d={des}", xy=(t, v),
                        xytext=(t + 5, v + 0.004),
                        fontsize=7, fontweight="bold", color=color,
                        zorder=9)

# Pod-count shading on a twin axis
ax_hpa_cnt = ax_hpa.twinx()
for bk_data, color, label in [
    (b_buckets, C_BASE, "Baseline pods"),
    (o_buckets, C_OMNI, "OmniFlow pods"),
]:
    ct, cc = pod_count_series(bk_data)
    if ct is not None:
        ax_hpa_cnt.step(ct, cc, where="post", color=color,
                        linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)
ax_hpa_cnt.set_ylabel("# pods", fontsize=8, alpha=0.6)
ax_hpa_cnt.tick_params(axis="y", labelsize=7, colors="grey")

ax_hpa.set_ylabel("CPU (core fraction)", fontsize=9)
ax_hpa.legend(loc="upper right", fontsize=8)

# -- panel 3: per-pod raw CPU (explains why the average varies) ------------
_POD_ALPHA = 0.55
_POD_LW = 0.8
_base_cmap = plt.cm.Blues
_omni_cmap = plt.cm.Oranges

if b_perpod:
    n_b = len(b_perpod)
    for i, (pname, (pt, pv)) in enumerate(sorted(b_perpod.items())):
        c = _base_cmap(0.4 + 0.5 * i / max(n_b - 1, 1))
        ax_raw.plot(pt, rolling(pv), color=c, linewidth=_POD_LW, alpha=_POD_ALPHA)
    # Single legend proxy
    ax_raw.plot([], [], color=C_BASE, linewidth=1.2, label=f"Baseline pods ({len(b_perpod)})")

if o_perpod:
    n_o = len(o_perpod)
    for i, (pname, (pt, pv)) in enumerate(sorted(o_perpod.items())):
        c = _omni_cmap(0.4 + 0.5 * i / max(n_o - 1, 1))
        ax_raw.plot(pt, rolling(pv), color=c, linewidth=_POD_LW, alpha=_POD_ALPHA)
    ax_raw.plot([], [], color=C_OMNI, linewidth=1.2, label=f"OmniFlow pods ({len(o_perpod)})")

ax_raw.axhline(HPA_THRESHOLD, color="red", linewidth=0.7, linestyle="--",
               alpha=0.5, zorder=5)
ax_raw.set_ylabel("Per-pod raw CPU", fontsize=9)
ax_raw.legend(loc="upper right", fontsize=8)

# # -- panel 4: tracker standard deviation --------------------------------------
# if o_sig is not None:
#     ot2, _, _, ostd, _, _, _ = o_sig
#     ax_std.plot(ot2, rolling(ostd), color=C_STD, linewidth=1.5,
#                 label="OmniFlow sigma (EMA std)", alpha=0.9, zorder=3)
#     ax_std.fill_between(ot2, 0, rolling(ostd), color=C_STD, alpha=0.12)
# ax_std.set_ylabel("Tracker sigma", fontsize=9)
# ax_std.legend(loc="upper right", fontsize=8)

# -- panel 5: urgency ---------------------------------------------------------
if o_sig is not None:
    ot2, _, _, _, ourg, oint, _ = o_sig
    ax_urg.fill_between(ot2, 0, rolling(ourg), color=C_URG, alpha=0.4, step="post")
    ax_urg.step(ot2, rolling(ourg), where="post", color=C_URG,
                linewidth=0.9, label="Urgency", zorder=3)
ax_urg.set_ylabel("Urgency", fontsize=9)
ax_urg.legend(loc="upper right", fontsize=8)

# -- panel 6: adaptive interval -----------------------------------------------
if o_sig is not None:
    ax_int.step(ot2, rolling(oint), where="post", color=C_INT,
                linewidth=0.9, label="Adaptive interval (steps)", zorder=3)
ax_int.set_ylabel("Adaptive interval", fontsize=9)
ax_int.legend(loc="upper right", fontsize=8)

# -- panel 7: load shape -------------------------------------------------------
ax_l.tick_params(axis="x", labelbottom=True)
if locust_phases:
    phase_t = [p[0] for p in locust_phases] + [max_t]
    phase_u = [p[1] for p in locust_phases] + [locust_phases[-1][1]]
    ax_l.step(phase_t, phase_u, where="post", color=C_LOAD, linewidth=1.8, zorder=3)
    ax_l.fill_between(phase_t, phase_u, step="post", alpha=0.4, color=C_LOAD)
    seen: set = set()
    for t, u in locust_phases:
        if u not in seen or u > 50:
            ax_l.annotate(f"{u}", xy=(t + 6, u + 5), color=C_LOAD, fontsize=8, alpha=0.9)
            seen.add(u)

ax_l.set_ylabel("Conc.\nUsers", fontsize=9)
ax_l.set_xlabel("Elapsed seconds", fontsize=10)
ax_l.set_ylim(0, 240)

# -- load phase shading across all panels -------------------------------------
phase_colours = {200: ("#ff4444", 0.06), 150: ("#ff8844", 0.045),
                 75: ("#ffcc44", 0.035), 50: ("#88aaff", 0.025)}
if locust_phases:
    for i in range(len(locust_phases)):
        t0 = locust_phases[i][0]
        t1 = locust_phases[i + 1][0] if i + 1 < len(locust_phases) else max_t
        u = locust_phases[i][1]
        if u in phase_colours:
            fc, alpha = phase_colours[u]
            for ax in all_axes:
                ax.axvspan(t0, t1, color=fc, alpha=alpha, zorder=1)

# -- panel labels (matches fontstyle used in plot.py) -------------------------
for ax, label in zip(all_axes,
                     ["(1) Replicas", "(2) HPA input signal",
                      "(3) Per-pod raw CPU",  # "(4) Tracker \u03c3",
                      "(4) Urgency", "(5) Adaptive interval",
                      "(6) Load shape"]):
    ax.text(0.005, 0.97, label, transform=ax.transAxes,
            fontsize=8, va="top", color="grey", style="italic")

# -- title ---------------------------------------------------------------------
run_id = run_dir.name
b_first_t = b_first[0] if b_first else "?"
o_first_t = o_first[0] if o_first else "?"

fig.suptitle(
    f"HPA Scaling - {run_id} - "
    f"Baseline: first scale t={b_first_t}s | peak {max(br)} replicas -"
    f"OmniFlow: first scale t={o_first_t}s | peak {max(or_)} replicas\n",
    fontsize=11,
)


def fmt_events(events):
    parts = []
    for t, prev, curr, direction, diff in events:
        arrow = "↑" if direction == "up" else "↓"
        parts.append(f"t={t}s {prev}{arrow}{curr}")
    return "  ->  ".join(parts)


# fig.text(0.08, 0.974, f"Baseline: {fmt_events(b_events)}", color=C_BASE, fontsize=7)
# fig.text(0.08, 0.968, f"OmniFlow: {fmt_events(o_events)}", color=C_OMNI, fontsize=7)

# -- save ----------------------------------------------------------------------
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig.tight_layout()
plt.savefig(out_path, format="pdf", dpi=_DPI, bbox_inches="tight")
print(f"Saved: {out_path}")

# png_path = out_path.with_suffix(".png")
# plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
# print(f"Saved: {png_path}")
