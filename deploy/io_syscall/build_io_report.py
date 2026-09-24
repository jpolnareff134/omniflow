#!/usr/bin/env python3
"""Build the I/O report and, when supplied, the pure replay paper table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    return json.loads(Path(path_str).read_text())


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _replay_stat(
    replay_workload: dict[str, Any] | None,
    metric: str,
    field: str = "mean",
) -> float | None:
    if not replay_workload:
        return None
    stats = _get_nested(replay_workload, "paper_metrics", metric)
    if not isinstance(stats, dict):
        return None
    return _maybe_float(stats.get(field))


def _replay_mean_urgency(replay_workload: dict[str, Any] | None) -> float | None:
    value = _replay_stat(replay_workload, "mean_urgency")
    if value is not None:
        return value
    return _maybe_float(_get_nested(replay_workload or {}, "paper_run_mean_urgency", "mean"))


def _pct_delta(reference: float | None, value: float | None) -> float | None:
    if reference in (None, 0) or value is None:
        return None
    return ((value / reference) - 1.0) * 100.0


def _delta_pct(full_value: float | None, adaptive_value: float | None) -> float | None:
    if full_value in (None, 0) or adaptive_value is None:
        return None
    return (1.0 - (adaptive_value / full_value)) * 100.0


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _tex_num(value: float | None, digits: int = 6) -> str:
    return "0" if value is None else f"{value:.{digits}f}"


def _escape_tex(text: str) -> str:
    return text.replace("_", r"\_")


def _available_labels(sections: dict[str, dict[str, Any]]) -> list[str]:
    return [label for label in ["floor", "latency", "postgres", "redis"] if label in sections]


def _load_eval_section(
    summary: dict[str, Any],
    replay_workload: dict[str, Any] | None = None,
    replay_enabled: bool = False,
) -> dict[str, Any]:
    comparisons = summary.get("overhead_comparisons") or summary.get("comparisons") or {}
    fidelity = summary.get("fidelity") or {}
    live = summary.get("live_comparability") or {}
    counts = live.get("counts") or {}
    comparable_count = int(counts.get("comparable", 0))
    total_count = int(live.get("count") or 0)
    live_urgency = _maybe_float(comparisons.get("adaptive_live_mean_urgency"))
    replay_urgency = _replay_mean_urgency(replay_workload)
    return {
        "kind": "eval",
        "label": str(summary.get("label") or "unknown"),
        "metric_label": "tracked signal",
        "metric": {"summary_label": "tracked signal", "higher_is_better": None},
        "baseline": {},
        "full": {
            "mean_urgency": _maybe_float(comparisons.get("full_live_mean_urgency")),
            "total_cpu_s": _maybe_float(comparisons.get("full_live_total_cpu_s")),
            "probe_fetch_total_s": _maybe_float(comparisons.get("full_live_probe_fetch_total_s")),
            "post_fetch_control_total_s": _maybe_float(comparisons.get("full_live_post_fetch_control_total_s")),
            "publish_path_total_s": _maybe_float(comparisons.get("full_live_publish_path_total_s")),
        },
        "adaptive": {
            "mean_urgency": replay_urgency if replay_enabled else live_urgency,
            "live_mean_urgency": live_urgency,
            "replay_mean_urgency": replay_urgency,
            "total_cpu_s": _maybe_float(comparisons.get("adaptive_live_total_cpu_s")),
            "probe_fetch_total_s": _maybe_float(comparisons.get("adaptive_live_probe_fetch_total_s")),
            "post_fetch_control_total_s": _maybe_float(comparisons.get("adaptive_live_post_fetch_control_total_s")),
            "publish_path_total_s": _maybe_float(comparisons.get("adaptive_live_publish_path_total_s")),
        },
        "throughput": {},
        "quality": {
            "sample_ratio_pct": _maybe_float(fidelity.get("adaptive_replay_sample_ratio_pct")),
            "data_saved_vs_full_pct": _maybe_float(fidelity.get("adaptive_replay_data_saved_vs_full_pct")),
            "peak_recall_top5_pct": _maybe_float(fidelity.get("adaptive_replay_peak_recall_top5_pct")),
            "correlation": _maybe_float(fidelity.get("adaptive_replay_correlation")),
            "nrmse_mean": _maybe_float(fidelity.get("adaptive_replay_nrmse_mean")),
            "live_comparable_pct": None if total_count == 0 else (comparable_count / total_count) * 100.0,
        },
    }


def _load_workload_section(
    summary: dict[str, Any],
    replay_workload: dict[str, Any] | None = None,
    replay_enabled: bool = False,
) -> dict[str, Any]:
    comparisons = summary.get("comparisons") or {}
    throughput = comparisons.get("throughput") or {}
    data_capture = comparisons.get("data_capture") or {}
    urgency = comparisons.get("urgency") or {}
    probe = comparisons.get("probe_fetch") or {}
    post_fetch = comparisons.get("post_fetch_control") or {}
    publish = comparisons.get("publish_path") or {}
    quality = comparisons.get("tracked_metric_quality") or {}
    metric = summary.get("metric") or {}
    label = str(summary.get("kind") or "unknown")
    default_metric_label = "p99 latency" if label == "latency" else "throughput"
    live_urgency = _maybe_float(urgency.get("adaptive_mean_urgency"))
    replay_urgency = _replay_mean_urgency(replay_workload)
    replay_sample_ratio = _replay_stat(replay_workload, "sample_ratio")
    replay_nrmse = _replay_stat(replay_workload, "nrmse_mean")
    replay_correlation = _replay_stat(replay_workload, "correlation")
    replay_max_gap = _replay_stat(replay_workload, "max_gap")
    return {
        "kind": "workload",
        "label": label,
        "metric_label": str(metric.get("summary_label") or default_metric_label),
        "metric": metric,
        "baseline": {
            "mean_output": _maybe_float(throughput.get("baseline_mean_throughput")),
        },
        "full": {
            "mean_urgency": _maybe_float(urgency.get("full_mean_urgency")),
            "total_cpu_s": _maybe_float(_get_nested(comparisons, "cpu", "full_total_cpu_s")),
            "probe_fetch_total_s": _maybe_float(probe.get("full_probe_fetch_total_s")),
            "post_fetch_control_total_s": _maybe_float(post_fetch.get("full_post_fetch_control_total_s")),
            "publish_path_total_s": _maybe_float(publish.get("full_publish_path_total_s")),
        },
        "adaptive": {
            "mean_urgency": replay_urgency if replay_enabled else live_urgency,
            "live_mean_urgency": live_urgency,
            "replay_mean_urgency": replay_urgency,
            "total_cpu_s": _maybe_float(_get_nested(comparisons, "cpu", "adaptive_total_cpu_s")),
            "probe_fetch_total_s": _maybe_float(probe.get("adaptive_probe_fetch_total_s")),
            "post_fetch_control_total_s": _maybe_float(post_fetch.get("adaptive_post_fetch_control_total_s")),
            "publish_path_total_s": _maybe_float(publish.get("adaptive_publish_path_total_s")),
        },
        "throughput": {
            "adaptive_vs_full_pct": _maybe_float(throughput.get("adaptive_vs_full_pct")),
            "adaptive_vs_baseline_pct": _maybe_float(throughput.get("adaptive_vs_baseline_pct")),
        },
        "quality": {
            "sample_ratio_pct": (
                replay_sample_ratio * 100.0
                if replay_enabled and replay_sample_ratio is not None
                else _maybe_float(data_capture.get("adaptive_sample_ratio_pct"))
            ),
            "corr": (
                replay_correlation
                if replay_enabled and replay_correlation is not None
                else _maybe_float(quality.get("corr"))
            ),
            "nrmse_mean": (
                replay_nrmse
                if replay_enabled and replay_nrmse is not None
                else _maybe_float(quality.get("nrmse_mean") or quality.get("nrmse_p95"))
            ),
            "max_gap": (
                replay_max_gap
                if replay_enabled and replay_max_gap is not None
                else None
            ),
        },
    }


def _build_sections(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    replay_summary = _load_json(args.replay_summary)
    replay_workloads = (replay_summary or {}).get("workloads") or {}
    replay_enabled = args.replay_summary is not None
    floor = _load_json(args.floor_summary)
    latency = _load_json(args.latency_summary)
    postgres = _load_json(args.postgres_summary)
    redis = _load_json(args.redis_summary)
    if floor:
        label = str(floor.get("label") or "floor")
        sections["floor"] = _load_eval_section(
            floor, replay_workloads.get(label), replay_enabled
        )
    if latency:
        label = str(latency.get("kind") or latency.get("label") or "latency")
        replay_workload = replay_workloads.get(label)
        sections["latency"] = (
            _load_workload_section(latency, replay_workload, replay_enabled)
            if isinstance(latency.get("kind"), str)
            else _load_eval_section(latency, replay_workload, replay_enabled)
        )
    if postgres:
        sections["postgres"] = _load_workload_section(
            postgres, replay_workloads.get("postgres"), replay_enabled
        )
    if redis:
        sections["redis"] = _load_workload_section(
            redis, replay_workloads.get("redis"), replay_enabled
        )
    return sections


PAPER_WORKLOAD_LABELS = {
    "postgres": "pgbench",
    "redis": "redis-benchmark",
    "latency": "fio",
}


def _format_pm(
    replay_workload: dict[str, Any],
    metric: str,
    digits: int,
    suffix: str = "",
    scale: float = 1.0,
) -> str:
    stats = (replay_workload.get("paper_metrics") or {}).get(metric) or {}
    mean = _maybe_float(stats.get("mean"))
    stdev = _maybe_float(stats.get("stdev"))
    if mean is None or stdev is None:
        return "n/a"
    return f"{mean * scale:.{digits}f} \\!\\pm\\! {stdev * scale:.{digits}f}{suffix}"


def _build_paper_table(replay_summary: dict[str, Any]) -> tuple[str, str]:
    workloads = replay_summary.get("workloads") or {}
    missing = [name for name in PAPER_WORKLOAD_LABELS if name not in workloads]
    if missing:
        raise ValueError(
            "replay summary is missing paper workloads: " + ", ".join(missing)
        )
    for name, workload in workloads.items():
        if name in PAPER_WORKLOAD_LABELS and not isinstance(workload.get("paper_metrics"), dict):
            raise ValueError(f"replay summary has no paper_metrics for {name!r}")

    columns = [PAPER_WORKLOAD_LABELS[name] for name in PAPER_WORKLOAD_LABELS]
    rows = [
        ("Mean urgency ($\\bar{u}$)", "mean_urgency", 3, "", 1.0),
        ("Sampling ratio ($r\\downarrow$)", "sample_ratio", 1, "\\%", 100.0),
        ("NRMSE$\\downarrow$", "nrmse_mean", 3, "", 1.0),
        ("Correlation ($\\rho\\uparrow$)", "correlation", 3, "", 1.0),
        ("Maximum gap ($g_{\\max}\\downarrow$)", "max_gap", 2, "", 1.0),
    ]
    tex_lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \renewcommand{\arraystretch}{1.15}",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \begin{tabular}{@{}l r r r@{}}",
        r"        \hline",
        "        Metric & " + " & ".join(f"\\texttt{{{column}}}" for column in columns) + r" \\",
        r"        \hline",
    ]
    markdown_lines = [
        "| Metric | " + " | ".join(columns) + " |",
        "|---|---:|---:|---:|",
    ]
    for label, metric, digits, suffix, scale in rows:
        values = [
            _format_pm(workloads[name], metric, digits, suffix, scale)
            for name in PAPER_WORKLOAD_LABELS
        ]
        tex_lines.append(f"        {label} & " + " & ".join(values) + r" \\")
        markdown_values = []
        for name in PAPER_WORKLOAD_LABELS:
            stats = (workloads[name].get("paper_metrics") or {}).get(metric) or {}
            mean = _maybe_float(stats.get("mean"))
            stdev = _maybe_float(stats.get("stdev"))
            markdown_values.append(
                "n/a" if mean is None or stdev is None else
                f"{mean * scale:.{digits}f} ± {stdev * scale:.{digits}f}{suffix}"
            )
        markdown_lines.append(f"| {label} | " + " | ".join(markdown_values) + " |")
    tex_lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"    \caption{I/O replay results: mean $\pm$ population standard deviation over the declared paper runs.}",
        r"    \label{tab:io-results}",
        r"\end{table}",
        "",
    ])
    return "\n".join(tex_lines), "\n".join(markdown_lines) + "\n"


def _build_vs_floor(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    floor = sections.get("floor")
    if not floor:
        return {}
    comparisons: dict[str, Any] = {}
    for label, section in sections.items():
        if label == "floor":
            continue
        comparisons[label] = {
            "full": {
                "total_cpu_vs_floor_pct": _pct_delta(_maybe_float(floor["full"].get("total_cpu_s")), _maybe_float(section["full"].get("total_cpu_s"))),
                "probe_fetch_vs_floor_pct": _pct_delta(_maybe_float(floor["full"].get("probe_fetch_total_s")), _maybe_float(section["full"].get("probe_fetch_total_s"))),
            },
            "adaptive": {
                "total_cpu_vs_floor_pct": _pct_delta(_maybe_float(floor["adaptive"].get("total_cpu_s")), _maybe_float(section["adaptive"].get("total_cpu_s"))),
                "probe_fetch_vs_floor_pct": _pct_delta(_maybe_float(floor["adaptive"].get("probe_fetch_total_s")), _maybe_float(section["adaptive"].get("probe_fetch_total_s"))),
            },
        }
    return comparisons


def _build_adaptive_savings(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for label, section in sections.items():
        full = section.get("full") or {}
        adaptive = section.get("adaptive") or {}
        entry = {
            "total_cpu_saved_vs_full_pct": _delta_pct(_maybe_float(full.get("total_cpu_s")), _maybe_float(adaptive.get("total_cpu_s"))),
            "probe_fetch_saved_vs_full_pct": _delta_pct(_maybe_float(full.get("probe_fetch_total_s")), _maybe_float(adaptive.get("probe_fetch_total_s"))),
            "post_fetch_control_saved_vs_full_pct": _delta_pct(_maybe_float(full.get("post_fetch_control_total_s")), _maybe_float(adaptive.get("post_fetch_control_total_s"))),
            "publish_path_saved_vs_full_pct": _delta_pct(_maybe_float(full.get("publish_path_total_s")), _maybe_float(adaptive.get("publish_path_total_s"))),
        }
        if section.get("kind") == "workload":
            throughput = section.get("throughput") or {}
            entry["adaptive_vs_full_output_pct"] = _maybe_float(throughput.get("adaptive_vs_full_pct"))
            entry["adaptive_vs_baseline_output_pct"] = _maybe_float(throughput.get("adaptive_vs_baseline_pct"))
        comparisons[label] = entry
    return comparisons


def _chart_coords(sections: dict[str, dict[str, Any]], arm: str, metric: str) -> str:
    coords: list[str] = []
    for label in ["floor", "latency", "postgres", "redis"]:
        section = sections.get(label)
        if not section:
            continue
        coords.append(f"({label},{_tex_num(_maybe_float(_get_nested(section, arm, metric)))})")
    return " ".join(coords)


def _build_tikz_fragment(report: dict[str, Any]) -> str:
    sections = report["sections"]
    labels = _available_labels(sections)
    symbolic = ",".join(labels)
    urgency_label = report.get("urgency_label", "Adaptive live urgency")

    floor_rows: list[str] = []
    tracked_rows: list[str] = []
    probe_rows: list[str] = []
    for label in ["latency", "postgres", "redis"]:
        section = sections.get(label)
        comp = report["vs_floor"].get(label)
        if section and comp:
            floor_rows.append(
                f"{_escape_tex(label)} & {_fmt(_maybe_float(comp['full']['probe_fetch_vs_floor_pct']), 2)} & {_fmt(_maybe_float(comp['adaptive']['probe_fetch_vs_floor_pct']), 2)} & {_fmt(_maybe_float(_get_nested(section, 'adaptive', 'mean_urgency')), 3)} \\\\"
            )
        if section:
            quality = section.get("quality") or {}
            savings = report["adaptive_savings"].get(label, {})
            output_reference = "dense full trace" if label == "latency" else "full trace + daemon-off throughput"
            tracked_rows.append(
                f"{_escape_tex(label)} & {_escape_tex(output_reference)} & {_fmt(_maybe_float(quality.get('sample_ratio_pct')), 1)} & {_fmt(_maybe_float(savings.get('probe_fetch_saved_vs_full_pct')), 1)} & {_fmt(_maybe_float(_get_nested(section, 'adaptive', 'mean_urgency')), 3)} & {_escape_tex(f'corr={_fmt(_maybe_float(quality.get("correlation") or quality.get("corr")), 3)}, NRMSE={_fmt(_maybe_float(quality.get("nrmse_mean")), 3)}')} & {_fmt(_maybe_float(savings.get('adaptive_vs_full_output_pct')), 2)} & {_fmt(_maybe_float(savings.get('adaptive_vs_baseline_output_pct')), 2)} \\\\"
            )
    for label in labels:
        section = sections[label]
        full = section.get("full") or {}
        adaptive = section.get("adaptive") or {}
        savings = report["adaptive_savings"].get(label, {})
        probe_rows.append(
            f"{_escape_tex(label)} & {_fmt(_maybe_float(full.get('probe_fetch_total_s')), 3)} & {_fmt(_maybe_float(adaptive.get('probe_fetch_total_s')), 3)} & {_fmt(_maybe_float(savings.get('probe_fetch_saved_vs_full_pct')), 1)} & {_fmt(_maybe_float(adaptive.get('mean_urgency')), 3)} \\\\"
        )

    floor_section = sections.get("floor")
    floor_summary_rows = []
    if floor_section:
        full = floor_section.get("full") or {}
        adaptive = floor_section.get("adaptive") or {}
        savings = report["adaptive_savings"].get("floor", {})
        floor_summary_rows.append(
            f"floor & {_fmt(_maybe_float(full.get('probe_fetch_total_s')), 3)} & {_fmt(_maybe_float(adaptive.get('probe_fetch_total_s')), 3)} & {_fmt(_maybe_float(savings.get('probe_fetch_saved_vs_full_pct')), 1)} & {_fmt(_maybe_float(adaptive.get('mean_urgency')), 3)} \\\\"
        )

    return rf"""
\begin{{tikzpicture}}[
    scale=0.86,
    every node/.style={{transform shape}},
    node distance=0.8cm and 0.55cm,
    box/.style={{draw, rounded corners, align=center, minimum width=2.25cm, minimum height=0.85cm, fill=black!3}},
    smallbox/.style={{draw, rounded corners, align=left, text width=3.25cm, minimum height=1.0cm, fill=black!2}},
    >=Latex
]
\node[box] (kernel) {{Tracepoints and eBPF map}};
\node[box, right=of kernel] (fetch) {{Probe fetch}};
\node[box, right=of fetch] (control) {{Tracker + adaptive policy}};
\node[box, right=of control] (publish) {{Metrics + JSON emit}};
\draw[->, thick] (kernel) -- (fetch);
\draw[->, thick] (fetch) -- (control);
\draw[->, thick] (control) -- (publish);
\node[smallbox, below=0.95cm of fetch] (fetchtext) {{\textbf{{probe-fetch}}: kernel-side counter read cost.}};
\node[smallbox, below=0.95cm of control] (controltext) {{\textbf{{post-fetch control}}: tracker update and adaptive scheduling logic.}};
\node[smallbox, below=0.95cm of publish] (publishtext) {{\textbf{{publish path}}: Prometheus updates and JSON emission.}};
\draw[dashed, ->] (fetch) -- (fetchtext.north);
\draw[dashed, ->] (control) -- (controltext.north);
\draw[dashed, ->] (publish) -- (publishtext.north);
\end{{tikzpicture}}

\begin{{tikzpicture}}
\begin{{axis}}[
    ybar=5pt,
    bar width=13pt,
    width=14.5cm,
    height=6.8cm,
    ylabel={{Estimated probe-fetch total (s)}},
    symbolic x coords={{{symbolic}}},
    xtick=data,
    ymin=0,
    enlarge x limits=0.16,
    ymajorgrids,
    grid style={{black!10}},
    axis line style={{black!60}},
    tick style={{black!60}},
    legend style={{at={{(0.5,1.02)}}, anchor=south, legend columns=-1}},
    nodes near coords,
    every node near coord/.append style={{font=\scriptsize, rotate=90, anchor=west}},
]
\addplot+[draw=black!45, fill=black!20] coordinates {{{_chart_coords(sections, 'full', 'probe_fetch_total_s')}}};
\addplot+[draw=black!45, fill=black!60] coordinates {{{_chart_coords(sections, 'adaptive', 'probe_fetch_total_s')}}};
\legend{{full,adaptive}}
\end{{axis}}
\end{{tikzpicture}}

\begin{{tikzpicture}}
\begin{{axis}}[
    ybar=5pt,
    bar width=13pt,
    width=14.5cm,
    height=6.8cm,
    ylabel={{Total CPU (s)}},
    symbolic x coords={{{symbolic}}},
    xtick=data,
    ymin=0,
    enlarge x limits=0.16,
    ymajorgrids,
    grid style={{black!10}},
    axis line style={{black!60}},
    tick style={{black!60}},
    legend style={{at={{(0.5,1.02)}}, anchor=south, legend columns=-1}},
    nodes near coords,
    every node near coord/.append style={{font=\scriptsize, rotate=90, anchor=west}},
]
\addplot+[draw=black!45, fill=black!20] coordinates {{{_chart_coords(sections, 'full', 'total_cpu_s')}}};
\addplot+[draw=black!45, fill=black!60] coordinates {{{_chart_coords(sections, 'adaptive', 'total_cpu_s')}}};
\legend{{full,adaptive}}
\end{{axis}}
\end{{tikzpicture}}

\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrr}}
\toprule
Experiment & Full probe vs floor (\%) & Adaptive probe vs floor (\%) & {urgency_label} \\
\midrule
{chr(10).join(floor_rows) if floor_rows else 'No floor comparisons available. \\\\'}
\bottomrule
\end{{tabular}}
}}

\par\bigskip
\textbf{{Floor baseline summary}}\par
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
Test & Full probe (s) & Adaptive probe (s) & Probe saved (\%) & {urgency_label} \\
\midrule
{chr(10).join(floor_summary_rows) if floor_summary_rows else 'floor & n/a & n/a & n/a & n/a \\\\'}
\bottomrule
\end{{tabular}}
}}

\par\bigskip
\textbf{{Tracked-output tests}}\par
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrr}}
\toprule
Test & Output reference & Adaptive sample (\%) & Probe saved (\%) & {urgency_label} & Quality & Adaptive vs full (\%) & Adaptive vs baseline (\%) \\
\midrule
{chr(10).join(tracked_rows) if tracked_rows else 'No tracked tests available. & n/a & n/a & n/a & n/a & n/a & n/a & n/a \\\\'}
\bottomrule
\end{{tabular}}
}}

\par\bigskip
    extbf{{Probe-fetch totals by test}}\par
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
Test & Full probe & Adaptive probe & Saved & {urgency_label} \\
\midrule
{chr(10).join(probe_rows) if probe_rows else 'No probe data available. & n/a & n/a & n/a & n/a \\\\'}
\bottomrule
\end{{tabular}}
}}
"""


def _build_tikz_standalone(fragment: str) -> str:
    return rf"""\documentclass[tikz,border=6pt]{{standalone}}
\usepackage{{graphicx}}
\usepackage{{pgfplots}}
\usepackage{{tikz}}
\usepackage{{booktabs}}
\usetikzlibrary{{arrows.meta,positioning}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
{fragment}
\end{{document}}
"""


def _build_latex_report(report: dict[str, Any]) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{pgfplots}}
\usepackage{{tikz}}
\usepackage{{hyperref}}
\usetikzlibrary{{arrows.meta,positioning}}
\pgfplotsset{{compat=1.18}}


\begin{{document}}
\input{{io_report_tikz.tex}}

\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a combined TikZ-ready report for floor, latency, postgres, and redis summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--floor-summary", default=None)
    parser.add_argument("--latency-summary", default=None)
    parser.add_argument("--postgres-summary", default=None)
    parser.add_argument("--redis-summary", default=None)
    parser.add_argument(
        "--replay-summary", "--replay-urgency-summary", dest="replay_summary",
        default=None,
        help="Use the pure replay summary for all paper-table metrics.",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = _build_sections(args)
    report = {
        "sections": sections,
        "vs_floor": _build_vs_floor(sections),
        "adaptive_savings": _build_adaptive_savings(sections),
        "urgency_label": (
            "Adaptive replay urgency"
            if args.replay_summary
            else "Adaptive live urgency"
        ),
    }

    summary_path = out_dir / "io_report_summary.json"
    tikz_path = out_dir / "io_report_tikz.tex"
    tikz_standalone_path = out_dir / "io_report_tikz_standalone.tex"
    report_path = out_dir / "io_report.tex"
    paper_table_path = None
    paper_markdown_path = None
    if args.replay_summary:
        replay_payload = _load_json(args.replay_summary)
        if replay_payload is None:
            raise ValueError("--replay-summary did not contain a JSON object")
        paper_tex, paper_markdown = _build_paper_table(replay_payload)
        paper_table_path = out_dir / "io_paper_table.tex"
        paper_markdown_path = out_dir / "io_paper_table.md"
        paper_table_path.write_text(paper_tex)
        paper_markdown_path.write_text(paper_markdown)

    tikz_fragment = _build_tikz_fragment(report)
    summary_path.write_text(json.dumps(report, indent=2))
    tikz_path.write_text(tikz_fragment)
    tikz_standalone_path.write_text(_build_tikz_standalone(tikz_fragment))
    report_path.write_text(_build_latex_report(report))

    print(json.dumps({
        "summary": str(summary_path),
        "tikz": str(tikz_path),
        "tikz_standalone": str(tikz_standalone_path),
        "report": str(report_path),
        "paper_table": None if paper_table_path is None else str(paper_table_path),
        "paper_table_markdown": None if paper_markdown_path is None else str(paper_markdown_path),
        "sections": sorted(sections.keys()),
    }))


if __name__ == "__main__":
    main()
