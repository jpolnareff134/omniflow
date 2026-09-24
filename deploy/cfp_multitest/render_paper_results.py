#!/usr/bin/env python3
"""Render CloudPerfTrace paper metrics, including maximum sampling gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_LABELS = {
    "redis_tx_bytes": "Redis, egress throughput",
    "websearch_involuntary_ctx_switches": "WebSearch, total context switches",
    "hbase_block_write_latency": "HBase, block write latency",
    "minio_block_read_bytes": "Minio, block read throughput",
    "flink_memory_rss": "Flink, resident memory",
}

POLICY_LABELS = {
    "huang_wavelet_rate_inspired": "Huang-inspired",
    "magalhaes_truncated_exponential_inspired": "Magalhaes-inspired",
    "daoud_time_variation_adapted": "Daoud-inspired",
}

CASE_TEX_LABELS = {
    "redis_tx_bytes": r"Redis trace, egress throughput --- {\scriptsize \texttt{task 5 | tr\_self[38]} --- $1,604,068$ pts}",
    "websearch_involuntary_ctx_switches": r"WebSearch trace, total context switches --- {\scriptsize \texttt{task 6 | tr\_self[21]} --- $1,007,616$ pts.}",
    "hbase_block_write_latency": r"HBase trace, block write latency --- {\scriptsize \texttt{task 11 | tr\_self[51]} --- $599,177$ pts.}",
    "minio_block_read_bytes": r"Minio trace, block read throughput --- {\scriptsize \texttt{task 14 | tr\_self[47]} --- $612,236$ pts.}",
    "flink_memory_rss": r"Flink trace, resident memory --- {\scriptsize \texttt{task 16 | tr\_self[22]} --- $2,329,388$ pts.}",
}

METRICS = (
    "sample_ratio",
    "nrmse_mean",
    "correlation",
    "peak_recall_top5",
    "max_gap",
)


def _phase_summary(summary: dict) -> dict:
    values = {key: summary[key]["mean"] for key in METRICS}
    values["max_gap"] = summary["max_gap"]["max"]
    return values


def _rows(case: dict):
    yield "OmniFlow", case["two_pass"]["adaptive"], "single adaptive run"
    fixed = case["two_pass"]["fixed"]
    yield f"Fixed (I={fixed['interval']})", fixed["report"], "single fixed phase"

    matched = case["budget_matched"]
    yield (
        f"Matched ({matched['n']}/{matched['m']})",
        _phase_summary(matched["summary"]),
        f"all {matched['phase_count']} phases",
    )

    if "n_every_m" in case:
        periodic = case["n_every_m"]
        yield (
            f"Fixed ({periodic['n']}/{periodic['m']})",
            _phase_summary(periodic["summary"]),
            f"all {periodic['m']} phases",
        )

    for key, policy in case["policies"].items():
        yield POLICY_LABELS.get(key, key), policy["report"], "single policy run"


def _paper_rows(case: dict):
    yield "OmniFlow", case["two_pass"]["adaptive"]
    matched = case["budget_matched"]
    yield f"Fixed (${matched['n']}/{matched['m']}$)", _phase_summary(matched["summary"])
    for key, policy in case["policies"].items():
        yield POLICY_LABELS.get(key, key), policy["report"]


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _tex_pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def render(payload: dict, source: str) -> str:
    lines = [
        "# CloudPerfTrace Results with Maximum Sampling Gap",
        "",
        f"Source: `{source}`",
        "",
        "`Max gap` is the longest distance, in dense-trace ticks, between consecutive sampled points.",
        "Phase-aggregated rows report quality-metric means over every possible phase of the stated periodic sampler; `Max gap` is the worst phase.",
        "",
    ]
    for case in payload["cases"]:
        name = case["case"]["name"]
        lines.extend((
            f"## {CASE_LABELS.get(name, name)}",
            "",
            "| Method | r | NRMSE | rho | R5% | Max gap | Evaluation |",
            "|---|---:|---:|---:|---:|---:|---|",
        ))
        for method, report, evaluation in _rows(case):
            lines.append(
                f"| {method} | {_pct(report['sample_ratio'])} | "
                f"{report['nrmse_mean']:.4f} | {report['correlation']:.3f} | "
                f"{_pct(report['peak_recall_top5'])} | "
                f"{report['max_gap']:.1f} | {evaluation} |"
            )
        matched = case["budget_matched"]
        lines.extend((
            "",
            f"Matched target: {_pct(matched['target_sample_ratio'])}; nominal budget: "
            f"{_pct(matched['nominal_sample_ratio'])}; error: "
            f"{100.0 * matched['budget_error']:+.3f} percentage points.",
            "",
        ))
    return "\n".join(lines)


def render_latex(payload: dict, source: str) -> str:
    lines = [
        f"% Generated from {source}",
        r"\begin{table}[t]",
        r"    \footnotesize",
        r"    \renewcommand{\arraystretch}{1.08}",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{p{3cm} r r r r r}",
        r"        \hline",
        "        Method & $r$ & NRMSE & $\\rho$ & $g_{\\max}$ & $R_{5\\%}$ \\\\",
        r"        \hline",
    ]
    for case in payload["cases"]:
        name = case["case"]["name"]
        lines.extend((
            f"        \\multicolumn{{6}}{{@{{}}l}}{{\\makecell[l]{{{CASE_TEX_LABELS[name]}}}}} \\\\",
            r"        \hline",
        ))
        for method, report in _paper_rows(case):
            lines.append(
                f"        {method:<18} & {_tex_pct(report['sample_ratio'])} & "
                f"{report['nrmse_mean']:.4f} & {report['correlation']:.3f} & "
                f"{report['max_gap']:.0f} & {_tex_pct(report['peak_recall_top5'])} \\\\"
            )
        lines.append(r"        \hline")
    lines.extend((
        r"    \end{tabular}",
        r"    \centering",
        r"    \caption{Information loss on five CloudPerfTrace traces. Each fixed $N/M$ baseline is within 0.1 percentage points of OmniFlow's budget. Its quality metrics are averaged over all $M$ phases, while $g_{\max}$ is the worst phase.}",
        r"    \label{tab:cloudperftrace-updated}",
        r"\end{table}",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-label", help="Source path shown in the generated report.")
    parser.add_argument("--format", choices=("markdown",
                        "latex"), default="markdown")
    args = parser.parse_args()

    payload = json.loads(args.run_json.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = args.source_label or str(args.run_json.resolve())
    output = render_latex(
        payload, source) if args.format == "latex" else render(payload, source)
    args.out.write_text(output)
    print(args.out)


if __name__ == "__main__":
    main()
