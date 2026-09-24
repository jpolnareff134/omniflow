#!/usr/bin/env python3
"""Pure synchronization helpers used by the Kubernetes experiment harness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def ready_pod_names(payload: dict, label_value: str = "nginx") -> list[str]:
    names = []
    for item in payload.get("items", []):
        metadata = item.get("metadata", {})
        if metadata.get("deletionTimestamp"):
            continue
        if metadata.get("labels", {}).get("app") != label_value:
            continue
        conditions = item.get("status", {}).get("conditions", [])
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions):
            names.append(metadata["name"])
    return sorted(names)


def metric_versions(
        payload: dict, mode: str, names: set[str] | None = None
) -> dict[str, str]:
    versions = {}
    for item in payload.get("items", []):
        name = item.get("metadata", {}).get("name") or item.get("describedObject", {}).get("name")
        timestamp = item.get("timestamp", "")
        if not name or not timestamp or (names is not None and name not in names):
            continue
        if mode == "resource":
            containers = item.get("containers", [])
            if not containers or not all(c.get("usage", {}).get("cpu") for c in containers):
                continue
        elif mode == "custom" and "value" not in item:
            continue
        versions[name] = timestamp
    return versions


def fresh_metrics(
        ready_pods: list[str], baseline: dict[str, str], current: dict[str, str]
) -> tuple[bool, str]:
    expected = set(ready_pods)
    if set(current) != expected:
        return False, f"metric pods {sorted(current)} != ready pods {sorted(expected)}"
    missing_baseline = expected - set(baseline)
    if missing_baseline:
        return False, f"baseline missing pods {sorted(missing_baseline)}"
    stale = sorted(name for name in expected if current[name] <= baseline[name])
    if stale:
        return False, f"timestamps did not advance for {stale}"
    return True, "fresh"


def daemon_sample_versions(
    payload: dict, namespace: str, names: set[str] | None = None
) -> dict[str, dict]:
    versions = {}
    for node in payload.get("nodes", []):
        for track in node.get("tracks", []):
            name = track.get("pod_name")
            if (
                track.get("namespace") != namespace
                or not name
                or (names is not None and name not in names)
            ):
                continue
            candidate = {
                "n_sampled": int(track.get("n_sampled", 0)),
                "last_sample_wall": track.get("last_sample_wall"),
            }
            previous = versions.get(name)
            if previous is None or candidate["n_sampled"] > previous["n_sampled"]:
                versions[name] = candidate
    return versions


def fresh_daemon_samples(
    ready_pods: list[str], baseline: dict[str, dict], current: dict[str, dict]
) -> tuple[bool, str]:
    expected = set(ready_pods)
    if set(current) != expected:
        return False, f"daemon tracks {sorted(current)} != ready pods {sorted(expected)}"
    missing_baseline = expected - set(baseline)
    if missing_baseline:
        return False, f"daemon baseline missing pods {sorted(missing_baseline)}"
    stale = sorted(
        name for name in expected
        if current[name]["n_sampled"] <= baseline[name]["n_sampled"]
        or current[name].get("last_sample_wall") is None
    )
    if stale:
        return False, f"OmniFlow did not take a new sample for {stale}"
    return True, "fresh"


def metrics_published_after_samples(
    ready_pods: list[str],
    metrics: dict[str, str],
    samples: dict[str, dict],
    minimum_delay_s: float,
) -> tuple[bool, str]:
    expected = set(ready_pods)
    if set(metrics) != expected or set(samples) != expected:
        return False, "metric, sample, and Ready pod sets differ"
    stale = []
    for name in ready_pods:
        sample_wall = samples[name].get("last_sample_wall")
        if sample_wall is None:
            stale.append(name)
            continue
        metric_wall = datetime.fromisoformat(
            metrics[name].replace("Z", "+00:00")
        ).timestamp()
        if metric_wall < float(sample_wall) + minimum_delay_s:
            stale.append(name)
    if stale:
        return False, f"custom metrics were not published after fresh samples for {stale}"
    return True, "published after fresh OmniFlow samples"


def stable_hpa(hpa: dict, deployment: dict) -> tuple[bool, str]:
    status = hpa.get("status", {})
    conditions = {item.get("type"): item for item in status.get("conditions", [])}
    for condition in ("ScalingActive", "AbleToScale"):
        if conditions.get(condition, {}).get("status") != "True":
            return False, f"{condition} is not true"
    if not status.get("currentMetrics"):
        return False, "currentMetrics is empty"
    if status.get("currentReplicas") != 1 or status.get("desiredReplicas") != 1:
        return False, "HPA replicas are not stable at one"

    metadata = deployment.get("metadata", {})
    spec = deployment.get("spec", {})
    dep_status = deployment.get("status", {})
    if dep_status.get("observedGeneration", 0) < metadata.get("generation", 0):
        return False, "deployment generation is not observed"
    if spec.get("replicas") != 1:
        return False, "deployment spec replicas is not one"
    for field in ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas"):
        if dep_status.get(field, 0) != 1:
            return False, f"deployment {field} is not one"
    return True, "stable"


def render_job(template: Path, output: Path, shape: str, seed: int) -> None:
    import yaml

    documents = list(yaml.safe_load_all(template.read_text()))
    job = next(document for document in documents if document and document.get("kind") == "Job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container.setdefault("env", [])}
    env["LOAD_SHAPE"]["value"] = shape
    env["EXPERIMENT_SEED"]["value"] = str(seed)
    output.write_text(yaml.safe_dump(job, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-job")
    render.add_argument("template", type=Path)
    render.add_argument("output", type=Path)
    render.add_argument("shape")
    render.add_argument("seed", type=int)
    args = parser.parse_args()
    if args.command == "render-job":
        render_job(args.template, args.output, args.shape, args.seed)


if __name__ == "__main__":
    main()
