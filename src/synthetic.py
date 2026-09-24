import argparse
import logging as log

import numpy as np
from numpy.typing import NDArray

from data.composer import TraceComposer
from data.specific.cloudperftrace import load_parquet_trace
from data.specific.gcloud import load_json_trace
from plot.plot import plot_composed_trace
from support.log import initialize_log
from tracker.pipeline import PipelineConfig, PipelineResult, Pipeline


def _run_batch(
        trace: NDArray[np.float64],
        pipe_config: PipelineConfig,
        save_dir: str,
        n_bins: int,
        n_zoom: int,
        zoom_window: int,
        comment: str = "",
        generator_info: dict | None = None,
) -> None:
    """Dual batch run: dense reference tracker + adaptive replay.

    Runs both passes so that information-loss metrics can be computed,
    then produces aggregated overview + auto-zoom plots.
    """
    from tracker.windowed import TickResult, evaluate_info_loss
    from plot.batch import (
        find_interesting_regions,
        plot_batch_overview,
        plot_batch_zooms,
    )

    n = len(trace)

    log.info("Batch mode: running dense reference tracker on %s points ...",
             f"{n:,}")
    log.info("Comment: %s", comment)
    full_tracker = pipe_config._make_tracker()
    full_results: list[TickResult] = full_tracker.track(trace)
    log.info("Dense reference pass complete.")

    log.info("Running adaptive poller ...")
    poller = pipe_config._make_poller()
    poll_results = poller.track(trace)

    n_sampled = sum(1 for pr in poll_results if pr.sampled)
    log.info("Poller done: %s / %s sampled (%.1f%%)",
             f"{n_sampled:,}", f"{n:,}",
             100.0 * n_sampled / n)

    log.info("Computing information loss ...")
    info_loss = evaluate_info_loss(
        trace,
        full_tracker=pipe_config._make_tracker(),
        poller=pipe_config._make_poller(),
    )
    for line in str(info_loss).split("\n"):
        log.info(line)

    regions = find_interesting_regions(
        poll_results,
        n_regions=n_zoom,
        window=zoom_window,
    )
    for r in regions:
        log.info("  Region: %s  (score %.0f)", r.label, r.score)

    overview_path = f"{save_dir}/batch_replay_overview.png"
    plot_batch_overview(
        trace, poll_results,
        full_results=full_results,
        n_bins=n_bins,
        regions=regions,
        title="Batch Replay Overview",
        save_path=overview_path,
    )

    if regions:
        zoom_path = f"{save_dir}/batch_replay_zooms.pdf"
        plot_batch_zooms(
            trace, poll_results, regions,
            full_results=full_results,
            sigma_band=pipe_config.sigma_band,
            save_path=zoom_path,
        )

    if save_dir:
        from tracker.pipeline import PipelineResult
        result = PipelineResult(
            trace=trace,
            config=pipe_config,
            full_results=full_results,
            poll_results=poll_results,
            info_loss=info_loss,
        )
        experiment_path = result.save_json(
            save_dir=save_dir,
            generator=generator_info,
        )
        log.info("Experiment manifest saved to %s", experiment_path)

    log.info("Batch replay plots written to %s", save_dir)


def _build_generator_info(args, trace_builder) -> dict:
    """Build the generator metadata dict for the experiment manifest."""
    if trace_builder is not None:
        return {**trace_builder.to_dict(), "comment": args.comment or None}
    if getattr(args, "parquet", None):
        return {
            "source": args.cloudperftrace,
            "task": args.task,
            "column": args.column,
            "metric": args.metric,
            "comment": args.comment or None,
        }
    return {
        "source": args.json,
        "field": args.field,
        "comment": args.comment or None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="OmniFlow synthetic pipeline."
                    "This tools allows you to run the tracker and adaptive poller on synthetic traces, "
                    "either generated from a YAML description or loaded from JSON / CloudPerfTrace Parquet files.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--yamltrace", type=str, default=None,
        help="Path to a YAML file describing the trace (segments, bridges, ...).",
    )
    src.add_argument(
        "--json", type=str, default=None,
        help="Path to a JSON file containing a raw time-series "
             "(flat array or array of objects).  Use --field to pick "
             "a column.",
    )
    src.add_argument(
        "--cloudperftrace", type=str, default=None,
        help="Path to a CloudPerfTrace Parquet directory (the folder "
             "that contains parquet_ds/).  Requires pyarrow.",
    )
    # ---- end mutually exclusive group ----

    parser.add_argument(
        "--field", type=str, default=None,
        help="Key to extract when --json contains objects  "
             "(e.g. 'cpus').  Auto-detected if omitted.",
    )

    # This is common to both JSON and YAML traces, so we put it outside the mutually exclusive group
    parser.add_argument(
        "--max-points", type=int, default=0,
        help="Truncate the trace to at most N points (0 = no limit). Valid for JSON files and CloudPerfTrace Parquet files (after filtering by task/column/metric).",
    )

    # CloudPerfTrace options
    parser.add_argument(
        "--task", type=int, default=None,
        help="CloudPerfTrace task ID to filter on "
             "(4=DataServing, 5=Redis, 6=WebSearch, 7=GraphAnalytics, "
             "9=DataAnalytics, 10=MLPerf, 11=HBase, 13=Alluxio, "
             "14=Minio, 15=TPC-C, 16=Flink).  All tasks if omitted.",
    )
    parser.add_argument(
        "--column", type=str, default="tr_self",
        help="Parquet list-column to read "
             "(tr_self, lin_self, td_self, tr_oth, lin_oth, td_oth).  "
             "Default: tr_self.",
    )
    parser.add_argument(
        "--metric", type=int, default=1,
        help="0-based metric index inside the column "
             "(0=timestamp, ...).  Default: 1.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Load at most N Parquet rows / execution windows "
             "(0 = no limit).",
    )

    # Batch mode options
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: aggregated overview + auto-zoom instead of "
             "the regular dense-reference + adaptive-replay plots. Designed for "
             "very long traces (100k+ points).",
    )
    parser.add_argument(
        "--bins", type=int, default=1500,
        help="Number of bins for the batch replay overview (default 1500).",
    )
    parser.add_argument(
        "--zoom-regions", type=int, default=5,
        help="Number of interesting regions to zoom into (default 5).",
    )
    parser.add_argument(
        "--zoom-window", type=int, default=2000,
        help="Width (in samples) of each zoom region (default 2000).",
    )
    # End batch options

    parser.add_argument(
        "--comment", type=str, default="",
        help="Free-text note describing this experiment.",
    )

    parser.add_argument(
        "--log-level", "-l", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    log_folder = initialize_log(
        log_level=args.log_level,
        name="main",
        console_only=False,
        application_type="pipeline",
        create_out_subfolders=True,
    )

    trace_builder = None
    if args.yamltrace:
        trace_builder = TraceComposer.from_yaml(args.yamltrace)
        log.info("Loaded trace from %s", args.yamltrace)
        trace = trace_builder.build()
    elif args.json:
        trace = load_json_trace(
            args.json, field=args.field, max_points=args.max_points,
        )
    elif args.cloudperftrace:
        trace = load_parquet_trace(
            args.cloudperftrace,
            task=args.task,
            column=args.column,
            metric=args.metric,
            max_points=args.max_points,
            max_rows=args.max_rows,
        )
    else:
        log.exception("No trace source provided.")
        return

    log.info("Trace: %s points.", f"{len(trace):,}")

    if args.comment:
        log.info("Comment: %s", args.comment)
        if log_folder:
            with open(f"{log_folder}/COMMENT.txt", "w") as f:
                f.write(args.comment + "\n")

    if args.batch:
        # Batch mode 
        generator_info = _build_generator_info(args, trace_builder)
        _run_batch(
            trace,
            pipe_config=PipelineConfig(),
            save_dir=log_folder,
            n_bins=args.bins,
            n_zoom=args.zoom_regions,
            zoom_window=args.zoom_window,
            comment=args.comment,
            generator_info=generator_info,
        )
    else:
        # Normal mode
        if trace_builder is not None:
            plot_composed_trace(
                trace, save_path=f"{log_folder}/composed_trace.png",
            )

        pipe = Pipeline()
        result: PipelineResult = pipe.run(trace)
        [log.info(summary) for summary in result.summary().split("\n")]
        result.plot(save_dir=log_folder)

        if log_folder:
            generator_info = _build_generator_info(args, trace_builder)
            experiment_path = result.save_json(
                save_dir=log_folder,
                generator=generator_info,
            )
            log.info("Experiment manifest saved to %s", experiment_path)


if __name__ == "__main__":
    main()
