"""CloudPerfTrace Parquet dataset loading.

The ``load_parquet_trace`` function handles the *CloudPerfTrace* dataset
(Hive-partitioned Parquet under ``parquet_ds/``). 

The loader reshapes the requested column, extracts
the requested metric index (0-based, where 0 = timestamp), and
concatenates all matching rows into a single 1-D time-series.
"""

from __future__ import annotations

import logging as log
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Column name -> (stride, feature count excl. timestamp)
_COLUMN_INFO: dict[str, int] = {
    "tr_self": 54,  # 1 ts + 53 VM metrics (libvirt)
    "lin_self": 39,  # 1 ts + 38 hw counters (Linux perf)
    "td_self": 13,  # 1 ts + 12 top-down (Intel TDA)
    "tr_oth": 54,
    "lin_oth": 39,
    "td_oth": 13,
}


def load_parquet_trace(
        parquet_dir: str | Path,
        task: int | None = None,
        column: str = "tr_self",
        metric: int = 1,
        max_points: int = 0,
        max_rows: int = 0,
) -> NDArray[np.float64]:
    """Load a 1-D time-series from a CloudPerfTrace Parquet dataset.

    Parameters
    ----------
    parquet_dir : str or Path
        Path to the top-level Parquet directory (contains ``parquet_ds/``).
        If a ``parquet_ds/`` sub-directory exists inside, it is used
        automatically.
    task : int, optional
        Application task ID to filter on (e.g. 6 for Web Search).
        When *None*, all tasks are loaded (may be very large).
    column : str
        Which list-column to read: ``tr_self``, ``lin_self``, ``td_self``,
        or their ``_oth`` counterparts.  Default ``tr_self``.
    metric : int
        0-based feature index inside the column.  Index 0 is always the
        Unix timestamp; useful metrics start at 1.  Default 1 (main CPU
        signal in ``tr_self``).
    max_points : int
        If > 0, truncate the final 1-D array to at most this many points.
    max_rows : int
        If > 0, load at most this many Parquet rows (execution windows).

    Returns
    -------
    NDArray[np.float64]
    """
    try:
        import pyarrow.dataset as pds
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Parquet support.  "
            "Install it with: pip install pyarrow"
        ) from exc

    parquet_dir = Path(parquet_dir)
    # Auto-detect parquet_ds/ sub-directory
    if (parquet_dir / "parquet_ds").is_dir():
        parquet_dir = parquet_dir / "parquet_ds"

    if column not in _COLUMN_INFO:
        raise ValueError(
            f"Unknown column '{column}'.  "
            f"Choose from: {', '.join(sorted(_COLUMN_INFO))}"
        )
    stride = _COLUMN_INFO[column]
    if metric < 0 or metric >= stride:
        raise ValueError(
            f"metric={metric} out of range for column '{column}' "
            f"(valid: 0-{stride - 1})"
        )

    log.info(
        "Loading Parquet dataset from %s  (task=%s  column=%s  metric=%d) ...",
        parquet_dir, task if task is not None else "*", column, metric,
    )

    dataset = pds.dataset(
        str(parquet_dir), format="parquet", partitioning="hive",
    )

    print("Dataset schema:")
    print(dataset.schema)

    # Build filter & column selection
    filt = pds.field("tasks") == task if task is not None else None
    table = dataset.to_table(filter=filt, columns=[column])

    if max_rows > 0:
        table = table.slice(0, max_rows)

    log.info("Read %s rows from Parquet.", f"{table.num_rows:,}")

    # Each row's column value is a flat list; reshape -> (T_i, stride)
    # and extract metric column, then concatenate.
    chunks: list[NDArray] = []
    col_array = table.column(column)
    for i in range(table.num_rows):
        flat = col_array[i].as_py()
        T_i = len(flat) // stride
        mat = np.array(flat, dtype=np.float64).reshape(T_i, stride)
        chunks.append(mat[:, metric])

    arr = np.concatenate(chunks)
    if max_points > 0:
        arr = arr[:max_points]

    log.info(
        "Loaded %s points from %s rows  "
        "(min=%.4g  max=%.4g  mean=%.4g)",
        f"{len(arr):,}", f"{table.num_rows:,}",
        arr.min(), arr.max(), arr.mean(),
    )
    return arr
