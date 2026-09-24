"""
Load a time-series trace from JSON or Parquet files.

JSON layouts supported
---------------------
* **Flat array** - ``[1.0, 2.5, 3.1, ...]``
* **Object array** - ``[{"cpus": "8e-4", "memory": "0.003"}, ...]``
  -> specify *field* to pick a column.
* **Nested object array** - ``[{"average_usage": {"cpus": "..."}}, ...]``
  -> use dot-notation: ``--field average_usage.cpus``
* **Wrapped array** - ``[[{...}, ...]]``  (e.g. Google cluster traces exported
  via BigQuery).  The loader peels one level of wrapping automatically.

All numeric strings (``"8.04901123046875E-4"``) are coerced to float.
"""

from __future__ import annotations

import json
import logging as log
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _resolve_path(obj: Any, path: str) -> Any:
    """Walk a dot-separated path into a nested dict/object.

    >>> _resolve_path({"a": {"b": {"c": 42}}}, "a.b.c")
    42
    """
    for key in path.split("."):
        obj = obj[key]
    return obj


def _auto_detect_field(sample: dict) -> str:
    """Find the first numeric-looking leaf in a (possibly nested) dict.

    Returns the dot-notation path, e.g. ``"average_usage.cpus"``.
    """

    def _walk(d: dict, prefix: str = "") -> str | None:
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                found = _walk(v, full)
                if found is not None:
                    return found
            else:
                try:
                    float(v)
                    return full
                except (TypeError, ValueError):
                    continue
        return None

    result = _walk(sample)
    if result is None:
        raise ValueError(
            "Could not auto-detect a numeric field.  "
            "Top-level keys: " + ", ".join(sample.keys())
        )
    return result


def load_json_trace(
        path: str | Path,
        field: str | None = None,
        max_points: int = 0,
) -> NDArray[np.float64]:
    """Read a JSON file and return a 1-D float64 array.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    field : str, optional
        Key (or dot-separated path) to extract from each object.
        Examples: ``"cpus"``, ``"average_usage.cpus"``.
        Auto-detected if omitted.  Ignored for flat numeric arrays.
    max_points : int
        If > 0, truncate the result to at most this many points.
        Useful for quick smoke-tests on very long traces.

    Returns
    -------
    NDArray[np.float64]
    """
    path = Path(path)
    log.info("Loading trace from %s ...", path)

    with open(path) as fh:
        raw = json.load(fh)

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Expected a non-empty JSON array in '{path}'")

    # Peel one level of wrapping if necessary: [[{...}, ...]] -> [{...}, ...]
    if isinstance(raw[0], list):
        log.info("Detected wrapped array; peeling one level.")
        raw = raw[0]

    # --- flat numeric array ---
    if not isinstance(raw[0], dict):
        values = [float(v) for v in raw]
    else:
        # --- array of objects (possibly nested) ---
        if field is None:
            field = _auto_detect_field(raw[0])
            log.info("Auto-selected field '%s'", field)
        else:
            log.info("Using field '%s'", field)

        values = [float(_resolve_path(item, field)) for item in raw]

    if max_points > 0:
        values = values[:max_points]

    arr = np.array(values, dtype=np.float64)
    log.info("Loaded %s points  (min=%.4g  max=%.4g  mean=%.4g)",
             f"{len(arr):,}", arr.min(), arr.max(), arr.mean())
    return arr
