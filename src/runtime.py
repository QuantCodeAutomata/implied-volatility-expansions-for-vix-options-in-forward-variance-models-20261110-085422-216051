"""Runtime benchmarking: median wall-clock time over repeated calls (Sec. 2.3)."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def median_runtime_ms(func: Callable[[], None], n_runs: int) -> float:
    """Median wall-clock runtime in milliseconds over n_runs calls to func()."""
    times = np.empty(n_runs)
    for i in range(n_runs):
        start = time.perf_counter()
        func()
        times[i] = (time.perf_counter() - start) * 1000.0
    return float(np.median(times))
