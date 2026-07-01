"""Pre-processing of polar LaserScan data.

Two cheap but effective filters:

* **Median filter** on the range vector (window 3) suppresses isolated
  drop-outs and mixed pixels without blurring trailer wall edges.
* **Range gating** rejects sensor self-returns and out-of-range readings.

We do NOT apply outlier removal in Cartesian space here — that is done
later by Euclidean clustering, which has a more meaningful notion of
"outlier" for our geometry.
"""

from __future__ import annotations

import numpy as np


def median_filter(ranges: np.ndarray, w: int = 3) -> np.ndarray:
    """1-D median filter on a range vector, treating ``inf`` as missing."""
    if w < 2:
        return ranges.copy()
    half = w // 2
    n = ranges.shape[0]
    out = ranges.copy()
    # Build a (n, w) sliding window with edge replication
    padded = np.pad(ranges, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, w)
    # Median ignoring inf
    with np.errstate(invalid="ignore"):
        out = np.nanmedian(np.where(np.isinf(win), np.nan, win), axis=1)
    out = np.where(np.isnan(out), np.inf, out)
    return out


def gate_range(ranges: np.ndarray, min_r: float, max_r: float) -> np.ndarray:
    out = ranges.copy()
    out[out < min_r] = np.inf
    out[out > max_r] = np.inf
    return out


def to_xy(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Polar -> Cartesian (sensor frame). Returns (M, 2) for finite ranges."""
    valid = np.isfinite(ranges)
    r = ranges[valid]
    a = angles[valid]
    return np.column_stack([r * np.cos(a), r * np.sin(a)])
