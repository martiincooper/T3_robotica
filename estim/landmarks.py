"""Landmark front-end for EKF-SLAM.

Each 2-D scan contains returns from both the bounding walls and the
cylindrical landmarks. This module segments the scan (sensor frame) into
clusters, keeps the compact, near-circular ones (cylinders) and rejects
the elongated ones (walls), and returns a range-bearing observation to
each estimated cylinder CENTRE.

A cluster is accepted as a cylinder if:
  * it has >= ``min_pts`` points,
  * its spatial extent (max pairwise chord) <= ``max_extent`` m,
  * a circle fit has small residual and radius in ``radius_range``.

The centre is recovered from the visible arc by a least-squares circle
fit (Kasa), which is robust for the >0.15 m radius cylinders used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LMObs:
    rng: float        # range to estimated centre [m]
    bearing: float    # bearing in sensor frame [rad]
    xy: np.ndarray    # estimated centre in sensor frame (2,)
    n: int            # supporting points


def _cluster(points: np.ndarray, eps: float) -> List[np.ndarray]:
    """Split points (assumed ordered by bearing) at gaps > eps."""
    if len(points) == 0:
        return []
    clusters, cur = [], [0]
    for i in range(1, len(points)):
        if np.hypot(*(points[i] - points[i - 1])) > eps:
            clusters.append(points[cur]); cur = []
        cur.append(i)
    clusters.append(points[cur])
    return [c for c in clusters if len(c) > 0]


def _fit_circle(pts: np.ndarray):
    """Kasa circle fit -> (cx, cy, r, rms_residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx * cx + cy * cy, 1e-9))
    res = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
    return cx, cy, r, res


def extract_landmarks(points: np.ndarray,
                      eps: float = 0.35,
                      min_pts: int = 4,
                      max_extent: float = 1.2,
                      radius_range=(0.12, 0.6),
                      max_resid: float = 0.08) -> List[LMObs]:
    obs: List[LMObs] = []
    for c in _cluster(points, eps):
        if len(c) < min_pts:
            continue
        extent = np.hypot(*(c.max(0) - c.min(0)))
        if extent > max_extent:
            continue                                   # wall / long segment
        cx, cy, r, res = _fit_circle(c)
        if not (radius_range[0] <= r <= radius_range[1]) or res > max_resid:
            continue                                   # not a clean cylinder
        obs.append(LMObs(rng=math.hypot(cx, cy),
                         bearing=math.atan2(cy, cx),
                         xy=np.array([cx, cy]), n=len(c)))
    return obs


def to_world(obs: LMObs, sensor_pose) -> np.ndarray:
    x, y, yaw = sensor_pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * obs.xy[0] - s * obs.xy[1],
                     y + s * obs.xy[0] + c * obs.xy[1]])
