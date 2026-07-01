"""RANSAC line fitting with end-point estimation.

A line in 2-D is parameterized in Hessian normal form ``n·x = d`` with
``|n| = 1``, recovered as the principal direction of the inlier set.

For each cluster we run a single-line RANSAC; trailer side walls are
clean enough that a single line dominates each cluster. If a cluster
contains two walls (the trailer corner is visible) we recurse on the
outliers and keep both lines. That recursion is bounded by
``max_iter_lines = 2`` because no scan can see more than the rear, side
and partial front of a single trailer simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class LineFit:
    """A fitted line segment with its inlier set."""
    direction: np.ndarray   # (2,) unit vector along the line
    point: np.ndarray       # (2,) a point on the line (centroid of inliers)
    inliers: np.ndarray     # (k, 2)
    p0: np.ndarray          # (2,) projected first endpoint
    p1: np.ndarray          # (2,) projected last  endpoint

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def yaw(self) -> float:
        """Heading (atan2) of the line direction in (-pi, pi]."""
        return math.atan2(self.direction[1], self.direction[0])

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)


def _fit_line_lsq(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Total least squares line fit; returns (direction, centroid)."""
    c = points.mean(axis=0)
    centered = points - c
    # SVD of (k, 2) — direction = right-singular vector with largest sigma
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    return direction / np.linalg.norm(direction), c


def _project_endpoints(points: np.ndarray, direction: np.ndarray,
                       centroid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``points`` onto the line and return the two extreme points."""
    t = (points - centroid) @ direction
    i_min = int(t.argmin())
    i_max = int(t.argmax())
    p0 = centroid + t[i_min] * direction
    p1 = centroid + t[i_max] * direction
    return p0, p1


def ransac_line(points: np.ndarray,
                distance_threshold: float = 0.05,
                iterations: int = 200,
                min_inliers_ratio: float = 0.5,
                rng: np.random.Generator | None = None) -> LineFit | None:
    """Run RANSAC for the dominant line in ``points``.

    Returns ``None`` if no model contains at least
    ``min_inliers_ratio * len(points)`` inliers.
    """
    if points.shape[0] < 2:
        return None
    rng = rng if rng is not None else np.random.default_rng(0)
    n = points.shape[0]
    best_inliers = None
    best_count = -1
    for _ in range(iterations):
        idx = rng.choice(n, size=2, replace=False)
        p, q = points[idx[0]], points[idx[1]]
        v = q - p
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v /= nv
        # Distance of every point to the line through (p, v)
        normal = np.array([-v[1], v[0]])
        dist = np.abs((points - p) @ normal)
        mask = dist < distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_inliers = mask
    if best_inliers is None or best_count < max(2, int(min_inliers_ratio * n)):
        return None
    inliers = points[best_inliers]
    direction, centroid = _fit_line_lsq(inliers)
    p0, p1 = _project_endpoints(inliers, direction, centroid)
    return LineFit(direction=direction, point=centroid,
                   inliers=inliers, p0=p0, p1=p1)


def extract_lines(cluster: np.ndarray,
                  distance_threshold: float,
                  iterations: int,
                  min_inliers_ratio: float,
                  max_lines: int = 2,
                  rng: np.random.Generator | None = None) -> List[LineFit]:
    """Extract up to ``max_lines`` lines from a cluster (recursive RANSAC).

    After fitting the dominant line we re-run RANSAC on its outliers; this
    captures the rear + side walls of a single trailer when its corner is
    in the LiDAR FOV.
    """
    lines: List[LineFit] = []
    remaining = cluster
    for _ in range(max_lines):
        fit = ransac_line(remaining, distance_threshold,
                          iterations, min_inliers_ratio, rng=rng)
        if fit is None:
            break
        lines.append(fit)
        # Remove inliers (compare by identity via boolean mask)
        in_set = {tuple(p) for p in fit.inliers}
        keep = np.array([tuple(p) not in in_set for p in remaining])
        remaining = remaining[keep]
        if remaining.shape[0] < 8:
            break
    return lines
