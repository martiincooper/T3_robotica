"""Synthetic 2D LiDAR sensor.

Models a Hokuyo UTM-30LX-EW class scanner. The sensor model is
intentionally simple but captures the four phenomena that matter for the
downstream perception pipeline (justified in ``docs/methodology.md``):

  1. **Quantization** of bearings (fixed angular resolution).
  2. **Gaussian range noise** with constant 1-sigma.
   3. **Random dropouts** (no-return rays).
  4. **Mixed pixels** that occasionally return a long, corrupted range,
     emulating the response near depth discontinuities.

The world is composed of:
  * Rectangular trailer side walls (4 line segments per trailer).
  * Cylindrical static obstacles.
  * Cylindrical dynamic distractors (pedestrians) whose pose at the
    scan timestamp is queried from the :class:`World`.

Why analytical ray-casting instead of a mesh-based simulator?
  * Determinism and reproducibility on every platform.
  * No GPU / installation overhead — runs in CI.
  * Closed-form ray–segment and ray–circle intersection are O(N) per ray
    and well below 1 ms for our 1080-beam scans.

Limitations
-----------
* No multi-bounce, no transparency, no reflectivity. The CoppeliaSim
  scene provided in ``ros_ws`` is the way to study those.
* Pedestrians are 2-D cylinders (no leg / body articulation).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class LidarParams:
    range_min: float = 0.05
    range_max: float = 30.0
    fov_deg: float = 270.0
    angular_resolution_deg: float = 0.25
    range_noise_std: float = 0.02
    dropout_prob: float = 0.01
    mixed_pixel_prob: float = 0.002
    rate_hz: float = 40.0

    @property
    def num_beams(self) -> int:
        return int(round(self.fov_deg / self.angular_resolution_deg)) + 1

    def angles(self) -> np.ndarray:
        """Beam angles in the sensor frame, symmetric around 0."""
        half = math.radians(self.fov_deg) / 2.0
        return np.linspace(-half, half, self.num_beams)


@dataclass
class LaserScan:
    """Plain-old-data laser scan, mirrors ``sensor_msgs/msg/LaserScan``."""

    stamp: float
    angles: np.ndarray            # (N,) sensor-frame bearings [rad]
    ranges: np.ndarray            # (N,) ranges [m], +inf == no return
    range_min: float
    range_max: float
    frame_id: str = "lidar"

    def to_xy(self) -> np.ndarray:
        """Project valid returns to Cartesian sensor-frame points (M,2)."""
        valid = np.isfinite(self.ranges) & (self.ranges < self.range_max)
        r = self.ranges[valid]
        a = self.angles[valid]
        return np.column_stack([r * np.cos(a), r * np.sin(a)])


class Lidar2D:
    """Analytical 2-D ray-caster against segments and circles."""

    def __init__(self, params: LidarParams, rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self._angles = params.angles()

    # ----------------------------------------------------------------- main
    def scan(
        self,
        sensor_pose: Tuple[float, float, float],
        segments: np.ndarray,
        circles: np.ndarray,
        stamp: float,
    ) -> LaserScan:
        """Produce a noisy scan (fully vectorized over beams)."""
        x0, y0, yaw = sensor_pose
        beam_world = self._angles + yaw
        cos_b = np.cos(beam_world)              # (B,)
        sin_b = np.sin(beam_world)
        B = self._angles.shape[0]
        ranges = np.full(B, np.inf, dtype=float)

        # ---------- segments: vectorized (B x S) --------------------------
        if segments.size:
            px = segments[:, 0]; py = segments[:, 1]
            qx = segments[:, 2]; qy = segments[:, 3]
            ex = qx - px;        ey = qy - py
            # det[B,S] = dx_b * (-ey_s) - dy_b * (-ex_s)
            det = cos_b[:, None] * (-ey)[None, :] - sin_b[:, None] * (-ex)[None, :]
            rx = (px - x0)[None, :]
            ry = (py - y0)[None, :]
            nz = np.abs(det) > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (rx * (-ey)[None, :] - ry * (-ex)[None, :]) / det
                u = (cos_b[:, None] * ry - sin_b[:, None] * rx) / det
            valid = nz & (t > 0) & (u >= 0.0) & (u <= 1.0)
            t = np.where(valid, t, np.inf)
            ranges = np.minimum(ranges, t.min(axis=1))

        # ---------- circles: vectorized (B x C) ---------------------------
        if circles.size:
            cx = circles[:, 0]; cy = circles[:, 1]; r = circles[:, 2]
            fx = x0 - cx; fy = y0 - cy
            b = 2.0 * (fx[None, :] * cos_b[:, None] + fy[None, :] * sin_b[:, None])
            c = (fx * fx + fy * fy - r * r)[None, :]
            disc = b * b - 4.0 * c
            sqd = np.sqrt(np.maximum(disc, 0.0))
            t1 = (-b - sqd) * 0.5
            t2 = (-b + sqd) * 0.5
            t1 = np.where((disc >= 0.0) & (t1 > 1e-6), t1, np.inf)
            t2 = np.where((disc >= 0.0) & (t2 > 1e-6), t2, np.inf)
            tc = np.minimum(t1, t2).min(axis=1)
            ranges = np.minimum(ranges, tc)

        # ---------- range clipping ----------------------------------------
        too_close = ranges < self.params.range_min
        ranges[too_close] = self.params.range_max  # treat as no return
        too_far = ranges > self.params.range_max
        ranges[too_far] = np.inf

        # ---------- noise model -------------------------------------------
        valid = np.isfinite(ranges)
        ranges[valid] += self.rng.normal(
            0.0, self.params.range_noise_std, size=valid.sum())

        drop = self.rng.random(ranges.shape[0]) < self.params.dropout_prob
        ranges[drop] = np.inf

        mp = self.rng.random(ranges.shape[0]) < self.params.mixed_pixel_prob
        ranges[mp] = self.rng.uniform(
            self.params.range_max * 0.5, self.params.range_max, size=mp.sum())

        return LaserScan(
            stamp=stamp,
            angles=self._angles.copy(),
            ranges=ranges,
            range_min=self.params.range_min,
            range_max=self.params.range_max,
        )


# ===================================================================== math
def _ray_segments_intersect(
    x0: float, y0: float, dx: float, dy: float, segs: np.ndarray
) -> float:
    """Smallest positive ray parameter t for intersection with any segment.

    Solves ``[dx, -ex; dy, -ey] * [t, u] = [px - x0, py - y0]`` for each
    segment, where ``(ex, ey) = (qx - px, qy - py)`` is the segment's
    direction. A hit is valid iff ``t > 0`` and ``0 <= u <= 1``.
    Returns +inf if there is no hit.
    """
    px, py = segs[:, 0], segs[:, 1]
    qx, qy = segs[:, 2], segs[:, 3]
    ex, ey = qx - px, qy - py

    det = dx * (-ey) - dy * (-ex)        # = -dx*ey + dy*ex
    # Avoid div-by-zero for parallel rays
    nz = np.abs(det) > 1e-12

    t = np.full(segs.shape[0], np.inf)
    u = np.full(segs.shape[0], np.inf)
    rx = px - x0
    ry = py - y0
    # Cramer's rule on the 2x2 system
    t[nz] = (rx[nz] * (-ey[nz]) - ry[nz] * (-ex[nz])) / det[nz]
    u[nz] = (dx * ry[nz] - dy * rx[nz]) / det[nz]

    valid = nz & (t > 0) & (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return float("inf")
    return float(t[valid].min())


def _ray_circles_intersect(
    x0: float, y0: float, dx: float, dy: float, circles: np.ndarray
) -> float:
    """Smallest positive ray parameter t for intersection with any circle.

    Solves ``|origin + t * d - c|^2 = r^2``; selects the smaller positive
    root if any. Direction ``(dx, dy)`` is assumed unit length.
    """
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    fx, fy = x0 - cx, y0 - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    hit = disc >= 0.0
    if not np.any(hit):
        return float("inf")
    sqd = np.sqrt(disc[hit])
    t1 = (-b[hit] - sqd) * 0.5
    t2 = (-b[hit] + sqd) * 0.5
    # Pick smallest positive root per circle, then min across circles
    t1 = np.where(t1 > 1e-6, t1, np.inf)
    t2 = np.where(t2 > 1e-6, t2, np.inf)
    return float(np.minimum(t1, t2).min())


# ================================================================ geometry
def rectangle_segments(cx: float, cy: float, yaw: float,
                       length: float, width: float) -> np.ndarray:
    """Return the 4 world-frame edges of a centered rectangle as segments."""
    hl, hw = length / 2.0, width / 2.0
    # Corners in body frame, ordered FL, FR, RR, RL
    corners_body = np.array([[ hl,  hw],
                             [ hl, -hw],
                             [-hl, -hw],
                             [-hl,  hw]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_world = corners_body @ R.T + np.array([cx, cy])
    segs = []
    for i in range(4):
        x0, y0 = corners_world[i]
        x1, y1 = corners_world[(i + 1) % 4]
        segs.append([x0, y0, x1, y1])
    return np.asarray(segs)
