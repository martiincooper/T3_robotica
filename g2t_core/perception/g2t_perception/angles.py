"""Angle-estimation pipeline.

Given the RANSAC-fitted line segments from one scan, we identify which
lines belong to trailer-1 and trailer-2 and read out the absolute heading
of each trailer in the **sensor frame**.

Geometric model
---------------
Each trailer is a rectangle of length ``L_body`` and width ``W_body`` whose
heading is the longitudinal axis of the rectangle. The LiDAR observes
two types of edges:

* **Side walls** (length = ``L_body`` ≈ 1.6 m): the fitted line direction
  is parallel to the trailer heading.
* **Front / rear walls** (length = ``W_body`` ≈ 0.7 m): the fitted line
  direction is *perpendicular* to the trailer heading.

We disambiguate by line length: lines longer than a threshold are treated
as side walls; shorter ones as end walls. This single discriminator
recovers the trailer heading with a residual 180° ambiguity, which we
resolve using the previous estimate (frame-to-frame continuity) or, on
the first frame, a hint that the trailers lie behind the LiDAR.

Identification (line → trailer)
-------------------------------
Lines are grouped by **midpoint distance** to the LiDAR origin: the
nearest cluster of lines belongs to trailer-1, the next nearest to
trailer-2. This works whenever the trailers are physically separated
along the longitudinal direction (always true for a G2T) and is robust
to partial occlusion because we use *all* detected lines, not just the
single longest one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np

from .ransac import LineFit


@dataclass
class TrailerLine:
    trailer_id: int           # 1 or 2
    line: LineFit
    is_side: bool             # True if treated as a side wall


SIDE_WALL_LENGTH_THRESHOLD = 0.9   # m — lines longer than this are side walls


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def line_to_heading_sensor(line: LineFit) -> Tuple[float, bool]:
    """Return ``(heading_sensor, is_side)`` — unambiguous up to sensor noise.

    The 180° flip of a fitted line direction is resolved geometrically:

    * **End wall** (line shorter than ``SIDE_WALL_LENGTH_THRESHOLD``):
      the trailer body lies *behind* the visible face from the LiDAR's
      point of view; therefore the heading direction is the line normal
      pointing **away from the body** = pointing **toward the LiDAR
      origin**.

    * **Side wall** (line longer): the LiDAR sees the entire length of
      the wall.  The endpoint that is **closer to the LiDAR origin** is
      the front/hitch corner of the trailer; the heading direction is
      therefore the vector from the far endpoint to the near endpoint.

    Both rules use only the LiDAR origin (no hint, no smoothing), so the
    returned heading is consistent across frames *without* relying on
    frame-to-frame continuity for sign disambiguation.
    """
    is_side = line.length >= SIDE_WALL_LENGTH_THRESHOLD
    if is_side:
        # Pick the endpoint closer to the origin as the "front"
        d0 = np.linalg.norm(line.p0)
        d1 = np.linalg.norm(line.p1)
        near, far = (line.p0, line.p1) if d0 <= d1 else (line.p1, line.p0)
        v = near - far
    else:
        # End wall: heading is perpendicular to the line. Pick the normal
        # that points from the line midpoint TOWARD the LiDAR origin
        # (i.e. away from the trailer body, which lies behind the face).
        d = line.direction
        normals = (np.array([d[1], -d[0]]),
                   np.array([-d[1],  d[0]]))
        to_origin = -line.midpoint
        v = normals[0] if normals[0] @ to_origin >= normals[1] @ to_origin \
            else normals[1]
    v = v / (np.linalg.norm(v) + 1e-12)
    return math.atan2(v[1], v[0]), is_side


def identify_trailers(lines: List[LineFit],
                      pick_top_k: int = 4) -> List[TrailerLine]:
    """Assign each detected line to trailer 1 or trailer 2.

    Strategy: keep the ``pick_top_k`` lines with the closest endpoint to
    the LiDAR origin; cluster them in two groups along the radial axis
    using the midpoint distance median; nearest group is trailer-1.
    Within each group we pick the line with the **most inliers**
    (i.e., the most reliable fit, regardless of side/end discrimination).
    """
    if not lines:
        return []
    cand = sorted(lines, key=lambda l: min(np.linalg.norm(l.p0),
                                           np.linalg.norm(l.p1)))[:pick_top_k]
    if len(cand) == 1:
        return [TrailerLine(trailer_id=1, line=cand[0],
                            is_side=cand[0].length >= SIDE_WALL_LENGTH_THRESHOLD)]
    mid_d = np.array([np.linalg.norm(l.midpoint) for l in cand])
    med = float(np.median(mid_d))
    near = [l for l, d in zip(cand, mid_d) if d <= med]
    far = [l for l, d in zip(cand, mid_d) if d > med]
    out: List[TrailerLine] = []
    if near:
        best = max(near, key=lambda l: l.inliers.shape[0])
        out.append(TrailerLine(1, best, best.length >= SIDE_WALL_LENGTH_THRESHOLD))
    if far:
        best = max(far, key=lambda l: l.inliers.shape[0])
        out.append(TrailerLine(2, best, best.length >= SIDE_WALL_LENGTH_THRESHOLD))
    return out


def estimate_articulation_angles(
    classified: List[TrailerLine],
    sensor_mount_yaw: float = math.pi,
    prev_psi1: Optional[float] = None,
    prev_psi2: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float],
           Optional[float], Optional[float]]:
    """Return ``(psi1, psi2, heading1_sensor, heading2_sensor)``.

    Convention
    ----------
    * ``psi_i`` is the *signed* articulation angle ``theta_{i-1} - theta_i``
      (i.e. positive if body i lags clockwise relative to body i-1).
    * ``sensor_mount_yaw`` is the yaw of the sensor frame in the tractor
      body frame. With the LiDAR pointing backwards, this is ``π``.

    Aligned-case derivation (used as the initial hint and as a sanity
    check)::

        theta_tractor_world           = theta_t
        theta_sensor_world            = theta_t + π
        theta_trailer_world (aligned) = theta_t
        therefore heading_in_sensor   = theta_trailer_world − theta_sensor_world
                                      = −π        (or equivalently +π)

    so the initial hint for the sensor-frame heading is ``-π``.
    """
    psi1 = psi2 = None
    h1 = h2 = None

    line1 = next((c for c in classified if c.trailer_id == 1), None)
    line2 = next((c for c in classified if c.trailer_id == 2), None)

    if line1 is not None:
        h1, _ = line_to_heading_sensor(line1.line)
        # psi1 = theta_tractor − theta_trailer1
        #      = -(theta_trailer1 − theta_tractor)
        #      = -(theta_trailer1_sensor + sensor_mount_yaw)    [convert]
        # With sensor_mount_yaw = π and aligned trailer (h1 = -π):
        #   psi1 = -(-π + π) = 0  ✓
        psi1 = _wrap(-(h1 + sensor_mount_yaw))

    if line2 is not None:
        # psi2 = theta_trailer1 − theta_trailer2. If trailer-1 was not
        # detected this scan we fall back to "aligned with tractor".
        h2, _ = line_to_heading_sensor(line2.line)
        h1_for_psi2 = h1 if h1 is not None else -math.pi
        psi2 = _wrap(h1_for_psi2 - h2)

    return psi1, psi2, h1, h2
