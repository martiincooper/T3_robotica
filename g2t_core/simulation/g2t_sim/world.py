"""World / obstacle representation for the synthetic environment.

The world holds:
  * the kinematic vehicle (whose body geometry is rebuilt every scan
    from the current state via :func:`Lidar2D` segment-list helpers),
  * static cylindrical obstacles,
  * dynamic cylindrical "pedestrians" with parametric trajectories.

Dynamic obstacles are deterministic functions of time (sinusoidal), which
keeps the dataset fully reproducible from a single seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

import numpy as np

from .lidar import rectangle_segments


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float

    def pose_at(self, t: float) -> np.ndarray:  # noqa: ARG002 — static
        return np.array([self.x, self.y, self.radius])


@dataclass
class Pedestrian:
    """Distractor cylinder oscillating along one axis."""
    x0: float
    y0: float
    amp: float
    freq_hz: float
    axis: str        # "x" or "y"
    radius: float

    def pose_at(self, t: float) -> np.ndarray:
        delta = self.amp * math.sin(2 * math.pi * self.freq_hz * t)
        if self.axis == "x":
            return np.array([self.x0 + delta, self.y0, self.radius])
        return np.array([self.x0, self.y0 + delta, self.radius])


@dataclass
class BodyGeom:
    length: float
    width: float


class World:
    """Container that materializes the segment / circle arrays expected
    by :class:`Lidar2D`.
    """

    def __init__(self,
                 tractor_geom: BodyGeom,
                 trailer1_geom: BodyGeom,
                 trailer2_geom: BodyGeom,
                 obstacles: List[Obstacle],
                 pedestrians: List[Pedestrian]):
        self.tractor = tractor_geom
        self.trailer1 = trailer1_geom
        self.trailer2 = trailer2_geom
        self.obstacles = obstacles
        self.pedestrians = pedestrians

    # ----------------------------------------------------------- segments
    def vehicle_segments(self,
                         tractor_pose: np.ndarray,
                         trailer1_pose: np.ndarray,
                         trailer2_pose: np.ndarray) -> np.ndarray:
        """Build world-frame edges of all three bodies.

        Each body pose is anchored at its **rear axle** (consistent with
        :meth:`G2TKinematics.body_poses`), so the rectangle is shifted by
        ``+length/2`` along the heading to obtain its geometric center.
        """
        def body(pose: np.ndarray, geom: BodyGeom) -> np.ndarray:
            cx = pose[0] + 0.5 * geom.length * math.cos(pose[2])
            cy = pose[1] + 0.5 * geom.length * math.sin(pose[2])
            return rectangle_segments(cx, cy, pose[2], geom.length, geom.width)

        segs = np.vstack([
            body(tractor_pose, self.tractor),
            body(trailer1_pose, self.trailer1),
            body(trailer2_pose, self.trailer2),
        ])
        return segs

    def trailer_only_segments(self,
                              trailer1_pose: np.ndarray,
                              trailer2_pose: np.ndarray) -> np.ndarray:
        """Body edges of the trailers only (excludes the tractor)."""
        def body(pose: np.ndarray, geom: BodyGeom) -> np.ndarray:
            cx = pose[0] + 0.5 * geom.length * math.cos(pose[2])
            cy = pose[1] + 0.5 * geom.length * math.sin(pose[2])
            return rectangle_segments(cx, cy, pose[2], geom.length, geom.width)
        return np.vstack([body(trailer1_pose, self.trailer1),
                          body(trailer2_pose, self.trailer2)])

    def circles_at(self, t: float) -> np.ndarray:
        circs = [o.pose_at(t) for o in self.obstacles] + \
                [p.pose_at(t) for p in self.pedestrians]
        if not circs:
            return np.zeros((0, 3))
        return np.asarray(circs)
