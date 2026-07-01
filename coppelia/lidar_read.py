"""Read a single 2-D scan from the native SICK S300 Fast scanner.

The scanner is two 135-degree perspective vision sensors (256x1 each).
``getVisionSensorDepth(h, 1)`` returns the **perpendicular** metric depth
per pixel; we back-project each pixel to a 3-D ray in the sensor frame,
transform it into a chosen reference frame using the sensor's queried
pose, and reduce to a horizontal (bearing, range) pair. Returning points
via the queried pose makes the result independent of the exact sensor
mounting orientation (we never hard-code the 90-degree rolls).
"""
from __future__ import annotations

import math
import struct
from typing import List, Tuple

import numpy as np


def _quat_to_R(q) -> np.ndarray:
    # CoppeliaSim pose quaternion order: (qx, qy, qz, qw)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def find_scanner(sim, root_alias: str = "LiDAR"):
    """Return (sensor_handles, ref_handle) for the mounted scanner."""
    root = sim.getObject("/" + root_alias)
    tree = sim.getObjectsInTree(root, sim.handle_all, 0)
    sensors, ref = [], root
    for h in tree:
        if sim.getObjectType(h) == sim.object_visionsensor_type:
            sensors.append(h)
        if sim.getObjectAlias(h, 0) == "ref":
            ref = h
    return sorted(sensors), ref


def set_range(sim, sensors, far_m: float) -> None:
    for h in sensors:
        sim.setObjectFloatParam(h, sim.visionfloatparam_far_clipping,
                                float(far_m))


def _unpack_depth(depth) -> np.ndarray:
    if isinstance(depth, (bytes, bytearray)):
        n = len(depth) // 4
        return np.array(struct.unpack(f"{n}f", depth), float)
    return np.asarray(depth, float)


def read_scan(sim, sensors, frame, far_m: float, view_sign: float = 1.0
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (angles[rad], ranges[m], points_xy) in ``frame`` coordinates,
    sorted by bearing. ``view_sign`` flips the optical axis (+Z/-Z) if the
    visual check comes out front/back mirrored."""
    ang: List[float] = []
    rng: List[float] = []
    pts: List[Tuple[float, float]] = []
    for h in sensors:
        depth, res = sim.getVisionSensorDepth(h, 1)  # metric range along beam
        rx, ry = res
        vals = _unpack_depth(depth)
        fov = sim.getObjectFloatParam(h, sim.visionfloatparam_perspective_angle)
        tan_half = math.tan(fov / 2.0)
        # Use the sensor's ACTUAL far clipping as the no-return threshold,
        # so a miss (clamped to far) is dropped whatever the current range.
        far_h = sim.getObjectFloatParam(h, sim.visionfloatparam_far_clipping)
        thresh = min(far_m, far_h) * 0.999
        pose = sim.getObjectPose(h, frame)
        R = _quat_to_R(pose[3:]); t = np.array(pose[:3])
        for i in range(rx):
            d = float(vals[i])
            if not math.isfinite(d) or d >= thresh or d <= 1e-3:
                continue
            # s_sign = -1 fixed by lidar_calibrate.py (score 1.00 vs scene).
            s = -((2.0 * (i + 0.5) / rx) - 1.0)        # (-1, 1)
            # getVisionSensorDepth returns PERPENDICULAR depth (z along the
            # optical axis), so the 3-D point is [s*tan_half*d, 0, d]. The
            # euclidean beam range then follows from the point itself.
            Pc = np.array([s * tan_half * d, 0.0, view_sign * d])
            Pr = R @ Pc + t
            ang.append(math.atan2(Pr[1], Pr[0]))
            rng.append(math.hypot(Pr[0], Pr[1]))
            pts.append((Pr[0], Pr[1]))
    if not ang:
        return np.zeros(0), np.zeros(0), np.zeros((0, 2))
    order = np.argsort(ang)
    return (np.asarray(ang)[order], np.asarray(rng)[order],
            np.asarray(pts)[order])
