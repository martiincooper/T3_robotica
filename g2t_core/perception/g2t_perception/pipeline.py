"""High-level estimator that ties together preprocessing, clustering,
RANSAC, identification and angle extraction.

Designed to be **stateless per scan** (apart from the optional smoother
and the previous-angle hint), so that single scans are easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional

import numpy as np

from .angles import (estimate_articulation_angles, identify_trailers,
                     TrailerLine)
from .clustering import euclidean_clusters
from .preprocess import gate_range, median_filter, to_xy
from .ransac import LineFit, extract_lines
from .smoothing import ScalarKalman1D


@dataclass
class EstimationResult:
    stamp: float
    psi1_raw: Optional[float] = None
    psi2_raw: Optional[float] = None
    psi1_smoothed: Optional[float] = None
    psi2_smoothed: Optional[float] = None
    # Diagnostics
    num_clusters: int = 0
    num_lines: int = 0
    trailer_lines: List[TrailerLine] = field(default_factory=list)
    points_xy: Optional[np.ndarray] = None


class GeometricEstimator:
    """Run the perception pipeline on a stream of scans."""

    def __init__(self, cfg: dict, sensor_mount_yaw: float = math.pi,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.sensor_mount_yaw = sensor_mount_yaw
        self.rng = rng if rng is not None else np.random.default_rng(0)
        smooth_cfg = cfg.get("smoothing", {})
        self._smooth_enabled = bool(smooth_cfg.get("enable", True))
        if self._smooth_enabled:
            self._k1 = ScalarKalman1D(
                process_std=float(smooth_cfg.get("process_std", 0.05)),
                measurement_std=float(smooth_cfg.get("measurement_std", 0.03)))
            self._k2 = ScalarKalman1D(
                process_std=float(smooth_cfg.get("process_std", 0.05)),
                measurement_std=float(smooth_cfg.get("measurement_std", 0.03)))
        self._last_t: Optional[float] = None
        self._prev_psi1: Optional[float] = None
        self._prev_psi2: Optional[float] = None

    # ----------------------------------------------------------------- run
    def step(self, ranges: np.ndarray, angles: np.ndarray,
             stamp: float) -> EstimationResult:
        cfg = self.cfg
        # 1. preprocess
        r = median_filter(ranges, cfg["preprocess"]["median_window"])
        r = gate_range(r, cfg["preprocess"]["min_range"], 1e6)
        xy = to_xy(r, angles)

        # 2. ROI in the sensor frame
        roi = cfg["roi"]
        # NOTE: ROI is expressed in the LiDAR sensor frame. With the
        # sensor pointing -X in the tractor body frame, the trailers
        # appear in the +X half plane of the sensor frame.
        mask = ((xy[:, 0] >= -roi["x_max"]) & (xy[:, 0] <= -roi["x_min"]) &
                (xy[:, 1] >= roi["y_min"]) & (xy[:, 1] <= roi["y_max"]))
        # The ROI in config is given in tractor body frame coordinates
        # (negative X = behind tractor). The sensor X axis points
        # backwards (mount_yaw = π), so tractor body X = -sensor X.
        # Hence we flip sign for the longitudinal axis.
        roi_xy = xy[mask]

        result = EstimationResult(stamp=stamp, points_xy=roi_xy)
        if roi_xy.shape[0] < cfg["cluster"]["min_points"]:
            self._smooth(result, stamp)
            return result

        # 3. clustering
        clusters = euclidean_clusters(
            roi_xy,
            eps=cfg["cluster"]["eps"],
            min_points=cfg["cluster"]["min_points"],
            max_points=cfg["cluster"]["max_points"])
        result.num_clusters = len(clusters)
        if not clusters:
            self._smooth(result, stamp)
            return result

        # 4. RANSAC line extraction per cluster
        ransac_cfg = cfg["ransac"]
        all_lines: List[LineFit] = []
        for c in clusters:
            all_lines.extend(extract_lines(
                c,
                distance_threshold=ransac_cfg["distance_threshold"],
                iterations=ransac_cfg["iterations"],
                min_inliers_ratio=ransac_cfg["min_inliers_ratio"],
                rng=self.rng))
        result.num_lines = len(all_lines)

        # 5. trailer identification + angle estimation
        classified = identify_trailers(
            all_lines, pick_top_k=cfg["trailer_id"]["pick_top_k_lines"])
        result.trailer_lines = classified
        psi1, psi2, _, _ = estimate_articulation_angles(
            classified, sensor_mount_yaw=self.sensor_mount_yaw,
            prev_psi1=self._prev_psi1, prev_psi2=self._prev_psi2)
        result.psi1_raw = psi1
        result.psi2_raw = psi2
        if psi1 is not None:
            self._prev_psi1 = psi1
        if psi2 is not None:
            self._prev_psi2 = psi2

        # 6. smoothing
        self._smooth(result, stamp)
        return result

    # ------------------------------------------------------------ smoother
    def _smooth(self, result: EstimationResult, stamp: float) -> None:
        if not self._smooth_enabled:
            return
        dt = 0.0 if self._last_t is None else max(1e-3, stamp - self._last_t)
        self._last_t = stamp
        if dt > 0.0 and self._k1.is_initialized:
            self._k1.predict(dt)
            self._k2.predict(dt)
        if result.psi1_raw is not None:
            self._k1.update(result.psi1_raw)
        if result.psi2_raw is not None:
            self._k2.update(result.psi2_raw)
        if self._k1.is_initialized:
            result.psi1_smoothed = self._k1.value
        if self._k2.is_initialized:
            result.psi2_smoothed = self._k2.value
