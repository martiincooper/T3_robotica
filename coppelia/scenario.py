"""Guía 3 scenario loader + map builder.

Single source of truth for the Guía 3 environment. Reads
``config/scenario_g3.yaml`` and materializes:

* ``landmarks``  : (N, 4) array [id, x, y, radius] — SLAM landmarks / obstacles.
* ``wall_segments`` : (M, 4) array [x0, y0, x1, y1] — bounding walls.
* start / goal poses, reference waypoints.

The same object is consumed by:
  * ``coppelia/build_scene.py``   (creates the CoppeliaSim scene),
  * ``coppelia/record_dataset.py`` (drives the G2T + records the dataset),
  * ``slam/`` and ``planning/``     (obstacle / landmark ground truth),
  * ``tools/preview_scenario.py``  (report figure, runnable without CoppeliaSim).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


@dataclass
class Pose:
    x: float
    y: float
    theta: float = 0.0
    psi1: float = 0.0
    psi2: float = 0.0

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Scenario:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.size = np.array(cfg["world"]["size"], float)

        s = cfg["start"]
        self.start = Pose(s["x"], s["y"], math.radians(s["theta_deg"]),
                          math.radians(s.get("psi1_deg", 0.0)),
                          math.radians(s.get("psi2_deg", 0.0)))
        g = cfg["goal"]
        self.goal = Pose(g["x"], g["y"], math.radians(g["theta_deg"]))

        # ---- landmarks / obstacles -------------------------------------
        lms = [[l["id"], l["x"], l["y"], l["radius"]] for l in cfg["landmarks"]]
        ng = cfg.get("narrow_gap", {})
        if ng.get("enable", False):
            cx, cy = ng["center"]
            half = ng["gap_width"] / 2.0 + ng["radius"]
            nid = max(l[0] for l in lms) + 1
            # two cylinders straddling the corridor centreline (perp. to +x)
            lms.append([nid,     cx, cy + half, ng["radius"]])
            lms.append([nid + 1, cx, cy - half, ng["radius"]])
        self.landmarks = np.asarray(lms, float)          # (N,4): id,x,y,r

        # ---- walls ------------------------------------------------------
        self.wall_segments = self._build_walls() if cfg.get("walls", {}).get(
            "enable", False) else np.zeros((0, 4))

        self.reference_waypoints = np.asarray(cfg["reference_waypoints"], float)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        with open(path, "r") as f:
            return cls(yaml.safe_load(f))

    # --------------------------------------------------------------- walls
    def _build_walls(self) -> np.ndarray:
        w, h = self.size
        x0, y0 = -w / 2.0, -h / 2.0
        x1, y1 = w / 2.0, h / 2.0
        return np.array([
            [x0, y0, x1, y0],
            [x1, y0, x1, y1],
            [x1, y1, x0, y1],
            [x0, y1, x0, y0],
        ], float)

    # ------------------------------------------------- geometry accessors
    def circles(self) -> np.ndarray:
        """(N,3) [x,y,radius] for LiDAR ray-casting / collision checks."""
        return self.landmarks[:, 1:4].copy()

    def landmark_xy(self) -> np.ndarray:
        return self.landmarks[:, 1:3].copy()

    def in_collision(self, x: float, y: float, clearance: float = 0.0) -> bool:
        """True if point (x,y) is within (radius+clearance) of any obstacle
        or outside the walls."""
        w, h = self.size
        if abs(x) > w / 2.0 - clearance or abs(y) > h / 2.0 - clearance:
            return True
        d = np.hypot(self.landmarks[:, 1] - x, self.landmarks[:, 2] - y)
        return bool(np.any(d <= self.landmarks[:, 3] + clearance))

    def min_obstacle_distance(self, x: float, y: float) -> float:
        d = np.hypot(self.landmarks[:, 1] - x, self.landmarks[:, 2] - y) \
            - self.landmarks[:, 3]
        return float(d.min())
