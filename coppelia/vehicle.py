"""Impose the G2T articulated pose on the CoppeliaSim bodies.

Co-simulation glue: the validated RK4 kinematics (Guía 2) own the state;
each step we place ``Tractor`` / ``Trailer1`` / ``Trailer2`` (and, as a
child, the LiDAR) at the poses returned by ``G2TKinematics.body_poses``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g2t_core.simulation.g2t_sim.kinematics import (  # noqa: E402
    G2TKinematics, G2TParams, G2TState)

BODY_Z = 0.25  # half of BODY_H used in build_scene


def make_kinematics(cfg: dict) -> G2TKinematics:
    v = cfg["vehicle"]
    import math
    p = G2TParams(L0=v["L0"], d0=v.get("d0", 0.0), L1=v["L1"],
                  d1=v.get("d1", 0.0), L2=v["L2"])
    return G2TKinematics(p, psi_limit=math.radians(v.get("psi_limit_deg", 75.0)))


def find_bodies(sim) -> dict:
    out = {}
    for name in ("Tractor", "Trailer1", "Trailer2"):
        out[name] = sim.getObject("/" + name)
    return out


def place(sim, bodies: dict, kin: G2TKinematics, state: G2TState) -> None:
    tr, t1, t2 = kin.body_poses(state)
    for name, pose in zip(("Tractor", "Trailer1", "Trailer2"), (tr, t1, t2)):
        h = bodies[name]
        sim.setObjectPosition(h, -1, [float(pose[0]), float(pose[1]), BODY_Z])
        sim.setObjectOrientation(h, -1, [0.0, 0.0, float(pose[2])])
