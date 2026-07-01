"""Objectively pick the LiDAR reconstruction convention.

Takes one scan at the start pose and tries the 4 sign conventions
(optical axis +Z/-Z x parallax +/-). For each it scores the fraction of
returns that land within a tolerance of a known wall segment or cylinder
surface, then reports the best and saves a 4-panel overlay.

    python coppelia/lidar_calibrate.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coppelia._client import connect, version_string
from coppelia.scenario import Scenario
from coppelia.vehicle import make_kinematics, find_bodies, place
from coppelia.lidar_read import find_scanner, set_range, _quat_to_R, _unpack_depth
from g2t_core.simulation.g2t_sim.kinematics import G2TState

FAR = 5.0
TOL = 0.35


def scan_variant(sim, sensors, frame, view_sign, s_sign):
    pts = []
    for h in sensors:
        depth, res = sim.getVisionSensorDepth(h, 1)
        rx, _ = res
        vals = _unpack_depth(depth)
        fov = sim.getObjectFloatParam(h, sim.visionfloatparam_perspective_angle)
        th = math.tan(fov / 2)
        far_h = sim.getObjectFloatParam(h, sim.visionfloatparam_far_clipping)
        pose = sim.getObjectPose(h, frame)
        R = _quat_to_R(pose[3:]); t = np.array(pose[:3])
        for i in range(rx):
            d = float(vals[i])
            if d >= far_h * 0.999 or d <= 1e-3:
                continue
            s = s_sign * ((2 * (i + 0.5) / rx) - 1)
            Pc = np.array([s * th * d, 0.0, view_sign * d])
            Pr = R @ Pc + t
            pts.append((Pr[0], Pr[1]))
    return np.asarray(pts) if pts else np.zeros((0, 2))


def dist_to_scene(sc: Scenario, pts):
    if len(pts) == 0:
        return np.zeros(0)
    d = np.full(len(pts), 1e9)
    # cylinders
    for _, x, y, r in sc.landmarks:
        d = np.minimum(d, np.abs(np.hypot(pts[:, 0] - x, pts[:, 1] - y) - r))
    # wall segments
    for x0, y0, x1, y1 in sc.wall_segments:
        a = np.array([x0, y0]); b = np.array([x1, y1]); ab = b - a
        tt = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        proj = a + tt[:, None] * ab
        d = np.minimum(d, np.hypot(pts[:, 0] - proj[:, 0], pts[:, 1] - proj[:, 1]))
    return d


def main() -> None:
    sc = Scenario.load(ROOT / "config/scenario_g3.yaml")
    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / "coppelia/g3_scene.ttt"))
    bodies = find_bodies(sim); kin = make_kinematics(sc.cfg); st = sc.start
    place(sim, bodies, kin, G2TState(st.x, st.y, st.theta, st.psi1, st.psi2))
    sensors, ref = find_scanner(sim)
    sim.setStepping(True); sim.startSimulation()
    sim.step(); set_range(sim, sensors, FAR); sim.step()

    combos = [(vs, ss) for vs in (1, -1) for ss in (1, -1)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    best = None
    for ax, (vs, ss) in zip(axes.ravel(), combos):
        pts = scan_variant(sim, sensors, sim.handle_world, vs, ss)
        dd = dist_to_scene(sc, pts)
        score = float(np.mean(dd < TOL)) if len(dd) else 0.0
        print(f"view_sign={vs:+d} s_sign={ss:+d}: {len(pts)} pts, "
              f"score(<{TOL}m)={score:.2f}")
        if best is None or score > best[0]:
            best = (score, vs, ss)
        for x0, y0, x1, y1 in sc.wall_segments:
            ax.plot([x0, x1], [y0, y1], "k-", lw=1.5)
        for _, x, y, r in sc.landmarks:
            ax.add_patch(Circle((x, y), r, color="0.8"))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=5, c="tab:red")
        ax.scatter([st.x], [st.y], c="tab:green", s=60)
        ax.set_aspect("equal"); ax.set_title(
            f"view={vs:+d} s={ss:+d}  score={score:.2f}")
        ax.grid(alpha=0.3)
    sim.stopSimulation()
    print(f"\nBEST: view_sign={best[1]:+d} s_sign={best[2]:+d} "
          f"(score={best[0]:.2f})")
    out = ROOT / "figures/lidar_calibration.png"
    fig.suptitle("LiDAR convention calibration vs known scene")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
