"""Numeric diagnostic for the LiDAR reader. Prints concrete per-pixel
world points so we can see WHAT the beams hit (floor? walls? cylinders?).

    python coppelia/lidar_diag.py
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coppelia._client import connect, version_string
from coppelia.scenario import Scenario
from coppelia.vehicle import make_kinematics, find_bodies, place
from coppelia.lidar_read import find_scanner, set_range, _quat_to_R
from g2t_core.simulation.g2t_sim.kinematics import G2TState

FAR = 15.0


def main() -> None:
    sc = Scenario.load(ROOT / "config/scenario_g3.yaml")
    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / "coppelia/g3_scene.ttt"))
    bodies = find_bodies(sim)
    kin = make_kinematics(sc.cfg)
    st = sc.start
    place(sim, bodies, kin, G2TState(st.x, st.y, st.theta, st.psi1, st.psi2))
    sensors, ref = find_scanner(sim)
    set_range(sim, sensors, FAR)
    sim.setStepping(True); sim.startSimulation()
    for _ in range(3):
        sim.step()

    print(f"tractor start = ({st.x},{st.y}), walls at x=+-{sc.size[0]/2}, "
          f"y=+-{sc.size[1]/2}")
    for h in sensors:
        depth, res = sim.getVisionSensorDepth(h, 1)
        rx, ry = res
        vals = (np.array(struct.unpack(f"{len(depth)//4}f", depth))
                if isinstance(depth, (bytes, bytearray)) else np.asarray(depth))
        fov = sim.getObjectFloatParam(h, sim.visionfloatparam_perspective_angle)
        tan_half = math.tan(fov / 2)
        pw = sim.getObjectPose(h, sim.handle_world)
        R = _quat_to_R(pw[3:]); t = np.array(pw[:3])
        # local axis world directions
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        print(f"\nsensor {h}: world pos={[round(v,2) for v in t]}")
        print(f"  FOV={math.degrees(fov):.1f}  depth[min/max]="
              f"{vals.min():.2f}/{vals.max():.2f}")
        print(f"  localX->world={[round(v,2) for v in ex]}")
        print(f"  localY->world={[round(v,2) for v in ey]}")
        print(f"  localZ->world={[round(v,2) for v in ez]}  (view axis?)")
        for i in [0, rx // 2, rx - 1]:
            d = float(vals[i]); s = (2 * (i + 0.5) / rx) - 1
            ray = np.array([s * tan_half, 0.0, 1.0]); ray /= np.linalg.norm(ray)
            P = R @ (d * ray) + t
            print(f"  px{i:3d} d={d:5.2f} -> world=({P[0]:6.2f},{P[1]:6.2f},"
                  f"{P[2]:5.2f})")
    sim.stopSimulation()


if __name__ == "__main__":
    main()
