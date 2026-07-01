"""Render a 3-D image of the G3 scene from CoppeliaSim and save it as
report/figures/coppelia_scene.png (visual evidence of the environment/model).

Adds a perspective vision sensor overlooking the scene, places the G2T at
the start pose so the articulated vehicle is visible, renders one frame and
writes the PNG. Works even when the interactive 3-D viewport is not framed,
because it uses the same off-screen vision-sensor renderer CoppeliaSim already
runs.

    python coppelia/capture_scene.py            # bird's-eye 3/4 view
    python coppelia/capture_scene.py --top      # top-down view
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coppelia._client import connect, version_string
from coppelia.scenario import Scenario
from coppelia.vehicle import make_kinematics, find_bodies, place
from g2t_core.simulation.g2t_sim.kinematics import G2TState


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Return CoppeliaSim pose [x,y,z,qx,qy,qz,qw] with local +Z looking at
    ``target`` (vision-sensor view axis is +Z)."""
    eye = np.array(eye, float); target = np.array(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    up = np.array(up, float)
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    # rotation matrix -> quaternion (x,y,z,w)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return [float(eye[0]), float(eye[1]), float(eye[2]),
            float(qx), float(qy), float(qz), float(qw)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="coppelia/g3_scene.ttt")
    ap.add_argument("--out", default="report/figures/coppelia_scene.png")
    ap.add_argument("--top", action="store_true", help="top-down instead of 3/4")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 900])
    args = ap.parse_args()

    sc = Scenario.load(ROOT / "config/scenario_g3.yaml")
    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / args.scene))

    # place the G2T at the start pose so the articulated vehicle is visible
    bodies = find_bodies(sim); kin = make_kinematics(sc.cfg); st = sc.start
    place(sim, bodies, kin, G2TState(st.x, st.y, st.theta, st.psi1, st.psi2))

    # brighten ambient light so the off-screen render is not dark even if the
    # scene's default lights were removed when the scene was built with --clear
    for setter in (
        lambda: sim.setArrayParam(sim.arrayparam_ambient_light, [0.8, 0.8, 0.8]),
    ):
        try:
            setter()
        except Exception as exc:
            print(f"[capture] ambient light note: {exc}")

    # camera (vision sensor) overlooking the whole scene
    w, h = sc.size
    cam = sim.createVisionSensor(1, [args.res[0], args.res[1], 0, 0],
                                 [0.05, 200.0, 60.0 * math.pi / 180.0,
                                  0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim.setObjectAlias(cam, "ReportCam")
    # force perspective projection + clipping + FoV explicitly (robust to
    # createVisionSensor param-layout differences across versions)
    for fn in (
        lambda: sim.setObjectInt32Param(cam, sim.visionintparam_perspective_operation, 1),
        lambda: sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, math.radians(60)),
        lambda: sim.setObjectFloatParam(cam, sim.visionfloatparam_near_clipping, 0.05),
        lambda: sim.setObjectFloatParam(cam, sim.visionfloatparam_far_clipping, 200.0),
    ):
        try:
            fn()
        except Exception as exc:
            print(f"[capture] cam param note: {exc}")
    if args.top:
        pose = _look_at([0, 0, max(w, h) * 1.3], [0, 0, 0], up=(0, 1, 0))
    else:
        pose = _look_at([-w * 0.75, -h * 1.1, max(w, h) * 0.7], [0, 0, 0])
    sim.setObjectPose(cam, -1, pose)

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(3):
        sim.step()
    sim.handleVisionSensor(cam)
    img, res = sim.getVisionSensorImg(cam)
    sim.stopSimulation()
    sim.removeObjects([cam])

    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(img, dtype=np.uint8)
    else:
        arr = np.array(img, dtype=np.uint8)
    arr = arr.reshape(res[1], res[0], 3)[::-1]  # flip vertical (bottom-up)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out), arr)
    print(f"wrote {out}  ({res[0]}x{res[1]})  mean brightness={arr.mean():.1f}")
    if arr.mean() < 8:
        print("[capture] WARNING: image is very dark. The scene likely has no "
              "lights (removed by build_scene --clear). Rebuild the scene "
              "WITHOUT --clear, or add a light in the CoppeliaSim GUI "
              "([Add > Light > Omnidirectional]) and rerun.")


if __name__ == "__main__":
    main()
