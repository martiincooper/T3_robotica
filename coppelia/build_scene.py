"""Build the Guía 3 G2T scene inside a running CoppeliaSim.

Co-simulation design
--------------------
CoppeliaSim owns the *scene, ground truth and visuals*; the validated
G2T RK4 kinematics (reused from Guía 2, ``g2t_core/simulation``) drive
the articulated pose each step (imposed via ``setObjectPose``). This
keeps the vehicle model identical to the one the EKF/EKF-SLAM assume,
while producing genuine CoppeliaSim sensor data and renderings — which
addresses the Guía 2 feedback about using a robotics simulator.

Objects created (all under a single ``G3_Scene`` dummy):
  * Floor + bounding walls.
  * One cylinder per landmark/obstacle (respondable, so the LiDAR sees
    them), aliased ``LM_<id>`` for ground-truth data association.
  * ``Tractor``, ``Trailer1``, ``Trailer2`` cuboids (kinematic).
  * ``LiDAR`` mounted on the tractor (built-in 2D scanner model if
    available, otherwise a script-based fan — see ``--no-model``).
  * ``Start`` and ``Goal`` dummies.

Usage (with CoppeliaSim open)::

    python coppelia/build_scene.py --config config/scenario_g3.yaml \
        --save coppelia/g3_scene.ttt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coppelia._client import connect, version_string
from coppelia.probe import CANDIDATE_LIDAR_MODELS, resources_dir
from coppelia.scenario import Scenario

FLOOR_Z = 0.0
BODY_H = 0.5          # visual height of vehicle bodies [m]
WALL_H = 1.5         # tall enough to be seen by the beam plane (~0.69 m)
LM_H = 1.5           # landmark cylinder height [m]


def _mk_cuboid(sim, sx, sy, sz):
    opts = 0
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [sx, sy, sz], opts)
    return h


def _mk_cylinder(sim, d, h):
    return sim.createPrimitiveShape(sim.primitiveshape_cylinder, [d, d, h], 0)


def _set_pose(sim, h, x, y, z, yaw=0.0, ref=-1):
    sim.setObjectPosition(h, ref, [float(x), float(y), float(z)])
    sim.setObjectOrientation(h, ref, [0.0, 0.0, float(yaw)])


def _color(sim, h, rgb):
    try:
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse,
                          [float(c) for c in rgb])
    except Exception:
        pass


def _static(sim, h, static=True, respondable=True):
    try:
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1 if static else 0)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable,
                                1 if respondable else 0)
    except Exception:
        pass


def build(sim, sc: Scenario, load_model: bool = True) -> dict:
    root = sim.createDummy(0.05)
    sim.setObjectAlias(root, "G3_Scene")

    handles: dict = {"root": root, "landmarks": {}}

    # ---- floor ----------------------------------------------------------
    w, h = sc.size
    floor = _mk_cuboid(sim, float(w), float(h), 0.05)
    _set_pose(sim, floor, 0, 0, FLOOR_Z - 0.025)
    _color(sim, floor, [0.85, 0.85, 0.85]); _static(sim, floor)
    sim.setObjectAlias(floor, "Floor"); sim.setObjectParent(floor, root, True)
    handles["floor"] = floor

    # ---- walls ----------------------------------------------------------
    th = sc.cfg.get("walls", {}).get("thickness", 0.2)
    for i, (x0, y0, x1, y1) in enumerate(sc.wall_segments):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        length = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        wall = _mk_cuboid(sim, length, th, WALL_H)
        _set_pose(sim, wall, cx, cy, WALL_H / 2, yaw)
        _color(sim, wall, [0.5, 0.5, 0.55]); _static(sim, wall)
        sim.setObjectAlias(wall, f"Wall_{i}")
        sim.setObjectParent(wall, root, True)

    # ---- landmarks / obstacles -----------------------------------------
    for lid, x, y, r in sc.landmarks:
        cyl = _mk_cylinder(sim, 2 * r, LM_H)
        _set_pose(sim, cyl, x, y, LM_H / 2)
        _color(sim, cyl, [0.30, 0.30, 0.35]); _static(sim, cyl)
        sim.setObjectAlias(cyl, f"LM_{int(lid)}")
        sim.setObjectParent(cyl, root, True)
        handles["landmarks"][int(lid)] = cyl

    # ---- vehicle bodies (kinematic; pose imposed at run time) -----------
    v = sc.cfg["vehicle"]
    bodies = {}
    for name, L, W, col in [
        ("Tractor", v["body_length_0"], v["W0"], [0.85, 0.20, 0.20]),
        ("Trailer1", v["body_length_1"], v["W1"], [0.20, 0.45, 0.85]),
        ("Trailer2", v["body_length_2"], v["W2"], [0.20, 0.70, 0.35]),
    ]:
        b = _mk_cuboid(sim, float(L), float(W), BODY_H)
        _color(sim, b, col); _static(sim, b, static=True, respondable=False)
        sim.setObjectAlias(b, name); sim.setObjectParent(b, root, True)
        bodies[name] = b
    handles["bodies"] = bodies

    # ---- LiDAR ----------------------------------------------------------
    lidar = None
    if load_model:
        try:
            base = Path(resources_dir(sim))
            for rel in CANDIDATE_LIDAR_MODELS:
                p = base / rel
                if p.exists():
                    lidar = sim.loadModel(str(p))
                    sim.setObjectAlias(lidar, "LiDAR")
                    print(f"[build_scene] mounted native LiDAR model: {rel}")
                    break
        except Exception as exc:
            print(f"[build_scene] LiDAR model load failed: {exc}")
    if lidar is None:
        # Fallback marker; the recorder will ray-cast analytically.
        lidar = sim.createDummy(0.1)
        sim.setObjectAlias(lidar, "LiDAR")
        print("[build_scene] Using dummy LiDAR marker (analytic ray-cast "
              "fallback in the recorder).")
    lp = sc.cfg["lidar"]
    mx, my = lp["mount_xy"]
    sim.setObjectParent(lidar, bodies["Tractor"], True)
    _set_pose(sim, lidar, mx, my, BODY_H / 2 + 0.15,
              math.radians(lp["mount_yaw_deg"]), ref=bodies["Tractor"])
    handles["lidar"] = lidar

    # ---- start / goal dummies ------------------------------------------
    st = sim.createDummy(0.3); sim.setObjectAlias(st, "Start")
    _set_pose(sim, st, sc.start.x, sc.start.y, 0.15, sc.start.theta)
    sim.setObjectParent(st, root, True)
    gl = sim.createDummy(0.3); sim.setObjectAlias(gl, "Goal")
    _set_pose(sim, gl, sc.goal.x, sc.goal.y, 0.15, sc.goal.theta)
    sim.setObjectParent(gl, root, True)

    return handles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/scenario_g3.yaml")
    ap.add_argument("--save", default="coppelia/g3_scene.ttt")
    ap.add_argument("--no-model", action="store_true",
                    help="skip loading a built-in LiDAR model")
    ap.add_argument("--clear", action="store_true",
                    help="purge current scene before building")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sc = Scenario.load(root / args.config)
    client, sim = connect()
    print(version_string(sim))

    if args.clear:
        sim.stopSimulation()
        # keep lights and cameras so the scene stays lit / viewable
        keep = set()
        for attr in ("object_light_type", "object_camera_type"):
            if hasattr(sim, attr):
                keep.add(getattr(sim, attr))
        for h in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0):
            try:
                if sim.getObjectType(h) in keep:
                    continue
                sim.removeObjects([h])
            except Exception:
                pass

    handles = build(sim, sc, load_model=not args.no_model)

    # Ensure the scene is lit even if lights were removed in a previous run:
    # a bright ambient term makes every shape visible in the 3-D viewport and
    # in any render (it is a scene property saved with the .ttt).
    try:
        sim.setArrayParam(sim.arrayparam_ambient_light, [0.5, 0.5, 0.5])
    except Exception as exc:
        print(f"[build_scene] ambient light note: {exc}")
    print(f"[build_scene] created {len(handles['landmarks'])} landmarks, "
          f"3 bodies, LiDAR, start/goal.")

    save_path = root / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sim.saveScene(str(save_path))
        print(f"[build_scene] scene saved to {save_path}")
    except Exception as exc:
        print(f"[build_scene] saveScene failed ({exc}); scene is still "
              "loaded in the running CoppeliaSim — save it manually.")


if __name__ == "__main__":
    main()
