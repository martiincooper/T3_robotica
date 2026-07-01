"""Second probe: confirm depth-buffer extraction from the SICK S300 Fast.

Reports, for each of the two vision sensors: perspective FOV, near/far
clipping, resolution, pose relative to the scanner reference frame, and a
sample of the depth buffer returned by ``getVisionSensorDepth`` (both
normalized and, if supported, in metres). From this the recorder can
assemble a single 270-degree scan in the LiDAR frame.

    python coppelia/lidar_probe2.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coppelia._client import connect, version_string
from coppelia.probe import CANDIDATE_LIDAR_MODELS, resources_dir


def main() -> None:
    client, sim = connect()
    print(version_string(sim))
    base = Path(resources_dir(sim))
    model_rel = next(r for r in CANDIDATE_LIDAR_MODELS if (base / r).exists())
    model = sim.loadModel(str(base / model_rel))
    print(f"model: {model_rel}  handle={model}")

    tree = sim.getObjectsInTree(model, sim.handle_all, 0)
    sensors, ref = [], model
    for h in tree:
        if sim.getObjectType(h) == sim.object_visionsensor_type:
            sensors.append(h)
        if sim.getObjectAlias(h, 0) == "ref":
            ref = h
    print(f"vision sensors: {sensors}, ref frame: {ref}")

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(3):
        sim.step()

    for h in sensors:
        print(f"\n=== sensor {h} ({sim.getObjectAlias(h, 1)}) ===")
        try:
            fov = sim.getObjectFloatParam(h, sim.visionfloatparam_perspective_angle)
            near = sim.getObjectFloatParam(h, sim.visionfloatparam_near_clipping)
            far = sim.getObjectFloatParam(h, sim.visionfloatparam_far_clipping)
            print(f"  FOV(perspective_angle) = {math.degrees(fov):.2f} deg")
            print(f"  near={near:.3f} m  far={far:.3f} m")
        except Exception as exc:
            print(f"  param read failed: {exc}")
        try:
            print(f"  resolution = {sim.getVisionSensorRes(h)}")
        except Exception as exc:
            print(f"  res failed: {exc}")
        try:
            pose = sim.getObjectPose(h, ref)
            eul = sim.getObjectOrientation(h, ref)
            print(f"  pose rel ref  = {[round(v,3) for v in pose]}")
            print(f"  euler rel ref = {[round(math.degrees(v),1) for v in eul]}")
        except Exception as exc:
            print(f"  pose failed: {exc}")

        # depth buffer — normalized then (try) metric
        for opt in (0, 1):
            try:
                depth, res = sim.getVisionSensorDepth(h, opt)
                # depth may be bytes-packed floats or a list
                if isinstance(depth, (bytes, bytearray)):
                    import struct
                    n = len(depth) // 4
                    vals = struct.unpack(f"{n}f", depth)
                else:
                    vals = list(depth)
                vmin, vmax = min(vals), max(vals)
                print(f"  getVisionSensorDepth(opt={opt}) res={res} "
                      f"n={len(vals)} min={vmin:.3f} max={vmax:.3f} "
                      f"sample={[round(v,3) for v in vals[:5]]}")
            except Exception as exc:
                print(f"  getVisionSensorDepth(opt={opt}) failed: {exc}")

    sim.stopSimulation()
    print("\ndone — paste this output.")


if __name__ == "__main__":
    main()
