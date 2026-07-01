"""Discover how the native 2D laser scanner exposes its data in this
CoppeliaSim build, so the recorder can read it correctly.

Loads the preferred scanner model into a *temporary* scene, inspects its
object tree (vision- vs proximity-sensor based), runs a few simulation
steps and tries every documented read path, printing what works and the
data shapes. Nothing is saved.

    python coppelia/lidar_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coppelia._client import connect, version_string
from coppelia.probe import CANDIDATE_LIDAR_MODELS, resources_dir


def _type_name(sim, h):
    t = sim.getObjectType(h)
    names = {
        sim.object_shape_type: "shape",
        sim.object_visionsensor_type: "visionSensor",
        sim.object_proximitysensor_type: "proximitySensor",
        sim.object_dummy_type: "dummy",
        sim.object_joint_type: "joint",
        sim.object_script_type: "script",
        sim.object_forcesensor_type: "forceSensor",
    }
    return names.get(t, f"type#{t}")


def main() -> None:
    client, sim = connect()
    print(version_string(sim))
    base = Path(resources_dir(sim))

    model_rel = next((r for r in CANDIDATE_LIDAR_MODELS
                      if (base / r).exists()), None)
    if model_rel is None:
        raise SystemExit("No candidate scanner model found on this install.")
    print(f"loading model: {model_rel}")
    model = sim.loadModel(str(base / model_rel))
    sim.setObjectAlias(model, "LiDAR_probe")

    # ---- enumerate the model tree --------------------------------------
    tree = sim.getObjectsInTree(model, sim.handle_all, 0)
    vision, prox = [], []
    print(f"\nmodel tree ({len(tree)} objects):")
    for h in tree:
        tn = _type_name(sim, h)
        try:
            alias = sim.getObjectAlias(h, 1)
        except Exception:
            alias = "?"
        print(f"  h={h:5d}  {tn:16s}  {alias}")
        if tn == "visionSensor":
            vision.append(h)
        elif tn == "proximitySensor":
            prox.append(h)

    # ---- run a few steps in stepping mode ------------------------------
    print("\nstarting stepped simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(5):
        sim.step()

    # ---- try vision-sensor reads ---------------------------------------
    for h in vision:
        print(f"\n[visionSensor {h}] alias={sim.getObjectAlias(h, 1)}")
        try:
            res, data, packets = sim.readVisionSensor(h)
            print(f"  readVisionSensor -> res={res}, "
                  f"len(data)={None if data is None else len(data)}, "
                  f"len(packets)={None if packets is None else len(packets)}")
            if packets:
                print(f"    packet[0] len={len(packets[0])}, "
                      f"first vals={packets[0][:6]}")
        except Exception as exc:
            print(f"  readVisionSensor failed: {exc}")
        try:
            r = sim.getVisionSensorRes(h)
            print(f"  resolution = {r}")
        except Exception as exc:
            print(f"  getVisionSensorRes failed: {exc}")

    # ---- try proximity-sensor reads ------------------------------------
    for h in prox:
        try:
            r = sim.readProximitySensor(h)
            print(f"\n[proximitySensor {h}] readProximitySensor -> {r}")
        except Exception as exc:
            print(f"\n[proximitySensor {h}] failed: {exc}")

    # ---- common signal names used by the built-in scanner scripts ------
    print("\nprobing string signals:")
    for name in ["measuredDataAtThisTime", "measuredData",
                 "SICK_S300_data", "Hokuyo_data", "scanData"]:
        try:
            val = sim.getStringSignal(name)
            print(f"  '{name}': "
                  f"{'None' if val is None else f'{len(val)} bytes'}")
        except Exception as exc:
            print(f"  '{name}': err {exc}")

    sim.stopSimulation()
    print("\ndone. Tell me which read path returned data + the shapes.")


if __name__ == "__main__":
    main()
