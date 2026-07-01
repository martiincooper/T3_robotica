"""Connectivity probe for CoppeliaSim.

Run this FIRST (with CoppeliaSim open) to confirm the environment before
building the scene or recording data:

    python coppelia/probe.py

Prints the CoppeliaSim version, confirms the ZMQ remote API works, and
lists which built-in 2D laser-scanner models are available on this
install (so we can pick the right one for the LiDAR mount).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coppelia._client import connect, version_string

# Relative to <app>/Contents/Resources on macOS (or <app>/ on Linux/Windows).
# Ordered by suitability for a 24x18 m map: long-range/wide-FOV first.
CANDIDATE_LIDAR_MODELS = [
    "models/components/sensors/SICK S300 Fast.ttm",   # 270 deg, ~30 m  (preferred)
    "models/components/sensors/2D laser scanner.ttm",  # generic, configurable
    "models/components/sensors/SICK TiM310 Fast.ttm",
    "models/components/sensors/Hokuyo URG 04LX UG01_Fast.ttm",  # 240 deg, ~5 m
    "models/components/sensors/Hokuyo URG 04LX UG01.ttm",
]


def resources_dir(sim) -> str:
    """Return the models base dir for this platform.

    ``application_path`` is ``.../Contents/MacOS`` on macOS (models live in
    the sibling ``Resources`` dir) and the app root elsewhere.
    """
    from pathlib import Path
    app = sim.getStringParam(sim.stringparam_application_path)
    p = Path(app)
    res = p.parent / "Resources"
    return str(res if (res / "models").exists() else p)


def main() -> None:
    client, sim = connect()
    print(version_string(sim))
    try:
        app = sim.getStringParam(sim.stringparam_application_path)
    except Exception:
        app = "?"
    print(f"application path : {app}")
    print(f"sim state        : {sim.getSimulationState()}")

    print("\nSearching for built-in 2D laser-scanner models:")
    base = Path(resources_dir(sim))
    for rel in CANDIDATE_LIDAR_MODELS:
        found = "FOUND" if (base / rel).exists() else "missing"
        print(f"  [{found:7}] {rel}")

    print("\nOK — remote API is working. Next: python coppelia/build_scene.py")


if __name__ == "__main__":
    main()
