"""Visual sanity check for the native LiDAR reader.

Loads the saved scene, places the G2T at the start pose, takes ONE scan
in the world frame, and overlays the returned points on the known
landmark map. If the points sit on the cylinders and walls, the reader
geometry is correct.

    python coppelia/test_scan.py           # try +Z optical axis
    python coppelia/test_scan.py --flip    # try -Z if mirrored
"""
from __future__ import annotations

import argparse
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
from coppelia.lidar_read import find_scanner, set_range, read_scan
from g2t_core.simulation.g2t_sim.kinematics import G2TState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/scenario_g3.yaml")
    ap.add_argument("--scene", default="coppelia/g3_scene.ttt")
    ap.add_argument("--far", type=float, default=15.0)
    ap.add_argument("--flip", action="store_true", help="use -Z optical axis")
    ap.add_argument("--out", default="figures/scan_check.png")
    args = ap.parse_args()

    sc = Scenario.load(ROOT / args.config)
    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / args.scene))

    bodies = find_bodies(sim)
    kin = make_kinematics(sc.cfg)
    st = sc.start
    place(sim, bodies, kin, G2TState(st.x, st.y, st.theta, st.psi1, st.psi2))

    sensors, ref = find_scanner(sim)
    print(f"scanner sensors={sensors} ref={ref}")

    sim.setStepping(True)
    sim.startSimulation()
    sim.step()
    set_range(sim, sensors, args.far)       # apply AFTER the model's init
    sim.step()
    far_now = sim.getObjectFloatParam(sensors[0],
                                      sim.visionfloatparam_far_clipping)
    print(f"far clipping now = {far_now:.2f} m (requested {args.far})")

    ang, rng, pts = read_scan(sim, sensors, sim.handle_world, args.far,
                              view_sign=-1.0 if args.flip else 1.0)
    sim.stopSimulation()
    print(f"scan: {len(ang)} returns, "
          f"range [{rng.min():.2f}, {rng.max():.2f}] m" if len(ang)
          else "scan: 0 returns")

    # ---- overlay -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for x0, y0, x1, y1 in sc.wall_segments:
        ax.plot([x0, x1], [y0, y1], "k-", lw=2)
    for lid, x, y, r in sc.landmarks:
        ax.add_patch(Circle((x, y), r, color="0.75"))
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="tab:red",
                   label="LiDAR returns (world)")
    ax.scatter([st.x], [st.y], marker="o", s=120, c="tab:green", label="tractor")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Scan check @ start pose ({'−Z' if args.flip else '+Z'} axis)")
    ax.legend(loc="upper left", fontsize=8)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
