"""Render + validate the Guía 3 scenario WITHOUT CoppeliaSim.

Produces ``figures/scenario_map.png`` (a report figure) and prints a
validation report: clearance of the reference waypoints, the narrow-gap
width, and start/goal feasibility. Runnable in any environment with
numpy + matplotlib.

    python tools/preview_scenario.py --config config/scenario_g3.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coppelia.scenario import Scenario


def resample_path(wps: np.ndarray, step: float = 0.1) -> np.ndarray:
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        d = np.hypot(*(b - a))
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/scenario_g3.yaml")
    ap.add_argument("--out", default="figures/scenario_map.png")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sc = Scenario.load(root / args.config)

    path = resample_path(sc.reference_waypoints, 0.05)
    clr = np.array([sc.min_obstacle_distance(x, y) for x, y in path])
    half_width = max(sc.cfg["vehicle"]["W0"], sc.cfg["vehicle"]["W1"]) / 2.0

    # ---- validation report ---------------------------------------------
    print("=== Scenario validation ===")
    print(f"world size            : {sc.size.tolist()} m")
    print(f"n landmarks/obstacles : {len(sc.landmarks)}")
    print(f"start                 : ({sc.start.x}, {sc.start.y}, "
          f"{math.degrees(sc.start.theta):.0f} deg)")
    print(f"goal                  : ({sc.goal.x}, {sc.goal.y}, "
          f"{math.degrees(sc.goal.theta):.0f} deg)")
    print(f"tractor half-width    : {half_width:.2f} m")
    print(f"ref-path min clearance: {clr.min():.2f} m "
          f"({'OK' if clr.min() > half_width else 'TIGHT/COLLISION'})")
    ng = sc.cfg.get("narrow_gap", {})
    if ng.get("enable"):
        print(f"narrow gap width      : {ng['gap_width']:.2f} m "
              f"(free space between the two cylinders)")
    print(f"start in collision    : {sc.in_collision(sc.start.x, sc.start.y)}")
    print(f"goal  in collision    : {sc.in_collision(sc.goal.x, sc.goal.y)}")

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for x0, y0, x1, y1 in sc.wall_segments:
        ax.plot([x0, x1], [y0, y1], "k-", lw=2)
    for lid, x, y, r in sc.landmarks:
        ax.add_patch(Circle((x, y), r, color="0.35"))
        ax.text(x, y, f"{int(lid)}", color="w", ha="center", va="center",
                fontsize=7)
    # narrow-gap highlight
    if ng.get("enable"):
        cx, cy = ng["center"]
        ax.add_patch(Circle((cx, cy), ng["gap_width"] / 2.0, fill=False,
                            ls="--", ec="tab:red", lw=1.2))
        ax.text(cx, cy - 1.4, "paso estrecho\n(oclusión/ambigüedad)",
                color="tab:red", ha="center", va="top", fontsize=8)

    ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=1.5,
            label="ruta de referencia (generación de datos)")
    ax.scatter(sc.reference_waypoints[:, 0], sc.reference_waypoints[:, 1],
               c="tab:blue", s=18, zorder=5)
    ax.scatter([sc.start.x], [sc.start.y], marker="o", s=120,
               c="tab:green", zorder=6, label="inicio")
    ax.scatter([sc.goal.x], [sc.goal.y], marker="*", s=260,
               c="tab:orange", zorder=6, label="meta")
    # heading arrows
    for p, col in [(sc.start, "tab:green"), (sc.goal, "tab:orange")]:
        ax.arrow(p.x, p.y, 1.2 * math.cos(p.theta), 1.2 * math.sin(p.theta),
                 head_width=0.3, color=col, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Guía 3 — Escenario G2T: mapa, landmarks, inicio y meta")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
