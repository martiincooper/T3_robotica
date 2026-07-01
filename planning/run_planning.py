"""Run RRT* path planning over the SLAM-built landmark map and plot results."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coppelia.scenario import Scenario
from estim.inputs import NoiseCfg
import estim.run_slam as run_slam
from planning.rrt_star import RRTStar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/scenario_g3.yaml")
    ap.add_argument("--dataset", default="datasets/g3_run/dataset.h5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sc = Scenario.load(ROOT / args.config)
    f = h5py.File(ROOT / args.dataset, "r")
    
    # 1. Run SLAM to get the built landmark map
    noise = NoiseCfg()
    inp, est, slam = run_slam.run(f, noise, args.seed)
    slam_lms = slam.landmarks()
    
    # 2. Get start/goal
    st = sc.start
    gl = sc.goal
    start_pos = (st.x, st.y)
    goal_pos = (gl.x, gl.y)
    
    print(f"Planning from start {start_pos} to goal {goal_pos}...")
    print(f"SLAM built map has {len(slam_lms)} landmarks")

    # 3. Setup RRT* planner using SLAM landmarks
    # Nominal obstacle radius is 0.30 m. World size from config.
    rrt = RRTStar(
        start=start_pos,
        goal=goal_pos,
        landmarks=slam_lms,
        landmark_radius=0.30,
        world_size=(sc.size[0], sc.size[1]),
        walls=sc.wall_segments,
        L0=sc.cfg["vehicle"]["L0"],
        max_steer_rad=math.radians(35.0),
        clearance=0.55,       # W0/2 (0.40m) + margin (0.15m)
        max_step=1.0,
        search_radius=2.0,
        goal_sample_rate=0.15,
        goal_tolerance=0.5,
        max_nodes=2000
    )

    # 4. Plan path
    path_raw = rrt.plan()
    if path_raw is None:
        print("Planning failed! Trying APF fallback or returning empty.")
        # Draw map anyway
        path_smoothed = []
        path_raw = []
    else:
        # Shortcut/Smooth
        path_smoothed = rrt.shortcut_path(path_raw, iterations=200)
        print(f"Raw path nodes: {len(path_raw)}, Smoothed path nodes: {len(path_smoothed)}")
        
        # Save planned path to file for Block 6 trajectory following
        planning_dir = ROOT / "planning"
        planning_dir.mkdir(exist_ok=True)
        path_file = planning_dir / "rrt_path.csv"
        np.savetxt(path_file, np.array(path_smoothed), delimiter=",", header="x,y", comments="")
        print(f"Saved planned path to {path_file}")

    # 5. Plotting results
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw walls
    for x0, y0, x1, y1 in sc.wall_segments:
        ax.plot([x0, x1], [y0, y1], "k-", lw=2, label="Walls" if 'Walls' not in ax.get_legend_handles_labels()[1] else "")
        
    # Draw SLAM estimated landmarks (with inflated clearance)
    for i, (lx, ly) in enumerate(slam_lms):
        # Nominal obstacle circle
        ax.add_patch(Circle((lx, ly), 0.30, color="orange", alpha=0.3, 
                            label="SLAM Obstacles" if i == 0 else ""))
        # Inflated collision boundary
        ax.add_patch(Circle((lx, ly), 0.30 + rrt.clearance, color="red", alpha=0.08, fill=True, ls=":", ec="red",
                            label="Inflated Margin" if i == 0 else ""))
        
    # Draw true landmarks for comparison
    gt_lms = f["meta/landmarks"][:]
    for i, (_, lx, ly, lr) in enumerate(gt_lms):
        ax.scatter(lx, ly, marker="+", c="blue", s=80, zorder=5,
                   label="True Landmarks" if i == 0 else "")
        
    # Draw RRT* Tree
    for node in rrt.nodes:
        if node.parent is not None:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "g-", alpha=0.15, lw=1)
            
    # Draw start/goal
    ax.scatter(start_pos[0], start_pos[1], c="green", marker="s", s=100, zorder=10, label="Start")
    ax.scatter(goal_pos[0], goal_pos[1], c="red", marker="*", s=200, zorder=10, label="Goal")
    
    # Draw raw path
    if len(path_raw) > 0:
        pr = np.array(path_raw)
        ax.plot(pr[:, 0], pr[:, 1], "r--", lw=1.5, label="Raw RRT* Path")
        
    # Draw smoothed path
    if len(path_smoothed) > 0:
        ps = np.array(path_smoothed)
        ax.plot(ps[:, 0], ps[:, 1], "m-", lw=2.5, label="Smoothed Path")
        ax.scatter(ps[:, 0], ps[:, 1], c="magenta", s=25, zorder=8)

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("RRT* Path Planning on SLAM-Built Map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "rrt_star_plan.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote planning plot to {out_path}")
    
    f.close()


if __name__ == "__main__":
    main()
