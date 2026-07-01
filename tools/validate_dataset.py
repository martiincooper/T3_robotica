"""Validate datasets/g3_run/dataset.h5 for correctness before SLAM.

Checks, all offline (no CoppeliaSim):
  1. Scan->world consistency: transform every scan into the world with the
     logged sensor_pose and measure distance to the nearest wall/cylinder.
  2. Landmark observability: how many scans see each landmark (needed so
     EKF-SLAM has range-bearing observations).
  3. Odometry drift: dead-reckon the noisy odometry and compare to GT
     (must drift, otherwise there is nothing for SLAM/EKF to correct).
  4. Articulation angles psi1/psi2 sanity (bounded, smooth).

Writes figures/dataset_validation.png and prints a PASS/FAIL report.
"""
from __future__ import annotations

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

TOL = 0.35


def seg_dist(pts, segs):
    d = np.full(len(pts), 1e9)
    for x0, y0, x1, y1 in segs:
        a = np.array([x0, y0]); b = np.array([x1, y1]); ab = b - a
        tt = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        proj = a + tt[:, None] * ab
        d = np.minimum(d, np.hypot(pts[:, 0] - proj[:, 0], pts[:, 1] - proj[:, 1]))
    return d


def main() -> None:
    sc = Scenario.load(ROOT / "config/scenario_g3.yaml")
    f = h5py.File(ROOT / "datasets/g3_run/dataset.h5", "r")
    dt = float(f.attrs["dt"])
    gt = f["gt/state"][:]
    gt_tr = f["gt/tractor"][:]
    lms = f["meta/landmarks"][:]
    counts = f["scan/count"][:]
    poses = f["scan/sensor_pose"][:]
    ranges = f["scan/ranges"][:]
    pts = f["scan/points"][:]
    odom_v = f["odom/v_noisy"][:]
    odom_w = f["odom/omega_noisy"][:]
    ctrl_v = f["ctrl/v"][:]
    t = f["gt/time"][:]

    off = np.concatenate([[0], np.cumsum(counts)])

    # ---- 1. scan->world consistency + 2. observability -----------------
    resid_all = []
    seen = np.zeros(len(lms), int)
    world_pts = []
    for i in range(len(counts)):
        p = pts[off[i]:off[i + 1]]
        if len(p) == 0:
            continue
        x, y, yaw = poses[i]
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * p[:, 0] - s * p[:, 1]
        wy = y + s * p[:, 0] + c * p[:, 1]
        w = np.column_stack([wx, wy])
        world_pts.append(w[::5])
        d_seg = seg_dist(w, sc.wall_segments)
        d_lm = np.full(len(w), 1e9)
        for j, (_, lx, ly, lr) in enumerate(lms):
            dj = np.abs(np.hypot(wx - lx, wy - ly) - lr)
            d_lm = np.minimum(d_lm, dj)
            if np.any(np.hypot(wx - lx, wy - ly) < lr + TOL):
                seen[j] += 1
        resid_all.append(np.minimum(d_seg, d_lm))
    resid = np.concatenate(resid_all)
    frac_ok = float(np.mean(resid < TOL))

    # ---- 3. odometry dead reckoning ------------------------------------
    dr = np.zeros((len(t), 3))
    dr[0] = gt_tr[0]
    for k in range(1, len(t)):
        th = dr[k - 1, 2]
        dr[k, 0] = dr[k - 1, 0] + odom_v[k] * math.cos(th) * dt
        dr[k, 1] = dr[k - 1, 1] + odom_v[k] * math.sin(th) * dt
        dr[k, 2] = th + odom_w[k] * dt
    drift = np.hypot(dr[:, 0] - gt_tr[:, 0], dr[:, 1] - gt_tr[:, 1])

    # ---- report --------------------------------------------------------
    print("=== DATASET VALIDATION ===")
    print(f"scan->world within {TOL} m : {frac_ok*100:.1f}%  "
          f"({'PASS' if frac_ok > 0.9 else 'CHECK'})")
    print(f"landmarks observed        : {int((seen>0).sum())}/{len(lms)} "
          f"(min scans/lm={seen.min()}, max={seen.max()})")
    print(f"odom dead-reckoning drift : final={drift[-1]:.2f} m, "
          f"max={drift.max():.2f} m  ({'PASS-drifts' if drift[-1] > 0.3 else 'too small'})")
    print(f"psi1 range [deg]          : "
          f"[{math.degrees(gt[:,3].min()):.1f}, {math.degrees(gt[:,3].max()):.1f}]")
    print(f"psi2 range [deg]          : "
          f"[{math.degrees(gt[:,4].min()):.1f}, {math.degrees(gt[:,4].max()):.1f}]")
    print(f"noise check: std(odom_v-ctrl_v)={np.std(odom_v-ctrl_v):.3f} "
          f"(cfg {sc.cfg['odometry']['v_noise_std']})")

    # ---- figure --------------------------------------------------------
    W = np.concatenate(world_pts)
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))

    a = ax[0, 0]
    a.scatter(W[:, 0], W[:, 1], s=1, c="tab:red", alpha=0.15)
    a.plot(gt_tr[:, 0], gt_tr[:, 1], "b-", lw=2, label="GT")
    for _, x, y, r in lms:
        a.add_patch(Circle((x, y), r, fill=False, ec="k"))
    a.set_title(f"1. Scan->world map  ({frac_ok*100:.0f}% on geometry)")
    a.set_aspect("equal"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(gt_tr[:, 0], gt_tr[:, 1], "b-", lw=2, label="GT tractor")
    a.plot(dr[:, 0], dr[:, 1], "r--", lw=2, label="odometry (dead-reckon)")
    a.scatter([gt_tr[0, 0]], [gt_tr[0, 1]], c="g", s=60)
    a.set_title(f"3. Odometry drift (final {drift[-1]:.2f} m) -> motivates SLAM")
    a.set_aspect("equal"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.bar(range(len(lms)), seen, color="tab:purple")
    a.set_xlabel("landmark id"); a.set_ylabel("# scans observed")
    a.set_title("2. Landmark observability")
    a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(t, np.degrees(gt[:, 3]), label="psi1")
    a.plot(t, np.degrees(gt[:, 4]), label="psi2")
    a.set_xlabel("t [s]"); a.set_ylabel("articulation [deg]")
    a.set_title("4. Articulation angles"); a.legend(fontsize=8); a.grid(alpha=0.3)

    out = ROOT / "figures/dataset_validation.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
