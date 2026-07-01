"""Run EKF-SLAM over datasets/g3_run/dataset.h5 (offline, no CoppeliaSim).

Compares odometry-only dead reckoning vs EKF-SLAM vs ground truth, and
the built map vs the true landmark positions. Writes figures and prints
RMSE. Reused by Block 3/4 as a component.
"""
from __future__ import annotations

import argparse
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
from estim.inputs import NoiseCfg, build_inputs
from estim.ekf_slam import EKFSLAM, SlamCfg, wrap
from estim.landmarks import extract_landmarks


def dead_reckon(inp):
    n = len(inp.t); p = np.zeros((n, 3)); p[0] = inp.gt_tractor[0]
    for k in range(1, n):
        th = p[k - 1, 2]
        p[k, 0] = p[k - 1, 0] + inp.v_odom[k] * inp.dt * math.cos(th)
        p[k, 1] = p[k - 1, 1] + inp.v_odom[k] * inp.dt * math.sin(th)
        p[k, 2] = wrap(th + inp.w_odom[k] * inp.dt)
    return p


def rmse_xy(est, gt):
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, axis=1))))


def rmse_ang(est, gt):
    e = np.arctan2(np.sin(est[:, 2] - gt[:, 2]), np.cos(est[:, 2] - gt[:, 2]))
    return float(np.sqrt(np.mean(e ** 2)))


def run(f, noise: NoiseCfg, seed: int = 0, slam_cfg: SlamCfg | None = None):
    rng = np.random.default_rng(seed)
    inp = build_inputs(f, noise, rng)
    counts = f["scan/count"][:]; pts = f["scan/points"][:]
    off = np.concatenate([[0], np.cumsum(counts)])
    # measured sensor-yaw offset relative to tractor heading
    spose = f["scan/sensor_pose"][:]
    myaw = float(np.mean(np.arctan2(
        np.sin(spose[:, 2] - inp.gt_tractor[:, 2]),
        np.cos(spose[:, 2] - inp.gt_tractor[:, 2]))))

    cfg = slam_cfg or SlamCfg()
    cfg.sigma_v, cfg.sigma_w = noise.sigma_v, noise.sigma_w
    cfg.myaw = myaw
    g0 = inp.gt_tractor[0]
    slam = EKFSLAM(cfg, g0[0], g0[1], g0[2])

    n = len(inp.t); est = np.zeros((n, 3)); est[0] = g0
    for k in range(1, n):
        slam.predict(inp.v_odom[k], inp.w_odom[k], inp.dt)
        p = pts[off[k]:off[k + 1]]
        obs = [(o.rng, o.bearing) for o in extract_landmarks(p)]
        slam.update(obs)
        est[k] = slam.pose()
    return inp, est, slam


def match_map(est_lms, gt_lms):
    """Return matched (est, gt) pairs by nearest neighbour + RMSE."""
    if len(est_lms) == 0:
        return 0, float("nan")
    errs = []
    for e in est_lms:
        d = np.hypot(gt_lms[:, 1] - e[0], gt_lms[:, 2] - e[1])
        if d.min() < 1.0:
            errs.append(d.min())
    return len(errs), (float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/g3_run/dataset.h5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    f = h5py.File(ROOT / args.dataset, "r")
    gt_lms = f["meta/landmarks"][:]

    noise = NoiseCfg()
    inp, est, slam = run(f, noise, args.seed)
    dr = dead_reckon(inp)
    gt = inp.gt_tractor

    print("=== EKF-SLAM vs odometry (seed %d) ===" % args.seed)
    print(f"odometry-only  RMSE pos = {rmse_xy(dr, gt):.3f} m   "
          f"ang = {math.degrees(rmse_ang(dr, gt)):.2f} deg")
    print(f"EKF-SLAM       RMSE pos = {rmse_xy(est, gt):.3f} m   "
          f"ang = {math.degrees(rmse_ang(est, gt)):.2f} deg")
    nlm, map_rmse = match_map(slam.landmarks(), gt_lms)
    print(f"map: {slam.num_landmarks()} landmarks, {nlm} matched to GT, "
          f"map RMSE = {map_rmse:.3f} m")

    # ---- figures --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    a = ax[0]
    for _, x, y, r in gt_lms:
        a.add_patch(Circle((x, y), r, color="0.85"))
    a.plot(gt[:, 0], gt[:, 1], "b-", lw=2, label="ground truth")
    a.plot(dr[:, 0], dr[:, 1], "r--", lw=1.6, label="odometry only")
    a.plot(est[:, 0], est[:, 1], "g-", lw=1.8, label="EKF-SLAM")
    lm = slam.landmarks()
    if len(lm):
        a.scatter(lm[:, 0], lm[:, 1], marker="x", c="k", s=60,
                  label="estimated map")
    a.scatter(gt_lms[:, 1], gt_lms[:, 2], marker="+", c="tab:purple", s=60,
              label="true landmarks")
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=8)
    a.set_title("EKF-SLAM: trajectory + map"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    a = ax[1]
    ep_dr = np.hypot(dr[:, 0] - gt[:, 0], dr[:, 1] - gt[:, 1])
    ep_es = np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1])
    a.plot(inp.t, ep_dr, "r--", label="odometry only")
    a.plot(inp.t, ep_es, "g-", label="EKF-SLAM")
    a.set_xlabel("t [s]"); a.set_ylabel("position error [m]")
    a.set_title("Position error vs time"); a.legend(fontsize=8); a.grid(alpha=0.3)

    out = ROOT / "figures/slam_result.png"
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
