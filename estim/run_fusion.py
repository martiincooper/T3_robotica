"""Run EKF sensor fusion over the dataset, combining odometry, IMU, SLAM pose, and noisy articulation."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from estim.inputs import NoiseCfg, build_inputs
from estim.ekf_fusion import EKFFusion, EKFFusionConfig, _wrap
import estim.run_slam as run_slam


def dead_reckon(inp):
    n = len(inp.t)
    p = np.zeros((n, 3))
    p[0] = inp.gt_tractor[0]
    for k in range(1, n):
        th = p[k - 1, 2]
        p[k, 0] = p[k - 1, 0] + inp.v_odom[k] * inp.dt * math.cos(th)
        p[k, 1] = p[k - 1, 1] + inp.v_odom[k] * inp.dt * math.sin(th)
        p[k, 2] = _wrap(th + inp.w_odom[k] * inp.dt)
    return p


def rmse_xy(est, gt):
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, axis=1))))


def rmse_ang(est_ang, gt_ang):
    e = np.arctan2(np.sin(est_ang - gt_ang), np.cos(est_ang - gt_ang))
    return float(np.sqrt(np.mean(e ** 2)))


def run_fusion(f, noise: NoiseCfg, seed: int = 0, fusion_cfg: EKFFusionConfig | None = None):
    # 1. Run EKF-SLAM first to get the SLAM poses
    inp, slam_est, slam_filter = run_slam.run(f, noise, seed)
    
    # 2. Set up fusion filter
    cfg = fusion_cfg or EKFFusionConfig()
    cfg.L0 = float(f["gt/tractor"].attrs.get("L0", 1.20)) # fallback to default if not in attrs
    # Let's get L0, L1, L2 from config if possible
    try:
        import json
        sc_cfg = json.loads(f.attrs["config"])
        cfg.L0 = sc_cfg["vehicle"]["L0"]
        cfg.L1 = sc_cfg["vehicle"]["L1"]
        cfg.L2 = sc_cfg["vehicle"]["L2"]
    except Exception:
        pass
        
    ekf = EKFFusion(cfg)
    
    # Initialize state to ground truth at t=0
    gt_s = inp.gt_state
    ekf.x[0] = gt_s[0, 0]
    ekf.x[1] = gt_s[0, 1]
    ekf.x[2] = gt_s[0, 2]
    ekf.x[3] = gt_s[0, 3]
    ekf.x[4] = gt_s[0, 4]
    ekf.x[5] = inp.w_odom[0] # initial yaw rate estimate
    
    n = len(inp.t)
    est_states = np.zeros((n, 6))
    est_states[0] = ekf.x.copy()
    
    # Set up random generator for simulated articulation sensor noise
    rng = np.random.default_rng(seed + 1000) # different seed for articulation noise
    
    # Get IMU gyro data from dataset
    gyro_z = f["imu/gyro_z"][:]
    
    for k in range(1, n):
        # A. Prediction step
        # v_odom is the speed control input. Propagation time is dt.
        ekf.predict(inp.v_odom[k], inp.dt)
        
        # B. Update steps
        # 1. IMU Gyro
        ekf.update_imu(gyro_z[k])
        
        # 2. Odometry yaw rate (as a secondary sensor measuring omega)
        ekf.update_odom_w(inp.w_odom[k])
        
        # 3. SLAM pose (measuring x_t, y_t, theta_t)
        ekf.update_slam(slam_est[k, 0], slam_est[k, 1], slam_est[k, 2])
        
        # 4. Articulation angles (measuring psi1, psi2 from ground truth degraded with noise)
        psi1_meas = gt_s[k, 3] + rng.normal(0, cfg.r_psi_lidar)
        psi2_meas = gt_s[k, 4] + rng.normal(0, cfg.r_psi_lidar)
        ekf.update_lidar(psi1_meas, psi2_meas)
        
        est_states[k] = ekf.x.copy()
        
    return inp, slam_est, est_states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/g3_run/dataset.h5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    dataset_path = ROOT / args.dataset
    if not dataset_path.exists():
        print(f"Error: dataset file not found at {dataset_path}")
        sys.exit(1)
        
    f = h5py.File(dataset_path, "r")
    noise = NoiseCfg()
    
    inp, slam_est, fusion_est = run_fusion(f, noise, args.seed)
    dr = dead_reckon(inp)
    gt = inp.gt_tractor
    gt_s = inp.gt_state
    
    # Articulation dead reckoning from odometry (for comparison)
    # We can simulate how articulation angles propagate using odometry only (no updates)
    dr_art = np.zeros((len(inp.t), 2))
    dr_art[0] = gt_s[0, 3:5]
    try:
        import json
        sc_cfg = json.loads(f.attrs["config"])
        L1 = sc_cfg["vehicle"]["L1"]
        L2 = sc_cfg["vehicle"]["L2"]
    except Exception:
        L1, L2 = 1.50, 1.50
        
    for k in range(1, len(inp.t)):
        v = inp.v_odom[k]
        w = inp.w_odom[k]
        dt = inp.dt
        psi1 = dr_art[k-1, 0]
        psi2 = dr_art[k-1, 1]
        psi1_dot = -(v / L1) * math.sin(psi1) + w
        v1 = v * math.cos(psi1)
        psi2_dot = -(v1 / L2) * math.sin(psi2) + (v / L1) * math.sin(psi1)
        dr_art[k, 0] = _wrap(psi1 + psi1_dot * dt)
        dr_art[k, 1] = _wrap(psi2 + psi2_dot * dt)
    
    # Compute RMSE metrics
    rmse_dr_pos = rmse_xy(dr, gt)
    rmse_dr_ang = rmse_ang(dr[:, 2], gt[:, 2])
    
    rmse_slam_pos = rmse_xy(slam_est, gt)
    rmse_slam_ang = rmse_ang(slam_est[:, 2], gt[:, 2])
    
    rmse_fus_pos = rmse_xy(fusion_est[:, :3], gt)
    rmse_fus_ang = rmse_ang(fusion_est[:, 2], gt[:, 2])
    rmse_fus_psi1 = rmse_ang(fusion_est[:, 3], gt_s[:, 3])
    rmse_fus_psi2 = rmse_ang(fusion_est[:, 4], gt_s[:, 4])
    
    rmse_dr_psi1 = rmse_ang(dr_art[:, 0], gt_s[:, 3])
    rmse_dr_psi2 = rmse_ang(dr_art[:, 1], gt_s[:, 4])
    
    print("=== EKF Sensor Fusion vs SLAM vs Odometry (seed %d) ===" % args.seed)
    print(f"Odometry-only  RMSE pos = {rmse_dr_pos:.3f} m   "
          f"ang = {math.degrees(rmse_dr_ang):.2f} deg   "
          f"psi1 = {math.degrees(rmse_dr_psi1):.2f} deg   "
          f"psi2 = {math.degrees(rmse_dr_psi2):.2f} deg")
    print(f"EKF-SLAM       RMSE pos = {rmse_slam_pos:.3f} m   "
          f"ang = {math.degrees(rmse_slam_ang):.2f} deg")
    print(f"EKF Fusion     RMSE pos = {rmse_fus_pos:.3f} m   "
          f"ang = {math.degrees(rmse_fus_ang):.2f} deg   "
          f"psi1 = {math.degrees(rmse_fus_psi1):.2f} deg   "
          f"psi2 = {math.degrees(rmse_fus_psi2):.2f} deg")
          
    # ---- figures --------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Trajectory comparison
    a = ax[0, 0]
    gt_lms = f["meta/landmarks"][:]
    from matplotlib.patches import Circle
    for _, lx, ly, lr in gt_lms:
        a.add_patch(Circle((lx, ly), lr, color="0.85"))
    a.plot(gt[:, 0], gt[:, 1], "b-", lw=2, label="Ground Truth")
    a.plot(dr[:, 0], dr[:, 1], "r--", lw=1.5, label="Odometry Only")
    a.plot(slam_est[:, 0], slam_est[:, 1], "g-", lw=1.8, label="EKF-SLAM")
    a.plot(fusion_est[:, 0], fusion_est[:, 1], "m-.", lw=1.8, label="EKF Fusion")
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=9)
    a.set_title("Trajectory Comparison"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    
    # 2. Position error vs time
    a = ax[0, 1]
    err_dr = np.hypot(dr[:, 0] - gt[:, 0], dr[:, 1] - gt[:, 1])
    err_slam = np.hypot(slam_est[:, 0] - gt[:, 0], slam_est[:, 1] - gt[:, 1])
    err_fus = np.hypot(fusion_est[:, 0] - gt[:, 0], fusion_est[:, 1] - gt[:, 1])
    a.plot(inp.t, err_dr, "r--", label="Odometry Only")
    a.plot(inp.t, err_slam, "g-", label="EKF-SLAM")
    a.plot(inp.t, err_fus, "m-.", label="EKF Fusion")
    a.set_xlabel("t [s]"); a.set_ylabel("Position Error [m]")
    a.set_title("Position Error vs Time"); a.legend(fontsize=9); a.grid(alpha=0.3)
    
    # 3. Heading error vs time
    a = ax[1, 0]
    ang_err_dr = np.degrees(np.abs(np.arctan2(np.sin(dr[:, 2] - gt[:, 2]), np.cos(dr[:, 2] - gt[:, 2]))))
    ang_err_slam = np.degrees(np.abs(np.arctan2(np.sin(slam_est[:, 2] - gt[:, 2]), np.cos(slam_est[:, 2] - gt[:, 2]))))
    ang_err_fus = np.degrees(np.abs(np.arctan2(np.sin(fusion_est[:, 2] - gt[:, 2]), np.cos(fusion_est[:, 2] - gt[:, 2]))))
    a.plot(inp.t, ang_err_dr, "r--", label="Odometry Only")
    a.plot(inp.t, ang_err_slam, "g-", label="EKF-SLAM")
    a.plot(inp.t, ang_err_fus, "m-.", label="EKF Fusion")
    a.set_xlabel("t [s]"); a.set_ylabel("Heading Error [deg]")
    a.set_title("Heading Error vs Time"); a.legend(fontsize=9); a.grid(alpha=0.3)
    
    # 4. Articulation error vs time
    a = ax[1, 1]
    psi1_err_dr = np.degrees(np.abs(np.arctan2(np.sin(dr_art[:, 0] - gt_s[:, 3]), np.cos(dr_art[:, 0] - gt_s[:, 3]))))
    psi1_err_fus = np.degrees(np.abs(np.arctan2(np.sin(fusion_est[:, 3] - gt_s[:, 3]), np.cos(fusion_est[:, 3] - gt_s[:, 3]))))
    psi2_err_dr = np.degrees(np.abs(np.arctan2(np.sin(dr_art[:, 1] - gt_s[:, 4]), np.cos(dr_art[:, 1] - gt_s[:, 4]))))
    psi2_err_fus = np.degrees(np.abs(np.arctan2(np.sin(fusion_est[:, 4] - gt_s[:, 4]), np.cos(fusion_est[:, 4] - gt_s[:, 4]))))
    a.plot(inp.t, psi1_err_dr, "r--", label="psi1 Odom")
    a.plot(inp.t, psi1_err_fus, "m-", label="psi1 Fusion")
    a.plot(inp.t, psi2_err_dr, "c--", label="psi2 Odom")
    a.plot(inp.t, psi2_err_fus, "y-", label="psi2 Fusion")
    a.set_xlabel("t [s]"); a.set_ylabel("Articulation Error [deg]")
    a.set_title("Articulation Errors vs Time"); a.legend(fontsize=9); a.grid(alpha=0.3)
    
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fusion_result.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")
    
    f.close()


if __name__ == "__main__":
    main()
