"""Monte Carlo simulation running multiple seeds, noise levels, and scenarios.

Compares Odometry-only, EKF-SLAM, and EKF Fusion, exporting a CSV and creating boxplots.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from estim.inputs import NoiseCfg
import estim.run_slam as run_slam
import estim.run_fusion as run_fusion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8, help="number of random seeds")
    args = ap.parse_args()

    scenarios = {
        "Scenario_1": "datasets/g3_run/dataset.h5",
        "Scenario_2": "datasets/g3_alternative/dataset.h5"
    }

    noise_levels = {
        "Default_Noise (1x)": 1.0,
        "High_Noise (2x)": 2.0
    }

    results = []

    print(f"Starting Monte Carlo simulation over {args.seeds} seeds...")
    
    for sc_name, sc_path in scenarios.items():
        dataset_path = ROOT / sc_path
        if not dataset_path.exists():
            print(f"Skipping {sc_name}: file not found at {dataset_path}")
            continue
            
        f = h5py.File(dataset_path, "r")
        
        for nl_name, scale in noise_levels.items():
            # Build custom NoiseCfg scaling the bias and variance parameters
            noise = NoiseCfg(
                s_v=1.0 + 0.02 * scale,      # scale error scales with noise level
                b_v=0.0,
                sigma_v=0.03 * scale,
                s_w=1.0,
                b_w=0.0105 * scale,
                sigma_w=0.02 * scale
            )
            
            print(f"--- Running {sc_name} | {nl_name} ---")
            
            for seed in range(args.seeds):
                # 1. Run SLAM and Fusion
                try:
                    inp, slam_est, fusion_est = run_fusion.run_fusion(f, noise, seed)
                    dr = run_fusion.dead_reckon(inp)
                    gt = inp.gt_tractor
                    gt_s = inp.gt_state
                    
                    # Articulation dead reckoning
                    dr_art = np.zeros((len(inp.t), 2))
                    dr_art[0] = gt_s[0, 3:5]
                    try:
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
                        dr_art[k, 0] = run_fusion._wrap(psi1 + psi1_dot * dt)
                        dr_art[k, 1] = run_fusion._wrap(psi2 + psi2_dot * dt)
                    
                    # 2. Compute RMSEs
                    rmse_dr_pos = run_fusion.rmse_xy(dr, gt)
                    rmse_dr_ang = run_fusion.rmse_ang(dr[:, 2], gt[:, 2])
                    rmse_dr_psi1 = run_fusion.rmse_ang(dr_art[:, 0], gt_s[:, 3])
                    rmse_dr_psi2 = run_fusion.rmse_ang(dr_art[:, 1], gt_s[:, 4])
                    
                    rmse_slam_pos = run_fusion.rmse_xy(slam_est, gt)
                    rmse_slam_ang = run_fusion.rmse_ang(slam_est[:, 2], gt[:, 2])
                    
                    rmse_fus_pos = run_fusion.rmse_xy(fusion_est[:, :3], gt)
                    rmse_fus_ang = run_fusion.rmse_ang(fusion_est[:, 2], gt[:, 2])
                    rmse_fus_psi1 = run_fusion.rmse_ang(fusion_est[:, 3], gt_s[:, 3])
                    rmse_fus_psi2 = run_fusion.rmse_ang(fusion_est[:, 4], gt_s[:, 4])
                    
                    results.append({
                        "scenario": sc_name,
                        "noise_level": nl_name,
                        "seed": seed,
                        
                        "odom_pos_rmse": rmse_dr_pos,
                        "odom_ang_rmse": math.degrees(rmse_dr_ang),
                        "odom_psi1_rmse": math.degrees(rmse_dr_psi1),
                        "odom_psi2_rmse": math.degrees(rmse_dr_psi2),
                        
                        "slam_pos_rmse": rmse_slam_pos,
                        "slam_ang_rmse": math.degrees(rmse_slam_ang),
                        
                        "fusion_pos_rmse": rmse_fus_pos,
                        "fusion_ang_rmse": math.degrees(rmse_fus_ang),
                        "fusion_psi1_rmse": math.degrees(rmse_fus_psi1),
                        "fusion_psi2_rmse": math.degrees(rmse_fus_psi2),
                    })
                except Exception as e:
                    print(f"Error on seed {seed}: {e}")
                    
        f.close()

    df = pd.DataFrame(results)
    out_dir = ROOT / "evaluation"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "monte_carlo_results.csv", index=False)
    print(f"Wrote CSV to {out_dir / 'monte_carlo_results.csv'}")

    # Display Summary Statistics
    summary = df.groupby(["scenario", "noise_level"]).agg({
        "odom_pos_rmse": ["mean", "std"],
        "slam_pos_rmse": ["mean", "std"],
        "fusion_pos_rmse": ["mean", "std"],
        "odom_ang_rmse": ["mean", "std"],
        "slam_ang_rmse": ["mean", "std"],
        "fusion_ang_rmse": ["mean", "std"],
        "odom_psi1_rmse": ["mean", "std"],
        "fusion_psi1_rmse": ["mean", "std"],
        "odom_psi2_rmse": ["mean", "std"],
        "fusion_psi2_rmse": ["mean", "std"],
    })
    print("\n=== MONTE CARLO SUMMARY ===")
    print(summary.to_string())

    # ---- Generar Boxplots ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # We will draw boxplots comparing Odom vs SLAM vs Fusion for Position and Heading,
    # and Odom vs Fusion for Articulation.
    
    # Data extraction helper
    def get_boxplot_data(df_group, columns):
        return [df_group[col].values for col in columns]

    # Subplot 1: Position RMSE (Scenario 1)
    a = axes[0, 0]
    df1 = df[df["scenario"] == "Scenario_1"]
    groups1 = [df1[df1["noise_level"] == "Default_Noise (1x)"], df1[df1["noise_level"] == "High_Noise (2x)"]]
    
    data_pos = []
    labels_pos = []
    for grp, name in zip(groups1, ["1x", "2x"]):
        data_pos.extend([grp["odom_pos_rmse"].values, grp["slam_pos_rmse"].values, grp["fusion_pos_rmse"].values])
        labels_pos.extend([f"Odom\n({name})", f"SLAM\n({name})", f"Fusion\n({name})"])
        
    a.boxplot(data_pos, labels=labels_pos, patch_artist=True, 
              boxprops=dict(facecolor="lightblue", color="blue"),
              medianprops=dict(color="red"))
    a.set_ylabel("Position RMSE [m]")
    a.set_title("Position RMSE Comparison (Scenario 1)")
    a.grid(alpha=0.3)

    # Subplot 2: Position RMSE (Scenario 2)
    a = axes[0, 1]
    df2 = df[df["scenario"] == "Scenario_2"]
    groups2 = [df2[df2["noise_level"] == "Default_Noise (1x)"], df2[df2["noise_level"] == "High_Noise (2x)"]]
    
    data_pos2 = []
    for grp, name in zip(groups2, ["1x", "2x"]):
        data_pos2.extend([grp["odom_pos_rmse"].values, grp["slam_pos_rmse"].values, grp["fusion_pos_rmse"].values])
        
    a.boxplot(data_pos2, labels=labels_pos, patch_artist=True, 
              boxprops=dict(facecolor="lightgreen", color="green"),
              medianprops=dict(color="red"))
    a.set_ylabel("Position RMSE [m]")
    a.set_title("Position RMSE Comparison (Scenario 2)")
    a.grid(alpha=0.3)

    # Subplot 3: Heading RMSE (Scenario 1)
    a = axes[1, 0]
    data_ang = []
    labels_ang = []
    for grp, name in zip(groups1, ["1x", "2x"]):
        data_ang.extend([grp["odom_ang_rmse"].values, grp["slam_ang_rmse"].values, grp["fusion_ang_rmse"].values])
        labels_ang.extend([f"Odom\n({name})", f"SLAM\n({name})", f"Fusion\n({name})"])
        
    a.boxplot(data_ang, labels=labels_ang, patch_artist=True, 
              boxprops=dict(facecolor="lightpink", color="purple"),
              medianprops=dict(color="red"))
    a.set_ylabel("Heading RMSE [deg]")
    a.set_title("Heading RMSE Comparison (Scenario 1)")
    a.grid(alpha=0.3)

    # Subplot 4: Articulation RMSE (psi1, Scenario 1)
    a = axes[1, 1]
    data_psi = []
    labels_psi = []
    for grp, name in zip(groups1, ["1x", "2x"]):
        data_psi.extend([grp["odom_psi1_rmse"].values, grp["fusion_psi1_rmse"].values,
                         grp["odom_psi2_rmse"].values, grp["fusion_psi2_rmse"].values])
        labels_psi.extend([f"O1\n({name})", f"F1\n({name})", f"O2\n({name})", f"F2\n({name})"])
        
    a.boxplot(data_psi, labels=labels_psi, patch_artist=True, 
              boxprops=dict(facecolor="lightyellow", color="orange"),
              medianprops=dict(color="red"))
    a.set_ylabel("Articulation RMSE [deg]")
    a.set_title("Articulation RMSE (psi1 & psi2) Comparison (Scenario 1)")
    a.grid(alpha=0.3)

    fig_path = ROOT / "figures/robustness_boxplots.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    print(f"Saved boxplots to {fig_path}")


if __name__ == "__main__":
    main()
