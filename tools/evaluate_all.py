"""Consolidates and formats evaluation metrics from SLAM, EKF Fusion, and Trajectory Following."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results_csv = ROOT / "evaluation/monte_carlo_results.csv"
    nav_json = ROOT / "evaluation/navigation_metrics.json"
    
    print("======================================================================")
    print("                   GUÍA 3 NAVIGATION & ESTIMATION EVALUATION")
    print("======================================================================\n")
    
    # 1. Estimation metrics (Monte Carlo)
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        summary = df.groupby(["scenario", "noise_level"]).mean()
        summary_std = df.groupby(["scenario", "noise_level"]).std()
        
        print("--- PART 1: Estimation Performance (Mean ± Std over 8 Seeds) ---")
        for sc in ["Scenario_1", "Scenario_2"]:
            print(f"\n{sc}:")
            for nl in ["Default_Noise (1x)", "High_Noise (2x)"]:
                print(f"  {nl}:")
                # Pos RMSE
                odom_p_m, odom_p_s = summary.loc[(sc, nl), "odom_pos_rmse"], summary_std.loc[(sc, nl), "odom_pos_rmse"]
                slam_p_m, slam_p_s = summary.loc[(sc, nl), "slam_pos_rmse"], summary_std.loc[(sc, nl), "slam_pos_rmse"]
                fuse_p_m, fuse_p_s = summary.loc[(sc, nl), "fusion_pos_rmse"], summary_std.loc[(sc, nl), "fusion_pos_rmse"]
                print(f"    Position RMSE:   Odom: {odom_p_m:.3f}±{odom_p_s:.3f} m  | SLAM: {slam_p_m:.3f}±{slam_p_s:.3f} m  | Fusion: {fuse_p_m:.3f}±{fuse_p_s:.3f} m")
                
                # Ang RMSE
                odom_a_m, odom_a_s = summary.loc[(sc, nl), "odom_ang_rmse"], summary_std.loc[(sc, nl), "odom_ang_rmse"]
                slam_a_m, slam_a_s = summary.loc[(sc, nl), "slam_ang_rmse"], summary_std.loc[(sc, nl), "slam_ang_rmse"]
                fuse_a_m, fuse_a_s = summary.loc[(sc, nl), "fusion_ang_rmse"], summary_std.loc[(sc, nl), "fusion_ang_rmse"]
                print(f"    Heading RMSE:    Odom: {odom_a_m:.2f}°±{odom_a_s:.2f}° | SLAM: {slam_a_m:.2f}°±{slam_a_s:.2f}° | Fusion: {fuse_a_m:.2f}°±{fuse_a_s:.2f}°")
                
                # Articulation RMSE
                odom_psi1_m, odom_psi1_s = summary.loc[(sc, nl), "odom_psi1_rmse"], summary_std.loc[(sc, nl), "odom_psi1_rmse"]
                fuse_psi1_m, fuse_psi1_s = summary.loc[(sc, nl), "fusion_psi1_rmse"], summary_std.loc[(sc, nl), "fusion_psi1_rmse"]
                odom_psi2_m, odom_psi2_s = summary.loc[(sc, nl), "odom_psi2_rmse"], summary_std.loc[(sc, nl), "odom_psi2_rmse"]
                fuse_psi2_m, fuse_psi2_s = summary.loc[(sc, nl), "fusion_psi2_rmse"], summary_std.loc[(sc, nl), "fusion_psi2_rmse"]
                print(f"    Articulation:    Odom psi1: {odom_psi1_m:.2f}°±{odom_psi1_s:.2f}° | Fusion psi1: {fuse_psi1_m:.2f}°±{fuse_psi1_s:.2f}°")
                print(f"                     Odom psi2: {odom_psi2_m:.2f}°±{odom_psi2_s:.2f}° | Fusion psi2: {fuse_psi2_m:.2f}°±{fuse_psi2_s:.2f}°")
        print()
    else:
        print("Warning: monte_carlo_results.csv not found.")
        
    # 2. Closed-loop Navigation Performance
    if nav_json.exists():
        with open(nav_json, "r") as f_nav:
            nav = json.load(f_nav)
        print("--- PART 2: Closed-Loop Trajectory Following (Online RRT* + Pure Pursuit) ---")
        print(f"  Time to Goal            : {nav['time_to_goal']:.2f} s")
        print(f"  Executed Path Length    : {nav['path_length']:.2f} m")
        print(f"  Final Goal Error        : {nav['goal_error']:.3f} m")
        print(f"  Cross-Track Error (RMSE): {nav['rmse_cte']:.3f} m")
        print(f"  Min Obstacle Distance   : {nav['min_clearance']:.3f} m")
        print(f"  Collisions Detected     : {nav['collisions']}")
        print()
    else:
        print("Warning: navigation_metrics.json not found.")
        
    print("======================================================================")


if __name__ == "__main__":
    main()
