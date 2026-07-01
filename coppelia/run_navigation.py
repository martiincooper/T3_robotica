"""Online Trajectory Tracking in CoppeliaSim using RRT* path, EKF SLAM, and EKF Fusion closed-loop control."""
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

from coppelia._client import connect, version_string
from coppelia.scenario import Scenario
from coppelia.vehicle import make_kinematics, find_bodies, place
from coppelia.lidar_read import find_scanner, set_range, read_scan
from g2t_core.simulation.g2t_sim.kinematics import G2TState, wrap_angle

from estim.inputs import NoiseCfg
from estim.ekf_slam import EKFSLAM, SlamCfg
from estim.ekf_fusion import EKFFusion, EKFFusionConfig, _wrap
from estim.landmarks import extract_landmarks
from planning.rrt_star import RRTStar


def pure_pursuit(pos, theta, wps, ld, v_nom, max_steer, L0):
    """Return (v, delta, done, cte). Steers the tractor toward a look-ahead
    point on the polyline ``wps``."""
    p = np.array(pos[:2])
    goal = wps[-1]
    if np.hypot(*(goal - p)) < 0.6:
        return 0.0, 0.0, True, 0.0
        
    # Find closest point on the path
    d = np.hypot(wps[:, 0] - p[0], wps[:, 1] - p[1])
    i0 = int(np.argmin(d))
    cte = float(d[i0]) # cross-track error approximation
    
    # Lookahead point
    target = wps[-1]
    acc = 0.0
    for j in range(i0, len(wps) - 1):
        seg = np.hypot(*(wps[j + 1] - wps[j]))
        acc += seg
        if acc >= ld:
            target = wps[j + 1]
            break
            
    alpha = wrap_angle(math.atan2(target[1] - p[1], target[0] - p[0]) - theta)
    delta = math.atan2(2.0 * L0 * math.sin(alpha), ld)
    delta = float(np.clip(delta, -max_steer, max_steer))
    return v_nom, delta, False, cte


def densify(wps, step=0.05):
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/scenario_g3.yaml")
    ap.add_argument("--scene", default="coppelia/g3_scene.ttt")
    ap.add_argument("--far", type=float, default=5.0)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sc = Scenario.load(ROOT / args.config)
    drv = sc.cfg["driving"]
    dt = drv["dt"]
    L0 = sc.cfg["vehicle"]["L0"]
    max_steer = math.radians(35.0)
    
    # Load or generate RRT* path
    rrt_path_file = ROOT / "planning/rrt_path.csv"
    if rrt_path_file.exists():
        print(f"Loading planned RRT* path from {rrt_path_file}")
        path = np.loadtxt(rrt_path_file, delimiter=",", skiprows=1)
    else:
        print("Planned RRT* path file not found. Running planner dynamically on GT landmarks...")
        rrt = RRTStar(
            start=(sc.start.x, sc.start.y),
            goal=(sc.goal.x, sc.goal.y),
            landmarks=sc.landmarks[:, 1:3],
            landmark_radius=0.30,
            world_size=(sc.size[0], sc.size[1]),
            walls=sc.wall_segments,
            L0=L0,
            max_steer_rad=max_steer
        )
        path = rrt.plan()
        if path is not None:
            path = np.array(rrt.shortcut_path(path, iterations=150))
            
    if path is None or len(path) == 0:
        print("Error: No valid path to track. Exiting.")
        sys.exit(1)
        
    wps = densify(path, 0.05)

    # Initialize Filters and Noise configurations
    noise = NoiseCfg()
    rng = np.random.default_rng(args.seed)
    
    # EKF-SLAM setup
    slam_cfg = SlamCfg()
    slam_cfg.sigma_v, slam_cfg.sigma_w = noise.sigma_v, noise.sigma_w
    slam_cfg.myaw = math.pi # SICK scanner faces backward
    st = sc.start
    slam = EKFSLAM(slam_cfg, st.x, st.y, st.theta)
    
    # EKF Fusion setup
    fusion_cfg = EKFFusionConfig()
    fusion_cfg.L0, fusion_cfg.L1, fusion_cfg.L2 = sc.cfg["vehicle"]["L0"], sc.cfg["vehicle"]["L1"], sc.cfg["vehicle"]["L2"]
    fusion_cfg.r_psi_lidar = 0.03
    ekf_fusion = EKFFusion(fusion_cfg)
    ekf_fusion.x[0] = st.x
    ekf_fusion.x[1] = st.y
    ekf_fusion.x[2] = st.theta
    ekf_fusion.x[3] = st.psi1
    ekf_fusion.x[4] = st.psi2
    ekf_fusion.x[5] = 0.0

    # Connect to CoppeliaSim
    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / args.scene))
    try:
        sim.setArrayParam(sim.arrayparam_ambient_light, [0.5, 0.5, 0.5])
    except Exception:
        pass
    bodies = find_bodies(sim)
    sensors, ref = find_scanner(sim)

    # Initialize robot state in sim
    state = G2TState(st.x, st.y, st.theta, st.psi1, st.psi2)
    place(sim, bodies, make_kinematics(sc.cfg), state)

    sim.setStepping(True)
    sim.startSimulation()
    sim.step(); set_range(sim, sensors, args.far); sim.step()

    # Logs
    log_time = []
    log_gt = []
    log_dr = []
    log_slam = []
    log_fusion = []
    log_cte = []
    log_min_dist = []
    
    dr_pos = np.array([st.x, st.y, st.theta])
    
    t = 0.0
    prev_theta = state.theta
    done = False
    
    print("Tracking RRT* path in closed-loop using EKF Fusion feedback...")
    for k in range(args.max_steps):
        # 1. Closed-loop control input from Estimated Pose (EKF Fusion)
        # Use estimated position for pure pursuit
        est_pose = ekf_fusion.x[:3]
        v, delta, goal_reached, cte = pure_pursuit(est_pose[:2], est_pose[2], wps, 
                                                  drv["lookahead"], drv["v_nominal"], max_steer, L0)
        
        if goal_reached:
            print(f"[navigation] Goal reached at t={t:.1f}s (step {k})")
            done = True
            break
            
        # 2. Advance true state cinematically
        kin = make_kinematics(sc.cfg)
        state = kin.step(state, v, delta, dt)
        place(sim, bodies, kin, state)
        sim.step()
        t += dt
        
        # 3. Simulate sensor readings and inject noise
        omega = wrap_angle(state.theta - prev_theta) / dt
        prev_theta = state.theta
        
        v_odom = v + rng.normal(0, noise.sigma_v)
        w_odom = omega + rng.normal(0, noise.sigma_w)
        gyro_z = omega + rng.normal(0, sc.cfg["imu"]["gyro_z_noise_std"])
        
        # 4. Dead reckoning propagation
        th_dr = dr_pos[2]
        dr_pos[0] += v_odom * dt * math.cos(th_dr)
        dr_pos[1] += v_odom * dt * math.sin(th_dr)
        dr_pos[2] = _wrap(th_dr + w_odom * dt)
        
        # 5. Read scanner and run SLAM filter
        ang, ranges_meas, pts_meas = read_scan(sim, sensors, ref, args.far)
        slam.predict(v_odom, w_odom, dt)
        obs = [(o.rng, o.bearing) for o in extract_landmarks(pts_meas)]
        slam.update(obs)
        
        # 6. Run EKF Fusion filter
        ekf_fusion.predict(v_odom, dt)
        ekf_fusion.update_imu(gyro_z)
        ekf_fusion.update_odom_w(w_odom)
        ekf_fusion.update_slam(slam.x[0], slam.x[1], slam.x[2])
        # Add noisy articulation angles
        psi1_meas = state.psi1 + rng.normal(0, 0.03)
        psi2_meas = state.psi2 + rng.normal(0, 0.03)
        ekf_fusion.update_lidar(psi1_meas, psi2_meas)
        
        # Logging
        log_time.append(t)
        log_gt.append(state.as_array()[:3]) # x, y, theta
        log_dr.append(dr_pos.copy())
        log_slam.append(slam.pose())
        log_fusion.append(ekf_fusion.x.copy())
        log_cte.append(cte)
        
        # Min distance to obstacles
        min_d = sc.min_obstacle_distance(state.x, state.y)
        log_min_dist.append(min_d)

    sim.stopSimulation()
    
    # Analyze and Output metrics
    log_gt = np.array(log_gt)
    log_dr = np.array(log_dr)
    log_slam = np.array(log_slam)
    log_fusion = np.array(log_fusion)
    log_cte = np.array(log_cte)
    log_min_dist = np.array(log_min_dist)
    
    path_len = float(np.sum(np.hypot(np.diff(log_gt[:, 0]), np.diff(log_gt[:, 1]))))
    goal_err = float(np.hypot(log_gt[-1, 0] - sc.goal.x, log_gt[-1, 1] - sc.goal.y))
    rmse_cte = float(np.sqrt(np.mean(log_cte ** 2)))
    min_clearance = float(log_min_dist.min())
    collisions = int(np.sum(log_min_dist < 0.0))
    
    print("\n=== TRAJECTORY FOLLOWING PERFORMANCE ===")
    print(f"Goal reached              : {done}")
    print(f"Time to goal              : {t:.2f} s")
    print(f"Executed path length      : {path_len:.2f} m")
    print(f"Final goal error          : {goal_err:.3f} m")
    print(f"Cross-track error (RMSE)  : {rmse_cte:.3f} m")
    print(f"Min obstacle distance     : {min_clearance:.3f} m")
    print(f"Collisions detected       : {collisions}")

    # Save metrics to json
    metrics = {
        "time_to_goal": t,
        "path_length": path_len,
        "goal_error": goal_err,
        "rmse_cte": rmse_cte,
        "min_clearance": min_clearance,
        "collisions": collisions
    }
    with open(ROOT / "evaluation/navigation_metrics.json", "w") as f_met:
        json.dump(metrics, f_met, indent=4)

    # ---- Plotting Results ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left subplot: Trajectory Map
    a = axes[0]
    # Draw walls
    for x0, y0, x1, y1 in sc.wall_segments:
        a.plot([x0, x1], [y0, y1], "k-", lw=1.5)
    # Draw obstacles
    for _, lx, ly, lr in sc.landmarks:
        a.add_patch(Circle((lx, ly), lr, color="0.85"))
        
    a.plot(path[:, 0], path[:, 1], "m-", lw=2, label="RRT* Reference Path")
    a.plot(log_gt[:, 0], log_gt[:, 1], "b-", lw=2.5, label="Executed Trajectory (GT)")
    a.plot(log_dr[:, 0], log_dr[:, 1], "r--", lw=1.2, label="Odometry Dead Reckoning")
    a.plot(log_slam[:, 0], log_slam[:, 1], "g-.", lw=1.5, label="Online EKF-SLAM")
    a.plot(log_fusion[:, 0], log_fusion[:, 1], "c:", lw=1.8, label="Online EKF Fusion")
    
    a.scatter([sc.start.x], [sc.start.y], c="g", s=80, marker="s", zorder=10, label="Start")
    a.scatter([sc.goal.x], [sc.goal.y], c="r", s=150, marker="*", zorder=10, label="Goal")
    
    a.set_aspect("equal")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="upper right")
    a.set_title("Closed-Loop Trajectory Following Map")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    
    # Right subplot: Errors & Clearance
    a = axes[1]
    color = 'tab:red'
    a.set_xlabel("t [s]")
    a.set_ylabel("Cross-Track Error [m]", color=color)
    line1 = a.plot(log_time, log_cte, color=color, lw=1.5, label="Cross-Track Error")
    a.tick_params(axis='y', labelcolor=color)
    
    a2 = a.twinx()
    color = 'tab:blue'
    a2.set_ylabel("Obstacle Clearance [m]", color=color)
    line2 = a2.plot(log_time, log_min_dist, color=color, lw=1.5, ls="--", label="Min Obstacle Distance")
    a2.axhline(0.40, color="orange", ls=":", alpha=0.8, label="Tractor Safety Limit (0.4m)")
    a2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    a.legend(lines, labels, loc='upper right', fontsize=8)
    a.set_title("Tracking Error and Obstacle Clearance vs Time")
    a.grid(alpha=0.3)
    
    fig_path = ROOT / "figures/navigation_following.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    print(f"Wrote trajectory tracking plot to {fig_path}")


if __name__ == "__main__":
    main()
