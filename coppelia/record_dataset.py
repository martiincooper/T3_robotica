"""Drive the G2T through the CoppeliaSim scene and record the dataset.

Pipeline per step:
  1. pure-pursuit controller picks (v, delta) toward the reference path,
  2. the validated RK4 kinematics advance the true state,
  3. the three bodies (and the child LiDAR) are placed in CoppeliaSim,
  4. one sim step renders the native SICK S300 scan,
  5. ground truth, noisy odometry/IMU and the scan are logged.

Output: datasets/g3_run/dataset.h5  (+ a quick verification figure).

    python coppelia/record_dataset.py --config config/scenario_g3.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coppelia._client import connect, version_string
from coppelia.scenario import Scenario
from coppelia.vehicle import make_kinematics, find_bodies, place
from coppelia.lidar_read import find_scanner, set_range, read_scan
from g2t_core.simulation.g2t_sim.kinematics import G2TState, wrap_angle


def pure_pursuit(state, wps, ld, v_nom, max_steer, L0):
    """Return (v, delta, done). Steers the tractor toward a look-ahead
    point on the polyline ``wps``."""
    p = np.array([state.x, state.y])
    goal = wps[-1]
    if np.hypot(*(goal - p)) < 0.6:
        return 0.0, 0.0, True
    # nearest point index on the polyline, then advance by look-ahead
    d = np.hypot(wps[:, 0] - p[0], wps[:, 1] - p[1])
    i0 = int(np.argmin(d))
    target = wps[-1]
    acc = 0.0
    for j in range(i0, len(wps) - 1):
        seg = np.hypot(*(wps[j + 1] - wps[j]))
        acc += seg
        if acc >= ld:
            target = wps[j + 1]
            break
    alpha = wrap_angle(math.atan2(target[1] - p[1], target[0] - p[0]) - state.theta)
    delta = math.atan2(2.0 * L0 * math.sin(alpha), ld)
    delta = float(np.clip(delta, -max_steer, max_steer))
    return v_nom, delta, False


def densify(wps, step=0.1):
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
    ap.add_argument("--max_steps", type=int, default=4000)
    args = ap.parse_args()

    sc = Scenario.load(ROOT / args.config)
    drv = sc.cfg["driving"]
    dt = drv["dt"]
    wps = densify(sc.reference_waypoints, 0.1)
    kin = make_kinematics(sc.cfg)
    L0 = sc.cfg["vehicle"]["L0"]
    max_steer = math.radians(35.0)
    rng = np.random.default_rng(sc.cfg["output"]["seed"])
    on = sc.cfg["odometry"]; imu = sc.cfg["imu"]

    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / args.scene))
    try:
        sim.setArrayParam(sim.arrayparam_ambient_light, [0.5, 0.5, 0.5])
    except Exception:
        pass
    bodies = find_bodies(sim)
    sensors, ref = find_scanner(sim)

    st = sc.start
    state = G2TState(st.x, st.y, st.theta, st.psi1, st.psi2)
    place(sim, bodies, kin, state)

    sim.setStepping(True)
    sim.startSimulation()
    sim.step(); set_range(sim, sensors, args.far); sim.step()

    # ---- logs -----------------------------------------------------------
    gt_t, gt_state = [], []
    gt_b = {"Tractor": [], "Trailer1": [], "Trailer2": []}
    ctrl_v, ctrl_delta = [], []
    odom_v, odom_w, imu_g = [], [], []
    sc_time, sc_count, sc_pose = [], [], []
    sc_ranges, sc_angles, sc_points = [], [], []

    t = 0.0
    prev_theta = state.theta
    for k in range(args.max_steps):
        v, delta, done = pure_pursuit(state, wps, drv["lookahead"],
                                      drv["v_nominal"], max_steer, L0)
        if done:
            print(f"[record] goal reached at t={t:.1f}s (step {k})")
            break
        state = kin.step(state, v, delta, dt)
        place(sim, bodies, kin, state)
        sim.step()
        t += dt

        # ground truth
        tr, t1, t2 = kin.body_poses(state)
        gt_t.append(t); gt_state.append(state.as_array())
        gt_b["Tractor"].append(tr); gt_b["Trailer1"].append(t1)
        gt_b["Trailer2"].append(t2)

        # controls + noisy odom / imu
        omega = wrap_angle(state.theta - prev_theta) / dt
        prev_theta = state.theta
        ctrl_v.append(v); ctrl_delta.append(delta)
        odom_v.append(v + rng.normal(0, on["v_noise_std"]))
        odom_w.append(omega + rng.normal(0, on["w_noise_std"]))
        imu_g.append(omega + rng.normal(0, imu["gyro_z_noise_std"]))

        # scan in LiDAR (ref) frame + ref world pose
        ang, r, pts = read_scan(sim, sensors, ref, args.far)
        pose = sim.getObjectPose(ref, sim.handle_world)
        yaw = math.atan2(2 * (pose[6] * pose[5] + pose[3] * pose[4]),
                         1 - 2 * (pose[4] ** 2 + pose[5] ** 2))
        sc_time.append(t); sc_count.append(len(r))
        sc_pose.append([pose[0], pose[1], yaw])
        sc_ranges.append(r); sc_angles.append(ang); sc_points.append(pts)

    sim.stopSimulation()
    n = len(gt_t)
    print(f"[record] logged {n} steps, mean returns/scan="
          f"{np.mean(sc_count):.0f}")

    # ---- write HDF5 -----------------------------------------------------
    out = ROOT / sc.cfg["output"]["dataset_dir"] / "dataset.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["config"] = json.dumps(sc.cfg)
        f.attrs["dt"] = dt
        f.create_dataset("meta/landmarks", data=sc.landmarks)   # id,x,y,r
        f.create_dataset("gt/time", data=np.array(gt_t))
        f.create_dataset("gt/state", data=np.array(gt_state))   # x,y,th,p1,p2
        for b in gt_b:
            f.create_dataset(f"gt/{b.lower()}", data=np.array(gt_b[b]))
        f.create_dataset("ctrl/v", data=np.array(ctrl_v))
        f.create_dataset("ctrl/delta", data=np.array(ctrl_delta))
        f.create_dataset("odom/time", data=np.array(gt_t))
        f.create_dataset("odom/v_noisy", data=np.array(odom_v))
        f.create_dataset("odom/omega_noisy", data=np.array(odom_w))
        f.create_dataset("imu/time", data=np.array(gt_t))
        f.create_dataset("imu/gyro_z", data=np.array(imu_g))
        # ragged scan storage: flat arrays + per-scan counts
        f.create_dataset("scan/time", data=np.array(sc_time))
        f.create_dataset("scan/count", data=np.array(sc_count, dtype=np.int32))
        f.create_dataset("scan/sensor_pose", data=np.array(sc_pose))
        f.create_dataset("scan/ranges",
                         data=np.concatenate(sc_ranges) if n else np.zeros(0))
        f.create_dataset("scan/angles",
                         data=np.concatenate(sc_angles) if n else np.zeros(0))
        f.create_dataset("scan/points",
                         data=(np.concatenate(sc_points) if n
                               else np.zeros((0, 2))))
    print(f"[record] wrote {out}")

    # ---- verification figure -------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    gts = np.array(gt_state)
    fig, ax = plt.subplots(figsize=(9, 7))
    for x0, y0, x1, y1 in sc.wall_segments:
        ax.plot([x0, x1], [y0, y1], "k-", lw=1.5)
    for _, x, y, rr in sc.landmarks:
        ax.add_patch(Circle((x, y), rr, color="0.8"))
    # accumulate every 10th scan into the world for a map preview
    for i in range(0, n, 10):
        p = sc_points[i]
        if len(p) == 0:
            continue
        yaw = sc_pose[i][2]; c, s = math.cos(yaw), math.sin(yaw)
        wx = sc_pose[i][0] + c * p[:, 0] - s * p[:, 1]
        wy = sc_pose[i][1] + s * p[:, 0] + c * p[:, 1]
        ax.scatter(wx, wy, s=1, c="tab:red", alpha=0.2)
    ax.plot(gts[:, 0], gts[:, 1], "-", color="tab:blue", lw=2, label="GT path")
    ax.scatter([st.x], [st.y], c="tab:green", s=80, label="start")
    ax.scatter([sc.goal.x], [sc.goal.y], marker="*", c="tab:orange", s=200,
               label="goal")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Recorded dataset: GT path + accumulated LiDAR (world)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    figp = ROOT / "figures/dataset_overview.png"
    fig.tight_layout(); fig.savefig(figp, dpi=140)
    print(f"[record] wrote {figp}")


if __name__ == "__main__":
    main()
