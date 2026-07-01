"""Top-level simulation driver — writes a self-contained HDF5 dataset.

Usage
-----
    python -m simulation.run_simulation --config config/simulation.yaml \
                                        --out datasets/sim_run

Output layout (one HDF5 file ``dataset.h5`` + companion ``meta.yaml``):

* ``/time``                  (N_steps,)   simulation timestamps [s]
* ``/gt/state``              (N_steps, 5) ground-truth full state
* ``/gt/tractor_pose``       (N_steps, 3) (x, y, yaw)
* ``/gt/trailer1_pose``      (N_steps, 3)
* ``/gt/trailer2_pose``      (N_steps, 3)
* ``/control/v``             (N_steps,)
* ``/control/delta``         (N_steps,)
* ``/odom/v_noisy``          (N_odom,)
* ``/odom/omega_noisy``      (N_odom,)
* ``/odom/time``             (N_odom,)
* ``/imu/gyro_z``            (N_imu,)
* ``/imu/time``              (N_imu,)
* ``/scan/time``             (N_scans,)
* ``/scan/ranges``           (N_scans, N_beams)
* ``/scan/angles``           (N_beams,)
* ``/scan/range_min``, ``/scan/range_max``  scalars

The dataset is sufficient to run the entire perception + fusion +
evaluation pipeline offline. A separate script (``scripts/h5_to_rosbag.py``)
converts it into a ROS 2 bag when desired.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from .g2t_sim.kinematics import G2TKinematics, G2TParams, G2TState
from .g2t_sim.lidar import Lidar2D, LidarParams
from .g2t_sim.trajectories import build_trajectory
from .g2t_sim.world import BodyGeom, Obstacle, Pedestrian, World


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cfg: dict, out_dir: str) -> str:
    rng = np.random.default_rng(int(cfg["output"]["seed"]))
    veh = cfg["vehicle"]
    params = G2TParams(L0=veh["L0"], d0=veh["d0"],
                       L1=veh["L1"], d1=veh["d1"], L2=veh["L2"])
    kin = G2TKinematics(params)
    init = cfg["scenario"]["initial_state"]
    state = G2TState(x=float(init["x0"]), y=float(init["y0"]),
                     theta=float(init["theta0"]),
                     psi1=float(init["psi1"]), psi2=float(init["psi2"]))

    # World
    world = World(
        tractor_geom=BodyGeom(veh["body_length_0"], veh["W0"]),
        trailer1_geom=BodyGeom(veh["body_length_1"], veh["W1"]),
        trailer2_geom=BodyGeom(veh["body_length_2"], veh["W2"]),
        obstacles=[Obstacle(**o) for o in cfg["scenario"]["obstacles"]],
        pedestrians=[Pedestrian(**p) for p in cfg["scenario"]["pedestrians"]],
    )

    # LiDAR
    lp = LidarParams(**{k: cfg["lidar"][k] for k in [
        "range_min", "range_max", "fov_deg", "angular_resolution_deg",
        "range_noise_std", "dropout_prob", "mixed_pixel_prob", "rate_hz"]})
    lidar = Lidar2D(lp, rng=rng)
    mount_xy = np.asarray(cfg["lidar"]["mount_xy"], dtype=float)
    mount_yaw = float(cfg["lidar"]["mount_yaw"])

    traj = build_trajectory(cfg["scenario"]["trajectory"])

    # Time discretisation
    ctrl_rate = float(cfg["scenario"]["control_rate_hz"])
    dt = 1.0 / ctrl_rate
    duration = float(cfg["scenario"]["duration_s"])
    n_steps = int(duration * ctrl_rate)

    odom_period = 1.0 / float(cfg["odometry"]["rate_hz"])
    imu_period = 1.0 / float(cfg["imu"]["rate_hz"])
    scan_period = 1.0 / lp.rate_hz

    # Buffers
    times = np.zeros(n_steps)
    states = np.zeros((n_steps, 5))
    p0 = np.zeros((n_steps, 3))
    p1 = np.zeros((n_steps, 3))
    p2 = np.zeros((n_steps, 3))
    ctrl_v = np.zeros(n_steps)
    ctrl_d = np.zeros(n_steps)

    odom_t, odom_v, odom_w = [], [], []
    imu_t, imu_gz = [], []
    scan_t, scan_r = [], []

    next_odom = 0.0
    next_imu = 0.0
    next_scan = 0.0

    onoise_v = float(cfg["odometry"]["v_noise_std"])
    onoise_w = float(cfg["odometry"]["w_noise_std"])
    inoise = float(cfg["imu"]["gyro_z_noise_std"])

    pole_cfg = cfg["scenario"].get("articulation_poles",
                                   {"enable": False, "radius": 0.05})
    pole_enabled = bool(pole_cfg.get("enable", False))
    pole_radius = float(pole_cfg.get("radius", 0.05))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "dataset.h5"

    for k in range(n_steps):
        t = k * dt
        v, delta = traj(t)
        # Step kinematics
        state = kin.step(state, v, delta, dt)
        times[k] = t
        states[k] = state.as_array()
        pa, pb, pc = kin.body_poses(state)
        p0[k] = pa; p1[k] = pb; p2[k] = pc
        ctrl_v[k] = v; ctrl_d[k] = delta

        # ----- Odometry samples (with noise) -----------------------
        if t >= next_odom:
            true_omega = v / params.L0 * math.tan(delta)
            odom_v.append(v + rng.normal(0.0, onoise_v))
            odom_w.append(true_omega + rng.normal(0.0, onoise_w))
            odom_t.append(t)
            next_odom += odom_period

        # ----- IMU samples ----------------------------------------
        if t >= next_imu:
            true_omega = v / params.L0 * math.tan(delta)
            imu_gz.append(true_omega + rng.normal(0.0, inoise))
            imu_t.append(t)
            next_imu += imu_period

        # ----- LiDAR scans ----------------------------------------
        if t >= next_scan:
            # Sensor pose in world = tractor pose composed with mount.
            cth, sth = math.cos(pa[2]), math.sin(pa[2])
            sx = pa[0] + cth * mount_xy[0] - sth * mount_xy[1]
            sy = pa[1] + sth * mount_xy[0] + cth * mount_xy[1]
            syaw = pa[2] + mount_yaw

            # Only the trailers (and obstacles/pedestrians) are in the LiDAR
            # FOV (it points backwards from the tractor rear). We include
            # the tractor body too for completeness; rays starting inside
            # the tractor cabin would otherwise hit it.
            segs = world.trailer_only_segments(pb, pc)
            circs = world.circles_at(t)
            if pole_enabled:
                # Add the two articulation-zone poles ("poste o marcador"
                # in the Guía 2 statement) as cylinders rigidly attached
                # to the current hitch points.
                h1, h2 = kin.hitch_points(state)
                pole_circles = np.array([
                    [float(h1[0]), float(h1[1]), pole_radius],
                    [float(h2[0]), float(h2[1]), pole_radius],
                ])
                circs = np.vstack([circs, pole_circles]) if circs.size \
                    else pole_circles
            scan = lidar.scan((sx, sy, syaw), segs, circs, t)
            scan_t.append(t)
            scan_r.append(scan.ranges.astype(np.float32))
            next_scan += scan_period

    # ---- Write HDF5 ------------------------------------------------------
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("gt/state", data=states)
        f.create_dataset("gt/tractor_pose", data=p0)
        f.create_dataset("gt/trailer1_pose", data=p1)
        f.create_dataset("gt/trailer2_pose", data=p2)
        f.create_dataset("control/v", data=ctrl_v)
        f.create_dataset("control/delta", data=ctrl_d)

        f.create_dataset("odom/time", data=np.asarray(odom_t))
        f.create_dataset("odom/v_noisy", data=np.asarray(odom_v))
        f.create_dataset("odom/omega_noisy", data=np.asarray(odom_w))

        f.create_dataset("imu/time", data=np.asarray(imu_t))
        f.create_dataset("imu/gyro_z", data=np.asarray(imu_gz))

        f.create_dataset("scan/time", data=np.asarray(scan_t))
        f.create_dataset("scan/ranges", data=np.asarray(scan_r))
        f.create_dataset("scan/angles", data=lp.angles())
        f.create_dataset("scan/range_min", data=lp.range_min)
        f.create_dataset("scan/range_max", data=lp.range_max)
        # Vehicle params for downstream consumers
        gp = f.create_group("params/vehicle")
        for k_, v_ in params.__dict__.items():
            gp.attrs[k_] = v_
        gm = f.create_group("params/lidar_mount")
        gm.attrs["x"] = mount_xy[0]
        gm.attrs["y"] = mount_xy[1]
        gm.attrs["yaw"] = mount_yaw
        # Articulation-pole metadata ("poste o marcador" from Guía 2)
        gpo = f.create_group("params/articulation_poles")
        gpo.attrs["enable"] = pole_enabled
        gpo.attrs["radius"] = pole_radius

    with open(out_dir / "meta.yaml", "w") as f:
        yaml.safe_dump({"source_config": cfg, "n_steps": n_steps,
                        "n_scans": len(scan_t), "duration_s": duration}, f)

    return str(h5_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    out = run(cfg, args.out)
    print(f"[run_simulation] wrote {out}")


if __name__ == "__main__":
    main()
