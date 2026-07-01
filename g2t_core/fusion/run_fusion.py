"""Batch fusion runner.

Reads the simulation HDF5 dataset and the perception CSV, runs the EKF
over the union of (odometry, IMU, LiDAR) events sorted by timestamp,
and writes the filtered state to a CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .g2t_fusion.ekf import EKFConfig, G2TEKF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--perception", required=True, help="CSV from run_perception")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim_config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        fcfg = yaml.safe_load(f)["ekf"]
    with open(args.sim_config, "r") as f:
        scfg = yaml.safe_load(f)

    cfg = EKFConfig(
        L0=scfg["vehicle"]["L0"], L1=scfg["vehicle"]["L1"], L2=scfg["vehicle"]["L2"],
        q_xy=fcfg["q_xy"], q_theta=fcfg["q_theta"],
        q_psi=fcfg["q_psi"], q_omega=fcfg["q_omega"],
        r_psi_lidar=fcfg["r_psi_lidar"], r_gyro_z=fcfg["r_gyro_z"],
        p0_xy=fcfg["p0"]["xy"], p0_theta=fcfg["p0"]["theta"],
        p0_psi=fcfg["p0"]["psi"], p0_omega=fcfg["p0"]["omega"],
        chi2_gate=fcfg["chi2_gate"])

    ekf = G2TEKF(cfg)

    with h5py.File(args.dataset, "r") as f:
        # Initialize state to ground truth at t=0
        gt0 = f["gt/state"][0]
        ekf.x[0] = gt0[0]; ekf.x[1] = gt0[1]; ekf.x[2] = gt0[2]
        ekf.x[3] = gt0[3]; ekf.x[4] = gt0[4]

        odom_t = f["odom/time"][:]; odom_v = f["odom/v_noisy"][:]
        imu_t = f["imu/time"][:]; imu_g = f["imu/gyro_z"][:]
        scan_t = f["scan/time"][:]

    p = pd.read_csv(args.perception)
    assert len(p) == len(scan_t), (len(p), len(scan_t))

    # Build a sorted event timeline: (timestamp, type, payload)
    events = []
    for i, t in enumerate(odom_t):
        events.append((t, "odom", float(odom_v[i])))
    for i, t in enumerate(imu_t):
        events.append((t, "imu", float(imu_g[i])))
    for i, t in enumerate(scan_t):
        events.append((t, "lidar", (p.psi1_raw.iloc[i], p.psi2_raw.iloc[i])))
    events.sort(key=lambda e: e[0])

    # Step the filter, logging the posterior at every scan timestamp
    last_t = events[0][0] if events else 0.0
    last_v = 0.0
    log_t = []; log_state = []; log_diag = []

    for t, typ, payload in events:
        dt = max(0.0, t - last_t)
        ekf.predict(last_v, dt)
        last_t = t
        if typ == "odom":
            last_v = payload
        elif typ == "imu":
            ekf.update_imu(payload)
        elif typ == "lidar":
            psi1, psi2 = payload
            psi1 = float(psi1) if pd.notna(psi1) else None
            psi2 = float(psi2) if pd.notna(psi2) else None
            ekf.update_lidar(psi1, psi2)
            log_t.append(t)
            log_state.append(ekf.x.copy())
            log_diag.append(np.diag(ekf.P).copy())

    arr = np.array(log_state)
    diag = np.array(log_diag)
    df = pd.DataFrame({
        "time": log_t,
        "x_t": arr[:, 0], "y_t": arr[:, 1], "theta_t": arr[:, 2],
        "psi1": arr[:, 3], "psi2": arr[:, 4], "omega_t": arr[:, 5],
        "var_psi1": diag[:, 3], "var_psi2": diag[:, 4],
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[run_fusion] wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
