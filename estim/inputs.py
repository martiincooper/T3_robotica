"""Build the noisy control/odometry inputs the estimators consume.

The recorded dataset is faithful but its odometry barely drifts under
pure white noise. Real wheel odometry drifts because of *systematic*
errors (wheel-scale, gyro bias). Since the dataset stores the true
controls and ground truth, we synthesise realistic odometry here:

    v_odom = s_v * v_true + b_v + N(0, sigma_v)
    w_odom = s_w * w_true + b_w + N(0, sigma_w)

with a scale (s), a constant bias (b) and white noise (sigma). This is
where Block 4 varies the noise; keeping it out of the dataset means one
recording serves every noise condition and every random seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class NoiseCfg:
    s_v: float = 1.02       # +2% velocity scale error
    b_v: float = 0.0        # m/s bias
    sigma_v: float = 0.03   # m/s white
    s_w: float = 1.0
    b_w: float = 0.0105     # rad/s gyro bias (~0.6 deg/s)
    sigma_w: float = 0.02   # rad/s white


@dataclass
class Inputs:
    t: np.ndarray
    dt: float
    v_true: np.ndarray
    w_true: np.ndarray
    v_odom: np.ndarray
    w_odom: np.ndarray
    gt_tractor: np.ndarray          # (N,3)
    gt_state: np.ndarray            # (N,5)


def build_inputs(f, noise: NoiseCfg, rng: np.random.Generator) -> Inputs:
    dt = float(f.attrs["dt"])
    t = f["gt/time"][:]
    v_true = f["ctrl/v"][:]
    gt_tr = f["gt/tractor"][:]
    gt_state = f["gt/state"][:]
    # true yaw rate from GT heading
    th = gt_tr[:, 2]
    w_true = np.zeros_like(th)
    w_true[1:] = np.arctan2(np.sin(th[1:] - th[:-1]),
                            np.cos(th[1:] - th[:-1])) / dt
    n = len(t)
    v_odom = noise.s_v * v_true + noise.b_v + rng.normal(0, noise.sigma_v, n)
    w_odom = noise.s_w * w_true + noise.b_w + rng.normal(0, noise.sigma_w, n)
    return Inputs(t, dt, v_true, w_true, v_odom, w_odom, gt_tr, gt_state)
