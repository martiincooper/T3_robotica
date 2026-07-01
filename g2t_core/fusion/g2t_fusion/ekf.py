"""Extended Kalman Filter fusing LiDAR-derived articulation angles with
simulated odometry and IMU.

State, motion model and observation model
=========================================

State (6-D)::

    x = [x_t, y_t, theta_t, psi1, psi2, omega_t]^T

with the tractor rear-axle position, tractor yaw, the two articulation
angles, and the tractor yaw rate.

Input::

    u = [v, delta]^T          (tractor speed and front-wheel steering)

Motion model (continuous time, on-axle hitch d0=d1=0 — see comments at
the bottom for the off-axle generalisation)::

    x_t_dot      = v cos(theta_t)
    y_t_dot      = v sin(theta_t)
    theta_t_dot  = omega_t
    psi1_dot     = -(v / L1) sin(psi1) + omega_t
    psi2_dot     = -(v1 / L2) sin(psi2) + omega_t - psi1_dot
                 = -(v cos(psi1)) / L2 * sin(psi2)
                   + (v / L1) sin(psi1)
    omega_t_dot  = 0           # random walk

Discretization uses **forward Euler with the control sampling period
dt** because LiDAR (40 Hz) and odometry (50 Hz) updates are well above
the bandwidth of the trailer dynamics (~ 1 Hz).  Forward Euler keeps the
Jacobian symbolic and human-checkable; switching to RK2 would only
matter for dt > 0.1 s.

Observation models
------------------
Two independent measurement types are handled:

* **LiDAR articulation** ``z = [psi1, psi2]``::

      h(x) = [psi1, psi2]     H = [[0,0,0,1,0,0],
                                   [0,0,0,0,1,0]]

  Each of ``psi1`` and ``psi2`` may be missing on a given scan; the
  filter updates only the available components.  A Mahalanobis χ² gate
  rejects gross outliers.

* **IMU gyroscope** ``z = omega_t``::

      h(x) = omega_t          H = [0, 0, 0, 0, 0, 1]

Tuning rationale (see ``docs/tuning.md``)
-----------------------------------------
* ``Q`` is built as ``Q = G diag(q^2) G^T * dt`` with ``G = I``; we
  approximate each state's process noise as Gaussian white acceleration
  scaled by the configured ``q_*`` standard deviation per √s.
* ``R`` values are taken directly from the simulation noise spec.
* Initial covariance is large enough to swallow the first few mis-fits
  but small enough that the filter converges within ~1 s of operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np


@dataclass
class EKFConfig:
    L0: float = 1.20
    L1: float = 1.50
    L2: float = 1.50

    q_xy: float = 0.05
    q_theta: float = 0.01
    q_psi: float = 0.02
    q_omega: float = 0.10

    r_psi_lidar: float = 0.03
    r_gyro_z: float = 0.01

    p0_xy: float = 0.1
    p0_theta: float = 0.05
    p0_psi: float = 0.1
    p0_omega: float = 0.1

    chi2_gate: float = 9.0


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class G2TEKF:
    cfg: EKFConfig = field(default_factory=EKFConfig)

    def __post_init__(self) -> None:
        self.x = np.zeros(6)
        self.P = np.diag([
            self.cfg.p0_xy ** 2, self.cfg.p0_xy ** 2, self.cfg.p0_theta ** 2,
            self.cfg.p0_psi ** 2, self.cfg.p0_psi ** 2, self.cfg.p0_omega ** 2])
        self._last_t: Optional[float] = None

    # ---------------------------------------------------------- prediction
    def predict(self, v: float, dt: float) -> None:
        """Forward Euler propagation with linearized covariance update."""
        if dt <= 0:
            return
        c = self.cfg
        theta = self.x[2]
        psi1 = self.x[3]
        psi2 = self.x[4]
        omega = self.x[5]

        # Pre-compute reused quantities
        s_psi1 = math.sin(psi1); c_psi1 = math.cos(psi1)
        s_psi2 = math.sin(psi2); c_psi2 = math.cos(psi2)
        v1 = v * c_psi1                       # on-axle longitudinal velocity of trailer-1

        f = np.array([
            v * math.cos(theta),                                # x_dot
            v * math.sin(theta),                                # y_dot
            omega,                                              # theta_dot
            -(v / c.L1) * s_psi1 + omega,                       # psi1_dot
            -(v1 / c.L2) * s_psi2 + omega - (-(v / c.L1) * s_psi1 + omega),
            0.0,                                                # omega_dot
        ])
        # Simplify psi2_dot: omega cancels with -omega, yielding:
        # psi2_dot = -(v cos psi1 / L2) sin psi2 + (v / L1) sin psi1
        f[4] = -(v / c.L2) * c_psi1 * s_psi2 + (v / c.L1) * s_psi1

        # ---------- Jacobian F = ∂f/∂x  (6 × 6) ------------------------
        F = np.zeros((6, 6))
        # d(x_dot)/d(theta)
        F[0, 2] = -v * math.sin(theta)
        # d(y_dot)/d(theta)
        F[1, 2] = v * math.cos(theta)
        # d(theta_dot)/d(omega) = 1
        F[2, 5] = 1.0
        # d(psi1_dot)/d(psi1) and d(psi1_dot)/d(omega)
        F[3, 3] = -(v / c.L1) * c_psi1
        F[3, 5] = 1.0
        # d(psi2_dot)/d(psi1)
        F[4, 3] = (v / c.L2) * s_psi1 * s_psi2 + (v / c.L1) * c_psi1
        # d(psi2_dot)/d(psi2)
        F[4, 4] = -(v / c.L2) * c_psi1 * c_psi2

        # Continuous → discrete: x_{k+1} = x_k + f * dt, P_{k+1} = Φ P Φ^T + Qd
        Phi = np.eye(6) + F * dt
        self.x = self.x + f * dt
        self.x[2] = _wrap(self.x[2])
        self.x[3] = _wrap(self.x[3])
        self.x[4] = _wrap(self.x[4])

        # Process noise covariance (discrete)
        Qd = np.diag([
            (c.q_xy * math.sqrt(dt)) ** 2, (c.q_xy * math.sqrt(dt)) ** 2,
            (c.q_theta * math.sqrt(dt)) ** 2,
            (c.q_psi * math.sqrt(dt)) ** 2,
            (c.q_psi * math.sqrt(dt)) ** 2,
            (c.q_omega * math.sqrt(dt)) ** 2,
        ])
        self.P = Phi @ self.P @ Phi.T + Qd
        # Enforce symmetry to fight numerical drift
        self.P = 0.5 * (self.P + self.P.T)

    # ---------------------------------------------------------- updates
    def update_lidar(self, psi1: Optional[float], psi2: Optional[float]) -> None:
        c = self.cfg
        rows = []
        z = []
        if psi1 is not None and math.isfinite(psi1):
            rows.append(3); z.append(psi1)
        if psi2 is not None and math.isfinite(psi2):
            rows.append(4); z.append(psi2)
        if not rows:
            return
        H = np.zeros((len(rows), 6))
        for i, r in enumerate(rows):
            H[i, r] = 1.0
        R = np.eye(len(rows)) * (c.r_psi_lidar ** 2)
        zp = H @ self.x
        innov = np.array([_wrap(zi - zpi) for zi, zpi in zip(z, zp)])
        S = H @ self.P @ H.T + R
        # Mahalanobis gating (component-wise; cheap and conservative)
        try:
            m2 = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return
        if m2 > c.chi2_gate:
            return
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[2] = _wrap(self.x[2])
        self.x[3] = _wrap(self.x[3])
        self.x[4] = _wrap(self.x[4])
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def update_imu(self, gyro_z: float) -> None:
        c = self.cfg
        H = np.array([[0, 0, 0, 0, 0, 1.0]])
        R = np.array([[c.r_gyro_z ** 2]])
        innov = np.array([gyro_z - self.x[5]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innov).ravel()
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
