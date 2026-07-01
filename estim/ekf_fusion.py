"""Extended Kalman Filter for fusing SLAM pose, IMU yaw rate, and articulation angles."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np


@dataclass
class EKFFusionConfig:
    L0: float = 1.20
    L1: float = 1.50
    L2: float = 1.50

    # Process noise standard deviations (continuous per sqrt(s))
    q_xy: float = 0.05
    q_theta: float = 0.01
    q_psi: float = 0.02
    q_omega: float = 0.10

    # Measurement noise standard deviations
    r_psi_lidar: float = 0.03   # simulated noisy articulation
    r_gyro_z: float = 0.01      # IMU gyro_z
    r_w_odom: float = 0.02      # wheel odometry yaw rate
    r_x_slam: float = 0.15      # SLAM position x
    r_y_slam: float = 0.15      # SLAM position y
    r_theta_slam: float = 0.02  # SLAM heading theta (rad)

    # Initial state covariance
    p0_xy: float = 0.1
    p0_theta: float = 0.05
    p0_psi: float = 0.1
    p0_omega: float = 0.1

    # Chi-square gating thresholds
    chi2_gate: float = 9.0       # 98.9% for 2 dof (articulation)
    chi2_gate_slam: float = 16.27 # 99.9% for 3 dof (SLAM pose)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class EKFFusion:
    cfg: EKFFusionConfig = field(default_factory=EKFFusionConfig)

    def __post_init__(self) -> None:
        # State: [x_t, y_t, theta_t, psi1, psi2, omega_t]
        self.x = np.zeros(6)
        self.P = np.diag([
            self.cfg.p0_xy ** 2, self.cfg.p0_xy ** 2, self.cfg.p0_theta ** 2,
            self.cfg.p0_psi ** 2, self.cfg.p0_psi ** 2, self.cfg.p0_omega ** 2
        ])

    def predict(self, v: float, dt: float) -> None:
        """Forward Euler propagation using the kinematics model with estimated yaw rate."""
        if dt <= 0:
            return
        c = self.cfg
        theta = self.x[2]
        psi1 = self.x[3]
        psi2 = self.x[4]
        omega = self.x[5]

        # Precompute trigonometric terms
        s_psi1 = math.sin(psi1); c_psi1 = math.cos(psi1)
        s_psi2 = math.sin(psi2); c_psi2 = math.cos(psi2)
        v1 = v * c_psi1

        # Continuous-time state derivative f(x, u)
        f = np.array([
            v * math.cos(theta),                                # x_dot
            v * math.sin(theta),                                # y_dot
            omega,                                              # theta_dot
            -(v / c.L1) * s_psi1 + omega,                       # psi1_dot
            -(v1 / c.L2) * s_psi2 + (v / c.L1) * s_psi1,         # psi2_dot (simplified)
            0.0,                                                # omega_dot (random walk)
        ])

        # Jacobian F = df/dx (6x6)
        F = np.zeros((6, 6))
        F[0, 2] = -v * math.sin(theta)
        F[1, 2] = v * math.cos(theta)
        F[2, 5] = 1.0
        F[3, 3] = -(v / c.L1) * c_psi1
        F[3, 5] = 1.0
        F[4, 3] = (v / c.L2) * s_psi1 * s_psi2 + (v / c.L1) * c_psi1
        F[4, 4] = -(v / c.L2) * c_psi1 * c_psi2

        # Discrete-time propagation: x_{k+1} = x_k + f * dt
        Phi = np.eye(6) + F * dt
        self.x = self.x + f * dt
        self.x[2] = _wrap(self.x[2])
        self.x[3] = _wrap(self.x[3])
        self.x[4] = _wrap(self.x[4])

        # Process noise covariance
        Qd = np.diag([
            (c.q_xy * math.sqrt(dt)) ** 2,
            (c.q_xy * math.sqrt(dt)) ** 2,
            (c.q_theta * math.sqrt(dt)) ** 2,
            (c.q_psi * math.sqrt(dt)) ** 2,
            (c.q_psi * math.sqrt(dt)) ** 2,
            (c.q_omega * math.sqrt(dt)) ** 2,
        ])
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    def update_lidar(self, psi1: Optional[float], psi2: Optional[float]) -> None:
        """Update articulation angles using simulated noisy articulation measurements."""
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
        self.P = 0.5 * (self.P + self.P.T)

    def update_imu(self, gyro_z: float) -> None:
        """Update the yaw rate using IMU gyro measurements."""
        c = self.cfg
        H = np.array([[0, 0, 0, 0, 0, 1.0]])
        R = np.array([[c.r_gyro_z ** 2]])
        innov = np.array([gyro_z - self.x[5]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innov).ravel()
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_odom_w(self, w_odom: float) -> None:
        """Update the yaw rate using wheel odometry yaw rate measurements."""
        c = self.cfg
        H = np.array([[0, 0, 0, 0, 0, 1.0]])
        R = np.array([[c.r_w_odom ** 2]])
        innov = np.array([w_odom - self.x[5]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innov).ravel()
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_slam(self, x_slam: float, y_slam: float, theta_slam: float) -> None:
        """Update tractor pose [x, y, theta] using SLAM pose output."""
        c = self.cfg
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.diag([c.r_x_slam ** 2, c.r_y_slam ** 2, c.r_theta_slam ** 2])
        z = np.array([x_slam, y_slam, theta_slam])
        zp = H @ self.x
        innov = np.array([
            z[0] - zp[0],
            z[1] - zp[1],
            _wrap(z[2] - zp[2])
        ])
        S = H @ self.P @ H.T + R
        try:
            m2 = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return
        if m2 > c.chi2_gate_slam:
            return
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[2] = _wrap(self.x[2])
        self.x[3] = _wrap(self.x[3])
        self.x[4] = _wrap(self.x[4])
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
