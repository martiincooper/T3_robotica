"""1-D scalar Kalman smoother — used as a baseline for psi1, psi2.

Constant-velocity model on the angle::

    [psi]      [1  dt] [psi]       [0.5 dt^2]
    [dpsi]  =  [0   1] [dpsi]  + Q [   dt   ]

Observation: ``z = psi``, noise variance ``r``.

Provides a fast comparison point against the full multi-state EKF without
requiring odometry or IMU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalarKalman1D:
    process_std: float = 0.05    # rad/sqrt(s) on angular velocity
    measurement_std: float = 0.03

    def __post_init__(self) -> None:
        self.x = np.zeros(2)        # [psi, dpsi]
        self.P = np.eye(2) * 1e-2
        self._initialized = False

    def predict(self, dt: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        q = self.process_std ** 2
        Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                          [dt ** 2 / 2.0, dt]])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: float) -> None:
        if not self._initialized:
            self.x[0] = z
            self.x[1] = 0.0
            self._initialized = True
            return
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.measurement_std ** 2]])
        # Innovation with angle wrap-around
        y = np.array([float(np.arctan2(np.sin(z - self.x[0]),
                                       np.cos(z - self.x[0])))])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.x[0] = float(np.arctan2(np.sin(self.x[0]), np.cos(self.x[0])))
        self.P = (np.eye(2) - K @ H) @ self.P

    @property
    def value(self) -> float:
        return float(self.x[0])

    @property
    def is_initialized(self) -> bool:
        """Public read of the internal initialisation flag."""
        return self._initialized
