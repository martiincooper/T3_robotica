"""EKF-SLAM for the G2T tractor with range-bearing cylinder landmarks.

State (grows as landmarks are discovered)::

    x = [ x_t, y_t, theta_t,  l1x, l1y,  l2x, l2y,  ... ]^T

Motion model (velocity / unicycle at the tractor rear axle, driven by the
odometry input u = [v, w])::

    x_t     <- x_t + v dt cos(theta_t)
    y_t     <- y_t + v dt sin(theta_t)
    theta_t <- theta_t + w dt

Landmarks are static, so only the 3x3 pose block moves. The control noise
is mapped through V = d f / d u to build the process covariance
``Q = V M V^T`` with ``M = diag(sigma_v^2, sigma_w^2)`` -- this is the
standard EKF-SLAM propagation (Thrun, Probabilistic Robotics, ch. 10).

Observation of landmark j (range-bearing in the sensor frame, which is
taken co-located with the tractor frame plus a fixed yaw offset absorbed
into the bearing)::

    r   = sqrt(dx^2 + dy^2)
    phi = atan2(dy, dx) - theta_t          (dx = l_jx - x_t, dy = l_jy - y_t)

Data association is nearest-neighbour with a Mahalanobis chi-square gate;
unmatched observations that also exceed a *creation* distance to every
map landmark initialise a new landmark via the inverse measurement model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class SlamCfg:
    sigma_v: float = 0.03
    sigma_w: float = 0.02
    sigma_r: float = 0.08        # range meas. noise [m]
    sigma_b: float = 0.03        # bearing meas. noise [rad]
    gate_chi2: float = 13.8      # 99.9% for 2 dof (associate if below)
    new_lm_dist: float = 1.4     # separation to treat obs as a new landmark [m]
    promote_count: int = 3       # confirmed sightings before a lm enters the map
    cand_radius: float = 0.6     # candidate association radius [m]
    mx: float = 0.9              # LiDAR mount offset from rear axle (body x)
    my: float = 0.0
    myaw: float = math.pi        # LiDAR faces backward


class EKFSLAM:
    def __init__(self, cfg: SlamCfg, x0, y0, th0):
        self.cfg = cfg
        self.x = np.array([x0, y0, th0], float)
        self.P = np.diag([1e-4, 1e-4, 1e-4])
        self.lm_ids: List[int] = []       # bookkeeping (index -> label)
        self._next = 0
        # provisional landmarks: list of [wx, wy, count]
        self._cand: List[list] = []

    # ---------------------------------------------------------- predict
    def predict(self, v: float, w: float, dt: float) -> None:
        th = self.x[2]
        self.x[0] += v * dt * math.cos(th)
        self.x[1] += v * dt * math.sin(th)
        self.x[2] = wrap(self.x[2] + w * dt)
        n = len(self.x)
        Fx = np.eye(n)
        Fx[0, 2] = -v * dt * math.sin(th)
        Fx[1, 2] = v * dt * math.cos(th)
        V = np.zeros((n, 2))
        V[0, 0] = dt * math.cos(th)
        V[1, 0] = dt * math.sin(th)
        V[2, 1] = dt
        M = np.diag([self.cfg.sigma_v ** 2, self.cfg.sigma_w ** 2])
        self.P = Fx @ self.P @ Fx.T + V @ M @ V.T

    # ------------------------------------------------ observation model
    def _sensor(self):
        xt, yt, th = self.x[0], self.x[1], self.x[2]
        c, s = math.cos(th), math.sin(th)
        sx = xt + c * self.cfg.mx - s * self.cfg.my
        sy = yt + s * self.cfg.mx + c * self.cfg.my
        # d(sx,sy)/d theta
        dsx = -s * self.cfg.mx - c * self.cfg.my
        dsy = c * self.cfg.mx - s * self.cfg.my
        return sx, sy, th, dsx, dsy

    def _h(self, j: int):
        """Predicted (r, phi_sensor) and Jacobian H (2 x n) for landmark j,
        accounting for the LiDAR lever arm (sensor offset from the axle)."""
        sx, sy, th, dsx, dsy = self._sensor()
        lx, ly = self.x[3 + 2 * j], self.x[4 + 2 * j]
        dx, dy = lx - sx, ly - sy
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        zhat = np.array([r, wrap(math.atan2(dy, dx) - th - self.cfg.myaw)])
        n = len(self.x)
        H = np.zeros((2, n))
        # range wrt pose
        H[0, 0] = -dx / r; H[0, 1] = -dy / r
        H[0, 2] = (dx * (-dsx) + dy * (-dsy)) / r
        # bearing wrt pose
        H[1, 0] = dy / q;  H[1, 1] = -dx / q
        H[1, 2] = (dx * (-dsy) - dy * (-dsx)) / q - 1.0
        # wrt landmark
        H[0, 3 + 2 * j] = dx / r;  H[0, 4 + 2 * j] = dy / r
        H[1, 3 + 2 * j] = -dy / q; H[1, 4 + 2 * j] = dx / q
        return zhat, H

    def num_landmarks(self) -> int:
        return (len(self.x) - 3) // 2

    def _add_landmark(self, r: float, b_sensor: float) -> None:
        sx, sy, th, _, _ = self._sensor()
        ang = th + self.cfg.myaw + b_sensor
        lx = sx + r * math.cos(ang)
        ly = sy + r * math.sin(ang)
        self.x = np.concatenate([self.x, [lx, ly]])
        n = len(self.x)
        P = np.zeros((n, n)); P[:n - 2, :n - 2] = self.P
        P[n - 2, n - 2] = 4.0; P[n - 1, n - 1] = 4.0   # large initial cov
        self.P = P
        self.lm_ids.append(self._next); self._next += 1

    # -------------------------------------------------------- update
    def update(self, observations) -> None:
        """observations: list of (range, bearing_in_sensor_frame)."""
        R = np.diag([self.cfg.sigma_r ** 2, self.cfg.sigma_b ** 2])
        for r_meas, b_sensor in observations:
            z = np.array([r_meas, wrap(b_sensor)])
            best_m2, best_H, best_S, best_innov = None, None, None, None
            for j in range(self.num_landmarks()):
                zhat, H = self._h(j)
                innov = np.array([z[0] - zhat[0], wrap(z[1] - zhat[1])])
                S = H @ self.P @ H.T + R
                m2 = float(innov @ np.linalg.solve(S, innov))
                if best_m2 is None or m2 < best_m2:
                    best_m2, best_H, best_S, best_innov = m2, H, S, innov
            if best_m2 is not None and best_m2 < self.cfg.gate_chi2:
                K = self.P @ best_H.T @ np.linalg.inv(best_S)
                self.x = self.x + K @ best_innov
                self.x[2] = wrap(self.x[2])
                I = np.eye(len(self.x))
                self.P = (I - K @ best_H) @ self.P
                self.P = 0.5 * (self.P + self.P.T)
            else:
                # not matched to a MAP landmark: route to provisional list
                sx, sy, th, _, _ = self._sensor()
                ang = th + self.cfg.myaw + b_sensor
                wx = sx + r_meas * math.cos(ang)
                wy = sy + r_meas * math.sin(ang)
                # ignore if very close to an existing map landmark (avoids dup)
                near_map = any(
                    math.hypot(wx - self.x[3 + 2 * j], wy - self.x[4 + 2 * j])
                    < self.cfg.new_lm_dist for j in range(self.num_landmarks()))
                if near_map:
                    continue
                self._register_candidate(wx, wy, r_meas, b_sensor)

    def _register_candidate(self, wx, wy, r_meas, b_sensor) -> None:
        for c in self._cand:
            if math.hypot(wx - c[0], wy - c[1]) < self.cfg.cand_radius:
                # running-average the candidate position, bump its count
                c[0] = 0.7 * c[0] + 0.3 * wx
                c[1] = 0.7 * c[1] + 0.3 * wy
                c[2] += 1
                if c[2] >= self.cfg.promote_count:
                    self._add_landmark(r_meas, b_sensor)
                    self._cand.remove(c)
                return
        self._cand.append([wx, wy, 1])

    # -------------------------------------------------------- accessors
    def pose(self) -> np.ndarray:
        return self.x[:3].copy()

    def landmarks(self) -> np.ndarray:
        m = self.num_landmarks()
        return self.x[3:3 + 2 * m].reshape(m, 2) if m else np.zeros((0, 2))
