"""Kinematic model of a tractor with two passive trailers (G2T).

Mathematical model
------------------
Tractor (front-wheel-steered bicycle model)::

    x_dot       =  v * cos(theta)
    y_dot       =  v * sin(theta)
    theta_dot   =  v / L0 * tan(delta)

with state ``(x, y, theta)`` at the tractor rear axle and control
``u = (v, delta)`` (longitudinal speed and front-wheel steering angle).

Trailers (off-axle hitch, see Altafini 2003, "Some properties of the
general n-trailer")::

    psi1_dot = v / L1 * sin(psi1) - (d0 / L1) * cos(psi1) * theta_dot
    psi2_dot = v1 / L2 * sin(psi2) - (d1 / L2) * cos(psi2) * theta1_dot

where ``psi_i`` is the articulation angle between body i-1 and body i,
``L_i`` is the wheelbase of trailer i (hitch-to-axle), ``d_{i-1}`` is the
off-axle offset of the towing point on the preceding body, ``v_i`` is the
longitudinal velocity of body i and ``theta_i = theta_{i-1} - psi_i`` is
its absolute orientation.

For an off-axle G2T the per-body velocities propagate as::

    v_i = v_{i-1} * cos(psi_i) + d_{i-1} * sin(psi_i) * theta_{i-1}_dot

The implementation below uses an explicit RK4 integrator with a fixed
time step.  RK4 was chosen over Euler because the tightly-curved
trajectories used in the experiments produce articulation rates that
make Euler diverge for dt > 0.01 s; RK4 remains stable up to dt = 0.05 s
in our tests (see ``docs/methodology.md``).

Assumptions / limitations
-------------------------
* No tire dynamics: pure kinematics, infinite friction, no slip.
* Massless trailers; the IEEE report will explicitly state this.
* Steering angle is the *commanded* angle (no actuator lag).
* Single-track ("bicycle") simplification — body widths are only used
  for visualization, not for dynamics or collisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Tuple

import numpy as np


@dataclass
class G2TState:
    """Full state of the G2T system in the world frame."""

    x: float = 0.0          # tractor rear-axle x [m]
    y: float = 0.0          # tractor rear-axle y [m]
    theta: float = 0.0      # tractor yaw [rad]
    psi1: float = 0.0       # tractor -> trailer-1 articulation [rad]
    psi2: float = 0.0       # trailer-1 -> trailer-2 articulation [rad]

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.psi1, self.psi2],
                        dtype=float)

    @classmethod
    def from_array(cls, a: np.ndarray) -> "G2TState":
        return cls(float(a[0]), float(a[1]), float(a[2]),
                   float(a[3]), float(a[4]))


@dataclass
class G2TParams:
    L0: float = 1.20
    d0: float = 0.30
    L1: float = 1.50
    d1: float = 0.20
    L2: float = 1.50


class G2TKinematics:
    """Integrator for the G2T kinematic model.

    Parameters
    ----------
    params : G2TParams
        Geometry of the vehicle (wheelbases + hitch offsets).
    psi_limit : float
        Hard saturation applied to each articulation angle. Real
        hitches cannot exceed ~80°; we default to 75° (1.31 rad).
    """

    def __init__(self, params: G2TParams, psi_limit: float = math.radians(75.0)):
        self.p = params
        self.psi_limit = psi_limit

    # ------------------------------------------------------------------ ODE
    def f(self, s: G2TState, v: float, delta: float) -> np.ndarray:
        """State derivative ``ds/dt`` for control ``(v, delta)``.

        Derivation (off-axle n-trailer, hitch BEHIND the towing axle by
        offset d_{i-1} ≥ 0; see ``docs/methodology.md``)::

            ω_i        =  ( v_{i-1} sin ψ_i − d_{i-1} ω_{i-1} cos ψ_i ) / L_i
            ψ_i_dot    =  ω_{i-1} − ω_i
                       = −v_{i-1} sin ψ_i / L_i
                          + ω_{i-1} ( 1 + d_{i-1} cos ψ_i / L_i )
            v_i        =  v_{i-1} cos ψ_i + d_{i-1} ω_{i-1} sin ψ_i

        For the on-axle limit d_{i-1} = 0 this reduces to the textbook
        n-trailer model and is asymptotically stable under straight-line
        motion (Murray & Sastry, 1990).
        """
        p = self.p
        # Tractor (body 0)
        omega0 = v / p.L0 * math.tan(delta)
        x_dot = v * math.cos(s.theta)
        y_dot = v * math.sin(s.theta)

        # Trailer 1 (body 1)
        psi1_dot = -(v / p.L1) * math.sin(s.psi1) \
                   + omega0 * (1.0 + (p.d0 / p.L1) * math.cos(s.psi1))

        # Longitudinal speed and yaw rate of trailer 1 (needed for trailer 2)
        v1 = v * math.cos(s.psi1) + p.d0 * omega0 * math.sin(s.psi1)
        omega1 = omega0 - psi1_dot

        # Trailer 2 (body 2)
        psi2_dot = -(v1 / p.L2) * math.sin(s.psi2) \
                   + omega1 * (1.0 + (p.d1 / p.L2) * math.cos(s.psi2))

        return np.array([x_dot, y_dot, omega0, psi1_dot, psi2_dot])

    # ------------------------------------------------------------------ RK4
    def step(self, s: G2TState, v: float, delta: float, dt: float) -> G2TState:
        """Advance state by ``dt`` using classical RK4."""
        a = s.as_array()

        def _f(arr: np.ndarray) -> np.ndarray:
            return self.f(G2TState.from_array(arr), v, delta)

        k1 = _f(a)
        k2 = _f(a + 0.5 * dt * k1)
        k3 = _f(a + 0.5 * dt * k2)
        k4 = _f(a + dt * k3)
        a_new = a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Saturate articulation angles (jack-knife guard).
        a_new[3] = float(np.clip(a_new[3], -self.psi_limit, self.psi_limit))
        a_new[4] = float(np.clip(a_new[4], -self.psi_limit, self.psi_limit))
        # Wrap yaw into (-pi, pi]
        a_new[2] = math.atan2(math.sin(a_new[2]), math.cos(a_new[2]))
        return G2TState.from_array(a_new)

    # ------------------------------------------------------------- FK helper
    def body_poses(self, s: G2TState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return absolute (x, y, yaw) poses of tractor, trailer-1, trailer-2.

        The convention is that the pose of each body is anchored at the
        **rear axle / hitch point of that body**, matching the kinematic
        derivation above. This is also the frame that the LiDAR mount is
        expressed in.
        """
        p = self.p
        theta1 = s.theta - s.psi1
        theta2 = theta1 - s.psi2

        # Hitch 1 = tractor rear-axle origin shifted by -d0 along tractor heading
        hx1 = s.x - p.d0 * math.cos(s.theta)
        hy1 = s.y - p.d0 * math.sin(s.theta)
        # Trailer-1 rear axle is L1 behind hitch 1 along trailer-1 heading
        x1 = hx1 - p.L1 * math.cos(theta1)
        y1 = hy1 - p.L1 * math.sin(theta1)

        # Hitch 2 = trailer-1 rear axle shifted by -d1 along trailer-1 heading
        hx2 = x1 - p.d1 * math.cos(theta1)
        hy2 = y1 - p.d1 * math.sin(theta1)
        # Trailer-2 rear axle is L2 behind hitch 2
        x2 = hx2 - p.L2 * math.cos(theta2)
        y2 = hy2 - p.L2 * math.sin(theta2)

        return (np.array([s.x, s.y, s.theta]),
                np.array([x1, y1, theta1]),
                np.array([x2, y2, theta2]))

    def hitch_points(self, s: G2TState) -> Tuple[np.ndarray, np.ndarray]:
        """Return the world-frame XY positions of hitch 1 and hitch 2.

        These are the physical articulation points between tractor/trailer-1
        and trailer-1/trailer-2 respectively. The IPD-482 Guía 2 statement
        explicitly mentions an "elemento físico (poste o marcador)" located
        in this region; the simulator places a small cylinder at each hitch
        so it appears in the LiDAR cloud (see ``run_simulation.py``).
        """
        p = self.p
        theta1 = s.theta - s.psi1
        # Hitch 1
        hx1 = s.x - p.d0 * math.cos(s.theta)
        hy1 = s.y - p.d0 * math.sin(s.theta)
        # Trailer-1 rear axle
        x1 = hx1 - p.L1 * math.cos(theta1)
        y1 = hy1 - p.L1 * math.sin(theta1)
        # Hitch 2
        hx2 = x1 - p.d1 * math.cos(theta1)
        hy2 = y1 - p.d1 * math.sin(theta1)
        return np.array([hx1, hy1]), np.array([hx2, hy2])


# ----------------------------------------------------------------------- util
def wrap_angle(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
