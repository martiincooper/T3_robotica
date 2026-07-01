"""Steering / velocity command generators for the simulator.

Each trajectory exposes the same interface::

    cmd(t: float) -> (v: float, delta: float)

so that ``run_simulation.py`` can swap them via a single config string.

The four trajectories below are designed to **explicitly exercise**
phenomena that stress both the geometric estimator and the EKF:

* ``snake``         — sinusoidal steering produces continuous, periodic
                       articulation angles spanning ±25° on both joints.
* ``figure_eight``  — sustained turning of opposite signs; tests symmetry
                       of the estimator and the EKF heading wrap-around.
* ``slalom``        — square-wave-like steering with fast counter-curves;
                       creates large psi rates that stress the motion model.
* ``parking``       — slow speed with abrupt reversals; checks behaviour
                       when v ≈ 0 (kinematic singularity for the trailer
                       sub-dynamics — see comments in ``kinematics.py``).
"""

from __future__ import annotations

import math
from typing import Callable, Tuple


CommandFn = Callable[[float], Tuple[float, float]]


def _wrap(period: float, t: float) -> float:
    return (t % period) / period


def snake(v: float, amp_deg: float, period: float) -> CommandFn:
    amp = math.radians(amp_deg)

    def cmd(t: float):
        return v, amp * math.sin(2 * math.pi * t / period)
    return cmd


def figure_eight(v: float, amp_deg: float, period: float) -> CommandFn:
    amp = math.radians(amp_deg)

    def cmd(t: float):
        # Two half-periods of opposite turning radius
        sign = 1.0 if _wrap(period, t) < 0.5 else -1.0
        return v, sign * amp
    return cmd


def slalom(v: float, amp_deg: float, period: float) -> CommandFn:
    """Smoothed square-wave (tanh edges) to avoid steering discontinuities
    that would make RK4 misbehave."""
    amp = math.radians(amp_deg)

    def cmd(t: float):
        phase = (t % period) / period   # 0..1
        # Trapezoidal pattern: +amp, ramp, -amp, ramp
        if phase < 0.25:
            d = amp
        elif phase < 0.5:
            d = amp * (1 - 8 * (phase - 0.25))
        elif phase < 0.75:
            d = -amp
        else:
            d = -amp * (1 - 8 * (phase - 0.75))
        return v, max(-amp, min(amp, d))
    return cmd


def parking(v: float, amp_deg: float, period: float) -> CommandFn:
    """Forward / reverse manoeuvre with fixed steering."""
    amp = math.radians(amp_deg)

    def cmd(t: float):
        # Reverse every half period
        forward = (t % period) < (period / 2.0)
        return (v if forward else -v), amp
    return cmd


def build_trajectory(spec: dict) -> CommandFn:
    """Factory used by ``run_simulation.py``."""
    t = spec["type"]
    v = float(spec.get("v_nominal", 1.0))
    a = float(spec.get("steering_amp_deg", 25.0))
    p = float(spec.get("steering_period_s", 8.0))
    return {
        "snake": snake,
        "figure_eight": figure_eight,
        "slalom": slalom,
        "parking": parking,
    }[t](v, a, p)
