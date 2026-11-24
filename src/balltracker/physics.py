"""Ball flight model, simulated for many parameter sets at once.

Forces: gravity, quadratic air drag (-k |v| v) and a constant sideways swing acceleration
before the bounce. Bounce: vertical speed reverses and shrinks by the restitution e, the
along-ground speed keeps a fraction h, and the seam can add sideways speed.

Fitting needs hundreds of simulations (multi-start search, finite-difference Jacobians),
so the integrator runs over a batch of parameter rows with NumPy instead of one Python
call per trajectory.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

from balltracker.pitch import BALL_RADIUS

G = 9.81
DT = 0.001  # integration step, seconds

PARAM_NAMES = ("x", "y", "z", "vx", "vy", "vz", "drag", "swing", "restitution", "retention", "seam")


@dataclass
class Delivery:
    """State at t = 0 plus the ball and pitch parameters."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    drag: float = 0.007  # k in 1/m; 0.5 * rho * Cd * area / mass for a cricket ball
    swing: float = 0.0  # sideways acceleration before the bounce, m/s^2
    restitution: float = 0.55
    retention: float = 0.88  # share of along-ground speed kept through the bounce
    seam: float = 0.0  # sideways speed added at the bounce, m/s

    def vector(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_vector(cls, values) -> Delivery:
        return cls(*map(float, values))

    @property
    def speed_kmh(self) -> float:
        return float(np.linalg.norm([self.vx, self.vy, self.vz]) * 3.6)


@dataclass
class Flight:
    positions: np.ndarray  # (n, T, 3) at the requested times
    bounce_time: np.ndarray  # (n,), NaN if no bounce before the last time
    bounce_point: np.ndarray  # (n, 3)


def simulate(params: np.ndarray, times: np.ndarray) -> Flight:
    """Positions at `times` (seconds, ascending, >= 0) for each parameter row."""
    params = np.atleast_2d(np.asarray(params, float))
    times = np.asarray(times, float)
    n = len(params)
    state = params[:, :6].copy()
    drag, swing = params[:, 6], params[:, 7]
    restitution, retention, seam = params[:, 8], params[:, 9], params[:, 10]
    bounced = state[:, 2] <= BALL_RADIUS
    bounce_time = np.full(n, np.nan)
    bounce_point = np.full((n, 3), np.nan)
    out = np.empty((n, len(times), 3))
    t, k = 0.0, 0
    while k < len(times) and times[k] <= 0:
        out[:, k] = state[:, :3]
        k += 1
    steps = int(np.ceil((times[-1] if len(times) else 0) / DT))
    for _ in range(steps):
        new = _rk4(state, DT, drag, np.where(bounced, 0.0, swing))
        crossing = ~bounced & (new[:, 2] < BALL_RADIUS)
        if crossing.any():
            new[crossing], bounce_time[crossing], bounce_point[crossing] = _bounce(
                state[crossing],
                new[crossing],
                t,
                drag[crossing],
                swing[crossing],
                restitution[crossing],
                retention[crossing],
                seam[crossing],
            )
            bounced |= crossing
        t_next = t + DT
        while k < len(times) and times[k] <= t_next + 1e-12:
            f = (times[k] - t) / DT
            out[:, k] = state[:, :3] + f * (new[:, :3] - state[:, :3])
            k += 1
        state, t = new, t_next
    return Flight(out, bounce_time, bounce_point)


def time_to_plane(params: np.ndarray, x_plane: float, horizon: float = 3.0) -> np.ndarray:
    """(n, 4): time and position where each trajectory first reaches x = x_plane (NaN if it
    never does within the horizon)."""
    times = np.arange(0, horizon, DT)
    flight = simulate(params, times)
    x = flight.positions[:, :, 0]
    out = np.full((len(x), 4), np.nan)
    for i in range(len(x)):
        idx = np.flatnonzero(x[i] >= x_plane)
        if len(idx) and idx[0] > 0:
            j = idx[0]
            f = (x_plane - x[i, j - 1]) / (x[i, j] - x[i, j - 1])
            out[i, 0] = times[j - 1] + f * DT
            out[i, 1:] = flight.positions[i, j - 1] + f * (
                flight.positions[i, j] - flight.positions[i, j - 1]
            )
    return out


def _acceleration(v: np.ndarray, drag: np.ndarray, swing: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    a = -drag[:, None] * speed * v
    a[:, 1] += swing
    a[:, 2] -= G
    return a


def _rk4(state: np.ndarray, dt, drag, swing) -> np.ndarray:
    """One RK4 step; dt is a scalar or one step length per row."""
    dt = np.broadcast_to(np.asarray(dt, float), (len(state),))[:, None]

    def deriv(s):
        return np.concatenate([s[:, 3:], _acceleration(s[:, 3:], drag, swing)], axis=1)

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _bounce(before, after, t, drag, swing, restitution, retention, seam):
    """Find the contact inside the step, apply the bounce there, and finish the step."""
    f = (before[:, 2] - BALL_RADIUS) / (before[:, 2] - after[:, 2])
    contact = _rk4(before, f * DT, drag, swing)
    # One Newton correction on the contact time: the straight-line guess above is off by
    # O(dt^2) because the path curves under gravity inside the step.
    f -= (contact[:, 2] - BALL_RADIUS) / (contact[:, 5] * DT)
    contact = _rk4(before, f * DT, drag, swing)
    contact[:, 2] = BALL_RADIUS
    contact[:, 3] *= retention
    contact[:, 4] = contact[:, 4] * retention + seam
    contact[:, 5] = -restitution * contact[:, 5]
    rest = _rk4(contact, (1 - f) * DT, drag, np.zeros_like(swing))
    return rest, t + f * DT, contact[:, :3].copy()
