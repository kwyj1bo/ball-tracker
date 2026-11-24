"""Fit the physics model to a single-camera ball track (system identification).

Unknowns: the ball's 3D position and velocity at the first tracked frame, plus drag, swing,
restitution, along-ground speed retention and seam movement. One camera cannot see depth
directly, but the physics constrains it: gravity fixes the vertical scale over time, and
the bounce must happen on the pitch plane at the moment the image path turns.

Cost: reprojection error in pixels (robust soft-L1 loss, so a few bad detections do not
drag the fit), plus weak Gaussian priors on the ball parameters so that the fit stays
physical when a clip barely constrains them. Starts come from a batched search over
plausible depths and speeds; the best few are refined with trust-region least squares,
using a Jacobian computed in one batched simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from balltracker.camera import Camera
from balltracker.physics import PARAM_NAMES, Delivery, simulate

# parameter -> (prior mean, prior sd); position and velocity have no prior
PRIORS = {
    "drag": (0.007, 0.003),
    "swing": (0.0, 1.5),
    "restitution": (0.55, 0.15),
    "retention": (0.88, 0.08),
    "seam": (0.0, 0.6),
}
LOWER = np.array([-5, -3, 0.0, 5, -8, -25, 0.0, -5, 0.1, 0.4, -3])
UPPER = np.array([25, 3, 3.5, 50, 8, 15, 0.03, 5, 0.95, 1.0, 3])


@dataclass
class FitConfig:
    pixel_sigma: float = 1.0  # expected detection error, px
    loss_scale_px: float = 2.0  # soft-L1 switches from squared to linear beyond this
    starts: int = 4  # refined starts
    fixed: dict[str, float] | None = None  # e.g. {"drag": 0.0} for the no-drag ablation


@dataclass
class FitResult:
    delivery: Delivery  # state at t = 0, the first observation
    rms_px: float
    variance: np.ndarray | None  # per parameter, from the Gauss-Newton approximation
    bounce_time: float
    bounce_point: np.ndarray
    converged: bool

    def sd(self, name: str) -> float:
        """Standard deviation of a fitted parameter (0 for a fixed one)."""
        if self.variance is None:
            return float("nan")
        return float(np.sqrt(max(self.variance[PARAM_NAMES.index(name)], 0.0)))


class TrajectoryFitter:
    def __init__(self, camera: Camera, config: FitConfig | None = None) -> None:
        self.camera = camera
        self.config = config or FitConfig()
        fixed = self.config.fixed or {}
        self.free = np.array([name not in fixed for name in PARAM_NAMES])
        self.fixed_values = np.array([fixed.get(name, 0.0) for name in PARAM_NAMES])

    def fit(self, times: np.ndarray, pixels: np.ndarray) -> FitResult:
        """times: seconds since the first observation (first entry 0); pixels: (n, 2)."""
        times, pixels = np.asarray(times, float), np.asarray(pixels, float)
        starts = self._initial_guesses(times, pixels)
        best = None
        for start in starts[: self.config.starts]:
            result = least_squares(
                self._residuals,
                start[self.free],
                jac=self._jacobian,
                args=(times, pixels),
                bounds=(LOWER[self.free], UPPER[self.free]),
                loss="soft_l1",
                f_scale=self.config.loss_scale_px / self.config.pixel_sigma,
                max_nfev=200,
            )
            if best is None or result.cost < best.cost:
                best = result
        return self._result(best, times, pixels)

    # --- cost -------------------------------------------------------------------------

    def _full(self, free_values: np.ndarray) -> np.ndarray:
        full = np.broadcast_to(self.fixed_values, free_values.shape[:-1] + (len(PARAM_NAMES),))
        full = full.copy()
        full[..., self.free] = free_values
        return full

    def _pixel_errors(self, params: np.ndarray, times, pixels) -> np.ndarray:
        """(rows, 2n) projected minus observed pixels, scaled by the pixel sigma."""
        positions = simulate(params, times).positions
        projected = self.camera.project(positions)
        behind = self.camera.depth(positions) <= 0.1
        projected[behind] = 1e4  # far off screen: a start that puts the ball behind the camera
        return ((projected - pixels) / self.config.pixel_sigma).reshape(len(params), -1)

    def _prior_errors(self, params: np.ndarray) -> np.ndarray:
        columns = [
            (params[:, PARAM_NAMES.index(n)] - m) / sd
            for n, (m, sd) in PRIORS.items()
            if self.free[PARAM_NAMES.index(n)]
        ]
        return np.stack(columns, axis=1) if columns else np.zeros((len(params), 0))

    def _residuals(self, free_values, times, pixels) -> np.ndarray:
        params = self._full(free_values[None, :])
        return np.concatenate(
            [self._pixel_errors(params, times, pixels)[0], self._prior_errors(params)[0]]
        )

    def _jacobian(self, free_values, times, pixels) -> np.ndarray:
        """Central differences, all perturbed parameter sets simulated in one batch."""
        steps = np.maximum(np.abs(free_values), 1.0) * 1e-5
        eye = np.diag(steps)
        batch = self._full(np.concatenate([free_values + eye, free_values - eye]))
        errors = np.concatenate(
            [self._pixel_errors(batch, times, pixels), self._prior_errors(batch)], axis=1
        )
        n = len(free_values)
        return ((errors[:n] - errors[n:]) / (2 * steps[:, None])).T

    # --- starts -----------------------------------------------------------------------

    def _initial_guesses(self, times, pixels) -> np.ndarray:
        """Batched search: the first ball position somewhere along its camera ray, aimed at
        a point along the last ray, with the speed that covers the distance in the observed
        time. Returns candidate full parameter vectors, best first."""
        rays = self.camera.rays(pixels[[0, -1]])
        origin = self.camera.position
        duration = max(times[-1] - times[0], 1e-3)
        candidates = []
        for start_x in np.linspace(-0.5, 16.0, 12):
            p0 = _point_on_ray_at_x(origin, rays[0], start_x)
            if p0 is None or not 0.0 < p0[2] < 3.2:
                continue
            for end_x in np.linspace(max(start_x + 2, 8.0), 24.0, 6):
                p1 = _point_on_ray_at_x(origin, rays[1], end_x)
                if p1 is None or not -0.5 < p1[2] < 4.0:
                    continue
                for vz in (-8.0, -4.0, -1.0, 2.0):
                    velocity = (p1 - p0) / duration
                    candidates.append(
                        [
                            *p0,
                            velocity[0],
                            velocity[1],
                            vz,
                            *(PRIORS[n][0] for n in PARAM_NAMES[6:]),
                        ]
                    )
        if not candidates:
            raise ValueError("no plausible starting point for the fit")
        params = np.array(candidates)
        params[:, ~self.free] = self.fixed_values[~self.free]
        params = np.clip(params, LOWER + 1e-9, UPPER - 1e-9)
        errors = self._pixel_errors(params, times, pixels)
        cost = np.sum(np.minimum(errors**2, 100.0), axis=1)  # cap: one bad point is not fatal
        return params[np.argsort(cost)]

    def _result(self, result, times, pixels) -> FitResult:
        params = self._full(result.x[None, :])
        flight = simulate(params, np.concatenate([times, [times[-1] + 2.0]]))
        pixel_errors = self._pixel_errors(params, times, pixels)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((pixel_errors * self.config.pixel_sigma) ** 2, 1))))
        variance = None
        try:
            variance = np.zeros(len(PARAM_NAMES))
            variance[self.free] = np.diag(np.linalg.inv(result.jac.T @ result.jac))
        except np.linalg.LinAlgError:
            variance = None
        return FitResult(
            Delivery.from_vector(params[0]),
            rms,
            variance,
            float(flight.bounce_time[0]),
            flight.bounce_point[0],
            bool(result.success),
        )


def _point_on_ray_at_x(origin, ray, x) -> np.ndarray | None:
    if abs(ray[0]) < 1e-6:
        return None
    s = (x - origin[0]) / ray[0]
    return origin + s * ray if s > 0 else None
