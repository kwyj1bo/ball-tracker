"""End-to-end evaluation on rendered deliveries with exact ground truth.

For each seeded delivery: render the clip, detect and track the ball, calibrate the camera
from clicked landmarks (with click noise), and fit the physics model in two ways:
- full track: every tracked frame, to score the 3D reconstruction;
- to impact: only frames before the ball reaches the batter's pad (x = 17.5 m), then
  predict where it would have crossed the stumps line, as a review system must.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from balltracker.calibrate_scene import clicked_calibration
from balltracker.detect import BallDetector
from balltracker.fit import FitConfig, TrajectoryFitter
from balltracker.physics import simulate, time_to_plane
from balltracker.pitch import STUMPS_X, hits_stumps
from balltracker.predict import predict
from balltracker.render import Renderer
from balltracker.scenario import random_delivery, random_setup
from balltracker.track import BallTracker

IMPACT_X = 17.5


@dataclass(frozen=True)
class EvalConfig:
    fps: float = 240.0
    click_noise_px: float = 1.5  # 0 means use the true camera
    fixed: tuple[tuple[str, float], ...] = ()  # model ablations, e.g. (("drag", 0.0),)
    name: str = "main"


@dataclass
class DeliveryResult:
    seed: int
    ball: str
    speed_kmh: float
    visible_frames: int
    detected_frames: int
    track_points: int
    track_error_px: float
    calibration_rms_px: float
    status: str = "ok"  # ok, no_track, short_track, fit_failed
    fit_rms_px: float = np.nan
    rmse_3d_cm: float = np.nan
    speed_error_kmh: float = np.nan
    bounce_error_cm: float = np.nan
    drag_error: float = np.nan
    restitution_error: float = np.nan
    stumps_error_cm: float = np.nan
    hitting_true: bool = False
    hitting_predicted: bool = False
    visible_fraction: float = np.nan
    notes: list[str] = field(default_factory=list)


def evaluate_delivery(seed: int, config: EvalConfig) -> DeliveryResult:
    rng = np.random.default_rng(seed)
    delivery = random_delivery(rng)
    setup = random_setup(rng, fps=config.fps)
    renderer = Renderer(setup, delivery, seed)
    truth = renderer.truth
    candidates = BallDetector().run(renderer.frame, len(renderer))
    track = BallTracker().run(candidates)
    visible = np.flatnonzero(truth.visible)
    detected = sum(_near(candidates[k], truth.pixels[k]) for k in visible)
    camera, cal_rms = _camera(setup.camera, config.click_noise_px, rng)
    result = DeliveryResult(
        seed,
        setup.ball_name,
        delivery.speed_kmh,
        len(visible),
        detected,
        0 if track is None else len(track),
        np.nan,
        cal_rms,
    )
    if track is None or len(track) < 20:
        result.status = "no_track" if track is None else "short_track"
        return result
    result.track_error_px = float(
        np.median(np.linalg.norm(track.pixels - truth.pixels[track.frames], axis=1))
    )
    times = truth.times[track.frames] + setup.shutter_s / 2
    fitter = TrajectoryFitter(camera, FitConfig(fixed=dict(config.fixed) or None))
    try:
        _score_full(result, fitter, track, times, truth, delivery)
        _score_impact(result, fitter, track, times, truth, delivery)
    except ValueError as error:
        result.status = "fit_failed"
        result.notes.append(str(error))
    return result


def evaluate_many(seeds, config: EvalConfig, workers: int = 8) -> list[DeliveryResult]:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(evaluate_delivery, seeds, [config] * len(seeds)))


def _score_full(result, fitter, track, times, truth, delivery) -> None:
    t0 = times[0]
    fit = fitter.fit(times - t0, track.pixels)
    result.fit_rms_px = fit.rms_px
    true_positions = truth.centres[track.frames]
    fitted = simulate(fit.delivery.vector(), times - t0).positions[0]
    result.rmse_3d_cm = float(np.sqrt(np.mean(np.sum((fitted - true_positions) ** 2, 1))) * 100)
    true_velocity = _velocity(delivery, t0)
    result.speed_error_kmh = fit.delivery.speed_kmh - float(np.linalg.norm(true_velocity) * 3.6)
    true_bounce = simulate(delivery.vector(), [2.0]).bounce_point[0]
    if np.isfinite(fit.bounce_point).all() and np.isfinite(true_bounce).all():
        result.bounce_error_cm = float(np.linalg.norm(fit.bounce_point[:2] - true_bounce[:2]) * 100)
    result.drag_error = fit.delivery.drag - delivery.drag
    result.restitution_error = fit.delivery.restitution - delivery.restitution


def _score_impact(result, fitter, track, times, truth, delivery) -> None:
    before = truth.centres[track.frames][:, 0] < IMPACT_X
    if before.sum() < 20:
        result.status = "short_track"
        return
    t0 = times[before][0]
    fit = fitter.fit(times[before] - t0, track.pixels[before])
    predicted = predict(fit.delivery)
    true_cross = time_to_plane(delivery.vector(), STUMPS_X)[0]
    result.hitting_true = bool(hits_stumps(*true_cross[2:]))
    result.hitting_predicted = predicted.hitting
    result.stumps_error_cm = float(np.linalg.norm(predicted.stumps_yz - true_cross[2:]) * 100)
    result.visible_fraction = float(
        before.sum() / max(1, (truth.centres[:, 0] < IMPACT_X)[truth.visible].sum())
    )


def _camera(true_camera, click_noise_px, rng):
    if click_noise_px <= 0:
        return true_camera, 0.0
    calibration = clicked_calibration(true_camera, click_noise_px, rng)
    return calibration.camera, calibration.rms_px


def _velocity(delivery, t) -> np.ndarray:
    a, b = simulate(delivery.vector(), [t, t + 1e-4]).positions[0]
    return (b - a) / 1e-4


def _near(candidates, pixel, radius=3.0) -> bool:
    return any(np.hypot(c.u - pixel[0], c.v - pixel[1]) <= radius for c in candidates)


def as_dicts(results: list[DeliveryResult]) -> list[dict]:
    return [asdict(r) for r in results]


def with_name(config: EvalConfig, name: str, **changes) -> EvalConfig:
    return replace(config, name=name, **changes)
