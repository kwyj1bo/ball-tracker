"""Random but plausible deliveries and camera setups, all from a seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from balltracker.camera import Camera
from balltracker.physics import Delivery, G


@dataclass
class Setup:
    camera: Camera
    fps: float = 240.0
    shutter_s: float = 1 / 1000
    ball_bgr: tuple[int, int, int] = (235, 235, 235)
    ball_name: str = "white"
    noise_sd: float = 3.0
    batter_y: float = 0.45  # where the batter stands, sideways from the middle stump


def random_delivery(rng: np.random.Generator) -> Delivery:
    """Release near the bowling crease, 95-145 km/h, pitching on a full to short length."""
    side = rng.choice([-1, 1])
    release = np.array(
        [rng.uniform(0.6, 1.2), side * rng.uniform(0.15, 0.65), rng.uniform(2.0, 2.35)]
    )
    bounce = np.array([rng.uniform(12.0, 18.0), rng.uniform(-0.5, 0.5)])
    speed = rng.uniform(26.0, 40.0)
    velocity = _aim(release, bounce, speed)
    return Delivery(*release, *velocity,
                    drag=rng.uniform(0.005, 0.009),
                    swing=rng.normal(0, 0.8),
                    restitution=rng.uniform(0.45, 0.65),
                    retention=rng.uniform(0.8, 0.95),
                    seam=rng.normal(0, 0.25))  # fmt: skip


def random_setup(rng: np.random.Generator, fps: float = 240.0) -> Setup:
    """A phone on a tripod behind the bowler's end, zoomed in down the pitch."""
    position = [rng.uniform(-16, -10), rng.uniform(-0.5, 0.5), rng.uniform(2.5, 4.5)]
    target = [rng.uniform(12, 16), 0.0, rng.uniform(0.0, 0.6)]
    camera = Camera.looking_at(position, target, (1280, 720), rng.uniform(30, 40))
    white = rng.random() < 0.7
    return Setup(
        camera, fps,
        ball_bgr=(235, 235, 235) if white else (40, 35, 165),
        ball_name="white" if white else "red",
        noise_sd=rng.uniform(2.0, 4.0),
        batter_y=rng.choice([-1, 1]) * rng.uniform(0.3, 0.6),
    )  # fmt: skip


def _aim(release: np.ndarray, bounce: np.ndarray, speed: float) -> np.ndarray:
    """Velocity that would land on `bounce` without drag, at roughly `speed`."""
    horizontal = bounce - release[:2]
    distance = np.linalg.norm(horizontal)
    t = distance / speed  # first guess: nearly flat
    for _ in range(20):
        vz = (0.036 - release[2] + 0.5 * G * t * t) / t
        t = distance / np.sqrt(max(speed**2 - vz**2, 1.0))
    v_h = horizontal / t
    return np.array([v_h[0], v_h[1], vz])
