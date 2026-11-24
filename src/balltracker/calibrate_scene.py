"""Calibrate the camera pose from pitch landmarks clicked in a frame.

In real use a person clicks the landmarks (stump bases and tops, crease corners) in one
frame. For evaluation the clicks are the true projections plus Gaussian noise, so the
error a real user would make carries through to every downstream number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from balltracker.camera import Calibration, Camera, calibrate_pose
from balltracker.pitch import landmarks


def clicked_calibration(
    true_camera: Camera, noise_px: float, rng: np.random.Generator
) -> Calibration:
    names = [n for n, p in landmarks().items() if true_camera.in_view(p[None, :])[0]]
    world = np.array([landmarks()[n] for n in names])
    clicks = true_camera.project(world) + rng.normal(0, noise_px, (len(world), 2))
    return calibrate_pose(world, clicks, true_camera.K, true_camera.dist, true_camera.size)


def calibration_from_file(
    path: str | Path, K: np.ndarray, dist: np.ndarray, size: tuple[int, int]
) -> Calibration:
    """Landmark clicks saved as {"landmark name": [u, v], ...}; names from pitch.landmarks()."""
    clicks = json.loads(Path(path).read_text())
    known = landmarks()
    unknown = sorted(set(clicks) - set(known))
    if unknown:
        raise ValueError(f"unknown landmarks {unknown}; known: {sorted(known)}")
    world = np.array([known[n] for n in clicks])
    return calibrate_pose(world, np.array(list(clicks.values()), float), K, dist, size)
