"""Real video in, results out: load a clip, run the tracker, fit, and draw an overlay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from balltracker.camera import Camera
from balltracker.detect import BallDetector
from balltracker.fit import FitResult, TrajectoryFitter
from balltracker.physics import simulate
from balltracker.predict import Prediction, predict
from balltracker.track import BallTracker, Track


@dataclass
class VideoResult:
    track: Track
    fit: FitResult
    prediction: Prediction
    fps: float


def load_frames(path: str | Path, start: int = 0, stop: int | None = None) -> tuple[list, float]:
    """Frames [start, stop) of a video, and its frame rate. Keep clips short: one
    delivery is under a second."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while stop is None or start + len(frames) < stop:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError("no frames read")
    return frames, fps


def analyse(frames: list[np.ndarray], fps: float, camera: Camera) -> VideoResult:
    candidates = BallDetector().run(lambda k: frames[k], len(frames))
    track = BallTracker().run(candidates)
    if track is None:
        raise ValueError("no ball track found")
    times = (track.frames - track.frames[0]) / fps
    fit = TrajectoryFitter(camera).fit(times, track.pixels)
    return VideoResult(track, fit, predict(fit.delivery), fps)


def overlay(frame: np.ndarray, camera: Camera, result: VideoResult) -> np.ndarray:
    """Tracked points in yellow, fitted path (and its continuation) in magenta."""
    image = frame.copy()
    duration = (result.track.frames[-1] - result.track.frames[0]) / result.fps + 0.3
    path = simulate(result.fit.delivery.vector(), np.linspace(0, duration, 200)).positions[0]
    pixels = camera.project(path[camera.depth(path) > 0])
    cv2.polylines(image, [np.round(pixels).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    for u, v in result.track.pixels:
        cv2.circle(image, (int(round(u)), int(round(v))), 2, (0, 255, 255), -1, cv2.LINE_AA)
    return image


def to_json(result: VideoResult) -> dict:
    p = result.prediction
    return {
        "tracked_frames": result.track.frames.tolist(),
        "fit_rms_px": result.fit.rms_px,
        "fitted": vars(result.fit.delivery),
        "speed_kmh": p.speed_kmh,
        "bounce_point_m": p.bounce_point.tolist(),
        "stumps_crossing_yz_m": p.stumps_yz.tolist(),
        "hitting": p.hitting,
    }


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text())
    return np.array(data["K"], float), np.array(data.get("dist", [0, 0, 0, 0, 0]), float)
