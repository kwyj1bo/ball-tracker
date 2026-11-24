"""Synthetic video of a delivery, with exact ground truth for every frame.

The static scene (outfield, pitch, creases, stumps) is rendered once by casting a ray
through every pixel. Each frame then adds the bowler, the ball with motion blur over the
shutter time, the batter (who can hide the ball), and sensor noise. Frames are
reproducible one at a time, so long clips never need to sit in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from balltracker import pitch
from balltracker.physics import Delivery, simulate
from balltracker.scenario import Setup

BLUR_SAMPLES = 8
NOISE_BANK = 6


@dataclass
class FrameTruth:
    times: np.ndarray  # frame time, seconds since release
    centres: np.ndarray  # (n, 3) ball centre mid-exposure
    pixels: np.ndarray  # (n, 2) projected centre
    radius_px: np.ndarray
    visible: np.ndarray  # in view and not hidden by the bowler or the batter


class Renderer:
    def __init__(
        self, setup: Setup, delivery: Delivery, seed: int, pre_s: float = 0.12, post_s: float = 0.08
    ) -> None:
        self.setup = setup
        self.cam = setup.camera
        self.delivery = delivery
        self.seed = seed
        end = self._time_at_stumps() + post_s
        self.times = np.arange(-pre_s, end, 1 / setup.fps)
        self.background = render_background(self.cam)
        rng = np.random.default_rng([seed, 1])
        h, w = self.cam.size[1], self.cam.size[0]
        self.noise = rng.normal(0, setup.noise_sd, (NOISE_BANK, h, w, 1)).astype(np.float32)
        self.truth = self._truth()

    def __len__(self) -> int:
        return len(self.times)

    def frame(self, k: int) -> np.ndarray:
        t = self.times[k]
        image = self.background.astype(np.float32)
        batter, batter_depth = self._batter_polygon()
        ball = self.truth.centres[k]
        ball_behind_batter = self.cam.depth(ball) > batter_depth
        if not ball_behind_batter:
            cv2.fillPoly(image, [batter], (225, 225, 230))
        if t >= 0:
            self._draw_ball(image, t)
        if ball_behind_batter:
            cv2.fillPoly(image, [batter], (225, 225, 230))
        cv2.fillPoly(image, [self._bowler_polygon(t)], (40, 45, 60))
        rng = np.random.default_rng([self.seed, 2, k])
        image *= rng.uniform(0.98, 1.02)  # exposure flicker
        image += self.noise[k % NOISE_BANK]
        return np.clip(image, 0, 255).astype(np.uint8)

    def _time_at_stumps(self) -> float:
        times = np.arange(0, 2.0, 0.002)
        x = simulate(self.delivery.vector(), times).positions[0, :, 0]
        past = np.flatnonzero(x >= pitch.STUMPS_X)
        return float(times[past[0]]) if len(past) else 1.2

    def _truth(self) -> FrameTruth:
        t = self.times + self.setup.shutter_s / 2
        centres = simulate(self.delivery.vector(), np.clip(t, 0, None)).positions[0]
        pixels = self.cam.project(centres)
        radius = self.cam.focal * pitch.BALL_RADIUS / self.cam.depth(centres)
        visible = (t >= 0) & self.cam.in_view(centres, margin=-2)
        _, batter_depth = self._batter_polygon()
        for i in np.flatnonzero(visible):
            hidden_by_bowler = (
                cv2.pointPolygonTest(
                    self._bowler_polygon(self.times[i]), tuple(map(float, pixels[i])), True
                )
                > -radius[i]
            )
            behind_batter = (
                self.cam.depth(centres[i]) > batter_depth
                and cv2.pointPolygonTest(
                    self._batter_polygon()[0], tuple(map(float, pixels[i])), True
                )
                > -radius[i]
            )
            visible[i] = not (hidden_by_bowler or behind_batter)
        return FrameTruth(self.times, centres, pixels, radius, visible)

    def _draw_ball(self, image: np.ndarray, t: float) -> None:
        sub = t + np.linspace(0, self.setup.shutter_s, BLUR_SAMPLES)
        centres = simulate(self.delivery.vector(), np.clip(sub, 0, None)).positions[0]
        if not self.cam.in_view(centres, margin=20).any():
            return
        pixels = self.cam.project(centres)
        radius = self.cam.focal * pitch.BALL_RADIUS / self.cam.depth(centres)
        lo = np.floor(pixels.min(0) - radius.max() - 2).astype(int)
        hi = np.ceil(pixels.max(0) + radius.max() + 2).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, np.array(self.cam.size))
        if np.any(hi <= lo):
            return
        mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.float32)
        layer = np.zeros_like(mask, dtype=np.uint8)
        for (u, v), r in zip(pixels - lo, radius, strict=True):
            layer[:] = 0
            centre = (int(round(u * 16)), int(round(v * 16)))
            cv2.circle(layer, centre, max(int(round(r * 16)), 8), 255, -1, cv2.LINE_AA, shift=4)
            mask += layer.astype(np.float32) / (255.0 * BLUR_SAMPLES)
        roi = image[lo[1] : hi[1], lo[0] : hi[0]]
        colour = np.array(self.setup.ball_bgr, np.float32)
        roi[:] = roi * (1 - mask[..., None]) + colour * mask[..., None]

    def _bowler_polygon(self, t: float) -> np.ndarray:
        """The bowler as a box that runs in, releases at t = 0, and moves off to the side."""
        d = self.delivery
        shift = np.clip(t, -0.2, 0.25) * 4.0
        side = np.sign(d.y) if d.y else 1.0
        x0 = d.x - 0.5 + shift
        y0 = d.y + side * (0.15 + max(t, 0) * 3.0)
        box = _box(x0, x0 + 0.45, y0 - 0.25, y0 + 0.25, 0.0, 2.05)
        return _hull(self.cam.project(box))

    def _batter_polygon(self) -> tuple[np.ndarray, float]:
        y = self.setup.batter_y
        box = _box(pitch.STUMPS_X - 1.6, pitch.STUMPS_X - 1.2, y - 0.25, y + 0.25, 0.0, 1.7)
        return _hull(self.cam.project(box)), float(self.cam.depth(box).min())


def render_background(cam) -> np.ndarray:
    """Outfield, pitch strip, crease lines and stumps, seen by `cam`."""
    w, h = cam.size
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = cam.rays(np.column_stack([u.ravel(), v.ravel()]))
    s = -cam.position[2] / np.where(rays[:, 2] < -1e-6, rays[:, 2], -1e-6)
    ground = cam.position + s[:, None] * rays
    sky = (rays[:, 2] >= -1e-6) | (s > 400)
    x, y = ground[:, 0], ground[:, 1]
    rng = np.random.default_rng(0)
    grain = rng.normal(0, 4, len(x))
    stripes = 8 * (np.floor((x + 200) / 6) % 2)
    grass = np.column_stack([45 + stripes + grain, 120 + stripes + grain, 60 + grain])
    strip = (np.abs(y) < pitch.PITCH_HALF_WIDTH) & (x > -2.5) & (x < pitch.STUMPS_X + 2.5)
    soil = np.column_stack([110 + grain, 150 + grain, 175 + grain])
    image = np.where(strip[:, None], soil, grass)
    image[sky] = (170, 140, 110)
    image = image.reshape(h, w, 3).astype(np.float32)
    for a, b in pitch.crease_lines():
        _ground_band(image, cam, a, b, 0.05, (240, 240, 240))
    for x_stumps in (0.0, pitch.STUMPS_X):
        for y in (-0.1, 0.0, 0.1):
            box = _box(
                x_stumps - 0.02, x_stumps + 0.02, y - 0.019, y + 0.019, 0, pitch.STUMPS_HEIGHT
            )
            cv2.fillPoly(image, [_hull(cam.project(box))], (200, 225, 235))
    return np.clip(image, 0, 255).astype(np.uint8)


def _ground_band(image, cam, a, b, width, colour) -> None:
    direction = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-direction[1], direction[0], 0]) * width / 2
    quad = np.array([a - normal, b - normal, b + normal, a + normal])
    if (cam.depth(quad) > 0.5).all():
        cv2.fillPoly(
            image, [np.round(cam.project(quad) * 16).astype(np.int32)], colour, cv2.LINE_AA, shift=4
        )


def _box(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def _hull(points: np.ndarray) -> np.ndarray:
    return cv2.convexHull(np.round(points).astype(np.int32))
