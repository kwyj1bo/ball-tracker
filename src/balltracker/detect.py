"""Ball candidates per frame, from background subtraction.

The camera is on a tripod, so the per-pixel median over frames spread through the clip is
an empty scene: the ball never stays on one pixel long enough to show up in it. Anything
that differs from that background is a candidate; size, shape and contrast filters keep
the ones that could be a ball. The bowler and batter are too big to pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    u: float
    v: float
    area: int
    contrast: float  # mean difference from the background, grey levels


@dataclass
class DetectorConfig:
    background_frames: int = 25
    min_threshold: float = 18.0  # grey levels
    noise_sigmas: float = 5.0  # threshold = max(min_threshold, noise_sigmas * noise sd)
    min_area: int = 2
    max_area: int = 400
    max_elongation: float = 8.0  # motion blur stretches the ball into a short streak


class BallDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def run(self, frame_at: Callable[[int], np.ndarray], count: int) -> list[list[Candidate]]:
        """frame_at(k) returns frame k; frames are fetched twice rather than all held."""
        background = self.background(frame_at, count)
        threshold = self._threshold(frame_at(count // 2), background)
        return [self.candidates(frame_at(k), background, threshold) for k in range(count)]

    def background(self, frame_at, count: int) -> np.ndarray:
        picks = np.linspace(0, count - 1, min(self.config.background_frames, count)).astype(int)
        return np.median(np.stack([frame_at(k) for k in picks]), axis=0).astype(np.int16)

    def candidates(self, frame, background, threshold) -> list[Candidate]:
        diff = np.abs(frame.astype(np.int16) - background).max(axis=2).astype(np.float32)
        mask = (diff > threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        found = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not self.config.min_area <= area <= self.config.max_area:
                continue
            if max(w, h) / max(min(w, h), 1) > self.config.max_elongation:
                continue
            ys, xs = np.nonzero(labels[y : y + h, x : x + w] == i)
            weights = diff[y + ys, x + xs]
            found.append(Candidate(
                float(x + 0.5 + np.average(xs, weights=weights)),
                float(y + 0.5 + np.average(ys, weights=weights)),
                int(area), float(weights.mean()),
            ))  # fmt: skip
        return found

    def _threshold(self, frame, background) -> float:
        """Robust noise level from the median absolute difference over the whole frame."""
        diff = np.abs(frame.astype(np.int16) - background).max(axis=2)
        sigma = 1.4826 * float(np.median(diff))
        return max(self.config.min_threshold, self.config.noise_sigmas * sigma)
