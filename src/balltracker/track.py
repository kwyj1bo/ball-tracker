"""Link per-frame candidates into one ball track.

The ball is the only small object that moves smoothly across many frames. Tracks are
grown from every pair of candidates in nearby frames, predicting each next position from
the last three points (constant acceleration in the image) and accepting the nearest
candidate inside a gate that widens with every missed frame. The longest, smoothest track
wins. A final pass drops points that sit far from a local quadratic through their
neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from balltracker.detect import Candidate


@dataclass
class Track:
    frames: np.ndarray  # frame indices
    pixels: np.ndarray  # (n, 2)

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class TrackerConfig:
    max_step_px: float = 40.0  # largest plausible move between consecutive frames
    gate_px: float = 4.0
    gate_growth_px: float = 3.0  # added per missed frame
    max_missed: int = 6
    min_length: int = 8
    outlier_px: float = 3.0


class BallTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

    def run(self, candidates: list[list[Candidate]]) -> Track | None:
        points = [np.array([[c.u, c.v] for c in frame]).reshape(-1, 2) for frame in candidates]
        best: list[tuple[int, np.ndarray]] = []
        for k, a, b, gap in self._seeds(points):
            if any(f == k and np.array_equal(p, a) for f, p in best):
                continue  # already part of the best track: growing it again finds nothing new
            track = self._grow([(k, a), (k + gap, b)], points)
            if len(track) > len(best):
                best = track
        if len(best) < self.config.min_length:
            return None
        frames = np.array([f for f, _ in best])
        pixels = np.array([p for _, p in best])
        keep = self._inliers(frames, pixels)
        return Track(frames[keep], pixels[keep])

    def _seeds(self, points):
        """Every pair of candidates one or two frames apart that is close enough."""
        for k in range(len(points) - 1):
            for gap in (1, 2):
                if k + gap >= len(points):
                    continue
                for a in points[k]:
                    for b in points[k + gap]:
                        if np.linalg.norm(b - a) <= self.config.max_step_px * gap:
                            yield k, a, b, gap

    def _grow(self, track, points):
        missed = 0
        k = track[-1][0]
        while k + 1 < len(points) and missed <= self.config.max_missed:
            k += 1
            guess = _predict(track, k)
            gate = self.config.gate_px + self.config.gate_growth_px * missed
            options = points[k]
            if len(options):
                distance = np.linalg.norm(options - guess, axis=1)
                j = int(np.argmin(distance))
                if distance[j] <= gate or self._bridges_kink(track, k, options[j]):
                    track.append((k, options[j]))
                    missed = 0
                    continue
            missed += 1
        return track

    def _bridges_kink(self, track, k: int, candidate: np.ndarray) -> bool:
        """At the bounce the image path turns sharply and the motion model misses. Accept
        a candidate anyway if it is no further from the last point than twice the recent
        speed allows: a different object would have to appear right on the path."""
        (f1, p1), (f2, p2) = track[-2], track[-1]
        speed = np.linalg.norm(p2 - p1) / (f2 - f1)
        reach = 2.0 * speed * (k - f2) + self.config.gate_px
        return np.linalg.norm(candidate - p2) <= min(reach, self.config.max_step_px * (k - f2))

    def _inliers(self, frames, pixels) -> np.ndarray:
        """Drop points far from a quadratic fitted to their 7 nearest neighbours in time."""
        keep = np.ones(len(frames), bool)
        for i in range(len(frames)):
            lo, hi = max(0, i - 3), min(len(frames), i + 4)
            others = [j for j in range(lo, hi) if j != i]
            if len(others) < 4:
                continue
            t = frames[others].astype(float)
            fit = [np.polyval(np.polyfit(t, pixels[others, d], 2), frames[i]) for d in (0, 1)]
            keep[i] = np.linalg.norm(pixels[i] - fit) <= self.config.outlier_px
        return keep


def _predict(track, k: int) -> np.ndarray:
    """Next position from the last points: constant acceleration if we have three."""
    frames = np.array([f for f, _ in track[-3:]], float)
    pixels = np.array([p for _, p in track[-3:]])
    degree = min(len(frames) - 1, 2)
    return np.array([np.polyval(np.polyfit(frames, pixels[:, d], degree), k) for d in (0, 1)])
