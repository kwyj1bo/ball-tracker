"""From a fitted delivery: how fast it was, where it bounced, and where it would have
crossed the stumps line."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from balltracker.physics import Delivery, simulate, time_to_plane
from balltracker.pitch import STUMPS_X, hits_stumps


@dataclass
class Prediction:
    speed_kmh: float  # at the first observation
    bounce_point: np.ndarray  # (3,), NaN for a full toss that reaches the stumps first
    stumps_yz: np.ndarray  # (2,) where the ball centre crosses the stumps plane
    hitting: bool


def predict(delivery: Delivery) -> Prediction:
    crossing = time_to_plane(delivery.vector(), STUMPS_X)[0]
    reaches = bool(np.isfinite(crossing).all())
    horizon = crossing[0] if reaches else 2.0
    flight = simulate(delivery.vector(), np.array([horizon]))
    bounce = flight.bounce_point[0]
    yz = crossing[2:]
    hitting = reaches and bool(hits_stumps(yz[0], yz[1]))
    return Prediction(delivery.speed_kmh, bounce, yz, hitting)
