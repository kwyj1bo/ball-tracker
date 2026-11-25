"""Cricket pitch geometry in metres.

World frame: origin on the ground at the middle stump of the bowler's end. x runs down the
pitch towards the batter, y points to the bowler's left, z points up. The batter's stumps
stand at x = STUMPS_X.
"""

from __future__ import annotations

import numpy as np

STUMPS_X = 20.12  # 22 yards, stumps to stumps
CREASE_TO_STUMPS = 1.22  # popping crease in front of each set of stumps
PITCH_HALF_WIDTH = 1.52
RETURN_CREASE_Y = 1.32  # return creases, each side of the middle stump
STUMPS_HALF_WIDTH = 0.1143  # three stumps and the gaps, 22.86 cm across
STUMPS_HEIGHT = 0.711
BALL_RADIUS = 0.036  # 72 mm diameter
BALL_MASS = 0.156


def landmarks() -> dict[str, np.ndarray]:
    """Named points that are easy to click in a frame: stump bases and tops, and the
    corners where popping and return creases meet, at both ends."""
    points = {}
    for end, x_stumps, sign in (("bowler", 0.0, 1), ("batter", STUMPS_X, -1)):
        x_crease = x_stumps + sign * CREASE_TO_STUMPS
        for side, y in (("off", -STUMPS_HALF_WIDTH), ("leg", STUMPS_HALF_WIDTH)):
            points[f"{end}_stump_{side}_base"] = np.array([x_stumps, y, 0.0])
            points[f"{end}_stump_{side}_top"] = np.array([x_stumps, y, STUMPS_HEIGHT])
        for side, y in (("left", RETURN_CREASE_Y), ("right", -RETURN_CREASE_Y)):
            points[f"{end}_crease_{side}"] = np.array([x_crease, y, 0.0])
    return points


def crease_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    """White lines painted on the pitch, as 3D segments on the ground."""
    lines = []
    for x_stumps, sign in ((0.0, 1), (STUMPS_X, -1)):
        x_pop = x_stumps + sign * CREASE_TO_STUMPS
        lines.append(
            (np.array([x_pop, -PITCH_HALF_WIDTH, 0]), np.array([x_pop, PITCH_HALF_WIDTH, 0]))
        )
        lines.append(
            (np.array([x_stumps, -RETURN_CREASE_Y, 0]), np.array([x_stumps, RETURN_CREASE_Y, 0]))
        )
        for y in (-RETURN_CREASE_Y, RETURN_CREASE_Y):
            lines.append(
                (np.array([x_stumps - sign * 1.0, y, 0]), np.array([x_pop + sign * 1.0, y, 0]))
            )
    return lines


def hits_stumps(y: float, z: float) -> bool:
    """Whether a ball centre crossing the stumps plane at (y, z) would hit the wicket."""
    return bool(abs(y) <= STUMPS_HALF_WIDTH + BALL_RADIUS and z <= STUMPS_HEIGHT + BALL_RADIUS)
