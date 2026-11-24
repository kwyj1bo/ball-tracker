"""Pinhole camera with lens distortion, and calibration from known points.

Conventions follow OpenCV: camera x to the right, y down, z forward. R and t map world
points into the camera frame: X_cam = R @ X_world + t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Camera:
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation
    size: tuple[int, int]  # width, height in pixels
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # k1 k2 p1 p2 k3

    @classmethod
    def looking_at(cls, position, target, size=(1280, 720), hfov_deg=40.0, dist=None) -> Camera:
        """Camera at `position` aimed at `target`, with world z as up."""
        position, target = np.asarray(position, float), np.asarray(target, float)
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        w, h = size
        f = w / 2 / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        d = np.zeros(5) if dist is None else np.asarray(dist, float)
        return cls(K, R, -R @ position, size, d)

    @property
    def position(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, float) @ self.R.T + self.t

    def project(self, points: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> (..., 2) pixels, with distortion. Vectorized."""
        cam = self.to_camera(points)
        x, y = cam[..., 0] / cam[..., 2], cam[..., 1] / cam[..., 2]
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        u = self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        return np.stack([u, v], axis=-1)

    def depth(self, points: np.ndarray) -> np.ndarray:
        return self.to_camera(points)[..., 2]

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """(n, 2) pixels -> (n, 3) unit ray directions in the world frame."""
        pts = np.asarray(pixels, float).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        directions = np.column_stack([norm, np.ones(len(norm))]) @ self.R
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def ground_point(self, pixel, height: float = 0.0) -> np.ndarray:
        """Where the ray through `pixel` meets the horizontal plane z = height."""
        ray = self.rays(np.atleast_2d(pixel))[0]
        s = (height - self.position[2]) / ray[2]
        return self.position + s * ray

    def in_view(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        uv = self.project(points)
        w, h = self.size
        return (
            (self.depth(points) > 0)
            & (uv[..., 0] >= -margin)
            & (uv[..., 0] < w + margin)
            & (uv[..., 1] >= -margin)
            & (uv[..., 1] < h + margin)
        )


@dataclass
class Calibration:
    camera: Camera
    rms_px: float  # reprojection error over the points used
    points: int


def calibrate_pose(
    world: np.ndarray, pixels: np.ndarray, K: np.ndarray, dist: np.ndarray, size: tuple[int, int]
) -> Calibration:
    """Camera pose from at least 4 known world points and their clicked pixels.

    SQPnP gives a global solution without an initial guess; Levenberg-Marquardt then
    polishes it on the same points.
    """
    world = np.asarray(world, np.float64).reshape(-1, 1, 3)
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(world, pixels, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("pose estimation failed")
    rvec, tvec = cv2.solvePnPRefineLM(world, pixels, K, dist, rvec, tvec)
    camera = Camera(K.copy(), cv2.Rodrigues(rvec)[0], tvec.ravel(), size, np.asarray(dist, float))
    error = camera.project(world.reshape(-1, 3)) - pixels.reshape(-1, 2)
    return Calibration(camera, float(np.sqrt((error**2).sum(axis=1).mean())), len(world))


def calibrate_intrinsics(
    images: list[np.ndarray],
    pattern: tuple[int, int],
    square_m: float,
    flags: int = cv2.CALIB_FIX_K3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """K, distortion and RMS reprojection error from checkerboard photos.
    pattern is the count of inner corners (columns, rows). k3 is fixed at zero by default:
    phone lenses rarely need it, and with few views it trades off against the focal
    length instead of describing the lens."""
    grid = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    object_points, image_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for image in images:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            object_points.append(grid)
            image_points.append(corners)
    if len(image_points) < 3:
        raise ValueError(f"checkerboard found in {len(image_points)} images; need at least 3")
    h, w = images[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, (w, h), None, None, flags=flags
    )
    return K, dist.ravel()[:5], float(rms)
