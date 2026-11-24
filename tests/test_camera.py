import cv2
import numpy as np
import pytest

from balltracker.camera import Camera, calibrate_intrinsics, calibrate_pose
from balltracker.pitch import landmarks

BEHIND_BOWLER = dict(position=[-12.0, 0.4, 3.2], target=[12.0, 0.0, 0.5])


def test_projection_matches_opencv_with_distortion():
    cam = Camera.looking_at(**BEHIND_BOWLER, dist=[-0.12, 0.05, 0.001, -0.002, 0.0])
    pts = np.random.default_rng(0).uniform([0, -1.5, 0], [20, 1.5, 2.5], (50, 3))
    rvec = cv2.Rodrigues(cam.R)[0]
    expected = cv2.projectPoints(pts, rvec, cam.t, cam.K, cam.dist)[0].reshape(-1, 2)
    np.testing.assert_allclose(cam.project(pts), expected, atol=1e-6)


def test_rays_invert_projection():
    cam = Camera.looking_at(**BEHIND_BOWLER, dist=[-0.1, 0.02, 0, 0, 0])
    ground = np.array([[5.0, 0.3, 0.0], [15.0, -0.8, 0.0], [19.0, 1.2, 0.0]])
    back = np.array([cam.ground_point(p) for p in cam.project(ground)])
    np.testing.assert_allclose(back, ground, atol=1e-6)


def test_pose_from_clicked_landmarks():
    truth = Camera.looking_at(**BEHIND_BOWLER)
    names = sorted(landmarks())
    world = np.array([landmarks()[n] for n in names])
    clicks = truth.project(world) + np.random.default_rng(1).normal(0, 1.0, (len(world), 2))
    cal = calibrate_pose(world, clicks, truth.K, truth.dist, truth.size)
    assert cal.points == 12
    assert cal.rms_px < 1.5
    assert np.linalg.norm(cal.camera.position - truth.position) < 0.3
    far = np.array([[19.0, 0.0, 0.5]])  # a point near the batter
    assert np.linalg.norm(cal.camera.project(far) - truth.project(far)) < 2.0


def _checkerboard(cam, pattern=(9, 6), square=0.03):
    """Render a checkerboard seen by `cam` at some pose by warping a flat image."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    px = 40
    board = np.kron((np.indices((rows, cols)).sum(0) % 2) * 255, np.ones((px, px))).astype(np.uint8)
    board = cv2.copyMakeBorder(board, px, px, px, px, cv2.BORDER_CONSTANT, value=255)
    corners_px = np.array(
        [[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]],
        np.float32,
    )
    scale = square / px
    corners_m = np.column_stack([(corners_px - px) * scale, np.zeros(4)])
    image_pts = cam.project(corners_m).astype(np.float32)
    H = cv2.getPerspectiveTransform(corners_px, image_pts)
    w, h = cam.size
    return cv2.warpPerspective(board, H, (w, h), borderValue=128)


def test_intrinsics_from_rendered_checkerboards():
    rng = np.random.default_rng(3)
    base = Camera.looking_at([0, 0, -0.5], [0.12, 0.08, 0], size=(960, 540), hfov_deg=60)
    images = []
    for _ in range(8):
        target = np.array([0.135, 0.09, 0]) + rng.uniform(-0.03, 0.03, 3) * [1, 1, 0]
        position = target + rng.uniform([-0.25, -0.25, -0.6], [0.25, 0.25, -0.4])
        cam = Camera.looking_at(position, target, size=(960, 540), hfov_deg=60)
        cam.K = base.K
        images.append(_checkerboard(cam))
    K, dist, rms = calibrate_intrinsics(images, (9, 6), 0.03)
    assert rms < 0.5
    assert K[0, 0] == pytest.approx(base.K[0, 0], rel=0.02)
    assert K[0, 2] == pytest.approx(480, abs=10)
