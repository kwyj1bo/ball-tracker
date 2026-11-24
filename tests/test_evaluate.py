import numpy as np

from balltracker.calibrate_scene import clicked_calibration
from balltracker.evaluate import EvalConfig, evaluate_delivery
from balltracker.scenario import random_setup


def test_one_delivery_end_to_end():
    result = evaluate_delivery(11, EvalConfig(fps=120))
    assert result.status == "ok"
    assert result.detected_frames >= 0.8 * result.visible_frames
    assert result.fit_rms_px < 2.0
    assert result.stumps_error_cm < 20
    assert abs(result.speed_error_kmh) < 15


def test_clicked_calibration_uses_only_landmarks_in_view():
    rng = np.random.default_rng(0)
    setup = random_setup(rng)
    calibration = clicked_calibration(setup.camera, 1.0, rng)
    assert 6 <= calibration.points <= 12
    assert calibration.rms_px < 2.5
