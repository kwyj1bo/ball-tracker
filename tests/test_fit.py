import numpy as np
import pytest

from balltracker.fit import FitConfig, TrajectoryFitter
from balltracker.physics import Delivery, simulate, time_to_plane
from balltracker.pitch import STUMPS_X
from balltracker.predict import predict
from balltracker.scenario import random_delivery, random_setup


def observations(seed, cut_x=17.5, noise=0.7):
    rng = np.random.default_rng(seed)
    delivery = random_delivery(rng)
    setup = random_setup(rng)
    times = np.arange(0.02, 0.9, 1 / 240)
    positions = simulate(delivery.vector(), times).positions[0]
    keep = positions[:, 0] < cut_x
    pixels = setup.camera.project(positions[keep]) + rng.normal(0, noise, (keep.sum(), 2))
    return delivery, setup.camera, times[keep], pixels


@pytest.mark.parametrize("seed", [0, 4])
def test_fit_predicts_the_stumps_crossing(seed):
    delivery, camera, times, pixels = observations(seed)
    fit = TrajectoryFitter(camera).fit(times - times[0], pixels)
    assert fit.rms_px < 1.5
    true_yz = time_to_plane(delivery.vector(), STUMPS_X)[0, 2:]
    assert np.linalg.norm(predict(fit.delivery).stumps_yz - true_yz) < 0.10


def test_fixed_parameters_stay_fixed():
    delivery, camera, times, pixels = observations(1)
    fit = TrajectoryFitter(camera, FitConfig(fixed={"drag": 0.0, "swing": 0.0})).fit(
        times - times[0], pixels
    )
    assert fit.delivery.drag == 0.0 and fit.delivery.swing == 0.0


def test_predict_on_a_straight_full_delivery():
    straight = Delivery(1.0, 0.0, 2.2, 30.0, 0.0, -3.5, drag=0.0)
    prediction = predict(straight)
    assert abs(prediction.stumps_yz[0]) < 1e-9
    assert prediction.bounce_point[0] < STUMPS_X
    assert prediction.speed_kmh == pytest.approx(np.hypot(30, 3.5) * 3.6)
