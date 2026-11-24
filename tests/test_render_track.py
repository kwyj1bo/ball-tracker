import numpy as np
import pytest

from balltracker.detect import BallDetector
from balltracker.render import Renderer
from balltracker.scenario import random_delivery, random_setup
from balltracker.track import BallTracker


@pytest.fixture(scope="module")
def clip():
    rng = np.random.default_rng(10)
    delivery = random_delivery(rng)
    setup = random_setup(rng, fps=120)
    return Renderer(setup, delivery, seed=10)


def test_frames_are_reproducible_and_show_the_ball_at_its_true_pixel(clip):
    k = int(np.flatnonzero(clip.truth.visible)[10])
    frame = clip.frame(k)
    assert np.array_equal(frame, clip.frame(k))
    u, v = np.round(clip.truth.pixels[k]).astype(int)
    patch = frame[v - 1 : v + 2, u - 1 : u + 2].astype(int)
    background = clip.background[v - 1 : v + 2, u - 1 : u + 2].astype(int)
    assert np.abs(patch - background).max() > 40


def test_no_ball_before_release(clip):
    k = int(np.flatnonzero(clip.times < 0)[-1])
    assert not clip.truth.visible[k]


def test_detection_and_tracking_follow_the_true_ball(clip):
    candidates = BallDetector().run(clip.frame, len(clip))
    track = BallTracker().run(candidates)
    visible = np.flatnonzero(clip.truth.visible)
    assert len(track) >= 0.8 * len(visible)
    error = np.linalg.norm(track.pixels - clip.truth.pixels[track.frames], axis=1)
    assert np.median(error) < 1.5
    assert (error < 4).mean() > 0.95
