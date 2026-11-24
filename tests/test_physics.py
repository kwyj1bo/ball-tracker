import numpy as np
import pytest
from scipy.integrate import solve_ivp

from balltracker.physics import Delivery, G, simulate, time_to_plane
from balltracker.pitch import BALL_RADIUS, STUMPS_X, hits_stumps

FAST = Delivery(x=1.0, y=0.2, z=2.2, vx=36.0, vy=-0.3, vz=-3.0)


def test_vacuum_flight_matches_the_parabola_and_bounce_time():
    d = Delivery(1.0, 0.0, 2.2, 30.0, 0.5, -2.0, drag=0.0)
    t = np.linspace(0, 0.3, 31)
    flight = simulate(d.vector(), t)
    expected_z = 2.2 - 2.0 * t - 0.5 * G * t**2
    np.testing.assert_allclose(flight.positions[0, :, 2], expected_z, atol=1e-9)
    np.testing.assert_allclose(flight.positions[0, :, 0], 1.0 + 30.0 * t, atol=1e-9)
    # z(t) = r  ->  0.5 g t^2 + 2 t - (2.2 - r) = 0
    t_bounce = (-2.0 + np.sqrt(4.0 + 2 * G * (2.2 - BALL_RADIUS))) / G
    flight = simulate(d.vector(), np.linspace(0, 1.0, 11))
    assert flight.bounce_time[0] == pytest.approx(t_bounce, abs=1e-9)


def test_bounce_scales_velocities():
    d = Delivery(1.0, 0.0, 2.2, 30.0, 0.0, -2.0, drag=0.0, restitution=0.5, retention=0.8, seam=0.4)
    # Outputs on integration step boundaries are exact for this polynomial motion, and a
    # finite difference over [0.600, 0.602] is the velocity at 0.601.
    flight = simulate(d.vector(), np.array([0.0, 0.6, 0.602]))
    tb = flight.bounce_time[0]
    vz_in = -2.0 - G * tb
    velocity = (flight.positions[0, 2] - flight.positions[0, 1]) / 0.002
    assert velocity[2] == pytest.approx(-0.5 * vz_in - G * (0.601 - tb), abs=1e-6)
    assert velocity[0] == pytest.approx(30.0 * 0.8, abs=1e-6)
    assert velocity[1] == pytest.approx(0.4, abs=1e-6)


def test_drag_and_swing_match_an_adaptive_reference_solver():
    d = Delivery(1.0, 0.0, 2.2, 38.0, 0.0, -1.0, drag=0.008, swing=1.2)

    def rhs(_, s):
        v = s[3:]
        a = -0.008 * np.linalg.norm(v) * v + [0, 1.2, -G]
        return np.concatenate([v, a])

    t = np.linspace(0, 0.25, 26)
    ref = solve_ivp(rhs, (0, 0.25), d.vector()[:6], t_eval=t, rtol=1e-11, atol=1e-12).y.T
    ours = simulate(d.vector(), t).positions[0]
    np.testing.assert_allclose(ours, ref[:, :3], atol=1e-7)


def test_batch_rows_are_independent():
    a = FAST.vector()
    b = Delivery(0.5, -0.3, 2.0, 28.0, 0.4, -1.0, swing=-1.0).vector()
    t = np.linspace(0, 0.8, 50)
    together = simulate(np.stack([a, b]), t).positions
    np.testing.assert_allclose(together[0], simulate(a, t).positions[0])
    np.testing.assert_allclose(together[1], simulate(b, t).positions[0])


def test_time_to_stumps_plane():
    hit = time_to_plane(FAST.vector(), STUMPS_X)[0]
    assert 0.5 < hit[0] < 0.7
    assert hit[1] == pytest.approx(STUMPS_X, abs=1e-9)
    assert hits_stumps(0.0, 0.5) and not hits_stumps(0.2, 0.5) and not hits_stumps(0.0, 0.8)
