import math

import numpy as np
import pytest

from synthia.privacy import (
    DEFAULT_ORDERS,
    Accountant,
    DPConfig,
    calibrate_noise_multiplier,
    clip_rows_l2,
    gaussian_mechanism,
    noisy_histogram,
    rdp_gaussian,
    rdp_sampled_gaussian,
    rdp_to_epsilon,
)


def test_gaussian_rdp_closed_form():
    orders = (2.0, 8.0, 32.0)
    np.testing.assert_allclose(rdp_gaussian(2.0, orders), np.array(orders) / 8.0)


def test_sampled_gaussian_reduces_to_gaussian_at_full_sampling():
    orders = (1.5, 2.0, 10.0)
    np.testing.assert_allclose(rdp_sampled_gaussian(1.0, 1.3, orders), rdp_gaussian(1.3, orders))
    assert np.all(rdp_sampled_gaussian(0.0, 1.3, orders) == 0)


def test_subsampling_amplifies_privacy():
    orders = DEFAULT_ORDERS
    full = rdp_gaussian(1.1, orders)
    sampled = rdp_sampled_gaussian(0.01, 1.1, orders)
    assert np.all(sampled <= full + 1e-12)
    assert sampled[0] < 0.01 * full[0]


@pytest.mark.parametrize(
    ("q", "sigma", "steps", "expected", "order"),
    [
        (250 / 60000, 1.3, 3600, 1.18, 17.0),
        (256 / 60000, 1.1, 14062, 3.01, 8.8),
    ],
)
def test_matches_published_tensorflow_privacy_values(q, sigma, steps, expected, order):
    """Reference points printed by TensorFlow Privacy's compute_dp_sgd_privacy for its MNIST tutorials."""
    acc = Accountant()
    acc.add_sampled_gaussian(q, sigma, steps)
    assert abs(acc.epsilon(1e-5) - expected) < 0.01
    assert acc.best_order(1e-5) == order


def test_epsilon_grows_with_steps_and_shrinks_with_noise():
    def eps(sigma, steps):
        acc = Accountant()
        acc.add_sampled_gaussian(0.05, sigma, steps)
        return acc.epsilon(1e-5)

    assert eps(1.0, 100) < eps(1.0, 1000)
    assert eps(2.0, 1000) < eps(1.0, 1000)


def test_composition_adds_rdp_linearly():
    a = Accountant()
    a.add_gaussian(3.0, count=4)
    b = Accountant()
    for _ in range(4):
        b.add_gaussian(3.0)
    np.testing.assert_allclose(a.rdp, b.rdp)
    assert len(a.queries) == 1 and len(b.queries) == 4


def test_rdp_to_epsilon_picks_best_order_and_validates_delta():
    rdp = rdp_gaussian(1.0, DEFAULT_ORDERS)
    eps, order = rdp_to_epsilon(rdp, DEFAULT_ORDERS, 1e-5)
    manual = min(r + math.log(1e5) / (a - 1) for r, a in zip(rdp, DEFAULT_ORDERS, strict=True))
    assert abs(eps - manual) < 1e-12
    assert order in DEFAULT_ORDERS
    with pytest.raises(ValueError):
        rdp_to_epsilon(rdp, DEFAULT_ORDERS, 0.0)


def test_calibration_hits_target_from_above():
    def build(sigma):
        acc = Accountant()
        acc.add_gaussian(sigma, count=5)
        return acc

    sigma = calibrate_noise_multiplier(1.0, 1e-5, build)
    assert build(sigma).epsilon(1e-5) <= 1.0
    assert build(sigma / 1.01).epsilon(1e-5) > 1.0


def test_calibration_rejects_unreachable_targets():
    def build(sigma):
        acc = Accountant()
        acc.add_gaussian(sigma)
        return acc

    with pytest.raises(ValueError, match="unreachable"):
        calibrate_noise_multiplier(1e-9, 1e-5, build, upper=10.0)


def test_mechanism_noise_scale_and_histogram_clipping():
    rng = np.random.default_rng(0)
    noisy = gaussian_mechanism(np.zeros(200_000), sensitivity=2.0, noise_multiplier=1.5, rng=rng)
    assert abs(noisy.std() - 3.0) < 0.05
    hist = noisy_histogram(np.zeros(1000), noise_multiplier=5.0, rng=rng)
    assert (hist >= 0).all() and (hist > 0).any()


def test_clip_rows_l2_bounds_norms():
    rows = np.random.default_rng(1).normal(size=(50, 6)) * 10
    clipped = clip_rows_l2(rows, 2.0)
    assert np.all(np.linalg.norm(clipped, axis=1) <= 2.0 + 1e-9)
    small = np.full((3, 2), 0.1)
    np.testing.assert_allclose(clip_rows_l2(small, 5.0), small)


def test_dp_config_validation():
    with pytest.raises(ValueError):
        DPConfig(0.0)
    with pytest.raises(ValueError):
        DPConfig(1.0, delta=1.0)
    assert DPConfig(2.0).delta == 1e-5
