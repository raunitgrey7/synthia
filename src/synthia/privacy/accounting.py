"""Renyi differential privacy accounting for the Gaussian and the Poisson-subsampled Gaussian mechanism.

The accountant keeps a vector of RDP guarantees, one per order ``alpha``, adds the guarantee of every query the
generator makes, and converts the total to ``(epsilon, delta)`` with the standard bound
``epsilon = min_alpha rdp(alpha) + log(1/delta) / (alpha - 1)``. The subsampled Gaussian computation follows the
exact integer-order expansion and the fractional-order series of Mironov, Talwar and Zhang (2019), which is the
same computation TensorFlow Privacy and Opacus use, so the numbers can be cross-checked against them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import special

DEFAULT_ORDERS: tuple[float, ...] = tuple([1 + x / 10.0 for x in range(1, 100)] + [float(a) for a in range(12, 64)])


def rdp_gaussian(noise_multiplier: float, orders: Sequence[float]) -> np.ndarray:
    """RDP of the Gaussian mechanism with sensitivity 1 and noise std ``noise_multiplier``: alpha / (2 sigma^2)."""
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    alphas = np.asarray(orders, dtype=np.float64)
    return alphas / (2.0 * noise_multiplier**2)


def rdp_sampled_gaussian(sampling_rate: float, noise_multiplier: float, orders: Sequence[float]) -> np.ndarray:
    """RDP of one step of the Poisson-subsampled Gaussian mechanism (Mironov, Talwar and Zhang 2019)."""
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError("sampling_rate must lie in [0, 1]")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if sampling_rate == 0.0:
        return np.zeros(len(orders))
    if sampling_rate == 1.0:
        return rdp_gaussian(noise_multiplier, orders)
    return np.array([_rdp_one_order(sampling_rate, noise_multiplier, float(a)) for a in orders])


def _rdp_one_order(q: float, sigma: float, alpha: float) -> float:
    if alpha <= 1.0:
        raise ValueError("RDP orders must be greater than 1")
    if math.isinf(alpha):
        return math.inf
    log_a = _log_a_int(q, sigma, int(alpha)) if float(alpha).is_integer() else _log_a_frac(q, sigma, alpha)
    return log_a / (alpha - 1.0)


def _log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log A(alpha) for integer alpha via the binomial expansion of the moment of the mixture ratio."""
    terms = []
    for i in range(alpha + 1):
        log_coef = math.lgamma(alpha + 1) - math.lgamma(i + 1) - math.lgamma(alpha - i + 1)
        log_coef += i * math.log(q) + (alpha - i) * math.log1p(-q)
        terms.append(log_coef + (i * i - i) / (2.0 * sigma**2))
    return float(special.logsumexp(terms))


def _log_erfc(x: float) -> float:
    return math.log(2.0) + float(special.log_ndtr(-x * math.sqrt(2.0)))


def _log_sub(a: float, b: float) -> float:
    """log(exp(a) - exp(b)) for a >= b."""
    if b == -math.inf:
        return a
    if a == b:
        return -math.inf
    if a < b:
        raise ValueError("log_sub needs a >= b")
    return a + math.log1p(-math.exp(b - a))


def _log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """log A(alpha) for fractional alpha: the two-sided series with generalised binomial coefficients."""
    log_a0, log_a1 = -math.inf, -math.inf
    z0 = sigma**2 * math.log(1.0 / q - 1.0) + 0.5
    i = 0
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log1p(-q)
        log_t1 = log_coef + j * math.log(q) + i * math.log1p(-q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2.0) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2.0) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2.0 * sigma**2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2.0 * sigma**2) + log_e1
        if coef > 0:
            log_a0 = float(np.logaddexp(log_a0, log_s0))
            log_a1 = float(np.logaddexp(log_a1, log_s1))
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30.0:
            break
    return float(np.logaddexp(log_a0, log_a1))


def rdp_to_epsilon(rdp: Sequence[float], orders: Sequence[float], delta: float) -> tuple[float, float]:
    """Convert accumulated RDP to the smallest epsilon over the orders; returns (epsilon, best order)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1")
    rdp_arr = np.asarray(rdp, dtype=np.float64)
    alphas = np.asarray(orders, dtype=np.float64)
    if rdp_arr.shape != alphas.shape:
        raise ValueError("rdp and orders must have the same length")
    eps = rdp_arr + math.log(1.0 / delta) / (alphas - 1.0)
    eps = np.where(np.isfinite(eps), eps, np.inf)
    idx = int(np.argmin(eps))
    return float(eps[idx]), float(alphas[idx])


@dataclass
class Accountant:
    """Accumulates RDP across queries; ``epsilon(delta)`` reports the composed guarantee."""

    orders: tuple[float, ...] = DEFAULT_ORDERS
    rdp: np.ndarray = field(init=False)
    queries: list[dict[str, float | str | int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rdp = np.zeros(len(self.orders), dtype=np.float64)

    def add_gaussian(self, noise_multiplier: float, count: int = 1, label: str = "gaussian") -> None:
        """Compose ``count`` Gaussian mechanism queries, each with noise ``noise_multiplier`` times its sensitivity."""
        if count < 1:
            raise ValueError("count must be at least 1")
        self.rdp = self.rdp + count * rdp_gaussian(noise_multiplier, self.orders)
        self.queries.append({"mechanism": label, "noise_multiplier": noise_multiplier, "count": count})

    def add_sampled_gaussian(
        self, sampling_rate: float, noise_multiplier: float, steps: int, label: str = "dp-sgd"
    ) -> None:
        """Compose ``steps`` Poisson-subsampled Gaussian queries (one per optimiser step)."""
        if steps < 1:
            raise ValueError("steps must be at least 1")
        self.rdp = self.rdp + steps * rdp_sampled_gaussian(sampling_rate, noise_multiplier, self.orders)
        self.queries.append(
            {
                "mechanism": label,
                "noise_multiplier": noise_multiplier,
                "sampling_rate": sampling_rate,
                "steps": steps,
            }
        )

    def epsilon(self, delta: float) -> float:
        return rdp_to_epsilon(self.rdp, self.orders, delta)[0]

    def best_order(self, delta: float) -> float:
        return rdp_to_epsilon(self.rdp, self.orders, delta)[1]


def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    build: Callable[[float], Accountant],
    lower: float = 0.05,
    upper: float = 1e5,
    tolerance: float = 1e-4,
) -> float:
    """Bisect the noise multiplier so that ``build(sigma).epsilon(delta) <= target_epsilon``.

    Epsilon decreases in sigma, so the search brackets the smallest multiplier that meets the target.
    """
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if build(upper).epsilon(delta) > target_epsilon:
        raise ValueError("target epsilon is unreachable even with the largest supported noise")
    if build(lower).epsilon(delta) <= target_epsilon:
        return lower
    lo, hi = lower, upper
    while hi / lo > 1.0 + tolerance:
        mid = math.sqrt(lo * hi)
        if build(mid).epsilon(delta) <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return hi
