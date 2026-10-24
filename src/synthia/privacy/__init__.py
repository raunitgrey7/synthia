"""Differential privacy: mechanisms, the RDP accountant and the shared configuration object.

Every DP generator in Synthia uses the same conventions so that reported epsilons are comparable:

* Neighbouring datasets differ by adding or removing one record (unbounded DP).
* Noise is Gaussian. A mechanism with L2 sensitivity ``s`` adds ``N(0, (sigma * s)^2)`` where ``sigma`` is the
  noise multiplier shared by every query the generator makes.
* Privacy loss is tracked in Renyi DP (Mironov 2017) and converted to ``(epsilon, delta)`` at the end.
  Poisson-subsampled queries (DP-SGD) use the sampled Gaussian mechanism bound of Mironov, Talwar and Zhang (2019).
* The noise multiplier is calibrated by bisection so the composed epsilon is at or below the target. The reported
  ``epsilon_spent`` is the accountant's value for the multiplier actually used, never the target itself.
"""

from synthia.privacy.accounting import (
    DEFAULT_ORDERS,
    Accountant,
    calibrate_noise_multiplier,
    rdp_gaussian,
    rdp_sampled_gaussian,
    rdp_to_epsilon,
)
from synthia.privacy.mechanisms import (
    DPConfig,
    PrivacyReport,
    clip_rows_l2,
    gaussian_mechanism,
    noisy_histogram,
)

__all__ = [
    "DEFAULT_ORDERS",
    "Accountant",
    "DPConfig",
    "PrivacyReport",
    "calibrate_noise_multiplier",
    "clip_rows_l2",
    "gaussian_mechanism",
    "noisy_histogram",
    "rdp_gaussian",
    "rdp_sampled_gaussian",
    "rdp_to_epsilon",
]
