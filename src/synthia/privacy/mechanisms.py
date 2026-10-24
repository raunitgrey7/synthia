"""Noise-adding primitives and the configuration and report objects shared by the DP generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DPConfig:
    """Target privacy budget. ``delta`` should be well below one over the number of records."""

    epsilon: float
    delta: float = 1e-5

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie strictly between 0 and 1")


@dataclass
class PrivacyReport:
    """What a DP fit actually spent, with the caveats a reader needs to judge the number."""

    epsilon_target: float
    epsilon_spent: float
    delta: float
    noise_multiplier: float
    rdp_order: float
    adjacency: str = "add or remove one record"
    queries: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon_target": self.epsilon_target,
            "epsilon_spent": self.epsilon_spent,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "rdp_order": self.rdp_order,
            "adjacency": self.adjacency,
            "queries": list(self.queries),
            "caveats": list(self.caveats),
        }


def gaussian_mechanism(
    value: np.ndarray, sensitivity: float, noise_multiplier: float, rng: np.random.Generator
) -> np.ndarray:
    """Add ``N(0, (noise_multiplier * sensitivity)^2)`` to every entry of an L2-sensitivity-``sensitivity`` query."""
    if sensitivity <= 0 or noise_multiplier <= 0:
        raise ValueError("sensitivity and noise_multiplier must be positive")
    arr = np.asarray(value, dtype=np.float64)
    return arr + rng.normal(0.0, noise_multiplier * sensitivity, size=arr.shape)


def noisy_histogram(counts: np.ndarray, noise_multiplier: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mechanism on a histogram (one record changes one cell by one, so L2 sensitivity is 1); clipped at 0."""
    noisy = gaussian_mechanism(np.asarray(counts, dtype=np.float64), 1.0, noise_multiplier, rng)
    return np.clip(noisy, 0.0, None)


def clip_rows_l2(matrix: np.ndarray, bound: float) -> np.ndarray:
    """Scale every row to L2 norm at most ``bound``; this bounds one record's contribution to a sum."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scale = np.minimum(1.0, bound / np.maximum(norms, 1e-12))
    return matrix * scale
