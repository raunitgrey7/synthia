"""Bias-corrected percentile bootstrap for metrics computed from one or more row sets.

Resampling rows with replacement biases non-negative distance statistics upward: two resamples of the same
table have a positive Kolmogorov-Smirnov distance, so the plain percentile interval of such a statistic can sit
entirely above the point estimate. The interval here is shifted by the bootstrap bias (mean of the replicates
minus the point estimate) and clipped to the statistic's valid range, which recentres it on the point estimate
while keeping the resampling spread. Unbiased statistics such as means and AUCs are left practically unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

RowSet = np.ndarray | pd.DataFrame | pd.Series


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its bootstrap percentile interval."""

    value: float
    low: float
    high: float
    n_boot: int
    level: float
    method: str = "bias-corrected percentile bootstrap"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _clean(self.value),
            "ci_low": _clean(self.low),
            "ci_high": _clean(self.high),
            "n_boot": self.n_boot,
            "level": self.level,
            "method": self.method,
        }

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _clean(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _take(rows: RowSet, idx: np.ndarray) -> RowSet:
    if isinstance(rows, pd.DataFrame | pd.Series):
        return rows.iloc[idx].reset_index(drop=True)
    return rows[idx]


def bootstrap(
    statistic: Callable[..., float],
    *row_sets: RowSet,
    n_boot: int = 200,
    seed: int = 0,
    level: float = 0.95,
    paired: bool = False,
    floor: float | None = None,
    ceiling: float | None = None,
) -> Estimate:
    """Resample each row set with replacement and recompute ``statistic``; ``paired`` shares one index set.

    The point estimate is the statistic on the full data. With ``n_boot`` of zero the interval collapses to it.
    ``floor`` and ``ceiling`` clip the interval to the statistic's valid range after the bias shift.
    """
    if not row_sets:
        raise ValueError("bootstrap needs at least one row set")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    lengths = [len(r) for r in row_sets]
    if paired and len(set(lengths)) != 1:
        raise ValueError("paired bootstrap needs row sets of equal length")
    value = float(statistic(*row_sets))
    if n_boot <= 0:
        return Estimate(value, value, value, 0, level)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        if paired:
            idx = rng.integers(lengths[0], size=lengths[0])
            samples = [_take(r, idx) for r in row_sets]
        else:
            samples = [_take(r, rng.integers(n, size=n)) for r, n in zip(row_sets, lengths, strict=True)]
        draws[b] = statistic(*samples)
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return Estimate(value, float("nan"), float("nan"), n_boot, level)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    bias = float(finite.mean()) - value
    low, high = _clip(low - bias, floor, ceiling), _clip(high - bias, floor, ceiling)
    return Estimate(value, min(low, value), max(high, value), n_boot, level)


def _clip(x: float, floor: float | None, ceiling: float | None) -> float:
    if floor is not None:
        x = max(x, floor)
    if ceiling is not None:
        x = min(x, ceiling)
    return float(x)
