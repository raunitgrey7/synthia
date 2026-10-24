"""Fidelity: per-column distances and pairwise association deltas between real and synthetic tables.

Numeric columns are compared with the two-sample Kolmogorov-Smirnov statistic, categorical columns with total
variation distance; both live in [0, 1] with 0 meaning identical distributions. Dependence is compared through an
association matrix that handles mixed types: Pearson correlation for numeric pairs, Cramer's V for categorical
pairs and the correlation ratio (eta) for mixed pairs. The correlation delta is the mean absolute difference of
the two matrices over the pairs; the maximum is also reported because a single lost dependency can matter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from synthia.evaluation.bootstrap import Estimate, bootstrap
from synthia.schema import ColumnSpec, TableSchema


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance computed from the pooled sorted sample."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / len(a)
    cdf_b = np.searchsorted(b, pooled, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def total_variation(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, categories: tuple[str, ...]) -> float:
    pa = _category_probs(a, categories)
    pb = _category_probs(b, categories)
    return float(0.5 * np.abs(pa - pb).sum())


def _category_probs(values: pd.Series | np.ndarray, categories: tuple[str, ...]) -> np.ndarray:
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    probs = np.array([counts.get(c, 0) for c in categories], dtype=np.float64)
    return probs / max(probs.sum(), 1.0)


def cramers_v_from_table(table: np.ndarray) -> float:
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    codes_a, ka = _as_codes(a)
    codes_b, kb = _as_codes(b)
    return cramers_v_from_table(np.bincount(codes_a * kb + codes_b, minlength=ka * kb).reshape(ka, kb).astype(float))


def correlation_ratio_from_codes(codes: np.ndarray, k: int, values: np.ndarray) -> float:
    """eta: square root of the share of numeric variance explained by the category means."""
    values = np.asarray(values, dtype=np.float64)
    total = ((values - values.mean()) ** 2).sum()
    if total <= 0:
        return 0.0
    counts = np.bincount(codes, minlength=k).astype(float)
    sums = np.bincount(codes, weights=values, minlength=k)
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), values.mean())
    between = (counts * (means - values.mean()) ** 2).sum()
    return float(np.sqrt(min(max(between / total, 0.0), 1.0)))


def correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    codes, k = _as_codes(categories)
    return correlation_ratio_from_codes(codes, k, values)


def _as_codes(values: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(np.asarray(values).astype(str)), sort=True)
    return codes.astype(np.int64), max(len(uniques), 1)


def _encode(df: pd.DataFrame, schema: TableSchema) -> tuple[np.ndarray, list[int]]:
    """Numeric columns as floats and categoricals as integer codes in one (n, d) float matrix."""
    out = np.empty((len(df), len(schema)), dtype=np.float64)
    cards = []
    for j, spec in enumerate(schema.columns):
        if spec.is_numeric:
            out[:, j] = df[spec.name].to_numpy(dtype=np.float64)
            cards.append(0)
        else:
            index = {c: i for i, c in enumerate(spec.categories)}
            out[:, j] = [index[str(v)] for v in df[spec.name].tolist()]
            cards.append(len(spec.categories))
    return out, cards


def _pair_association(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
    if kx == 0 and ky == 0:
        if x.std() <= 0 or y.std() <= 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    if kx > 0 and ky > 0:
        table = np.bincount(x.astype(np.int64) * ky + y.astype(np.int64), minlength=kx * ky).reshape(kx, ky)
        return cramers_v_from_table(table.astype(float))
    if kx > 0:
        return correlation_ratio_from_codes(x.astype(np.int64), kx, y)
    return correlation_ratio_from_codes(y.astype(np.int64), ky, x)


def association_matrix(df: pd.DataFrame, schema: TableSchema) -> np.ndarray:
    """Symmetric matrix of pairwise associations in schema order, with ones on the diagonal."""
    data, cards = _encode(df, schema)
    d = len(schema)
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = _pair_association(data[:, i], data[:, j], cards[i], cards[j])
    return out


def correlation_delta(real: pd.DataFrame, synthetic: pd.DataFrame, schema: TableSchema) -> tuple[float, float]:
    """(mean, max) absolute difference between the two association matrices over the column pairs."""
    diff = np.abs(association_matrix(real, schema) - association_matrix(synthetic, schema))
    upper = diff[np.triu_indices(len(schema), k=1)]
    if len(upper) == 0:
        return 0.0, 0.0
    return float(upper.mean()), float(upper.max())


@dataclass
class FidelityResult:
    per_column: dict[str, Estimate]
    column_metric: dict[str, str]
    mean_ks: Estimate
    mean_tvd: Estimate
    correlation_delta_mean: Estimate
    correlation_delta_max: Estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_column": {k: {"metric": self.column_metric[k], **v.to_dict()} for k, v in self.per_column.items()},
            "mean_ks": self.mean_ks.to_dict(),
            "mean_tvd": self.mean_tvd.to_dict(),
            "correlation_delta_mean": self.correlation_delta_mean.to_dict(),
            "correlation_delta_max": self.correlation_delta_max.to_dict(),
        }


def _column_distance(spec: ColumnSpec) -> Callable[[pd.Series, pd.Series], float]:
    if spec.is_numeric:
        return lambda a, b: ks_statistic(a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64))
    return lambda a, b: total_variation(a, b, spec.categories)


def evaluate_fidelity(
    real: pd.DataFrame, synthetic: pd.DataFrame, schema: TableSchema, n_boot: int = 200, seed: int = 0
) -> FidelityResult:
    """Per-column distances, their means by column type and the association deltas, each with a bootstrap CI."""
    per_column: dict[str, Estimate] = {}
    metric: dict[str, str] = {}
    for k, spec in enumerate(schema.columns):
        fn = _column_distance(spec)
        per_column[spec.name] = bootstrap(
            fn, real[spec.name], synthetic[spec.name], n_boot=n_boot, seed=seed + k, floor=0.0, ceiling=1.0
        )
        metric[spec.name] = "ks" if spec.is_numeric else "tvd"
    numeric, categorical = schema.numeric_names, schema.categorical_names

    def mean_ks(r: pd.DataFrame, s: pd.DataFrame) -> float:
        if not numeric:
            return 0.0
        return float(np.mean([_column_distance(schema.get(c))(r[c], s[c]) for c in numeric]))

    def mean_tvd(r: pd.DataFrame, s: pd.DataFrame) -> float:
        if not categorical:
            return 0.0
        return float(np.mean([_column_distance(schema.get(c))(r[c], s[c]) for c in categorical]))

    corr_boot = min(n_boot, 100)  # association matrices are the expensive part
    return FidelityResult(
        per_column=per_column,
        column_metric=metric,
        mean_ks=bootstrap(mean_ks, real, synthetic, n_boot=n_boot, seed=seed + 101, floor=0.0, ceiling=1.0),
        mean_tvd=bootstrap(mean_tvd, real, synthetic, n_boot=n_boot, seed=seed + 102, floor=0.0, ceiling=1.0),
        correlation_delta_mean=bootstrap(
            lambda r, s: correlation_delta(r, s, schema)[0],
            real,
            synthetic,
            n_boot=corr_boot,
            seed=seed + 103,
            floor=0.0,
        ),
        correlation_delta_max=bootstrap(
            lambda r, s: correlation_delta(r, s, schema)[1],
            real,
            synthetic,
            n_boot=corr_boot,
            seed=seed + 104,
            floor=0.0,
        ),
    )
