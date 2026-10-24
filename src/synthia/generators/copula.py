"""Gaussian copula generator with an optional differentially private fit.

Each column gets a marginal (a piecewise-linear quantile function for numeric columns, a probability vector
for categorical ones) and every record is mapped to normal scores; the dependence is a correlation matrix of
those scores. Sampling draws from the multivariate normal and pushes the draws back through the marginals.

With DP the fit touches the data only through Gaussian-mechanism queries: one histogram per column on public
bins (sensitivity 1), one noisy row count (sensitivity 1) and one second-moment matrix of the row-clipped normal
scores (sensitivity ``d * clip_factor^2``). The statistics queries receive ``stats_noise_weight`` times more noise
than the moment query because counts are large while correlation entries are not; the accountant composes the
whole plan and the noise multiplier is calibrated to the requested epsilon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from synthia.generators.base import Generator
from synthia.privacy import Accountant, DPConfig, PrivacyReport, calibrate_noise_multiplier, clip_rows_l2
from synthia.privacy.mechanisms import gaussian_mechanism, noisy_histogram
from synthia.schema import ColumnKind, ColumnSpec, TableSchema

_EPS = 1e-9


class NumericMarginal:
    """Piecewise-linear quantile function through knots (u_k, x_k); the CDF is its inverse."""

    def __init__(self, knots_u: np.ndarray, knots_x: np.ndarray) -> None:
        self.knots_u = np.asarray(knots_u, dtype=np.float64)
        self.knots_x = np.asarray(knots_x, dtype=np.float64)

    @classmethod
    def from_sample(cls, values: np.ndarray, n_knots: int, spec: ColumnSpec) -> NumericMarginal:
        grid = np.linspace(0.0, 1.0, n_knots + 1)
        knots = np.quantile(values, grid)
        knots = np.clip(knots, spec.lower, spec.upper)
        return cls(grid, np.maximum.accumulate(knots))

    @classmethod
    def from_histogram(cls, masses: np.ndarray, edges: np.ndarray) -> NumericMarginal:
        total = float(masses.sum())
        probs = masses / total if total > 0 else np.full(len(masses), 1.0 / len(masses))
        cum = np.concatenate([[0.0], np.cumsum(probs)])
        cum[-1] = 1.0
        return cls(cum, edges)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.interp(u, self.knots_u, self.knots_x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Inverse of the quantile function; flat stretches map to the top of their plateau."""
        xs, us = self.knots_x, self.knots_u
        keep = np.concatenate([xs[1:] > xs[:-1], [True]])
        return np.interp(x, xs[keep], us[keep])


class CategoricalMarginal:
    """Probability vector over the public category domain; categories occupy consecutive uniform intervals."""

    def __init__(self, probs: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        total = probs.sum()
        self.probs = probs / total if total > 0 else np.full(len(probs), 1.0 / len(probs))
        self.cum = np.concatenate([[0.0], np.cumsum(self.probs)])
        self.cum[-1] = 1.0

    def scores(self, codes: np.ndarray) -> np.ndarray:
        """Expected standard normal score of a draw whose uniform lies in the category's interval."""
        a, b = self.cum[codes], self.cum[codes + 1]
        width = b - a
        za = stats.norm.ppf(np.clip(a, _EPS, 1 - _EPS))
        zb = stats.norm.ppf(np.clip(b, _EPS, 1 - _EPS))
        num = stats.norm.pdf(za) * (a > 0) - stats.norm.pdf(zb) * (b < 1)
        return np.where(width > _EPS, num / np.maximum(width, _EPS), 0.0)

    def decode(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.cum, u, side="right") - 1, 0, len(self.probs) - 1)


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues at ``floor`` and rescale to unit diagonal; cheap projection onto valid correlations."""
    sym = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(sym)
    rebuilt = (vectors * np.maximum(values, floor)) @ vectors.T
    d = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(d, d)


class GaussianCopulaGenerator(Generator):
    """Gaussian copula with empirical marginals; see the module docstring for the DP plan."""

    name = "copula"

    def __init__(
        self,
        schema: TableSchema,
        dp: DPConfig | None = None,
        seed: int = 0,
        n_knots: int = 64,
        dp_bins: int = 16,
        clip_factor: float = 1.5,
        stats_noise_weight: float = 4.0,
    ) -> None:
        super().__init__(schema, dp=dp, seed=seed)
        if n_knots < 2 or dp_bins < 2:
            raise ValueError("n_knots and dp_bins must be at least 2")
        if clip_factor <= 0 or stats_noise_weight < 1:
            raise ValueError("clip_factor must be positive and stats_noise_weight at least 1")
        self.n_knots = n_knots
        self.dp_bins = dp_bins
        self.clip_factor = clip_factor
        self.stats_noise_weight = stats_noise_weight
        self.marginals: dict[str, NumericMarginal | CategoricalMarginal] = {}
        self.correlation = np.eye(len(schema))

    def _fit(self, df: pd.DataFrame) -> None:
        if self.dp is None:
            scores = self._fit_marginals_exact(df)
            self.correlation = _exact_correlation(scores)
        else:
            self._fit_dp(df)

    def _fit_marginals_exact(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        scores = np.empty((n, len(self.schema)))
        for j, col in enumerate(self.schema.columns):
            if col.is_numeric:
                values = df[col.name].to_numpy(dtype=np.float64)
                self.marginals[col.name] = NumericMarginal.from_sample(values, self.n_knots, col)
                ranks = stats.rankdata(values, method="average")
                scores[:, j] = stats.norm.ppf((ranks - 0.5) / n)
            else:
                codes = _codes(df[col.name], col)
                counts = np.bincount(codes, minlength=len(col.categories)).astype(np.float64)
                marginal = CategoricalMarginal(counts)
                self.marginals[col.name] = marginal
                scores[:, j] = marginal.scores(codes)
        return scores

    def _accountant(self, sigma: float) -> Accountant:
        acc = Accountant()
        acc.add_gaussian(sigma * self.stats_noise_weight, count=len(self.schema) + 1, label="histograms and count")
        acc.add_gaussian(sigma, count=1, label="second moment of clipped normal scores")
        return acc

    def _fit_dp(self, df: pd.DataFrame) -> None:
        assert self.dp is not None
        sigma = calibrate_noise_multiplier(self.dp.epsilon, self.dp.delta, self._accountant)
        sigma_stats = sigma * self.stats_noise_weight
        rng = self._rng
        d = len(self.schema)
        scores = np.empty((len(df), d))
        for j, col in enumerate(self.schema.columns):
            scores[:, j] = self._dp_marginal(df[col.name], col, sigma_stats, rng)
        noisy_n = max(float(gaussian_mechanism(np.array(len(df), dtype=np.float64), 1.0, sigma_stats, rng)), 1.0)
        radius = self.clip_factor * np.sqrt(d)
        clipped = clip_rows_l2(scores, radius)
        moment = clipped.T @ clipped
        noise = gaussian_mechanism(np.zeros((d, d)), radius**2, sigma, rng)
        noise = np.triu(noise) + np.triu(noise, 1).T
        moment = (moment + noise) / noisy_n
        diag = np.sqrt(np.maximum(np.diag(moment), 1e-6))
        self.correlation = nearest_correlation(np.clip(moment / np.outer(diag, diag), -1.0, 1.0))
        acc = self._accountant(sigma)
        self.privacy_report = PrivacyReport(
            epsilon_target=self.dp.epsilon,
            epsilon_spent=acc.epsilon(self.dp.delta),
            delta=self.dp.delta,
            noise_multiplier=sigma,
            rdp_order=acc.best_order(self.dp.delta),
            queries=acc.queries,
            caveats=self._dp_caveats(),
        )

    def _dp_marginal(self, series: pd.Series, col: ColumnSpec, sigma: float, rng: np.random.Generator) -> np.ndarray:
        if col.is_numeric:
            values = series.to_numpy(dtype=np.float64)
            edges = np.linspace(col.lower, col.upper, self.dp_bins + 1)
            counts, _ = np.histogram(np.clip(values, col.lower, col.upper), bins=edges)
            marginal = NumericMarginal.from_histogram(noisy_histogram(counts, sigma, rng), edges)
            self.marginals[col.name] = marginal
            u = np.clip(marginal.cdf(values), _EPS, 1 - _EPS)
            return stats.norm.ppf(u)
        codes = _codes(series, col)
        counts = np.bincount(codes, minlength=len(col.categories)).astype(np.float64)
        marginal = CategoricalMarginal(noisy_histogram(counts, sigma, rng))
        self.marginals[col.name] = marginal
        return marginal.scores(codes)

    def _sample(self, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
        chol = np.linalg.cholesky(self.correlation + 1e-9 * np.eye(len(self.schema)))
        z = rng.standard_normal((n_rows, len(self.schema))) @ chol.T
        u = np.clip(stats.norm.cdf(z), _EPS, 1 - _EPS)
        out: dict[str, np.ndarray] = {}
        for j, col in enumerate(self.schema.columns):
            marginal = self.marginals[col.name]
            if isinstance(marginal, NumericMarginal):
                values = marginal.quantile(u[:, j])
                out[col.name] = np.rint(values) if col.kind is ColumnKind.INTEGER else values
            else:
                out[col.name] = np.asarray(col.categories, dtype=object)[marginal.decode(u[:, j])]
        return pd.DataFrame(out)


def _codes(series: pd.Series, col: ColumnSpec) -> np.ndarray:
    index = {c: i for i, c in enumerate(col.categories)}
    return np.array([index[str(v)] for v in series.tolist()], dtype=np.int64)


def _exact_correlation(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=0)
    std = centered.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    corr = (centered / std).T @ (centered / std) / len(scores)
    np.fill_diagonal(corr, 1.0)
    return nearest_correlation(np.clip(corr, -1.0, 1.0))
