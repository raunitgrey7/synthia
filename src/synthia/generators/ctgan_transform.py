"""Data representation for the CTGAN-style generator: mode-specific normalisation, one-hot and condition vectors.

Continuous columns follow the CTGAN recipe: a variational Gaussian mixture is fitted per column, each value is
expressed as a normalised offset from the mode it is assigned to plus a one-hot of that mode. Under DP that
mixture would be a data-dependent statistic, so the DP transformer instead scales each numeric column to
``[-1, 1]`` with the public bounds and spends no privacy budget. Categorical columns are one-hot in both cases.

The condition vector is a one-hot over all categories of all categorical columns with a mask saying which column
is active. ``ConditionSampler`` implements CTGAN's training-by-sampling (pick a column, pick a category by
log-frequency, pick a real row that has it). ``FrequencySampler`` draws condition vectors from a (possibly noisy)
frequency table and never looks at rows, which is what the DP training loop and the sampling path use.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

from synthia.schema import ColumnKind, ColumnSpec, TableSchema
from synthia.transforms import one_hot


@dataclass(frozen=True)
class Span:
    """A contiguous block of the transformed row with one output activation."""

    column: str
    start: int
    size: int
    activation: str  # "tanh" or "softmax"


class _ContinuousModes:
    """Per-column mixture used for mode-specific normalisation."""

    def __init__(self, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> None:
        self.means = means
        self.stds = np.maximum(stds, 1e-6)
        self.weights = weights

    @classmethod
    def fit(cls, values: np.ndarray, max_modes: int, seed: int) -> _ContinuousModes:
        mixture = BayesianGaussianMixture(
            n_components=max_modes,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=500,
            tol=1e-3,
            n_init=1,
            random_state=seed,
        )
        with warnings.catch_warnings():
            # A mixture that stops early still gives usable modes; the encoding is a re-parametrisation.
            warnings.simplefilter("ignore", ConvergenceWarning)
            mixture.fit(values.reshape(-1, 1))
        keep = mixture.weights_ > 0.005
        if not keep.any():
            keep[np.argmax(mixture.weights_)] = True
        means = mixture.means_.ravel()[keep]
        stds = np.sqrt(mixture.covariances_.ravel()[keep])
        weights = mixture.weights_[keep] / mixture.weights_[keep].sum()
        return cls(means, stds, weights)

    @property
    def n_modes(self) -> int:
        return len(self.means)

    def encode(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return [alpha, one-hot(mode)] per value, the mode drawn from the responsibilities."""
        dens = self.weights * np.exp(-0.5 * ((values[:, None] - self.means) / self.stds) ** 2) / self.stds
        probs = dens / np.maximum(dens.sum(axis=1, keepdims=True), 1e-300)
        cumulative = np.cumsum(probs, axis=1)
        modes = np.minimum((rng.random(len(values))[:, None] > cumulative).sum(axis=1), self.n_modes - 1)
        alpha = np.clip((values - self.means[modes]) / (4.0 * self.stds[modes]), -0.99, 0.99)
        out = np.zeros((len(values), 1 + self.n_modes))
        out[:, 0] = alpha
        out[np.arange(len(values)), 1 + modes] = 1.0
        return out

    def decode(self, block: np.ndarray) -> np.ndarray:
        alpha = np.clip(block[:, 0], -1.0, 1.0)
        modes = np.argmax(block[:, 1:], axis=1)
        return alpha * 4.0 * self.stds[modes] + self.means[modes]


class DataTransformer:
    """Frame to float matrix and back; ``spans`` describe activations, ``cond_spans`` the conditionable blocks."""

    def __init__(self, schema: TableSchema, max_modes: int = 5, public_scaling: bool = False, seed: int = 0) -> None:
        if max_modes < 1:
            raise ValueError("max_modes must be at least 1")
        self.schema = schema
        self.max_modes = max_modes
        self.public_scaling = public_scaling
        self.seed = seed
        self.spans: list[Span] = []
        self.cond_spans: list[Span] = []
        self.modes: dict[str, _ContinuousModes] = {}
        self.output_dim = 0
        self.fitted = False

    def fit(self, df: pd.DataFrame | None) -> DataTransformer:
        """Needs data for mode-specific normalisation; with ``public_scaling`` the frame may be ``None``."""
        rng = np.random.default_rng(self.seed)
        spans: list[Span] = []
        offset = 0
        for col in self.schema.columns:
            if col.is_numeric:
                if self.public_scaling or df is None:
                    size = 1
                else:
                    modes = _ContinuousModes.fit(
                        df[col.name].to_numpy(dtype=np.float64), self.max_modes, int(rng.integers(2**31))
                    )
                    self.modes[col.name] = modes
                    size = 1 + modes.n_modes
                spans.append(Span(col.name, offset, size, "numeric"))
            else:
                spans.append(Span(col.name, offset, len(col.categories), "softmax"))
            offset += spans[-1].size
        self.spans = spans
        self.cond_spans = [s for s in spans if s.activation == "softmax"]
        self.output_dim = offset
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        self._check()
        out = np.zeros((len(df), self.output_dim), dtype=np.float32)
        for span, col in zip(self.spans, self.schema.columns, strict=True):
            if col.is_numeric:
                values = df[col.name].to_numpy(dtype=np.float64)
                if col.name in self.modes:
                    block = self.modes[col.name].encode(values, rng)
                else:
                    block = (2.0 * (np.clip(values, col.lower, col.upper) - col.lower) / col.span - 1.0)[:, None]
            else:
                block = one_hot(df[col.name], col.categories)
            out[:, span.start : span.start + span.size] = block
        return out

    def inverse(self, data: np.ndarray) -> pd.DataFrame:
        self._check()
        out: dict[str, np.ndarray] = {}
        for span, col in zip(self.spans, self.schema.columns, strict=True):
            block = np.asarray(data[:, span.start : span.start + span.size], dtype=np.float64)
            if col.is_numeric:
                out[col.name] = self._decode_numeric(block, col)
            else:
                out[col.name] = np.asarray(col.categories, dtype=object)[np.argmax(block, axis=1)]
        return pd.DataFrame(out)

    def _decode_numeric(self, block: np.ndarray, col: ColumnSpec) -> np.ndarray:
        if col.name in self.modes:
            values = self.modes[col.name].decode(block)
        else:
            values = (np.clip(block[:, 0], -1.0, 1.0) + 1.0) / 2.0 * col.span + col.lower
        values = np.clip(values, col.lower, col.upper)
        return np.rint(values) if col.kind is ColumnKind.INTEGER else values

    @property
    def cond_dim(self) -> int:
        return sum(s.size for s in self.cond_spans)

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError("DataTransformer.fit must run before transform or inverse")


def condition_vectors(transformer: DataTransformer, column_idx: np.ndarray, category_idx: np.ndarray) -> np.ndarray:
    """Build condition vectors from per-row (conditionable column index, category index) pairs."""
    cond = np.zeros((len(column_idx), transformer.cond_dim), dtype=np.float32)
    starts = np.cumsum([0, *[s.size for s in transformer.cond_spans]])[:-1]
    cond[np.arange(len(column_idx)), starts[column_idx] + category_idx] = 1.0
    return cond


class FrequencySampler:
    """Draw (column, category) conditions from per-column frequency tables; rows are never touched."""

    def __init__(self, transformer: DataTransformer, frequencies: list[np.ndarray]) -> None:
        self.transformer = transformer
        self.probs = []
        for freq in frequencies:
            total = float(np.sum(freq))
            self.probs.append(freq / total if total > 0 else np.full(len(freq), 1.0 / len(freq)))

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (condition matrix, column index, category index); empty when nothing is conditionable."""
        if not self.probs:
            return (
                np.zeros((n, 0), dtype=np.float32),
                np.zeros(n, dtype=np.int64),
                np.zeros(n, dtype=np.int64),
            )
        cols = rng.integers(len(self.probs), size=n)
        cats = np.empty(n, dtype=np.int64)
        for j, probs in enumerate(self.probs):
            mask = cols == j
            if mask.any():
                cats[mask] = rng.choice(len(probs), size=int(mask.sum()), p=probs)
        return condition_vectors(self.transformer, cols, cats), cols, cats


class ConditionSampler:
    """CTGAN training-by-sampling: condition on a log-frequency-weighted category and fetch matching rows."""

    def __init__(self, transformer: DataTransformer, data: np.ndarray) -> None:
        self.transformer = transformer
        self.n = len(data)
        self.row_pools: list[list[np.ndarray]] = []
        self.log_probs: list[np.ndarray] = []
        self.frequencies: list[np.ndarray] = []
        for span in transformer.cond_spans:
            block = data[:, span.start : span.start + span.size]
            codes = np.argmax(block, axis=1)
            counts = np.bincount(codes, minlength=span.size).astype(np.float64)
            self.frequencies.append(counts)
            log_freq = np.log1p(counts)
            self.log_probs.append(
                log_freq / log_freq.sum() if log_freq.sum() > 0 else np.full(span.size, 1.0 / span.size)
            )
            self.row_pools.append([np.flatnonzero(codes == k) for k in range(span.size)])

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (condition matrix, column index, category index, matching real row index)."""
        if not self.log_probs:
            rows = rng.integers(self.n, size=n)
            return (
                np.zeros((n, 0), dtype=np.float32),
                np.zeros(n, dtype=np.int64),
                np.zeros(n, dtype=np.int64),
                rows,
            )
        cols = rng.integers(len(self.log_probs), size=n)
        cats = np.empty(n, dtype=np.int64)
        rows = np.empty(n, dtype=np.int64)
        for j, probs in enumerate(self.log_probs):
            mask = np.flatnonzero(cols == j)
            if len(mask) == 0:
                continue
            cats[mask] = rng.choice(len(probs), size=len(mask), p=probs)
            for k in np.unique(cats[mask]):
                pool = self.row_pools[j][k]
                sub = mask[cats[mask] == k]
                rows[sub] = rng.choice(pool, size=len(sub)) if len(pool) else rng.integers(self.n, size=len(sub))
        return condition_vectors(self.transformer, cols, cats), cols, cats, rows
