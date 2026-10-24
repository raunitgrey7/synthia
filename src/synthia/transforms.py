"""Encoders shared by generators and metrics: one-hot, binning and a mixed-type numeric encoding.

The DP generators need encodings whose parameters are public: ``Discretizer`` with the ``uniform`` strategy bins
on the declared bounds and never looks at the data, which is why it exists next to the data-driven ``quantile``
strategy. ``MixedEncoder`` turns a frame into the Euclidean space in which the privacy metrics measure distances:
numeric columns are standardised with reference statistics, categoricals are one-hot columns scaled so that a
category change costs the same as a one standard deviation move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthia.schema import ColumnKind, TableSchema


def one_hot(series: pd.Series, categories: tuple[str, ...]) -> np.ndarray:
    """One-hot encode with a fixed category order; unknown values raise rather than silently vanish."""
    index = {c: i for i, c in enumerate(categories)}
    try:
        codes = np.array([index[str(v)] for v in series.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown category {exc.args[0]!r} in column {series.name!r}") from exc
    out = np.zeros((len(codes), len(categories)), dtype=np.float64)
    out[np.arange(len(codes)), codes] = 1.0
    return out


class Discretizer:
    """Map every column to an integer code: bins for numeric columns, category index for categoricals.

    Numeric columns carry a second, finer histogram (``n_sub_bins`` cells per coarse bin) called the profile.
    Decoding draws the sub-bin from the profile and then a uniform value inside it, so a coarse top bin that
    runs to the public upper bound does not spray values across the whole tail. Under DP the generator
    replaces the exact profile counts with noisy ones through ``set_profile``.
    """

    def __init__(self, schema: TableSchema, n_bins: int = 10, strategy: str = "quantile", n_sub_bins: int = 8) -> None:
        if n_bins < 2 or n_sub_bins < 1:
            raise ValueError("n_bins must be at least 2 and n_sub_bins at least 1")
        if strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")
        self.schema = schema
        self.n_bins = n_bins
        self.n_sub_bins = n_sub_bins
        self.strategy = strategy
        self.edges: dict[str, np.ndarray] = {}
        self.fine_edges: dict[str, np.ndarray] = {}
        self.profiles: dict[str, np.ndarray] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame | None = None) -> Discretizer:
        """Quantile edges need data; uniform edges come from the schema bounds. Profiles use data when given."""
        for col in self.schema.columns:
            if not col.is_numeric:
                continue
            lo, hi = float(col.lower or 0.0), float(col.upper or 1.0)
            if self.strategy == "uniform" or df is None:
                edges = np.linspace(lo, hi, self.n_bins + 1)
            else:
                values = df[col.name].to_numpy(dtype=np.float64)
                inner = np.quantile(values, np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1])
                edges = np.maximum.accumulate(np.concatenate([[lo], inner, [hi]]))
            self.edges[col.name] = edges
            fine = [np.linspace(edges[k], edges[k + 1], self.n_sub_bins + 1)[:-1] for k in range(self.n_bins)]
            self.fine_edges[col.name] = np.concatenate([*fine, [edges[-1]]])
            if df is None:
                self.profiles[col.name] = np.ones(self.n_bins * self.n_sub_bins)
            else:
                self.profiles[col.name] = self.profile_counts(col.name, df[col.name].to_numpy(dtype=np.float64))
        self.fitted = True
        return self

    def profile_counts(self, name: str, values: np.ndarray) -> np.ndarray:
        """Exact counts of ``values`` in the fine bins of column ``name`` (one record adds one to one cell)."""
        fine = self.fine_edges[name]
        idx = np.clip(np.searchsorted(fine, values, side="right") - 1, 0, len(fine) - 2)
        return np.bincount(idx, minlength=len(fine) - 1).astype(np.float64)

    def set_profile(self, name: str, counts: np.ndarray) -> None:
        """Replace a profile, for example with noisy counts; negative entries are clipped at zero."""
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != (self.n_bins * self.n_sub_bins,):
            raise ValueError("profile has the wrong length")
        self.profiles[name] = np.clip(counts, 0.0, None)

    @property
    def cardinalities(self) -> list[int]:
        """Number of codes per column in schema order."""
        out = []
        for col in self.schema.columns:
            if col.is_numeric:
                out.append(len(self.edges[col.name]) - 1)
            else:
                out.append(len(col.categories))
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        codes = np.empty((len(df), len(self.schema)), dtype=np.int64)
        for j, col in enumerate(self.schema.columns):
            if col.is_numeric:
                edges = self.edges[col.name]
                values = df[col.name].to_numpy(dtype=np.float64)
                codes[:, j] = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
            else:
                index = {c: i for i, c in enumerate(col.categories)}
                codes[:, j] = [index[str(v)] for v in df[col.name].tolist()]
        return codes

    def inverse(self, codes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        """Decode codes; numeric values come from the profile inside their bin, integers are rounded."""
        self._check_fitted()
        out: dict[str, np.ndarray] = {}
        for j, col in enumerate(self.schema.columns):
            column_codes = codes[:, j]
            if col.is_numeric:
                values = self._decode_numeric(col.name, column_codes, rng)
                out[col.name] = np.rint(values) if col.kind is ColumnKind.INTEGER else values
            else:
                out[col.name] = np.asarray(col.categories, dtype=object)[column_codes]
        return self.schema.coerce(pd.DataFrame(out))

    def _decode_numeric(self, name: str, codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        profile = self.profiles[name].reshape(self.n_bins, self.n_sub_bins)
        totals = profile.sum(axis=1, keepdims=True)
        probs = np.where(totals > 0, profile / np.maximum(totals, 1e-12), 1.0 / self.n_sub_bins)
        cumulative = np.cumsum(probs[codes], axis=1)
        sub = np.minimum((rng.random(len(codes))[:, None] > cumulative).sum(axis=1), self.n_sub_bins - 1)
        fine = self.fine_edges[name]
        start = codes * self.n_sub_bins + sub
        lo, hi = fine[start], fine[start + 1]
        return lo + rng.random(len(codes)) * (hi - lo)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Discretizer.fit must run before transform or inverse")


class MixedEncoder:
    """Standardise numerics with reference statistics and one-hot categoricals for distance computations."""

    def __init__(self, schema: TableSchema) -> None:
        self.schema = schema
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> MixedEncoder:
        for col in self.schema.columns:
            if col.is_numeric:
                values = df[col.name].to_numpy(dtype=np.float64)
                self.means[col.name] = float(values.mean())
                std = float(values.std())
                self.scales[col.name] = std if std > 1e-12 else 1.0
        self.fitted = True
        return self

    @property
    def width(self) -> int:
        return sum(1 if c.is_numeric else len(c.categories) for c in self.schema.columns)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MixedEncoder.fit must run before transform")
        parts = []
        for col in self.schema.columns:
            if col.is_numeric:
                values = df[col.name].to_numpy(dtype=np.float64)
                parts.append(((values - self.means[col.name]) / self.scales[col.name])[:, None])
            else:
                parts.append(one_hot(df[col.name], col.categories) / np.sqrt(2.0))
        return np.hstack(parts)
