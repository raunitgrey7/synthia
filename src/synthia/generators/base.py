"""The generator interface and the registry the CLI, API and benchmark build from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from synthia.privacy import DPConfig, PrivacyReport
from synthia.schema import TableSchema


class Generator(ABC):
    """Fit on a conforming frame, then sample. Subclasses set ``privacy_report`` when fitted with DP."""

    name: str = "generator"

    def __init__(self, schema: TableSchema, dp: DPConfig | None = None, seed: int = 0) -> None:
        self.schema = schema
        self.dp = dp
        self.seed = seed
        self.privacy_report: PrivacyReport | None = None
        self.fitted = False
        self.n_train = 0
        self._rng = np.random.default_rng(seed)

    def fit(self, df: pd.DataFrame) -> Generator:
        """Validate, coerce and delegate to ``_fit``; records the training size for the metrics."""
        frame = self.schema.coerce(df)
        if len(frame) < 2:
            raise ValueError("fitting needs at least two rows")
        self.n_train = len(frame)
        self._fit(frame)
        self.fitted = True
        return self

    def sample(self, n_rows: int, seed: int | None = None) -> pd.DataFrame:
        """Draw ``n_rows`` rows. A seed gives reproducible draws; without one the internal stream advances."""
        if not self.fitted:
            raise RuntimeError(f"{self.name} must be fitted before sampling")
        if n_rows <= 0:
            raise ValueError("n_rows must be positive")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        frame = self._sample(n_rows, rng)
        return self.schema.coerce(frame)

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def _sample(self, n_rows: int, rng: np.random.Generator) -> pd.DataFrame: ...

    def describe(self) -> dict[str, Any]:
        """Fitted-state summary used by the API and the report."""
        return {
            "generator": self.name,
            "fitted": self.fitted,
            "n_train": self.n_train,
            "dp": None if self.dp is None else {"epsilon": self.dp.epsilon, "delta": self.dp.delta},
            "privacy": None if self.privacy_report is None else self.privacy_report.to_dict(),
        }

    def _dp_caveats(self) -> list[str]:
        caveats = ["category domains and numeric bounds are treated as public"]
        if self.schema.bounds_inferred:
            caveats.append("numeric bounds were inferred from the data, so they are not covered by the epsilon")
        return caveats
