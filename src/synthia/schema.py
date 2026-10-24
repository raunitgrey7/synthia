"""Column metadata shared by every generator and metric.

A ``TableSchema`` is the contract between data and models. It records the kind of every column, the category
domain of categorical columns and the public value bounds of numeric columns. The bounds matter for differential
privacy: every DP mechanism in this package clips or bins values to the declared range, so the range must be a
public fact about the domain (age is between 18 and 95) and not a statistic read off the private data. When a
schema is inferred from data the bounds are marked as inferred and the DP generators report that caveat.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class ColumnKind(StrEnum):
    """The three column kinds a generator has to handle."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class ColumnSpec:
    """Metadata for one column. Numeric columns carry bounds, categorical columns carry their domain."""

    name: str
    kind: ColumnKind
    categories: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind is ColumnKind.CATEGORICAL:
            if not self.categories:
                raise ValueError(f"categorical column {self.name!r} needs at least one category")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError(f"categorical column {self.name!r} has duplicate categories")
        elif self.lower is None or self.upper is None:
            raise ValueError(f"numeric column {self.name!r} needs lower and upper bounds")
        elif not self.lower < self.upper:
            raise ValueError(f"numeric column {self.name!r} needs lower < upper, got {self.lower}, {self.upper}")

    @property
    def is_numeric(self) -> bool:
        return self.kind is not ColumnKind.CATEGORICAL

    @property
    def span(self) -> float:
        """Width of the public range; used as the clipping range by DP mechanisms."""
        if self.lower is None or self.upper is None:
            raise ValueError(f"column {self.name!r} has no numeric bounds")
        return float(self.upper - self.lower)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind.value}
        if self.is_numeric:
            out["lower"] = self.lower
            out["upper"] = self.upper
        else:
            out["categories"] = list(self.categories)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnSpec:
        return cls(
            name=str(data["name"]),
            kind=ColumnKind(data["kind"]),
            categories=tuple(str(c) for c in data.get("categories", ())),
            lower=data.get("lower"),
            upper=data.get("upper"),
        )


@dataclass(frozen=True)
class TableSchema:
    """Ordered collection of column specs plus a flag telling whether bounds came from the data."""

    columns: tuple[ColumnSpec, ...]
    bounds_inferred: bool = False
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("schema has duplicate column names")
        if not names:
            raise ValueError("schema needs at least one column")
        object.__setattr__(self, "_index", {name: i for i, name in enumerate(names)})

    def __iter__(self) -> Iterable[ColumnSpec]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric]

    @property
    def categorical_names(self) -> list[str]:
        return [c.name for c in self.columns if not c.is_numeric]

    def get(self, name: str) -> ColumnSpec:
        try:
            return self.columns[self._index[name]]
        except KeyError as exc:
            raise KeyError(f"unknown column {name!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"columns": [c.to_dict() for c in self.columns], "bounds_inferred": self.bounds_inferred}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableSchema:
        return cls(
            columns=tuple(ColumnSpec.from_dict(c) for c in data["columns"]),
            bounds_inferred=bool(data.get("bounds_inferred", False)),
        )

    @classmethod
    def infer(cls, df: pd.DataFrame, max_categories: int = 20) -> TableSchema:
        """Infer kinds and domains from a frame; numeric bounds are read from the data and flagged as such."""
        if df.shape[1] == 0:
            raise ValueError("cannot infer a schema from a frame with no columns")
        specs = [_infer_column(df[name], str(name), max_categories) for name in df.columns]
        return cls(columns=tuple(specs), bounds_inferred=True)

    def with_bounds(self, bounds: dict[str, tuple[float, float]]) -> TableSchema:
        """Return a copy with public bounds declared for the given numeric columns."""
        new_cols = []
        for col in self.columns:
            if col.name in bounds:
                lo, hi = bounds[col.name]
                new_cols.append(replace(col, lower=float(lo), upper=float(hi)))
            else:
                new_cols.append(col)
        still_inferred = any(c.is_numeric and c.name not in bounds for c in self.columns) and self.bounds_inferred
        return TableSchema(columns=tuple(new_cols), bounds_inferred=still_inferred)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return human-readable problems; an empty list means the frame conforms."""
        problems = [f"missing column {name!r}" for name in self.names if name not in df.columns]
        for col in self.columns:
            if col.name not in df.columns:
                continue
            series = df[col.name]
            if series.isna().any():
                problems.append(f"column {col.name!r} has {int(series.isna().sum())} missing values")
                continue
            if col.is_numeric:
                if not pd.api.types.is_numeric_dtype(series):
                    problems.append(f"column {col.name!r} should be numeric")
            else:
                unknown = set(series.astype(str).unique()) - set(col.categories)
                if unknown:
                    problems.append(f"column {col.name!r} has unknown categories {sorted(unknown)[:5]}")
        return problems

    def coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast a conforming frame to canonical dtypes and column order (floats, ints, plain strings)."""
        problems = self.validate(df)
        if problems:
            raise ValueError("; ".join(problems))
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            series = df[col.name]
            if col.kind is ColumnKind.CONTINUOUS:
                out[col.name] = series.astype(np.float64)
            elif col.kind is ColumnKind.INTEGER:
                out[col.name] = np.rint(series.astype(np.float64)).astype(np.int64)
            else:
                out[col.name] = series.astype(str).astype(object)
        return out


def _infer_column(series: pd.Series, name: str, max_categories: int) -> ColumnSpec:
    clean = series.dropna()
    if clean.empty:
        raise ValueError(f"column {name!r} has no non-missing values")
    if pd.api.types.is_bool_dtype(clean) or not pd.api.types.is_numeric_dtype(clean):
        cats = tuple(sorted(clean.astype(str).unique().tolist()))
        if len(cats) > max_categories:
            raise ValueError(f"column {name!r} has {len(cats)} distinct values; too many for a categorical column")
        return ColumnSpec(name=name, kind=ColumnKind.CATEGORICAL, categories=cats)
    values = clean.to_numpy(dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    if np.all(np.isclose(values, np.rint(values))):
        return ColumnSpec(name=name, kind=ColumnKind.INTEGER, lower=lo, upper=hi)
    return ColumnSpec(name=name, kind=ColumnKind.CONTINUOUS, lower=lo, upper=hi)
