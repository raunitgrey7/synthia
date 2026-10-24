"""Constrained sampling: request rows that satisfy column conditions by rejection, with CTGAN conditioning.

A ``Constraint`` restricts one column: an exact value, a set of allowed categories, or a numeric range. Sampling
over-generates, keeps the rows that satisfy every constraint and repeats with a batch sized from the observed
acceptance rate until ``n_rows`` are collected or ``max_rounds`` is exhausted. For the CTGAN a single categorical
equality constraint is pushed into the condition vector first, which raises the acceptance rate; rejection still
runs afterwards because the generator is not guaranteed to honour its condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from synthia.generators.base import Generator
from synthia.generators.ctgan import CTGANGenerator
from synthia.schema import TableSchema


class ConstraintError(ValueError):
    """Raised when the constraints are invalid for the schema or cannot be satisfied in the allowed rounds."""


@dataclass(frozen=True)
class Constraint:
    column: str
    equals: str | float | None = None
    one_of: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, schema: TableSchema) -> None:
        if self.column not in schema:
            raise ConstraintError(f"unknown column {self.column!r}")
        spec = schema.get(self.column)
        if self.equals is None and not self.one_of and self.minimum is None and self.maximum is None:
            raise ConstraintError(f"constraint on {self.column!r} restricts nothing")
        if spec.is_numeric:
            if self.one_of:
                raise ConstraintError(f"one_of applies to categorical columns, {self.column!r} is numeric")
            if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
                raise ConstraintError(f"minimum exceeds maximum for {self.column!r}")
            if self.equals is not None and isinstance(self.equals, str):
                raise ConstraintError(f"equals for numeric column {self.column!r} must be a number")
        else:
            if self.minimum is not None or self.maximum is not None:
                raise ConstraintError(f"ranges apply to numeric columns, {self.column!r} is categorical")
            allowed = set(spec.categories)
            wanted = set(str(v) for v in self.one_of) | ({str(self.equals)} if self.equals is not None else set())
            unknown = wanted - allowed
            if unknown:
                raise ConstraintError(f"unknown categories {sorted(unknown)} for {self.column!r}")

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        series = df[self.column]
        keep = np.ones(len(df), dtype=bool)
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=np.float64)
            if self.equals is not None:
                keep &= np.isclose(values, float(self.equals))
            if self.minimum is not None:
                keep &= values >= self.minimum
            if self.maximum is not None:
                keep &= values <= self.maximum
        else:
            values = series.astype(str).to_numpy()
            if self.equals is not None:
                keep &= values == str(self.equals)
            if self.one_of:
                keep &= np.isin(values, [str(v) for v in self.one_of])
        return keep

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"column": self.column}
        if self.equals is not None:
            out["equals"] = self.equals
        if self.one_of:
            out["one_of"] = list(self.one_of)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out


@dataclass
class SamplingInfo:
    rounds: int
    generated: int
    accepted: int
    acceptance_rate: float
    conditioned_on: dict[str, str] | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "generated": self.generated,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "conditioned_on": self.conditioned_on,
            "constraints": self.constraints,
        }


def _joint_mask(df: pd.DataFrame, constraints: list[Constraint]) -> np.ndarray:
    keep = np.ones(len(df), dtype=bool)
    for c in constraints:
        keep &= c.mask(df)
    return keep


def _ctgan_condition(generator: Generator, constraints: list[Constraint]) -> tuple[str, str] | None:
    if not isinstance(generator, CTGANGenerator):
        return None
    for c in constraints:
        spec = generator.schema.get(c.column)
        if not spec.is_numeric and c.equals is not None:
            return c.column, str(c.equals)
        if not spec.is_numeric and len(c.one_of) == 1:
            return c.column, str(c.one_of[0])
    return None


def sample_with_constraints(
    generator: Generator,
    n_rows: int,
    constraints: list[Constraint],
    seed: int | None = None,
    max_rounds: int = 10,
    max_batch: int = 20_000,
) -> tuple[pd.DataFrame, SamplingInfo]:
    """Rejection sampling with adaptive batch sizes; raises ``ConstraintError`` when the target is not reached."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    for c in constraints:
        c.validate(generator.schema)
    rng = np.random.default_rng(seed)
    condition = _ctgan_condition(generator, constraints)
    kept: list[pd.DataFrame] = []
    generated = accepted = 0
    batch = n_rows
    for round_no in range(1, max_rounds + 1):
        draw_seed = int(rng.integers(2**31))
        if condition is not None and isinstance(generator, CTGANGenerator):
            frame = generator.sample_conditioned(batch, condition[0], condition[1], seed=draw_seed)
        else:
            frame = generator.sample(batch, seed=draw_seed)
        mask = _joint_mask(frame, constraints)
        generated += len(frame)
        accepted += int(mask.sum())
        kept.append(frame[mask])
        if accepted >= n_rows:
            result = pd.concat(kept, ignore_index=True).iloc[:n_rows].reset_index(drop=True)
            info = SamplingInfo(
                round_no,
                generated,
                accepted,
                accepted / generated,
                _cond_dict(condition),
                [c.to_dict() for c in constraints],
            )
            return result, info
        rate = max(accepted / generated, 1.0 / max(generated, 1))
        batch = int(min(max_batch, max(n_rows, math.ceil((n_rows - accepted) / rate * 1.5))))
    raise ConstraintError(
        f"collected {accepted} of {n_rows} rows after {max_rounds} rounds and {generated} draws; "
        "the constraints are too selective for this generator"
    )


def _cond_dict(condition: tuple[str, str] | None) -> dict[str, str] | None:
    return None if condition is None else {condition[0]: condition[1]}
