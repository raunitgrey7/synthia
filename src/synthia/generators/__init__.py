"""The three generators behind one interface: fit a frame, sample rows, report the privacy spent."""

from __future__ import annotations

from typing import Any

from synthia.generators.base import Generator
from synthia.generators.bayesnet import BayesianNetworkGenerator
from synthia.generators.copula import GaussianCopulaGenerator
from synthia.generators.ctgan import CTGANGenerator
from synthia.privacy import DPConfig
from synthia.schema import TableSchema

__all__ = [
    "BayesianNetworkGenerator",
    "CTGANGenerator",
    "GaussianCopulaGenerator",
    "Generator",
    "available_generators",
    "make_generator",
]


def available_generators() -> dict[str, type[Generator]]:
    return {
        GaussianCopulaGenerator.name: GaussianCopulaGenerator,
        CTGANGenerator.name: CTGANGenerator,
        BayesianNetworkGenerator.name: BayesianNetworkGenerator,
    }


def make_generator(
    name: str,
    schema: TableSchema,
    dp: DPConfig | None = None,
    seed: int = 0,
    **kwargs: Any,  # noqa: ANN401  (hyperparameters forwarded to the chosen generator)
) -> Generator:
    """Build a generator by registry name; unknown names list the valid ones."""
    registry = available_generators()
    try:
        cls = registry[name]
    except KeyError as exc:
        raise ValueError(f"unknown generator {name!r}; choose from {sorted(registry)}") from exc
    return cls(schema, dp=dp, seed=seed, **kwargs)
