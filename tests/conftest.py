"""Shared fixtures: a small reference cohort and a fitted copula, built once per session."""

from __future__ import annotations

import pandas as pd
import pytest

from synthia.generators.copula import GaussianCopulaGenerator
from synthia.reference import generate_patients, patient_schema, train_holdout_split
from synthia.schema import TableSchema


@pytest.fixture(scope="session")
def schema() -> TableSchema:
    return patient_schema()


@pytest.fixture(scope="session")
def cohort() -> pd.DataFrame:
    return generate_patients(1600, seed=7)


@pytest.fixture(scope="session")
def split(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_holdout_split(cohort, 0.25, seed=7)


@pytest.fixture(scope="session")
def train(split: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return split[0]


@pytest.fixture(scope="session")
def holdout(split: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return split[1]


@pytest.fixture(scope="session")
def copula(schema: TableSchema, train: pd.DataFrame) -> GaussianCopulaGenerator:
    return GaussianCopulaGenerator(schema, seed=1).fit(train)


@pytest.fixture(scope="session")
def synthetic(copula: GaussianCopulaGenerator, train: pd.DataFrame) -> pd.DataFrame:
    return copula.sample(len(train), seed=11)
