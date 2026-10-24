"""Seeded reference data: a synthetic patient cohort with documented dependencies, plus CSV import.

The cohort is generated from an explicit causal graph so that every dependency a generator is supposed to
reproduce is known in advance. The graph, in topological order, is:

    age, sex                      exogenous
    bmi        <- age, sex        log-normal around 26.5 kg/m2, rising with age, slightly higher for men
    smoker     <- age, sex        three states (never, former, current); current smoking falls with age
    diabetes   <- age, bmi, sex, smoker          logistic
    systolic   <- age, bmi, smoker               linear + Gaussian noise
    diastolic  <- systolic, bmi                  linear + Gaussian noise
    heart_rate <- smoker, bmi, age               linear + Gaussian noise
    glucose    <- bmi, age, diabetes             linear + Gaussian noise; diabetes adds about 60 mg/dL
    cholesterol<- age, bmi, smoker               quadratic in age + Gaussian noise
    hypertension <- systolic, age                logistic, 50 percent at systolic 140
    cad        <- age, sex, smoker, cholesterol, hypertension, diabetes   logistic
    length_of_stay <- age, cad, diabetes, hypertension                   negative binomial
    readmitted_30d <- age, cad, diabetes, length_of_stay, smoker         logistic

Effect sizes are plausible for an adult hospital cohort and are documented in docs/EVALUATION.md. They are not
fitted to any real dataset; the point of the cohort is a known ground truth with mixed column types, skew,
count data and a rare binary outcome, which is what a tabular generator has to cope with in practice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synthia.schema import ColumnKind, ColumnSpec, TableSchema

YES_NO = ("no", "yes")
SEXES = ("F", "M")
SMOKER_STATES = ("never", "former", "current")


def patient_schema() -> TableSchema:
    """Public schema of the cohort. Bounds are domain facts, so DP mechanisms may rely on them."""
    return TableSchema(
        columns=(
            ColumnSpec("age", ColumnKind.INTEGER, lower=18, upper=95),
            ColumnSpec("sex", ColumnKind.CATEGORICAL, categories=SEXES),
            ColumnSpec("bmi", ColumnKind.CONTINUOUS, lower=14.0, upper=60.0),
            ColumnSpec("smoker", ColumnKind.CATEGORICAL, categories=SMOKER_STATES),
            ColumnSpec("systolic_bp", ColumnKind.CONTINUOUS, lower=80.0, upper=220.0),
            ColumnSpec("diastolic_bp", ColumnKind.CONTINUOUS, lower=40.0, upper=130.0),
            ColumnSpec("heart_rate", ColumnKind.CONTINUOUS, lower=40.0, upper=160.0),
            ColumnSpec("glucose", ColumnKind.CONTINUOUS, lower=50.0, upper=400.0),
            ColumnSpec("cholesterol", ColumnKind.CONTINUOUS, lower=100.0, upper=400.0),
            ColumnSpec("hypertension", ColumnKind.CATEGORICAL, categories=YES_NO),
            ColumnSpec("diabetes", ColumnKind.CATEGORICAL, categories=YES_NO),
            ColumnSpec("cad", ColumnKind.CATEGORICAL, categories=YES_NO),
            ColumnSpec("length_of_stay", ColumnKind.INTEGER, lower=0, upper=60),
            ColumnSpec("readmitted_30d", ColumnKind.CATEGORICAL, categories=YES_NO),
        ),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _yes_no(flag: np.ndarray) -> np.ndarray:
    return np.where(flag.astype(bool), "yes", "no").astype(object)


def _negative_binomial(rng: np.random.Generator, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Negative binomial with the given mean and shape ``dispersion`` (smaller means more overdispersed)."""
    p = dispersion / (dispersion + mean)
    return rng.negative_binomial(dispersion, p)


def generate_patients(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Draw ``n_rows`` patients from the documented causal graph. Same seed, same frame."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(seed)
    age = np.clip(np.rint(rng.normal(58.0, 16.0, n_rows)), 18, 95)
    male = rng.random(n_rows) < 0.48
    bmi = np.exp(rng.normal(np.log(26.5) + 0.004 * (age - 55.0) + 0.02 * male, 0.18))
    bmi = np.clip(bmi, 14.0, 60.0)
    smoker_idx = _draw_smoker(rng, age, male)
    current = smoker_idx == 2
    former = smoker_idx == 1
    diabetes_logit = -2.9 + 0.07 * (bmi - 27.0) + 0.035 * (age - 50.0) + 0.4 * male + 0.3 * current
    diabetes = rng.random(n_rows) < _sigmoid(diabetes_logit)
    systolic = 95.0 + 0.45 * age + 0.9 * (bmi - 27.0) + 6.0 * current + rng.normal(0.0, 11.0, n_rows)
    systolic = np.clip(systolic, 80.0, 220.0)
    diastolic = np.clip(0.52 * systolic + 12.0 + 0.3 * (bmi - 27.0) + rng.normal(0.0, 7.0, n_rows), 40.0, 130.0)
    heart_rate = 72.0 + 7.0 * current + 0.35 * (bmi - 27.0) - 0.08 * (age - 50.0) + rng.normal(0.0, 9.0, n_rows)
    heart_rate = np.clip(heart_rate, 40.0, 160.0)
    glucose_noise = rng.normal(0.0, 1.0, n_rows) * (10.0 + 18.0 * diabetes)
    glucose = 92.0 + 0.9 * (bmi - 27.0) + 0.2 * (age - 50.0) + 60.0 * diabetes + glucose_noise
    glucose = np.clip(glucose, 50.0, 400.0)
    cholesterol = 170.0 + 0.6 * (age - 50.0) - 0.004 * (age - 50.0) ** 2 + 1.2 * (bmi - 27.0) + 8.0 * current
    cholesterol = np.clip(cholesterol + rng.normal(0.0, 30.0, n_rows), 100.0, 400.0)
    hypertension = rng.random(n_rows) < _sigmoid(-14.0 + 0.1 * systolic + 0.02 * (age - 50.0))
    cad_logit = (
        -4.6
        + 0.05 * (age - 50.0)
        + 0.9 * male
        + 0.8 * current
        + 0.4 * former
        + 0.006 * (cholesterol - 180.0)
        + 0.9 * hypertension
        + 0.8 * diabetes
    )
    cad = rng.random(n_rows) < _sigmoid(cad_logit)
    los_mean = np.exp(0.6 + 0.012 * (age - 50.0) + 0.5 * cad + 0.3 * diabetes + 0.2 * hypertension)
    length_of_stay = np.clip(_negative_binomial(rng, los_mean, dispersion=3.0), 0, 60)
    readmit_logit = -3.6 + 0.03 * (age - 50.0) + 1.2 * cad + 0.8 * diabetes + 0.22 * length_of_stay + 0.4 * current
    readmitted = rng.random(n_rows) < _sigmoid(readmit_logit)
    frame = pd.DataFrame(
        {
            "age": age.astype(np.int64),
            "sex": np.where(male, "M", "F").astype(object),
            "bmi": bmi,
            "smoker": np.asarray(SMOKER_STATES, dtype=object)[smoker_idx],
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "heart_rate": heart_rate,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "hypertension": _yes_no(hypertension),
            "diabetes": _yes_no(diabetes),
            "cad": _yes_no(cad),
            "length_of_stay": length_of_stay.astype(np.int64),
            "readmitted_30d": _yes_no(readmitted),
        }
    )
    return patient_schema().coerce(frame)


def _draw_smoker(rng: np.random.Generator, age: np.ndarray, male: np.ndarray) -> np.ndarray:
    """Softmax over (never, former, current); former rises with age, current falls with it."""
    score_former = -1.0 + 0.03 * (age - 50.0) + 0.3 * male
    score_current = -1.0 - 0.025 * (age - 50.0) + 0.5 * male
    scores = np.column_stack([np.zeros_like(age, dtype=float), score_former, score_current])
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random(len(age))[:, None]
    return (draws > cumulative).sum(axis=1).clip(0, 2)


def train_holdout_split(df: pd.DataFrame, holdout_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle once with the seed and split; the holdout never touches a generator."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_holdout = round(len(df) * holdout_fraction)
    if n_holdout == 0 or n_holdout == len(df):
        raise ValueError("split leaves one side empty; use more rows or a different fraction")
    holdout = df.iloc[order[:n_holdout]].reset_index(drop=True)
    train = df.iloc[order[n_holdout:]].reset_index(drop=True)
    return train, holdout


def load_csv(
    path: str | Path, schema: TableSchema | None = None, max_categories: int = 20
) -> tuple[pd.DataFrame, TableSchema]:
    """Read a CSV, drop rows with missing values, infer or validate the schema and coerce dtypes."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} has no rows")
    frame = frame.dropna(axis=0, how="any").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} has no complete rows")
    if schema is None:
        schema = TableSchema.infer(frame, max_categories=max_categories)
    return schema.coerce(frame), schema
