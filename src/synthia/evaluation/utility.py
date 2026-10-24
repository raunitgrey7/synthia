"""Utility: train on synthetic, test on real (TSTR) against the train-on-real baseline (TRTR).

Two classifiers of different families are trained to predict a binary target: a regularised logistic regression
and a histogram gradient boosting model. Both are trained once on the real training split and once on the
synthetic table, and both are scored with ROC AUC on the real holdout, which no generator ever saw. The utility
ratio TSTR / TRTR is the headline: 1.0 means the synthetic table trains a model as good as the real one. The
confidence intervals come from a paired bootstrap over holdout rows, so the two AUCs are resampled together.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from synthia.evaluation.bootstrap import Estimate, bootstrap
from synthia.schema import ColumnKind, ColumnSpec, TableSchema
from synthia.transforms import MixedEncoder


@dataclass
class ModelUtility:
    model: str
    trtr_auc: Estimate
    tstr_auc: Estimate
    ratio: Estimate
    degenerate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "trtr_auc": self.trtr_auc.to_dict(),
            "tstr_auc": self.tstr_auc.to_dict(),
            "ratio": self.ratio.to_dict(),
            "degenerate": self.degenerate,
        }


@dataclass
class UtilityResult:
    target: str
    positive: str
    models: list[ModelUtility]

    @property
    def mean_ratio(self) -> float:
        return float(np.mean([m.ratio.value for m in self.models]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "positive": self.positive,
            "mean_ratio": self.mean_ratio,
            "models": [m.to_dict() for m in self.models],
        }


def _feature_schema(schema: TableSchema, target: str) -> TableSchema:
    return TableSchema(
        columns=tuple(c for c in schema.columns if c.name != target), bounds_inferred=schema.bounds_inferred
    )


def _labels(df: pd.DataFrame, spec: ColumnSpec, positive: str) -> np.ndarray:
    if spec.kind is ColumnKind.CATEGORICAL:
        return (df[spec.name].astype(str) == positive).to_numpy()
    return (df[spec.name].to_numpy(dtype=np.float64) > float(positive)).astype(bool)


Classifier = LogisticRegression | HistGradientBoostingClassifier


def _build_models(seed: int) -> dict[str, Classifier]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed),
    }


def _fit_scores(model: Classifier, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray | None:
    if len(np.unique(y_train)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def pick_target(schema: TableSchema, preferred: str | None = None) -> tuple[str, str]:
    """Choose the binary target and its positive category: the preferred column, else the last binary column."""
    candidates = [c for c in schema.columns if c.kind is ColumnKind.CATEGORICAL and len(c.categories) == 2]
    if preferred is not None:
        spec = schema.get(preferred)
        if spec.kind is not ColumnKind.CATEGORICAL or len(spec.categories) != 2:
            raise ValueError(f"target {preferred!r} must be a binary categorical column")
        return spec.name, spec.categories[-1]
    if not candidates:
        raise ValueError("no binary categorical column available as a utility target")
    return candidates[-1].name, candidates[-1].categories[-1]


def evaluate_utility(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: TableSchema,
    target: str | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> UtilityResult:
    """TSTR against TRTR for two model families, scored on the real holdout with paired bootstrap intervals."""
    target_name, positive = pick_target(schema, target)
    spec = schema.get(target_name)
    features = _feature_schema(schema, target_name)
    encoder = MixedEncoder(features).fit(real_train)
    x_train, x_hold, x_syn = (encoder.transform(d) for d in (real_train, real_holdout, synthetic))
    y_train, y_hold, y_syn = (_labels(d, spec, positive) for d in (real_train, real_holdout, synthetic))
    results = []
    for k, (name, model) in enumerate(_build_models(seed).items()):
        trtr_scores = _fit_scores(model, x_train, y_train, x_hold)
        tstr_scores = _fit_scores(_build_models(seed)[name], x_syn, y_syn, x_hold)
        degenerate = tstr_scores is None or trtr_scores is None
        if trtr_scores is None:
            trtr_scores = np.full(len(y_hold), 0.5)
        if tstr_scores is None:
            tstr_scores = np.full(len(y_hold), 0.5)
        trtr = bootstrap(_auc, y_hold, trtr_scores, n_boot=n_boot, seed=seed + k, paired=True, floor=0.0, ceiling=1.0)
        tstr = bootstrap(_auc, y_hold, tstr_scores, n_boot=n_boot, seed=seed + k, paired=True, floor=0.0, ceiling=1.0)
        ratio = bootstrap(
            lambda y, a, b: _auc(y, b) / _auc(y, a),
            y_hold,
            trtr_scores,
            tstr_scores,
            n_boot=n_boot,
            seed=seed + k,
            paired=True,
            floor=0.0,
        )
        results.append(ModelUtility(name, trtr, tstr, ratio, degenerate))
    return UtilityResult(target_name, positive, results)
