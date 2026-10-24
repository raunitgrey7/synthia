"""Run the three evaluation families on one synthetic table and collect them in one serialisable result."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from synthia.evaluation.fidelity import FidelityResult, evaluate_fidelity
from synthia.evaluation.privacy import PrivacyResult, evaluate_privacy
from synthia.evaluation.utility import UtilityResult, evaluate_utility
from synthia.schema import TableSchema


@dataclass
class EvaluationResult:
    fidelity: FidelityResult
    utility: UtilityResult
    privacy: PrivacyResult
    n_train: int
    n_holdout: int
    n_synthetic: int
    n_boot: int
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "n_synthetic": self.n_synthetic,
            "n_boot": self.n_boot,
            "seconds": round(self.seconds, 3),
            "fidelity": self.fidelity.to_dict(),
            "utility": self.utility.to_dict(),
            "privacy": self.privacy.to_dict(),
            **self.extra,
        }

    def headline(self) -> dict[str, float]:
        """The numbers the benchmark table and the quality gate look at."""
        return {
            "mean_ks": self.fidelity.mean_ks.value,
            "mean_tvd": self.fidelity.mean_tvd.value,
            "correlation_delta_mean": self.fidelity.correlation_delta_mean.value,
            "utility_ratio": self.utility.mean_ratio,
            "membership_auc": self.privacy.membership_auc.value,
            "dcr_share_train": self.privacy.dcr_share_train.value,
            "nndr_synthetic": self.privacy.nndr_synthetic.value,
            "exact_match_rate": self.privacy.exact_match_rate.value,
        }


def evaluate(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: TableSchema,
    target: str | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> EvaluationResult:
    """Coerce the three frames to the schema, then run fidelity, utility and privacy with the same seed."""
    started = time.perf_counter()
    train = schema.coerce(real_train)
    holdout = schema.coerce(real_holdout)
    syn = schema.coerce(synthetic)
    fidelity = evaluate_fidelity(train, syn, schema, n_boot=n_boot, seed=seed)
    utility = evaluate_utility(train, holdout, syn, schema, target=target, n_boot=n_boot, seed=seed)
    privacy = evaluate_privacy(train, holdout, syn, schema, n_boot=n_boot, seed=seed)
    return EvaluationResult(
        fidelity=fidelity,
        utility=utility,
        privacy=privacy,
        n_train=len(train),
        n_holdout=len(holdout),
        n_synthetic=len(syn),
        n_boot=n_boot,
        seconds=time.perf_counter() - started,
    )
