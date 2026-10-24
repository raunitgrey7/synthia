"""Privacy: membership inference, nearest-neighbour distance ratios and exact-match leakage.

All distances are Euclidean in the ``MixedEncoder`` space fitted on the real training split. The attacks are
black-box: they see only the synthetic table, which is the realistic threat model for a released dataset.

* Membership inference AUC: the attacker scores a candidate record by minus its distance to the nearest
  synthetic row (the full black-box attack of Chen et al. 2020, GAN-Leaks). Members are the training rows,
  non-members the holdout rows; 0.5 means the release reveals nothing about membership.
* DCR share: the fraction of synthetic rows closer to a training row than to a holdout row, with the training
  set subsampled to the holdout size so the comparison is fair; 0.5 is the ideal, higher means memorisation.
* NNDR: for each synthetic row, the distance to its nearest training row divided by the distance to the second
  nearest. Values near zero mean the row sits on one specific training record. The holdout's own NNDR against
  the training set is reported as the natural baseline.
* Exact matches: the share of synthetic rows identical to a training row after rounding numeric columns to two
  decimals, next to the same share for holdout rows, which is the rate chance alone produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from synthia.evaluation.bootstrap import Estimate, bootstrap
from synthia.schema import TableSchema
from synthia.transforms import MixedEncoder


@dataclass
class PrivacyResult:
    membership_auc: Estimate
    dcr_share_train: Estimate
    nndr_synthetic: Estimate
    nndr_holdout: Estimate
    exact_match_rate: Estimate
    exact_match_rate_holdout: Estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "membership_auc": self.membership_auc.to_dict(),
            "dcr_share_train": self.dcr_share_train.to_dict(),
            "nndr_synthetic": self.nndr_synthetic.to_dict(),
            "nndr_holdout": self.nndr_holdout.to_dict(),
            "exact_match_rate": self.exact_match_rate.to_dict(),
            "exact_match_rate_holdout": self.exact_match_rate_holdout.to_dict(),
        }


def nearest_distances(query: np.ndarray, reference: np.ndarray, k: int = 1) -> np.ndarray:
    """Distances from each query row to its ``k`` nearest reference rows, shape (n_query, k)."""
    k = min(k, len(reference))
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(reference)
    distances, _ = nn.kneighbors(query)
    return distances


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def _auc_from(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def membership_scores(
    train_x: np.ndarray, holdout_x: np.ndarray, synthetic_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(labels, scores) for the distance attack: members get label 1, score is minus the nearest distance."""
    members = -nearest_distances(train_x, synthetic_x)[:, 0]
    non_members = -nearest_distances(holdout_x, synthetic_x)[:, 0]
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
    return labels, np.concatenate([members, non_members])


def nndr(query_x: np.ndarray, reference_x: np.ndarray) -> np.ndarray:
    """Per-row ratio of nearest to second-nearest distance; 1.0 when the reference has a single row."""
    if len(reference_x) < 2:
        return np.ones(len(query_x))
    dist = nearest_distances(query_x, reference_x, k=2)
    return dist[:, 0] / np.maximum(dist[:, 1], 1e-12)


def exact_match_flags(
    candidates: pd.DataFrame, reference: pd.DataFrame, schema: TableSchema, decimals: int = 2
) -> np.ndarray:
    """Boolean per candidate row: does it equal some reference row after rounding numeric columns."""

    def keys(df: pd.DataFrame) -> np.ndarray:
        rounded = df[schema.names].copy()
        for name in schema.numeric_names:
            rounded[name] = np.round(rounded[name].to_numpy(dtype=np.float64), decimals)
        return pd.util.hash_pandas_object(rounded, index=False).to_numpy()

    return np.isin(keys(candidates), keys(reference))


def evaluate_privacy(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: TableSchema,
    n_boot: int = 200,
    seed: int = 0,
) -> PrivacyResult:
    """Distance-based attacks and leakage rates in the standardised mixed space, each with a bootstrap CI."""
    encoder = MixedEncoder(schema).fit(real_train)
    train_x, hold_x, syn_x = (encoder.transform(d) for d in (real_train, real_holdout, synthetic))
    rng = np.random.default_rng(seed)
    labels, scores = membership_scores(train_x, hold_x, syn_x)
    unit = {"floor": 0.0, "ceiling": 1.0}
    membership = bootstrap(_auc_from, labels, scores, n_boot=n_boot, seed=seed, paired=True, **unit)

    n_ref = min(len(train_x), len(hold_x))
    train_sub = train_x[rng.choice(len(train_x), size=n_ref, replace=False)]
    hold_sub = hold_x[rng.choice(len(hold_x), size=n_ref, replace=False)]
    closer_to_train = nearest_distances(syn_x, train_sub)[:, 0] < nearest_distances(syn_x, hold_sub)[:, 0]
    dcr_share = bootstrap(_mean, closer_to_train.astype(float), n_boot=n_boot, seed=seed + 1, **unit)

    nndr_syn = bootstrap(_median, nndr(syn_x, train_x), n_boot=n_boot, seed=seed + 2, **unit)
    nndr_hold = bootstrap(_median, nndr(hold_x, train_x), n_boot=n_boot, seed=seed + 3, **unit)

    syn_flags = exact_match_flags(synthetic, real_train, schema).astype(float)
    hold_flags = exact_match_flags(real_holdout, real_train, schema).astype(float)
    exact = bootstrap(_mean, syn_flags, n_boot=n_boot, seed=seed + 4, **unit)
    exact_hold = bootstrap(_mean, hold_flags, n_boot=n_boot, seed=seed + 5, **unit)
    return PrivacyResult(membership, dcr_share, nndr_syn, nndr_hold, exact, exact_hold)
