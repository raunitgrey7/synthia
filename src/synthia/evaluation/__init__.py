"""Fidelity, utility and privacy evaluation of a synthetic table against the real train and holdout splits.

Every headline number comes with a percentile bootstrap confidence interval. The bootstrap resamples rows
(paired where the statistic needs paired rows, independently otherwise), so the intervals describe sampling
uncertainty of the metric at the given table sizes; they say nothing about variation across generator seeds,
which the benchmark reports separately.
"""

from synthia.evaluation.bootstrap import Estimate, bootstrap
from synthia.evaluation.fidelity import FidelityResult, association_matrix, evaluate_fidelity
from synthia.evaluation.privacy import PrivacyResult, evaluate_privacy
from synthia.evaluation.runner import EvaluationResult, evaluate
from synthia.evaluation.utility import UtilityResult, evaluate_utility

__all__ = [
    "Estimate",
    "EvaluationResult",
    "FidelityResult",
    "PrivacyResult",
    "UtilityResult",
    "association_matrix",
    "bootstrap",
    "evaluate",
    "evaluate_fidelity",
    "evaluate_privacy",
    "evaluate_utility",
]
