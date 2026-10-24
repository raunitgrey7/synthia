import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthia.evaluation import bootstrap, evaluate, evaluate_fidelity, evaluate_privacy, evaluate_utility
from synthia.evaluation.fidelity import (
    association_matrix,
    correlation_delta,
    correlation_ratio,
    cramers_v,
    ks_statistic,
    total_variation,
)
from synthia.evaluation.privacy import exact_match_flags, membership_scores, nndr
from synthia.evaluation.utility import pick_target


def test_ks_matches_scipy_and_is_zero_for_identical():
    from scipy import stats

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(0.5, size=200)
    assert abs(ks_statistic(a, b) - stats.ks_2samp(a, b).statistic) < 1e-12
    assert ks_statistic(a, a) == 0.0


def test_total_variation_bounds_and_values():
    assert total_variation(["a", "a", "b", "b"], ["a", "a", "b", "b"], ("a", "b")) == 0.0
    assert total_variation(["a"] * 4, ["b"] * 4, ("a", "b")) == 1.0
    assert abs(total_variation(["a", "a", "a", "b"], ["a", "b", "b", "b"], ("a", "b")) - 0.5) < 1e-12


def test_association_measures_on_constructed_data():
    x = np.repeat(["a", "b"], 50)
    assert abs(cramers_v(x, x) - 1.0) < 1e-9
    assert cramers_v(x, np.tile(["u", "v"], 50)) < 1e-9
    values = np.where(x == "a", 0.0, 10.0)
    assert abs(correlation_ratio(x, values) - 1.0) < 1e-9
    assert correlation_ratio(x, np.ones(100)) == 0.0


def test_association_matrix_shape_and_identity(schema, train):
    matrix = association_matrix(train, schema)
    assert matrix.shape == (len(schema), len(schema))
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix, matrix.T)
    i, j = schema.names.index("systolic_bp"), schema.names.index("diastolic_bp")
    assert matrix[i, j] > 0.6
    assert correlation_delta(train, train, schema) == (0.0, 0.0)


def test_bootstrap_interval_contains_point_and_handles_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    est = bootstrap(lambda a: float(np.mean(a)), x, n_boot=100, seed=1)
    assert est.low <= est.value <= est.high
    assert est.low < est.high
    y = x * 2
    paired = bootstrap(lambda a, b: float(np.corrcoef(a, b)[0, 1]), x, y, n_boot=20, seed=1, paired=True)
    assert paired.value == pytest.approx(1.0)
    assert paired.low == pytest.approx(1.0)
    assert bootstrap(lambda a: 1.0, x, n_boot=0).to_dict()["n_boot"] == 0
    with pytest.raises(ValueError):
        bootstrap(lambda a, b: 0.0, x, x[:10], paired=True)


def test_fidelity_identical_tables_score_zero(schema, train):
    res = evaluate_fidelity(train, train, schema, n_boot=10)
    assert res.mean_ks.value == 0.0 and res.mean_tvd.value == 0.0
    assert res.correlation_delta_mean.value == 0.0
    assert set(res.per_column) == set(schema.names)
    assert res.column_metric["age"] == "ks" and res.column_metric["sex"] == "tvd"


def test_fidelity_of_copula_is_small(schema, train, synthetic):
    res = evaluate_fidelity(train, synthetic, schema, n_boot=20)
    assert res.mean_ks.value < 0.08
    assert res.mean_tvd.value < 0.05
    assert res.correlation_delta_mean.value < 0.12
    assert res.mean_ks.low <= res.mean_ks.value <= res.mean_ks.high


def test_utility_target_selection_and_ratio(schema, train, holdout, synthetic):
    assert pick_target(schema) == ("readmitted_30d", "yes")
    assert pick_target(schema, "diabetes") == ("diabetes", "yes")
    with pytest.raises(ValueError):
        pick_target(schema, "age")
    res = evaluate_utility(train, holdout, synthetic, schema, target="readmitted_30d", n_boot=20)
    assert {m.model for m in res.models} == {"logistic_regression", "gradient_boosting"}
    for m in res.models:
        assert 0.5 < m.trtr_auc.value < 1.0
        assert m.ratio.low <= m.ratio.value <= m.ratio.high
        assert not m.degenerate


def test_utility_degenerate_synthetic_target_is_flagged(schema, train, holdout, synthetic):
    collapsed = synthetic.copy()
    collapsed["readmitted_30d"] = "no"
    res = evaluate_utility(train, holdout, collapsed, schema, target="readmitted_30d", n_boot=5)
    assert all(m.degenerate for m in res.models)
    assert all(abs(m.tstr_auc.value - 0.5) < 1e-9 for m in res.models)


def test_privacy_attacks_detect_a_copy_and_clear_an_independent_sample(schema, train, holdout, synthetic):
    copied = evaluate_privacy(train, holdout, train.copy(), schema, n_boot=10)
    assert copied.membership_auc.value > 0.95
    assert copied.exact_match_rate.value == 1.0
    assert copied.nndr_synthetic.value < 1e-6
    clean = evaluate_privacy(train, holdout, synthetic, schema, n_boot=10)
    assert abs(clean.membership_auc.value - 0.5) < 0.06
    assert clean.exact_match_rate.value == 0.0
    assert 0.3 < clean.dcr_share_train.value < 0.7
    assert clean.nndr_synthetic.value > 0.5


def test_membership_scores_and_nndr_shapes(schema, train, holdout):
    from synthia.transforms import MixedEncoder

    enc = MixedEncoder(schema).fit(train)
    labels, scores = membership_scores(enc.transform(train), enc.transform(holdout), enc.transform(holdout))
    assert labels.sum() == len(train) and len(scores) == len(train) + len(holdout)
    assert np.allclose(scores[len(train) :], 0.0, atol=1e-5)
    ratios = nndr(enc.transform(holdout), enc.transform(train))
    assert ratios.shape == (len(holdout),) and np.all((ratios >= 0) & (ratios <= 1 + 1e-9))
    assert np.all(nndr(enc.transform(holdout), enc.transform(train.head(1))) == 1.0)


def test_exact_match_flags_round_numeric_columns(schema, train):
    jittered = train.head(20).copy()
    jittered["bmi"] = jittered["bmi"] + 1e-6
    assert exact_match_flags(jittered, train, schema).all()
    shifted = train.head(20).copy()
    shifted["bmi"] = shifted["bmi"] + 1.0
    assert not exact_match_flags(shifted, train, schema).any()


def test_full_evaluate_headline_keys(schema, train, holdout, synthetic):
    res = evaluate(train, holdout, synthetic, schema, target="readmitted_30d", n_boot=5)
    head = res.headline()
    assert set(head) == {
        "mean_ks",
        "mean_tvd",
        "correlation_delta_mean",
        "utility_ratio",
        "membership_auc",
        "dcr_share_train",
        "nndr_synthetic",
        "exact_match_rate",
    }
    payload = res.to_dict()
    assert payload["n_train"] == len(train) and payload["n_boot"] == 5
    assert payload["fidelity"]["per_column"]["age"]["metric"] == "ks"


@settings(max_examples=30, deadline=None)
@given(
    a=st.lists(st.sampled_from(["x", "y", "z"]), min_size=1, max_size=30),
    b=st.lists(st.sampled_from(["x", "y", "z"]), min_size=1, max_size=30),
)
def test_total_variation_is_a_bounded_symmetric_distance(a, b):
    cats = ("x", "y", "z")
    d = total_variation(a, b, cats)
    assert 0.0 <= d <= 1.0 + 1e-12
    assert abs(d - total_variation(b, a, cats)) < 1e-12
    assert total_variation(a, a, cats) == 0.0
