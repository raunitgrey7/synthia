import numpy as np
import pandas as pd
import pytest

from synthia.generators.copula import (
    CategoricalMarginal,
    GaussianCopulaGenerator,
    NumericMarginal,
    nearest_correlation,
)
from synthia.privacy import DPConfig
from synthia.schema import ColumnKind, ColumnSpec


def test_sample_conforms_to_schema(copula, schema, synthetic, train):
    assert schema.validate(synthetic) == []
    assert len(synthetic) == len(train)
    for col in schema.columns:
        if col.is_numeric:
            assert synthetic[col.name].between(col.lower, col.upper).all()
        else:
            assert set(synthetic[col.name]) <= set(col.categories)


def test_same_seed_same_sample(copula):
    a = copula.sample(200, seed=5)
    b = copula.sample(200, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(copula.sample(200, seed=6))


def test_refit_is_deterministic(schema, train):
    a = GaussianCopulaGenerator(schema, seed=3).fit(train)
    b = GaussianCopulaGenerator(schema, seed=3).fit(train)
    np.testing.assert_array_equal(a.correlation, b.correlation)
    pd.testing.assert_frame_equal(a.sample(100, seed=1), b.sample(100, seed=1))


def test_marginals_and_dependence_are_close(copula, train, synthetic):
    for col in ("age", "bmi", "glucose"):
        assert abs(train[col].mean() - synthetic[col].mean()) < 0.1 * train[col].std()
    real_corr = np.corrcoef(train["systolic_bp"], train["diastolic_bp"])[0, 1]
    syn_corr = np.corrcoef(synthetic["systolic_bp"], synthetic["diastolic_bp"])[0, 1]
    assert abs(real_corr - syn_corr) < 0.1
    rate_real = (train["diabetes"] == "yes").mean()
    rate_syn = (synthetic["diabetes"] == "yes").mean()
    assert abs(rate_real - rate_syn) < 0.04


def test_requires_fit_and_positive_rows(schema, copula):
    with pytest.raises(RuntimeError):
        GaussianCopulaGenerator(schema).sample(5)
    with pytest.raises(ValueError):
        copula.sample(0)


def test_dp_fit_reports_accounting_within_target(schema, train):
    gen = GaussianCopulaGenerator(schema, seed=0, dp=DPConfig(1.0)).fit(train)
    report = gen.privacy_report
    assert report is not None
    assert report.epsilon_spent <= 1.0 + 1e-9
    assert report.epsilon_spent > 0.99
    assert report.delta == 1e-5
    assert {q["mechanism"] for q in report.queries} == {
        "histograms and count",
        "second moment of clipped normal scores",
    }
    assert report.queries[0]["count"] == len(schema) + 1
    sample = gen.sample(300, seed=0)
    assert schema.validate(sample) == []


def test_more_privacy_means_more_noise(schema, train):
    tight = GaussianCopulaGenerator(schema, seed=0, dp=DPConfig(0.5)).fit(train)
    loose = GaussianCopulaGenerator(schema, seed=0, dp=DPConfig(8.0)).fit(train)
    assert tight.privacy_report.noise_multiplier > loose.privacy_report.noise_multiplier
    exact = GaussianCopulaGenerator(schema, seed=0).fit(train).correlation
    err_tight = np.abs(tight.correlation - exact).mean()
    err_loose = np.abs(loose.correlation - exact).mean()
    assert err_loose < err_tight


def test_inferred_bounds_caveat_is_reported(train):
    from synthia.schema import TableSchema

    inferred = TableSchema.infer(train)
    gen = GaussianCopulaGenerator(inferred, seed=0, dp=DPConfig(2.0)).fit(train)
    assert any("inferred" in c for c in gen.privacy_report.caveats)


def test_numeric_marginal_roundtrip_and_histogram_form():
    spec = ColumnSpec("x", ColumnKind.CONTINUOUS, lower=0, upper=10)
    values = np.random.default_rng(0).uniform(0, 10, 500)
    marg = NumericMarginal.from_sample(values, 16, spec)
    u = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(marg.cdf(marg.quantile(u)), u, atol=1e-9)
    hist = NumericMarginal.from_histogram(np.array([1.0, 1.0, 2.0, 0.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert hist.quantile(np.array([0.0, 0.5, 1.0])).tolist() == [0.0, 2.0, 4.0]


def test_categorical_marginal_scores_and_decoding():
    marg = CategoricalMarginal(np.array([3.0, 1.0]))
    scores = marg.scores(np.array([0, 1]))
    assert scores[0] < 0 < scores[1]
    assert abs(0.75 * scores[0] + 0.25 * scores[1]) < 1e-9
    assert marg.decode(np.array([0.1, 0.9])).tolist() == [0, 1]
    uniform = CategoricalMarginal(np.zeros(3))
    np.testing.assert_allclose(uniform.probs, 1 / 3)


def test_nearest_correlation_repairs_indefinite_matrix():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = nearest_correlation(bad)
    assert np.all(np.linalg.eigvalsh(fixed) > 0)
    np.testing.assert_allclose(np.diag(fixed), 1.0)


def test_constant_numeric_column_does_not_break_fit():
    from synthia.schema import TableSchema

    schema = TableSchema(
        columns=(
            ColumnSpec("x", ColumnKind.CONTINUOUS, lower=0, upper=10),
            ColumnSpec("c", ColumnKind.CATEGORICAL, categories=("a", "b")),
        )
    )
    frame = pd.DataFrame({"x": np.full(100, 5.0), "c": ["a", "b"] * 50})
    gen = GaussianCopulaGenerator(schema, seed=0).fit(frame)
    out = gen.sample(50, seed=0)
    assert (out["x"] == 5.0).all()
