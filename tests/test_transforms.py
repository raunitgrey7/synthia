import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthia.schema import ColumnKind, ColumnSpec, TableSchema
from synthia.transforms import Discretizer, MixedEncoder, one_hot


def test_one_hot_fixed_order_and_unknown_raises():
    out = one_hot(pd.Series(["b", "a", "b"], name="c"), ("a", "b"))
    assert out.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="unknown category"):
        one_hot(pd.Series(["z"], name="c"), ("a", "b"))


def test_discretizer_quantile_codes_cover_all_bins(schema, train):
    disc = Discretizer(schema, n_bins=8).fit(train)
    codes = disc.transform(train)
    assert codes.shape == (len(train), len(schema))
    assert codes.min() >= 0
    for j, card in enumerate(disc.cardinalities):
        assert codes[:, j].max() < card


def test_discretizer_uniform_ignores_data(schema, train):
    a = Discretizer(schema, n_bins=5, strategy="uniform").fit(train)
    b = Discretizer(schema, n_bins=5, strategy="uniform").fit(None)
    for name in schema.numeric_names:
        np.testing.assert_allclose(a.edges[name], b.edges[name])


def test_discretizer_inverse_stays_inside_bins(schema, train):
    disc = Discretizer(schema, n_bins=6, n_sub_bins=4).fit(train)
    codes = disc.transform(train)
    decoded = disc.inverse(codes, np.random.default_rng(0))
    recoded = disc.transform(decoded)
    numeric = [j for j, c in enumerate(schema.columns) if c.kind is ColumnKind.CONTINUOUS]
    assert (codes[:, numeric] == recoded[:, numeric]).all()
    categorical = [j for j, c in enumerate(schema.columns) if not c.is_numeric]
    assert (codes[:, categorical] == recoded[:, categorical]).all()


def test_discretizer_profile_shapes_and_validation(schema, train):
    disc = Discretizer(schema, n_bins=6, n_sub_bins=4).fit(train)
    assert disc.profiles["bmi"].shape == (24,)
    assert disc.profiles["bmi"].sum() == len(train)
    with pytest.raises(ValueError, match="wrong length"):
        disc.set_profile("bmi", np.ones(5))
    disc.set_profile("bmi", -np.ones(24))
    assert (disc.profiles["bmi"] == 0).all()


def test_discretizer_requires_fit_and_valid_args(schema):
    with pytest.raises(RuntimeError):
        Discretizer(schema).transform(pd.DataFrame())
    with pytest.raises(ValueError):
        Discretizer(schema, n_bins=1)
    with pytest.raises(ValueError):
        Discretizer(schema, strategy="odd")


def test_mixed_encoder_width_and_scaling(schema, train):
    enc = MixedEncoder(schema).fit(train)
    x = enc.transform(train)
    assert x.shape == (len(train), enc.width)
    numeric_idx = 0
    assert abs(x[:, numeric_idx].mean()) < 1e-9 and abs(x[:, numeric_idx].std() - 1) < 1e-9
    one_hot_block = x[:, enc.width - 2 :]
    assert np.allclose(one_hot_block.sum(axis=1), 1 / np.sqrt(2))


@settings(max_examples=30, deadline=None)
@given(values=st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=5, max_size=60))
def test_discretizer_codes_are_monotone_in_value(values):
    spec = TableSchema(columns=(ColumnSpec("v", ColumnKind.CONTINUOUS, lower=0, upper=100),))
    frame = pd.DataFrame({"v": values})
    disc = Discretizer(spec, n_bins=4).fit(frame)
    codes = disc.transform(frame)[:, 0]
    order = np.argsort(values, kind="stable")
    assert np.all(np.diff(codes[order]) >= 0)
