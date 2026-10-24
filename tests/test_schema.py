import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthia.schema import ColumnKind, ColumnSpec, TableSchema


def test_categorical_spec_requires_categories():
    with pytest.raises(ValueError, match="needs at least one category"):
        ColumnSpec("c", ColumnKind.CATEGORICAL)


def test_numeric_spec_requires_ordered_bounds():
    with pytest.raises(ValueError, match="lower < upper"):
        ColumnSpec("x", ColumnKind.CONTINUOUS, lower=5, upper=1)
    with pytest.raises(ValueError, match="needs lower and upper"):
        ColumnSpec("x", ColumnKind.INTEGER, lower=0)


def test_duplicate_categories_and_columns_rejected():
    with pytest.raises(ValueError, match="duplicate categories"):
        ColumnSpec("c", ColumnKind.CATEGORICAL, categories=("a", "a"))
    spec = ColumnSpec("x", ColumnKind.CONTINUOUS, lower=0, upper=1)
    with pytest.raises(ValueError, match="duplicate column names"):
        TableSchema(columns=(spec, spec))


def test_schema_roundtrip_through_dict(schema):
    rebuilt = TableSchema.from_dict(schema.to_dict())
    assert rebuilt == schema
    assert rebuilt.names == schema.names
    assert "age" in rebuilt and "nope" not in rebuilt


def test_infer_detects_kinds_and_flags_bounds():
    frame = pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": [1, 2, 3], "c": ["x", "y", "x"], "d": [True, False, True]})
    inferred = TableSchema.infer(frame)
    kinds = {c.name: c.kind for c in inferred.columns}
    assert kinds == {
        "a": ColumnKind.CONTINUOUS,
        "b": ColumnKind.INTEGER,
        "c": ColumnKind.CATEGORICAL,
        "d": ColumnKind.CATEGORICAL,
    }
    assert inferred.bounds_inferred is True
    assert inferred.get("a").lower == 1.5 and inferred.get("a").upper == 3.5


def test_infer_rejects_high_cardinality_strings():
    frame = pd.DataFrame({"id": [f"u{i}" for i in range(50)]})
    with pytest.raises(ValueError, match="too many"):
        TableSchema.infer(frame, max_categories=20)


def test_with_bounds_clears_inferred_flag_when_all_numeric_bounds_declared():
    frame = pd.DataFrame({"a": [1.0, 2.0], "c": ["x", "y"]})
    inferred = TableSchema.infer(frame)
    declared = inferred.with_bounds({"a": (0.0, 10.0)})
    assert declared.bounds_inferred is False
    assert declared.get("a").span == 10.0


def test_validate_reports_missing_unknown_and_nan(schema):
    frame = pd.DataFrame({"age": [30, np.nan], "sex": ["F", "Z"]})
    problems = schema.validate(frame)
    assert any("missing column" in p for p in problems)
    assert any("missing values" in p for p in problems)
    assert any("unknown categories" in p for p in problems)


def test_coerce_casts_types_and_orders_columns(schema, cohort):
    shuffled = cohort[list(reversed(cohort.columns))].copy()
    shuffled["age"] = shuffled["age"].astype(float)
    out = schema.coerce(shuffled)
    assert list(out.columns) == schema.names
    assert out["age"].dtype == np.int64
    assert out["bmi"].dtype == np.float64
    assert out["sex"].dtype == object


def test_coerce_raises_on_nonconforming(schema):
    with pytest.raises(ValueError, match="missing column"):
        schema.coerce(pd.DataFrame({"age": [1]}))


@settings(max_examples=25, deadline=None)
@given(
    values=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=2, max_size=40),
    labels=st.lists(st.sampled_from(["a", "b", "c"]), min_size=2, max_size=40),
)
def test_infer_then_coerce_is_idempotent(values, labels):
    n = min(len(values), len(labels))
    frame = pd.DataFrame({"v": values[:n], "l": labels[:n]})
    inferred = TableSchema.infer(frame)
    once = inferred.coerce(frame)
    twice = inferred.coerce(once)
    pd.testing.assert_frame_equal(once, twice)
    assert inferred.validate(once) == []
