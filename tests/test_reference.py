import numpy as np
import pandas as pd
import pytest

from synthia.reference import generate_patients, load_csv, patient_schema, train_holdout_split


def test_generation_is_deterministic():
    a = generate_patients(500, seed=3)
    b = generate_patients(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate_patients(500, seed=4))


def test_cohort_conforms_to_schema(cohort, schema):
    assert schema.validate(cohort) == []
    for col in schema.columns:
        if col.is_numeric:
            assert cohort[col.name].between(col.lower, col.upper).all()


def test_documented_dependencies_hold():
    df = generate_patients(6000, seed=0)
    assert np.corrcoef(df["age"], df["systolic_bp"])[0, 1] > 0.4
    assert np.corrcoef(df["systolic_bp"], df["diastolic_bp"])[0, 1] > 0.6
    glucose = df.groupby("diabetes", observed=True)["glucose"].mean()
    assert glucose["yes"] - glucose["no"] > 40
    readmit = df.groupby("cad", observed=True)["readmitted_30d"].apply(lambda s: (s == "yes").mean())
    assert readmit["yes"] > readmit["no"]
    hyper = df.groupby("hypertension", observed=True)["systolic_bp"].mean()
    assert hyper["yes"] > hyper["no"] + 10


def test_outcome_prevalences_are_plausible():
    df = generate_patients(6000, seed=1)
    for col, low, high in (("diabetes", 0.05, 0.25), ("cad", 0.03, 0.2), ("readmitted_30d", 0.05, 0.25)):
        rate = (df[col] == "yes").mean()
        assert low < rate < high, (col, rate)


def test_invalid_sizes_raise():
    with pytest.raises(ValueError):
        generate_patients(0)
    with pytest.raises(ValueError):
        train_holdout_split(generate_patients(10), 0.0, seed=0)
    with pytest.raises(ValueError, match="empty"):
        train_holdout_split(generate_patients(2), 0.1, seed=0)


def test_split_is_disjoint_and_seeded(cohort):
    a_train, a_hold = train_holdout_split(cohort, 0.25, seed=1)
    b_train, b_hold = train_holdout_split(cohort, 0.25, seed=1)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_hold, b_hold)
    assert len(a_train) + len(a_hold) == len(cohort)
    assert len(a_hold) == round(0.25 * len(cohort))


def test_csv_roundtrip_infers_schema(tmp_path, cohort):
    path = tmp_path / "patients.csv"
    cohort.head(300).to_csv(path, index=False)
    frame, schema = load_csv(path)
    assert len(frame) == 300
    assert schema.bounds_inferred is True
    assert schema.get("sex").categories == ("F", "M")
    frame2, _ = load_csv(path, schema=patient_schema())
    pd.testing.assert_frame_equal(frame2, patient_schema().coerce(cohort.head(300)))


def test_csv_with_missing_rows_drops_them(tmp_path, cohort):
    broken = cohort.head(50).copy()
    broken.loc[0, "bmi"] = np.nan
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    frame, _ = load_csv(path)
    assert len(frame) == 49
    with pytest.raises(ValueError, match="no complete rows"):
        empty = cohort.head(3).copy()
        empty["bmi"] = np.nan
        empty.to_csv(path, index=False)
        load_csv(path)
