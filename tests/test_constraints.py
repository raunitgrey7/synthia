import pytest

from synthia.constraints import Constraint, ConstraintError, sample_with_constraints
from synthia.generators.ctgan import CTGANGenerator


def test_constraint_validation_against_schema(schema):
    Constraint("age", minimum=30).validate(schema)
    with pytest.raises(ConstraintError, match="unknown column"):
        Constraint("zzz", minimum=1).validate(schema)
    with pytest.raises(ConstraintError, match="restricts nothing"):
        Constraint("age").validate(schema)
    with pytest.raises(ConstraintError, match="one_of applies"):
        Constraint("age", one_of=("1",)).validate(schema)
    with pytest.raises(ConstraintError, match="ranges apply"):
        Constraint("sex", minimum=1).validate(schema)
    with pytest.raises(ConstraintError, match="unknown categories"):
        Constraint("sex", equals="X").validate(schema)
    with pytest.raises(ConstraintError, match="exceeds"):
        Constraint("age", minimum=5, maximum=1).validate(schema)


def test_masks_apply_each_clause(cohort):
    assert Constraint("age", minimum=60, maximum=70).mask(cohort).sum() == cohort["age"].between(60, 70).sum()
    assert Constraint("sex", one_of=("F",)).mask(cohort).sum() == (cohort["sex"] == "F").sum()
    assert Constraint("length_of_stay", equals=0).mask(cohort).sum() == (cohort["length_of_stay"] == 0).sum()


def test_sampling_satisfies_constraints_and_reports_rate(copula):
    constraints = [
        Constraint("age", minimum=60, maximum=75),
        Constraint("sex", equals="F"),
        Constraint("smoker", one_of=("never", "former")),
    ]
    frame, info = sample_with_constraints(copula, 120, constraints, seed=0)
    assert len(frame) == 120
    assert frame["age"].between(60, 75).all()
    assert (frame["sex"] == "F").all()
    assert frame["smoker"].isin(["never", "former"]).all()
    assert 0 < info.acceptance_rate < 1 and info.rounds >= 1
    assert info.to_dict()["constraints"][0] == {"column": "age", "minimum": 60, "maximum": 75}


def test_sampling_is_seeded(copula):
    a, _ = sample_with_constraints(copula, 30, [Constraint("sex", equals="M")], seed=3)
    b, _ = sample_with_constraints(copula, 30, [Constraint("sex", equals="M")], seed=3)
    assert a.equals(b)


def test_unconstrained_request_is_a_plain_sample(copula):
    frame, info = sample_with_constraints(copula, 25, [], seed=1)
    assert len(frame) == 25 and info.acceptance_rate == 1.0 and info.rounds == 1


def test_impossible_constraints_fail_fast(copula):
    with pytest.raises(ConstraintError, match="too selective"):
        sample_with_constraints(copula, 10, [Constraint("age", minimum=200)], seed=0, max_rounds=3)
    with pytest.raises(ValueError):
        sample_with_constraints(copula, 0, [], seed=0)


def test_ctgan_pushes_categorical_equality_into_the_condition(schema, train):
    gen = CTGANGenerator(
        schema,
        seed=0,
        epochs=1,
        batch_size=50,
        pac=5,
        embedding_dim=8,
        generator_dim=(16,),
        discriminator_dim=(16,),
        torch_threads=2,
    )
    gen.fit(train.head(200))
    frame, info = sample_with_constraints(
        gen, 20, [Constraint("sex", equals="F"), Constraint("age", minimum=18)], seed=0
    )
    assert info.conditioned_on == {"sex": "F"}
    assert (frame["sex"] == "F").all() and len(frame) == 20
