import numpy as np
import pandas as pd
import pytest

from synthia.generators.bayesnet import (
    BayesianNetworkGenerator,
    maximum_spanning_tree,
    mutual_information,
    orient_tree,
)
from synthia.privacy import DPConfig


def test_mutual_information_known_values():
    independent = np.array([[25.0, 25.0], [25.0, 25.0]])
    assert abs(mutual_information(independent)) < 1e-12
    dependent = np.array([[50.0, 0.0], [0.0, 50.0]])
    assert abs(mutual_information(dependent) - np.log(2)) < 1e-12
    assert mutual_information(np.zeros((2, 2))) == 0.0


def test_spanning_tree_has_d_minus_one_edges_and_prefers_heavy_edges():
    weights = np.array([[0, 5, 1, 1], [5, 0, 4, 1], [1, 4, 0, 3], [1, 1, 3, 0]], dtype=float)
    edges = maximum_spanning_tree(weights)
    assert len(edges) == 3
    assert set(edges) == {(0, 1), (1, 2), (2, 3)}
    order, parents = orient_tree(edges, 4, root=0)
    assert order[0] == 0 and parents[0] is None
    assert all(order.index(parents[n]) < order.index(n) for n in range(1, 4))


def test_structure_is_acyclic_and_parent_counts_bounded(schema, train):
    gen = BayesianNetworkGenerator(schema, seed=0).fit(train)
    position = {node: k for k, node in enumerate(gen.order)}
    for node, parents in enumerate(gen.parents):
        assert len(parents) <= 2
        assert all(position[p] < position[node] for p in parents)
    names = set(gen.structure())
    assert names == set(schema.names)


def test_recovers_strong_documented_dependencies(schema, train):
    gen = BayesianNetworkGenerator(schema, seed=0).fit(train)
    structure = gen.structure()
    linked = {frozenset((child, parent)) for child, parents in structure.items() for parent in parents}
    assert frozenset(("systolic_bp", "diastolic_bp")) in linked
    assert frozenset(("glucose", "diabetes")) in linked


def test_tables_are_normalised(schema, train):
    gen = BayesianNetworkGenerator(schema, seed=0).fit(train)
    for table in gen.tables:
        np.testing.assert_allclose(table.sum(axis=1), 1.0)
        assert (table > 0).all()


def test_sample_conforms_and_is_deterministic(schema, train):
    gen = BayesianNetworkGenerator(schema, seed=2).fit(train)
    a = gen.sample(300, seed=4)
    b = gen.sample(300, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert schema.validate(a) == []
    other = BayesianNetworkGenerator(schema, seed=2).fit(train).sample(300, seed=4)
    pd.testing.assert_frame_equal(a, other)


def test_conditional_mean_is_preserved(schema, train):
    gen = BayesianNetworkGenerator(schema, seed=0).fit(train)
    syn = gen.sample(4000, seed=1)
    real = train.groupby("diabetes", observed=True)["glucose"].mean()
    fake = syn.groupby("diabetes", observed=True)["glucose"].mean()
    assert abs(real["yes"] - fake["yes"]) < 20
    assert abs(real["no"] - fake["no"]) < 5


def test_dp_defaults_shrink_the_model_and_report_budget(schema, train):
    gen = BayesianNetworkGenerator(schema, seed=0, dp=DPConfig(2.0)).fit(train)
    assert gen.max_parents == 1 and gen.n_bins == 6
    assert gen.discretizer.strategy == "uniform"
    report = gen.privacy_report
    assert 1.99 < report.epsilon_spent <= 2.0 + 1e-9
    assert len(report.queries) == 3
    assert schema.validate(gen.sample(200, seed=0)) == []


def test_invalid_hyperparameters(schema):
    with pytest.raises(ValueError):
        BayesianNetworkGenerator(schema, max_parents=3)
    with pytest.raises(ValueError):
        BayesianNetworkGenerator(schema, pseudocount=0)
