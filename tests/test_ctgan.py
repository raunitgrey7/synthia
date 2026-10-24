import numpy as np
import pandas as pd
import pytest
import torch

from synthia.generators.ctgan import CTGANGenerator, apply_activations, condition_loss
from synthia.generators.ctgan_transform import DataTransformer, FrequencySampler, condition_vectors
from synthia.privacy import DPConfig

SMALL = {
    "epochs": 2,
    "batch_size": 60,
    "pac": 5,
    "embedding_dim": 16,
    "generator_dim": (32,),
    "discriminator_dim": (32,),
    "torch_threads": 2,
}


@pytest.fixture(scope="module")
def small_train(train):
    return train.head(300)


@pytest.fixture(scope="module")
def fitted(schema, small_train):
    return CTGANGenerator(schema, seed=0, **SMALL).fit(small_train)


def test_transformer_roundtrip_recovers_categories_and_bounds(schema, small_train):
    transformer = DataTransformer(schema, max_modes=3, seed=0).fit(small_train)
    data = transformer.transform(small_train, np.random.default_rng(0))
    assert data.shape == (len(small_train), transformer.output_dim)
    back = transformer.inverse(data)
    for name in schema.categorical_names:
        assert back[name].tolist() == small_train[name].astype(str).tolist()
    for col in schema.columns:
        if col.is_numeric:
            assert back[col.name].between(col.lower, col.upper).all()
            assert np.corrcoef(back[col.name], small_train[col.name])[0, 1] > 0.95


def test_public_scaling_transformer_needs_no_data(schema, small_train):
    transformer = DataTransformer(schema, public_scaling=True).fit(None)
    assert transformer.output_dim == len(schema.numeric_names) + sum(
        len(c.categories) for c in schema.columns if not c.is_numeric
    )
    data = transformer.transform(small_train, np.random.default_rng(0))
    assert np.all(data[:, 0] >= -1) and np.all(data[:, 0] <= 1)
    back = transformer.inverse(data)
    np.testing.assert_allclose(back["bmi"], small_train["bmi"], atol=1e-3)


def test_condition_vectors_and_frequency_sampler(schema, small_train):
    transformer = DataTransformer(schema, public_scaling=True).fit(None)
    cond = condition_vectors(transformer, np.array([0, 1]), np.array([1, 2]))
    assert cond.shape == (2, transformer.cond_dim)
    assert cond.sum(axis=1).tolist() == [1.0, 1.0]
    sampler = FrequencySampler(
        transformer, [np.array([0.0, 1.0]), *[np.ones(s.size) for s in transformer.cond_spans[1:]]]
    )
    matrix, cols, cats = sampler.sample(50, np.random.default_rng(0))
    assert matrix.shape == (50, transformer.cond_dim)
    assert all(cats[cols == 0] == 1)


def test_activations_and_condition_loss(schema):
    transformer = DataTransformer(schema, public_scaling=True).fit(None)
    raw = torch.randn(8, transformer.output_dim)
    out = apply_activations(raw, transformer.spans, 0.2, torch.Generator().manual_seed(0))
    assert out.shape == raw.shape
    for span in transformer.cond_spans:
        block = out[:, span.start : span.start + span.size]
        torch.testing.assert_close(block.sum(dim=1), torch.ones(8))
    loss = condition_loss(raw, transformer.cond_spans, np.zeros(8, dtype=int), np.zeros(8, dtype=int))
    assert loss.item() > 0
    assert condition_loss(raw, [], np.zeros(8, dtype=int), np.zeros(8, dtype=int)).item() == 0


def test_fit_runs_and_samples_conform(fitted, schema):
    assert fitted.epochs_completed == 2
    assert fitted.steps_completed == 2 * (300 // 60)
    sample = fitted.sample(120, seed=1)
    assert schema.validate(sample) == []
    assert len(sample) == 120


def test_sampling_is_seeded_and_refit_is_deterministic(fitted, schema, small_train):
    a = fitted.sample(50, seed=9)
    b = fitted.sample(50, seed=9)
    pd.testing.assert_frame_equal(a, b)
    again = CTGANGenerator(schema, seed=0, **SMALL).fit(small_train)
    pd.testing.assert_frame_equal(again.sample(50, seed=9), a)


def test_conditioned_sampling_validates_inputs(fitted):
    out = fitted.sample_conditioned(40, "sex", "F", seed=0)
    assert len(out) == 40
    with pytest.raises(ValueError, match="not a categorical column"):
        fitted.sample_conditioned(5, "age", "1")
    with pytest.raises(ValueError, match="not a category"):
        fitted.sample_conditioned(5, "sex", "X")


def test_dp_fit_uses_pac_one_and_accounts_every_step(schema, small_train):
    gen = CTGANGenerator(schema, seed=0, dp=DPConfig(5.0), **{**SMALL, "pac": 10}).fit(small_train)
    assert gen.pac == 1
    report = gen.privacy_report
    assert report.epsilon_spent <= 5.0 + 1e-9
    sgd = next(q for q in report.queries if q["mechanism"] == "dp-sgd discriminator")
    assert sgd["steps"] == gen.steps_completed
    assert sgd["sampling_rate"] == pytest.approx(60 / 300)
    assert any("clipped" in c for c in report.caveats)
    assert schema.validate(gen.sample(30, seed=0)) == []


def test_per_sample_gradients_are_clipped(schema, small_train):
    gen = CTGANGenerator(schema, seed=0, dp=DPConfig(5.0), **SMALL)
    gen.fit(small_train)
    disc = gen.discriminator_net
    data = gen.transformer.transform(small_train.head(20), np.random.default_rng(0))
    cols, cats = gen._conditions_of(data)
    cond = condition_vectors(gen.transformer, cols, cats)
    real = torch.cat([torch.as_tensor(data), torch.as_tensor(cond)], dim=1)
    fake = torch.randn_like(real)
    grads = gen._per_sample_grads(real, fake)
    assert next(iter(grads.values())).shape[0] == 20
    gen._apply_clipped_noisy_grads(grads, expected_batch=20.0)
    assert all(p.grad is not None for p in disc.parameters())
    assert 0.0 <= gen._last_clip_fraction <= 1.0
