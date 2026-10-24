"""CTGAN-style conditional GAN in PyTorch, CPU only, with an optional DP-SGD discriminator.

The model follows Xu et al. (2019): mode-specific normalisation, a condition vector with training-by-sampling,
a residual generator, a packed (PAC) discriminator with the WGAN-GP objective and a cross-entropy term that
makes the generator honour its condition. The sizes and epoch count are chosen so a fit stays within a few
minutes on a laptop CPU; the benchmark records the epochs actually run and the wall time.

Under DP only the discriminator sees records, so it is the only network trained with DP-SGD: Poisson sampling of
rows, per-record gradients through ``torch.func`` (vmap over grad), clipping to ``clip_norm`` and Gaussian noise
with multiplier ``noise_multiplier``. The condition-frequency histograms get ``stats_noise_weight`` times that
multiplier: their counts are in the thousands, so the extra noise is harmless, while a full-batch query at the
SGD multiplier would dominate the budget. The gradient penalty and packing are dropped in that mode (PAC would make a
record part of several discriminator inputs) and the plain non-saturating GAN loss is used instead. The generator
update only touches the discriminator, so it is post-processing. Condition frequencies come from noisy histograms
and numeric scaling uses the public bounds, so the accountant covers every data access.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as fn

from synthia.generators.base import Generator
from synthia.generators.ctgan_transform import (
    ConditionSampler,
    DataTransformer,
    FrequencySampler,
    Span,
    condition_vectors,
)
from synthia.privacy import Accountant, DPConfig, PrivacyReport, calibrate_noise_multiplier
from synthia.privacy.mechanisms import noisy_histogram
from synthia.schema import TableSchema


class Residual(nn.Module):
    """Linear, batch norm, ReLU, then concatenate the input (the CTGAN generator block)."""

    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(inp, out)
        self.bn = nn.BatchNorm1d(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn.relu(self.bn(self.fc(x))), x], dim=1)


class GeneratorNet(nn.Module):
    def __init__(self, embedding_dim: int, dims: tuple[int, ...], data_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = embedding_dim
        for d in dims:
            layers.append(Residual(width, d))
            width += d
        layers.append(nn.Linear(width, data_dim))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DiscriminatorNet(nn.Module):
    """MLP critic that scores ``pac`` rows at once."""

    def __init__(self, input_dim: int, dims: tuple[int, ...], pac: int) -> None:
        super().__init__()
        self.pac = pac
        layers: list[nn.Module] = []
        width = input_dim * pac
        for d in dims:
            layers += [nn.Linear(width, d), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            width = d
        layers.append(nn.Linear(width, 1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x.reshape(-1, x.shape[1] * self.pac))


def gumbel_softmax(logits: torch.Tensor, tau: float, gen: torch.Generator) -> torch.Tensor:
    u = torch.rand(logits.shape, generator=gen).clamp_(1e-10, 1.0)
    return torch.softmax((logits - torch.log(-torch.log(u))) / tau, dim=-1)


def apply_activations(raw: torch.Tensor, spans: list[Span], tau: float, gen: torch.Generator) -> torch.Tensor:
    """tanh on numeric scalars, Gumbel softmax on mode indicators and categories."""
    parts = []
    for span in spans:
        block = raw[:, span.start : span.start + span.size]
        if span.activation == "numeric":
            parts.append(torch.tanh(block[:, :1]))
            if span.size > 1:
                parts.append(gumbel_softmax(block[:, 1:], tau, gen))
        else:
            parts.append(gumbel_softmax(block, tau, gen))
    return torch.cat(parts, dim=1)


def gradient_penalty(
    disc: DiscriminatorNet, real: torch.Tensor, fake: torch.Tensor, gen: torch.Generator
) -> torch.Tensor:
    pac = disc.pac
    alpha = (
        torch.rand(real.shape[0] // pac, 1, 1, generator=gen).repeat(1, pac, real.shape[1]).reshape(-1, real.shape[1])
    )
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = disc(mixed)
    grads = torch.autograd.grad(score, mixed, grad_outputs=torch.ones_like(score), create_graph=True)[0]
    norms = grads.reshape(-1, pac * real.shape[1]).norm(2, dim=1)
    return ((norms - 1.0) ** 2).mean() * 10.0


def condition_loss(raw: torch.Tensor, cond_spans: list[Span], col_idx: np.ndarray, cat_idx: np.ndarray) -> torch.Tensor:
    """Cross-entropy between the generated block of the conditioned column and the requested category."""
    if not cond_spans:
        return raw.new_zeros(())
    total = raw.new_zeros(())
    cols = torch.as_tensor(col_idx)
    cats = torch.as_tensor(cat_idx)
    for j, span in enumerate(cond_spans):
        mask = cols == j
        if mask.any():
            logits = raw[mask][:, span.start : span.start + span.size]
            total = total + fn.cross_entropy(logits, cats[mask], reduction="sum")
    return total / raw.shape[0]


class CTGANGenerator(Generator):
    """Conditional tabular GAN; see the module docstring for the training recipe and the DP variant."""

    name = "ctgan"

    def __init__(
        self,
        schema: TableSchema,
        dp: DPConfig | None = None,
        seed: int = 0,
        *,
        embedding_dim: int = 128,
        generator_dim: tuple[int, ...] = (256, 256),
        discriminator_dim: tuple[int, ...] = (256, 256),
        batch_size: int = 500,
        epochs: int = 100,
        pac: int = 10,
        lr: float = 2e-4,
        max_modes: int = 5,
        clip_norm: float = 1.0,
        stats_noise_weight: float = 20.0,
        tau: float = 0.2,
        max_seconds: float | None = None,
        torch_threads: int = 4,
    ) -> None:
        super().__init__(schema, dp=dp, seed=seed)
        if epochs < 1 or batch_size < 2 or pac < 1:
            raise ValueError("epochs and batch_size and pac must be positive")
        self.embedding_dim = embedding_dim
        self.generator_dim = tuple(generator_dim)
        self.discriminator_dim = tuple(discriminator_dim)
        self.pac = 1 if dp is not None else pac
        self.batch_size = max(self.pac, (batch_size // self.pac) * self.pac)
        self.epochs = epochs
        self.lr = lr
        self.max_modes = max_modes
        self.clip_norm = clip_norm
        self.stats_noise_weight = stats_noise_weight
        self.tau = tau
        self.max_seconds = max_seconds
        self.torch_threads = torch_threads
        self.transformer = DataTransformer(schema, max_modes=max_modes, public_scaling=dp is not None, seed=seed)
        self.freq_sampler: FrequencySampler | None = None
        self.generator_net: GeneratorNet | None = None
        self.discriminator_net: DiscriminatorNet | None = None
        self.epochs_completed = 0
        self.steps_completed = 0
        self.train_seconds = 0.0
        self.loss_history: list[tuple[float, float]] = []

    # Fitting -----------------------------------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        torch.set_num_threads(self.torch_threads)
        torch.manual_seed(self.seed)
        self._torch_gen = torch.Generator().manual_seed(self.seed)
        self.transformer.fit(None if self.dp is not None else df)
        data = self.transformer.transform(df, self._rng)
        self.generator_net = GeneratorNet(
            self.embedding_dim + self.transformer.cond_dim, self.generator_dim, data.shape[1]
        )
        self.discriminator_net = DiscriminatorNet(
            data.shape[1] + self.transformer.cond_dim, self.discriminator_dim, self.pac
        )
        started = time.perf_counter()
        if self.dp is None:
            sampler = ConditionSampler(self.transformer, data)
            self.freq_sampler = FrequencySampler(self.transformer, sampler.frequencies)
            self._train(data, lambda: self._disc_step_wgan(data, sampler))
        else:
            self._fit_dp(data)
        self.train_seconds = time.perf_counter() - started

    def _steps_per_epoch(self, n: int) -> int:
        return max(n // self.batch_size, 1)

    def _train(self, data: np.ndarray, disc_step: Callable[[], float]) -> None:
        assert self.generator_net is not None and self.discriminator_net is not None
        opt_g = torch.optim.Adam(self.generator_net.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
        opt_d = torch.optim.Adam(self.discriminator_net.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
        self._opt_d = opt_d
        started = time.perf_counter()
        for _epoch in range(self.epochs):
            for _step in range(self._steps_per_epoch(len(data))):
                loss_d = disc_step()
                loss_g = self._gen_step(opt_g)
                self.steps_completed += 1
            self.loss_history.append((loss_d, loss_g))
            self.epochs_completed += 1
            if self.max_seconds is not None and time.perf_counter() - started > self.max_seconds:
                break

    def _fake_batch(self, cond: np.ndarray, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw generator output and its activated version for ``n`` rows under the given conditions."""
        assert self.generator_net is not None
        noise = torch.randn(n, self.embedding_dim, generator=self._torch_gen)
        cond_t = torch.as_tensor(cond)
        raw = self.generator_net(torch.cat([noise, cond_t], dim=1))
        return raw, torch.cat(
            [apply_activations(raw, self.transformer.spans, self.tau, self._torch_gen), cond_t], dim=1
        )

    def _disc_step_wgan(self, data: np.ndarray, sampler: ConditionSampler) -> float:
        assert self.discriminator_net is not None
        cond, _cols, _cats, rows = sampler.sample(self.batch_size, self._rng)
        perm = self._rng.permutation(self.batch_size)
        real = torch.cat([torch.as_tensor(data[rows[perm]]), torch.as_tensor(cond[perm])], dim=1)
        _raw, fake = self._fake_batch(cond, self.batch_size)
        y_fake = self.discriminator_net(fake.detach())
        y_real = self.discriminator_net(real)
        loss = -(y_real.mean() - y_fake.mean()) + gradient_penalty(
            self.discriminator_net, real, fake.detach(), self._torch_gen
        )
        self._opt_d.zero_grad(set_to_none=False)
        loss.backward()
        self._opt_d.step()
        return float(loss.detach())

    def _gen_step(self, opt_g: torch.optim.Optimizer) -> float:
        assert self.discriminator_net is not None and self.freq_sampler is not None
        cond, cols, cats = self.freq_sampler.sample(self.batch_size, self._rng)
        raw, fake = self._fake_batch(cond, self.batch_size)
        score = self.discriminator_net(fake)
        adv = -score.mean() if self.dp is None else fn.softplus(-score).mean()
        loss = adv + condition_loss(raw, self.transformer.cond_spans, cols, cats)
        opt_g.zero_grad(set_to_none=False)
        loss.backward()
        opt_g.step()
        return float(loss.detach())

    # DP-SGD --------------------------------------------------------------------------------------------------

    def _accountant(self, sigma: float, n: int, steps: int) -> Accountant:
        acc = Accountant()
        if self.transformer.cond_spans:
            acc.add_gaussian(
                sigma * self.stats_noise_weight,
                count=len(self.transformer.cond_spans),
                label="condition frequencies",
            )
        acc.add_sampled_gaussian(min(self.batch_size / n, 1.0), sigma, steps, label="dp-sgd discriminator")
        return acc

    def _fit_dp(self, data: np.ndarray) -> None:
        assert self.dp is not None
        n = len(data)
        planned = self.epochs * self._steps_per_epoch(n)
        sigma = calibrate_noise_multiplier(self.dp.epsilon, self.dp.delta, lambda s: self._accountant(s, n, planned))
        frequencies = []
        for span in self.transformer.cond_spans:
            counts = data[:, span.start : span.start + span.size].sum(axis=0)
            frequencies.append(noisy_histogram(counts, sigma * self.stats_noise_weight, self._rng))
        self.freq_sampler = FrequencySampler(self.transformer, frequencies)
        self._sigma = sigma
        self._train(data, lambda: self._disc_step_dp(data))
        acc = self._accountant(sigma, n, max(self.steps_completed, 1))
        self.privacy_report = PrivacyReport(
            epsilon_target=self.dp.epsilon,
            epsilon_spent=acc.epsilon(self.dp.delta),
            delta=self.dp.delta,
            noise_multiplier=sigma,
            rdp_order=acc.best_order(self.dp.delta),
            queries=acc.queries,
            caveats=[*self._dp_caveats(), f"per-record gradients clipped to L2 norm {self.clip_norm}"],
        )

    def _disc_step_dp(self, data: np.ndarray) -> float:
        assert self.discriminator_net is not None
        n = len(data)
        q = min(self.batch_size / n, 1.0)
        idx = np.flatnonzero(self._rng.random(n) < q)
        if len(idx) == 0:
            return float("nan")
        real_rows = data[idx]
        cols, cats = self._conditions_of(real_rows)
        cond = condition_vectors(self.transformer, cols, cats)
        real = torch.cat([torch.as_tensor(real_rows), torch.as_tensor(cond)], dim=1)
        _raw, fake = self._fake_batch(cond, len(idx))
        per_sample = self._per_sample_grads(real, fake.detach())
        self._apply_clipped_noisy_grads(per_sample, expected_batch=q * n)
        self._opt_d.step()
        with torch.no_grad():
            yr, yf = self.discriminator_net(real), self.discriminator_net(fake)
            return float((fn.softplus(-yr) + fn.softplus(yf)).mean())

    def _conditions_of(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each sampled record conditions on one of its own categories (a per-record function, so DP-safe)."""
        spans = self.transformer.cond_spans
        if not spans:
            return np.zeros(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
        cols = self._rng.integers(len(spans), size=len(rows))
        cats = np.empty(len(rows), dtype=np.int64)
        for j, span in enumerate(spans):
            mask = cols == j
            cats[mask] = np.argmax(rows[mask][:, span.start : span.start + span.size], axis=1)
        return cols, cats

    def _per_sample_grads(self, real: torch.Tensor, fake: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.discriminator_net is not None
        disc = self.discriminator_net
        params = {k: v.detach() for k, v in disc.named_parameters()}

        def loss_fn(p: dict[str, torch.Tensor], r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
            yr = functional_call(disc, p, (r.unsqueeze(0),))
            yf = functional_call(disc, p, (f.unsqueeze(0),))
            return (fn.softplus(-yr) + fn.softplus(yf)).squeeze()

        return vmap(grad(loss_fn), in_dims=(None, 0, 0), randomness="different")(params, real, fake)

    def _apply_clipped_noisy_grads(self, per_sample: dict[str, torch.Tensor], expected_batch: float) -> None:
        assert self.discriminator_net is not None
        squared = torch.zeros(next(iter(per_sample.values())).shape[0])
        for g in per_sample.values():
            squared = squared + g.flatten(1).pow(2).sum(1)
        scale = torch.clamp(self.clip_norm / (squared.sqrt() + 1e-6), max=1.0)
        self._last_clip_fraction = float((scale < 1.0).float().mean())
        for name, param in self.discriminator_net.named_parameters():
            g = per_sample[name]
            summed = (g * scale.view(-1, *([1] * (g.dim() - 1)))).sum(0)
            noise = torch.randn(param.shape, generator=self._torch_gen) * (self._sigma * self.clip_norm)
            param.grad = (summed + noise) / expected_batch

    # Sampling ------------------------------------------------------------------------------------------------

    def _sample(self, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
        assert self.freq_sampler is not None
        gen = torch.Generator().manual_seed(int(rng.integers(2**62)))
        self._torch_gen = gen
        self.generator_net.eval()  # type: ignore[union-attr]
        blocks = []
        with torch.no_grad():
            for start in range(0, n_rows, self.batch_size):
                cond, _c, _k = self.freq_sampler.sample(min(self.batch_size, n_rows - start), rng)
                _raw, fake = self._fake_batch(cond, len(cond))
                blocks.append(fake[:, : self.transformer.output_dim].numpy())
        self.generator_net.train()  # type: ignore[union-attr]
        return self.transformer.inverse(np.vstack(blocks))

    def sample_conditioned(self, n_rows: int, column: str, value: str, seed: int | None = None) -> pd.DataFrame:
        """Sample with the condition vector fixed to one category; the inverse is not guaranteed to honour it."""
        if not self.fitted:
            raise RuntimeError("ctgan must be fitted before sampling")
        spans = self.transformer.cond_spans
        j = next((k for k, s in enumerate(spans) if s.column == column), None)
        if j is None:
            raise ValueError(f"{column!r} is not a categorical column")
        cats = self.schema.get(column).categories
        if value not in cats:
            raise ValueError(f"{value!r} is not a category of {column!r}")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        self._torch_gen = torch.Generator().manual_seed(int(rng.integers(2**62)))
        cond = condition_vectors(self.transformer, np.full(n_rows, j), np.full(n_rows, cats.index(value)))
        self.generator_net.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            _raw, fake = self._fake_batch(cond, n_rows)
        self.generator_net.train()  # type: ignore[union-attr]
        return self.schema.coerce(self.transformer.inverse(fake[:, : self.transformer.output_dim].numpy()))

    def describe(self) -> dict[str, object]:
        out = super().describe()
        out.update(
            {
                "epochs_completed": self.epochs_completed,
                "steps_completed": self.steps_completed,
                "train_seconds": round(self.train_seconds, 3),
                "pac": self.pac,
                "batch_size": self.batch_size,
            }
        )
        return out
