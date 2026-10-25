# REPORT: what was built, how every number was measured, and what is missing

Project: `synthia`, privacy-preserving synthetic tabular data for healthcare. Built against my project charter
and its spec. Everything below was produced by commands in this repository on this
machine; nothing is estimated.

## 1. What was built

| Spec item | Delivered | Where |
|---|---|---|
| Reference data: seeded patient tables with documented dependencies, optional CSV import | 14-column cohort (demographics, vitals, diagnoses, outcomes) drawn from an explicit causal graph with fixed effect sizes; `load_csv` with schema inference and public-bounds caveat | `src/synthia/reference.py`, `docs/EVALUATION.md` |
| Gaussian copula | empirical quantile-function marginals, expected normal scores for categories, PSD-projected correlation | `src/synthia/generators/copula.py` |
| CTGAN-style conditional GAN in PyTorch, CPU, capped | mode-specific normalisation (variational GMM), condition vectors with training-by-sampling, residual generator, PAC-10 critic, WGAN-GP, cross-entropy condition loss, epoch cap and wall-time cap | `src/synthia/generators/ctgan.py`, `ctgan_transform.py` |
| Bayesian-network sampler | Chow-Liu tree with optional second parent, CPTs with pseudocounts, within-bin profiles, ancestral sampling | `src/synthia/generators/bayesnet.py` |
| Differential privacy with measured epsilon accounting | Gaussian mechanism on sufficient statistics (copula, network), DP-SGD with per-record gradients via `torch.func` (GAN), one RDP accountant with Poisson-subsampling bound, bisection calibration, `PrivacyReport` listing every query | `src/synthia/privacy/` |
| Fidelity, utility, privacy evaluation with confidence intervals | KS and TVD per column, association delta (Pearson, Cramer's V, correlation ratio), TSTR vs TRTR for two model families, membership-inference AUC, DCR share, NNDR, exact matches; bias-corrected percentile bootstrap | `src/synthia/evaluation/` |
| HTML report | single file, inline SVG, table twin for every chart, dark mode, DP accounting section | `src/synthia/report/` |
| API with constraints | FastAPI: `/v1/fit`, `/v1/synthesize` (equals, one_of, range constraints; CTGAN condition push-down), `/v1/evaluate`, `/v1/report/{id}`, `/health`, `/metrics` (Prometheus text), JSON logs, request ids, `X-API-Key` when `APP_API_KEY` is set | `src/synthia/api/` |
| Benchmark harness and gates | `python -m synthia.bench [--quick] [--check] [--readme README.md]`, 15-config matrix, explicit thresholds per family, determinism check, `results.json` and `results.md`, README table refresh | `src/synthia/bench.py` |
| CI, Docker, Makefile, docs | ruff, pytest, quick benchmark gate, Docker smoke test; multi-stage non-root image; `make setup lint test bench serve`; README, ARCHITECTURE, EVALUATION | `.github/workflows/ci.yml`, `Dockerfile`, `Makefile`, `docs/` |

## 2. Environment

Darwin arm64, 15 CPUs, Python 3.13.17, torch 2.14.1 (4 threads for the benchmark), numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1. Other builds were running on the machine during every timed run (load averages between 3 and 27 on 15 cores), so the wall times below are upper bounds for an idle machine.

## 3. Commands and outputs

The final run was `make lint test bench` from a fresh login shell (`zsh -lc`) on 2026-10-25T00:10:04+00:00.

### Lint

```
uv run ruff check src tests
All checks passed!
uv run ruff format --check src tests
43 files already formatted
```

### Tests

```
118 passed in 55.43s
```

118 tests: unit, integration (API through the ASGI test client, CLI end to end), property-based (hypothesis: schema inference idempotence, discretiser monotonicity, TVD distance axioms), adversarial (constant column, impossible constraints, rows with missing values, unknown categories, copied training table) and determinism (same seed, identical samples for every generator; refit determinism for copula, network and GAN).

### Benchmark (`uv run python -m synthia.bench --check --readme README.md`)

```
[bench] copula-dp2 ...
[bench] copula-dp2: ks=0.0387 tvd=0.0039 assoc=0.0700 utility=0.928 mia=0.501 fit=0.0s eval=19.3s pass
[bench] bayesnet-dp2 ...
[bench] bayesnet-dp2: ks=0.0191 tvd=0.0066 assoc=0.1237 utility=0.877 mia=0.501 fit=0.0s eval=20.4s pass
[bench] copula-dp4 ...
[bench] copula-dp4: ks=0.0373 tvd=0.0041 assoc=0.0678 utility=0.946 mia=0.500 fit=0.0s eval=24.6s pass
[bench] bayesnet-dp4 ...
[bench] bayesnet-dp4: ks=0.0158 tvd=0.0029 assoc=0.1085 utility=0.832 mia=0.501 fit=0.0s eval=21.1s pass
[bench] copula-dp8 ...
[bench] copula-dp8: ks=0.0366 tvd=0.0037 assoc=0.0670 utility=0.952 mia=0.500 fit=0.0s eval=33.9s pass
[bench] bayesnet-dp8 ...
[bench] bayesnet-dp8: ks=0.0164 tvd=0.0032 assoc=0.1015 utility=0.768 mia=0.501 fit=0.0s eval=19.7s pass
[bench] ctgan-dp1 ...
[bench] ctgan-dp1: ks=0.1997 tvd=0.0449 assoc=0.1970 utility=0.789 mia=0.498 fit=49.9s eval=27.2s pass
[bench] ctgan-dp4 ...
[bench] ctgan-dp4: ks=0.2348 tvd=0.0316 assoc=0.1566 utility=0.928 mia=0.498 fit=48.8s eval=19.5s pass
[bench] wrote benchmarks/results.json and benchmarks/results.md in 519s; gates passed: True
make-exit=0
```

Total wall time 519 s for 15 configurations: 12000 rows, 9000 train / 3000 holdout, 9000 synthetic rows per generator, 200 bootstrap replicates, seed 0.

## 4. Measured results

### Headline table (identical to `benchmarks/results.md` and the README)

| Generator | epsilon spent | mean KS | mean TVD | assoc. delta | utility ratio | MIA AUC | NNDR | exact match | fit s | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| copula | off | 0.012 [0.009, 0.015] | 0.004 [0.001, 0.007] | 0.057 [0.051, 0.063] | 0.963 | 0.502 [0.491, 0.513] | 0.929 [0.926, 0.931] | 0.0000 | 0.0 | pass |
| bayesnet | off | 0.017 [0.014, 0.020] | 0.009 [0.005, 0.013] | 0.051 [0.046, 0.056] | 0.984 | 0.501 [0.490, 0.511] | 0.928 [0.926, 0.930] | 0.0000 | 0.0 | pass |
| ctgan | off | 0.089 [0.084, 0.093] | 0.064 [0.057, 0.070] | 0.069 [0.064, 0.073] | 0.934 | 0.491 [0.481, 0.501] | 0.934 [0.932, 0.936] | 0.0000 | 63.8 | pass |
| copula-dp0.5 | 0.500 | 0.049 [0.045, 0.053] | 0.004 [0.001, 0.007] | 0.083 [0.078, 0.088] | 0.785 | 0.502 [0.490, 0.514] | 0.930 [0.928, 0.932] | 0.0000 | 0.0 | pass |
| bayesnet-dp0.5 | 0.500 | 0.040 [0.036, 0.044] | 0.010 [0.006, 0.015] | 0.138 [0.134, 0.143] | 0.828 | 0.503 [0.491, 0.514] | 0.938 [0.936, 0.940] | 0.0000 | 0.0 | pass |
| copula-dp1 | 1.000 | 0.042 [0.039, 0.046] | 0.004 [0.001, 0.007] | 0.074 [0.068, 0.079] | 0.888 | 0.501 [0.490, 0.512] | 0.930 [0.928, 0.932] | 0.0000 | 0.0 | pass |
| bayesnet-dp1 | 1.000 | 0.031 [0.027, 0.034] | 0.011 [0.007, 0.017] | 0.144 [0.139, 0.150] | 0.822 | 0.499 [0.488, 0.511] | 0.936 [0.934, 0.938] | 0.0000 | 0.0 | pass |
| copula-dp2 | 2.000 | 0.039 [0.035, 0.042] | 0.004 [0.001, 0.008] | 0.070 [0.064, 0.076] | 0.928 | 0.501 [0.490, 0.513] | 0.931 [0.929, 0.933] | 0.0000 | 0.0 | pass |
| bayesnet-dp2 | 2.000 | 0.019 [0.016, 0.023] | 0.007 [0.003, 0.011] | 0.124 [0.117, 0.129] | 0.877 | 0.501 [0.490, 0.513] | 0.934 [0.932, 0.936] | 0.0000 | 0.0 | pass |
| copula-dp4 | 4.000 | 0.037 [0.034, 0.041] | 0.004 [0.001, 0.008] | 0.068 [0.061, 0.074] | 0.946 | 0.500 [0.489, 0.511] | 0.931 [0.929, 0.933] | 0.0000 | 0.0 | pass |
| bayesnet-dp4 | 4.000 | 0.016 [0.013, 0.019] | 0.003 [0.000, 0.006] | 0.109 [0.103, 0.114] | 0.832 | 0.501 [0.490, 0.512] | 0.931 [0.929, 0.933] | 0.0000 | 0.0 | pass |
| copula-dp8 | 8.000 | 0.037 [0.033, 0.040] | 0.004 [0.001, 0.007] | 0.067 [0.061, 0.073] | 0.952 | 0.500 [0.489, 0.511] | 0.931 [0.929, 0.933] | 0.0000 | 0.0 | pass |
| bayesnet-dp8 | 8.000 | 0.016 [0.013, 0.020] | 0.003 [0.000, 0.007] | 0.102 [0.097, 0.108] | 0.768 | 0.501 [0.491, 0.512] | 0.934 [0.931, 0.935] | 0.0000 | 0.0 | pass |
| ctgan-dp1 | 1.000 | 0.200 [0.195, 0.205] | 0.045 [0.040, 0.049] | 0.197 [0.193, 0.201] | 0.789 | 0.498 [0.487, 0.511] | 0.951 [0.950, 0.953] | 0.0000 | 49.9 | pass |
| ctgan-dp4 | 4.000 | 0.235 [0.228, 0.241] | 0.032 [0.027, 0.035] | 0.157 [0.153, 0.160] | 0.928 | 0.498 [0.488, 0.510] | 0.944 [0.943, 0.946] | 0.0000 | 48.8 | pass |

### Details per configuration

| Config | epsilon spent | noise multiplier | RDP order | DCR share train [CI] | LR AUC real / synthetic | GB AUC real / synthetic | max assoc. delta | fit s | eval s |
|---|---|---|---|---|---|---|---|---|---|
| copula |  | off |  | 0.487 [0.477, 0.496] | 0.758 / 0.747 | 0.738 / 0.695 | 0.413 | 0.0 | 32.8 |
| bayesnet |  | off |  | 0.503 [0.493, 0.512] | 0.758 / 0.754 | 0.738 / 0.718 | 0.176 | 0.0 | 21.5 |
| ctgan |  | off |  | 0.494 [0.486, 0.503] | 0.758 / 0.707 | 0.738 / 0.690 | 0.359 | 63.8 | 18.9 |
| copula-dp0.5 | 0.49998 | 13.503 | 48 | 0.475 [0.464, 0.486] | 0.758 / 0.574 | 0.738 / 0.600 | 0.420 | 0.0 | 19.7 |
| bayesnet-dp0.5 | 0.49998 | 16.803 | 48 | 0.491 [0.481, 0.501] | 0.758 / 0.704 | 0.738 / 0.536 | 0.550 | 0.0 | 27.4 |
| copula-dp1 | 0.99996 | 6.823 | 25 | 0.484 [0.474, 0.495] | 0.758 / 0.671 | 0.738 / 0.657 | 0.396 | 0.0 | 31.6 |
| bayesnet-dp1 | 0.99995 | 8.490 | 25 | 0.520 [0.510, 0.531] | 0.758 / 0.696 | 0.738 / 0.536 | 0.484 | 0.0 | 18.7 |
| copula-dp2 | 1.99998 | 3.479 | 13 | 0.486 [0.476, 0.495] | 0.758 / 0.710 | 0.738 / 0.678 | 0.378 | 0.0 | 19.3 |
| bayesnet-dp2 | 1.99996 | 4.329 | 13 | 0.493 [0.484, 0.504] | 0.758 / 0.709 | 0.738 / 0.605 | 0.380 | 0.0 | 20.4 |
| copula-dp4 | 3.99981 | 1.804 | 7.2 | 0.486 [0.478, 0.495] | 0.758 / 0.726 | 0.738 / 0.690 | 0.367 | 0.0 | 24.6 |
| bayesnet-dp4 | 3.99977 | 2.245 | 7.2 | 0.506 [0.495, 0.516] | 0.758 / 0.632 | 0.738 / 0.612 | 0.313 | 0.0 | 21.1 |
| copula-dp8 | 7.99958 | 0.961 | 4.3 | 0.488 [0.480, 0.497] | 0.758 / 0.731 | 0.738 / 0.693 | 0.361 | 0.0 | 33.9 |
| bayesnet-dp8 | 7.99999 | 1.196 | 4.3 | 0.502 [0.492, 0.510] | 0.758 / 0.557 | 0.738 / 0.591 | 0.311 | 0.0 | 19.7 |
| ctgan-dp1 | 0.99995 | 12.776 | 24 | 0.520 [0.511, 0.530] | 0.758 / 0.600 | 0.738 / 0.580 | 0.796 | 49.9 | 27.2 |
| ctgan-dp4 | 3.99991 | 3.497 | 7 | 0.510 [0.500, 0.521] | 0.758 / 0.721 | 0.738 / 0.668 | 0.648 | 48.8 | 19.5 |

GAN training: {"ctgan": {"epochs_completed": 300, "steps_completed": 5400, "train_seconds": 57.665, "pac": 10, "batch_size": 500}, "ctgan-dp1": {"epochs_completed": 60, "steps_completed": 540, "train_seconds": 49.934, "pac": 1, "batch_size": 1000}, "ctgan-dp4": {"epochs_completed": 60, "steps_completed": 540, "train_seconds": 48.793, "pac": 1, "batch_size": 1000}}.

Reading the table: the holdout's own NNDR against the training set is 0.924 and its exact-match rate is 0.0000; every synthetic table sits at or above that NNDR and at zero exact matches, and every membership AUC interval covers 0.5. The DP-SGD GAN rows (mean KS 0.200 and 0.235) are the weak point and are reported as such.

### Accountant cross-check

`tests/test_privacy_accounting.py` compares the accountant with two values printed by TensorFlow Privacy's `compute_dp_sgd_privacy` for its MNIST tutorials:

| Configuration | Published | Synthia | Best order |
|---|---|---|---|
| q = 250/60000, sigma 1.3, 3600 steps, delta 1e-5 | 1.18 | 1.1799 | 17 |
| q = 256/60000, sigma 1.1, 14062 steps, delta 1e-5 | 3.01 | 3.0083 | 8.8 |

### Reproducibility: first full run against the final run

The benchmark ran twice on this machine (first at 2026-10-24T23:55:59Z, then the final run above). Maximum absolute difference over the eight headline metrics per configuration, and the fit times of both runs:

| Config | max abs diff | fit s (run 1) | fit s (final) |
|---|---|---|---|
| copula | 0.000000 | 0.0 | 0.0 |
| bayesnet | 0.000000 | 0.0 | 0.0 |
| ctgan | 0.000000 | 64.1 | 63.8 |
| copula-dp0.5 | 0.000000 | 0.0 | 0.0 |
| bayesnet-dp0.5 | 0.000000 | 0.0 | 0.0 |
| copula-dp1 | 0.000000 | 0.0 | 0.0 |
| bayesnet-dp1 | 0.000000 | 0.0 | 0.0 |
| copula-dp2 | 0.000000 | 0.0 | 0.0 |
| bayesnet-dp2 | 0.000000 | 0.0 | 0.0 |
| copula-dp4 | 0.000000 | 0.0 | 0.0 |
| bayesnet-dp4 | 0.000000 | 0.0 | 0.0 |
| copula-dp8 | 0.000000 | 0.0 | 0.0 |
| bayesnet-dp8 | 0.000000 | 0.0 | 0.0 |
| ctgan-dp1 | 0.000000 | 55.9 | 49.9 |
| ctgan-dp4 | 0.000000 | 57.9 | 48.8 |

Determinism check inside the final run (same seed, refit and resample twice): {'copula': True, 'bayesnet': True}.

## 5. Decisions that came out of measurement

* Bootstrap intervals for distances. The plain percentile interval of mean KS for the copula sat at [0.034, 0.054] with a point estimate of 0.029: resampling rows with replacement biases a non-negative distance upward. The intervals are now shifted by the bootstrap bias and clipped to the metric's range (`evaluation/bootstrap.py`); means and AUCs are unaffected in practice.
* DP-CTGAN budget split. With the condition-frequency histograms at the same noise multiplier as DP-SGD, a 36-step fit at epsilon 1 needed sigma 12.1 because six full-data Gaussian queries dominated the RDP sum. Giving those histograms 20 times the multiplier (counts are in the thousands, so the extra noise is harmless) moved the budget to the gradient steps.
* DP-CTGAN batch size. Probes at epsilon 1 on the same split: batch 500 for 60 epochs (1080 steps, sigma 9.06) gave mean KS 0.21 and utility 0.74; batch 2000 for 100 epochs (400 steps, sigma 21.9) gave KS 0.24 and utility 0.65; batch 1000 for 150 epochs (1350 steps, sigma 20.1) gave KS 0.21 and utility 0.87; batch 1000 for 60 epochs (540 steps, sigma 12.8) gave KS 0.21 and utility 0.74. None is competitive with the statistics-based DP generators, so the benchmark keeps the cheapest honest setting (batch 1000, 60 epochs) and reports it.
* Bayesian network binning. Uniform-within-bin decoding spread the top bins of skewed columns to the public upper bound (glucose given diabetes came out at 246 against 157 real). A fine within-bin profile (8 sub-bins, one extra DP query) fixed it: 157.7 against 156.9.
* Evaluation speed. The pandas-based association matrix and the row-wise exact-match key made one evaluation take 70 to 100 s at 9,000 rows; vectorising both with numpy and `hash_pandas_object` brought it to the 19 to 65 s seen in the table under load.

## 6. Docker verification

`docker build -t synthia:latest .` succeeded (image 1.74 GB, dominated by the CPU torch wheel). `docker run -d -p 8766:8000 synthia:latest` answered `/health` with `{"status":"ok","version":"0.1.0","models_cached":0,"auth_enabled":false}`, `id` inside the container reported `uid=999(synthia) gid=999(synthia)`, and a `POST /v1/fit` for the copula on 400 rows returned a model id from inside the container.

## 7. Known gaps and limitations

* The DP-SGD GAN is not useful at this cohort size: 540 critic steps at sampling rate 0.11 need sigma 12.8 for epsilon 1 and 3.5 for epsilon 4, and mean KS stays above 0.2. The copula and the network are the practical DP generators here; the GAN rows are kept for honesty.
* The DP network's utility ratio is non-monotone in epsilon (0.828, 0.822, 0.877, 0.832, 0.768 for epsilon 0.5, 1, 2, 4, 8). Each cell is a single seed and the readmission target sits several hops from its predictors in a tree, so which noisy edge wins matters more than the noise on the tables. A multi-seed sweep would characterise this but was out of the time budget.
* The non-DP GAN is capped at 300 epochs (58 s of training in the final run); it reaches mean KS 0.089 and utility 0.934, clearly behind the copula and network. More epochs would help but the charter caps training at a few minutes.
* GAN numbers reproduced exactly between the two full runs on this machine with the same thread count; across machines the floating point reduction order can differ, so the GAN rows should be expected to move within their intervals elsewhere.
* Confidence intervals cover row sampling only, not seed-to-seed variation of the generators.
* The copula's single correlation matrix smooths higher-order effects (for example the diabetes-dependent variance of glucose); the network with two parents captures some of these, as its higher utility ratio shows.
* Fitted models live in memory in the API; there is no persistence. Rows with missing values are dropped at import; free text is not handled.
* The reference cohort is synthetic by construction, so the benchmark measures how well a generator reproduces a known graph, not clinical realism. Any CSV can be substituted with `--csv`, in which case numeric bounds are inferred and the DP report flags that they are not covered by the epsilon.
* The Docker image is 1.74 GB because of the torch wheel; a copula-only image would be far smaller but was not built.
* The HTML report was validated structurally (well-formed, self-contained, 18 inline SVGs) and by reading the generated SVG coordinates, not by viewing it in a browser, which this build was not allowed to use.
