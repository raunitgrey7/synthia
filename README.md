# Synthia: privacy-preserving synthetic tabular data

Synthia fits a generator to a patient table, samples new rows that keep the marginals and the dependencies of the
original, and measures what came out: fidelity against the real data, utility for a downstream model, and how
much the synthetic rows leak about the people in the training set. Three generators share one interface, a
Gaussian copula, a Bayesian network and a CTGAN-style GAN, and each has a differentially private fit whose
epsilon is computed by one Renyi DP accountant rather than quoted. Everything runs on a CPU with no network and
no API key; a fixed seed gives the same table twice.

Why it exists: synthetic health data is only useful if someone can say how faithful it is and how private it is,
in the same breath and with error bars. Most toolkits give the first number and wave at the second. Synthia
reports both, with bootstrap confidence intervals, and ships the privacy accounting next to the attack results so
a reader can see when they disagree.

## Architecture

```
 reference cohort (seeded causal graph) or CSV  ->  TableSchema (kinds, domains, public bounds)
                                                            |
                   +----------------------------------------v---------------------------------------+
                   | generators: copula | bayesnet | ctgan           privacy: DPConfig, mechanisms, |
                   |   fit(train) -> sample(n, seed)        <------  RDP accountant, calibration    |
                   +----------------------------------------+---------------------------------------+
                                                            | synthetic rows
             constraints (rejection + CTGAN conditioning)   |
                                                            v
                   evaluation: fidelity (KS, TVD, association delta), utility (TSTR vs TRTR),
                               privacy (membership AUC, DCR share, NNDR, exact matches), bootstrap CIs
                                                            |
                      +-------------------------------------+------------------------------------+
                      v                                     v                                    v
            HTML report (inline SVG)             FastAPI (/v1/fit, /v1/synthesize,        bench (results.json,
                                                  /v1/evaluate, /v1/report, /metrics)       results.md, gates)
```

Modules and data flow are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); metrics, dataset and
reproduction steps in [docs/EVALUATION.md](docs/EVALUATION.md); every measured number with its command in
[REPORT.md](REPORT.md).

## Quickstart

```bash
make setup                                                            # uv sync, Python 3.13, CPU torch
make test                                                             # 118 tests
make bench                                                            # full matrix -> benchmarks/results.json, refreshes the table below
uv run synthia report --generator copula --epsilon 1 --out reports/copula_dp1.html
make serve                                                            # FastAPI on :8000, then see the API section
```

`uv run synthia --help` lists the other commands (`data`, `synthesize`, `evaluate`), all of which also accept
`--csv your_table.csv`.

## Measured results

Reference cohort of 12,000 patients (14 columns: demographics, vitals, diagnoses, outcomes, see
[docs/EVALUATION.md](docs/EVALUATION.md) for the generating graph), split 9,000 train / 3,000 holdout. Each
generator is fitted on the training split and asked for 9,000 rows. Lower is better for KS, TVD and the
association delta; the utility ratio is TSTR AUC over TRTR AUC on the holdout for the `readmitted_30d` target
(1.0 means no loss); membership AUC near 0.5 and an exact-match rate of 0 mean the attacks found nothing. NNDR
is the median nearest-to-second-nearest distance ratio of synthetic rows to the training set (near 0 would mean
copies).

<!-- benchmark-table:start -->
Measured 2026-10-25T00:10:04+00:00 by `python -m synthia.bench --check --readme README.md` on Darwin arm64 with torch 2.14.1 (4 threads); 9000 training rows, 3000 holdout rows, 200 bootstrap replicates, seed 0. Brackets are 95 percent bootstrap intervals.

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
<!-- benchmark-table:end -->

How to read it:

* The copula and the Bayesian network reproduce marginals almost exactly (mean KS 0.012 and 0.017) and keep
  most of the dependence structure (association delta 0.057 and 0.051); both train a readmission model at 96
  to 98 percent of the real AUC. The CPU-capped GAN is weaker on marginals (mean KS 0.089, mean TVD 0.064) and
  reaches 93 percent utility after 300 epochs and 64 seconds, which is the expected trade for a GAN on a laptop.
* Under DP the statistics-based generators degrade gracefully: at epsilon 1 the copula keeps mean KS at 0.042
  and trains a readmission model at 89 percent of the real AUC; at epsilon 4 it is back to 95 percent. The DP
  network keeps marginals (KS 0.016 to 0.040) but loses more dependence (association delta 0.10 to 0.14 against
  0.051 exact) and its utility ratio moves non-monotonically with epsilon (0.83, 0.82, 0.88, 0.83, 0.77 for
  epsilon 0.5 to 8), a single-seed effect of which tree edges the noisy mutual information picks. The DP-SGD GAN
  at this cohort size does not degrade gracefully: mean KS 0.200 at epsilon 1 and 0.235 at epsilon 4. Those rows
  are kept because they are the honest result; see Limitations.
* No configuration shows a membership signal (every AUC interval covers 0.5; point estimates 0.491 to 0.503),
  the DCR share stays between 0.475 and 0.520, NNDR medians (0.928 to 0.951) sit at or above the holdout's own
  0.924, and no synthetic row equals a training row. The holdout's exact-match rate against the training set is
  also 0, so the comparison baseline is clean.
* Intervals are 95 percent bias-corrected bootstrap intervals over rows; they do not cover variation across
  generator seeds.

## Differential privacy: what the epsilon covers

Every DP fit goes through the same accountant (`synthia.privacy`): neighbouring datasets differ by one record
(add or remove), every statistic that touches a record passes through a Gaussian mechanism, privacy loss is
tracked in Renyi DP and converted to (epsilon, delta) at delta = 1e-5, and the noise multiplier is bisected so the
composed epsilon lands at or just below the target. The report lists each query, its sensitivity, its noise
multiplier, and the epsilon actually spent.

| Generator | Data accesses under DP |
|---|---|
| copula | one histogram per column on public bins, one noisy row count, one second-moment matrix of row-clipped normal scores |
| bayesnet | all pairwise contingency tables (one query), all conditional count tables (one query), within-bin numeric profiles (one query) |
| ctgan | noisy condition frequencies, then DP-SGD on the critic with Poisson sampling, per-record clipping and Gaussian noise; the generator update is post-processing |

What is treated as public and therefore not covered: the category domains, the declared numeric bounds and (for
the copula and network) the number of rows. If the schema was inferred from the data the bounds are
data-dependent; the privacy report says so in its caveats. The accountant is cross-checked against two values
published by TensorFlow Privacy (`tests/test_privacy_accounting.py`).

## API

```bash
make serve
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/fit -H 'content-type: application/json' \
  -d '{"generator": "copula", "n_rows": 4000, "seed": 0, "dp": {"epsilon": 1.0}}'
# -> {"model_id": "...", "privacy": {"epsilon_spent": 0.99999, ...}}
curl -s -X POST localhost:8000/v1/synthesize -H 'content-type: application/json' \
  -d '{"model_id": "<id>", "n_rows": 50, "seed": 1,
       "constraints": [{"column": "age", "minimum": 65}, {"column": "diabetes", "equals": "yes"}]}'
curl -s -X POST localhost:8000/v1/evaluate -H 'content-type: application/json' -d '{"model_id": "<id>", "n_boot": 100}'
open http://localhost:8000/v1/report/<id>
curl -s localhost:8000/metrics | head
```

`POST /v1/fit` also accepts `"rows": [...]` to fit on uploaded data (schema inferred). Set `APP_API_KEY` to
require `X-API-Key` on every route except `/health` and `/metrics`. Every response carries `X-Request-ID`, logs
are JSON lines, and `/metrics` is Prometheus text format. Docker: `make docker && docker run -p 8000:8000 synthia`.

## Limitations

* The DP-SGD GAN is not competitive at 9,000 rows: 540 critic steps at sampling rate 0.11 need a noise
  multiplier of 12.8 for epsilon 1 and 3.5 for epsilon 4, and the marginals drift (mean KS 0.200 and 0.235).
  DP-GANs need far more rows or a much larger epsilon; the copula and network are the practical DP choices here.
* The DP Bayesian network learns a tree from noisy pairwise tables; at epsilon 1 the association delta is 0.144
  against 0.051 for the exact fit, and the utility ratio does not improve monotonically with epsilon.
* The non-DP GAN is capped at 300 epochs on a CPU and its numbers are machine-dependent in the last digits
  (reduction order); the copula and network are bit-for-bit reproducible.
* The copula models dependence through a single correlation matrix, so conditional effects beyond second order
  (such as the diabetes shift in glucose variance) are smoothed.
* Confidence intervals cover row sampling, not seed-to-seed variation of the generators.
* Models live in memory; there is no persistence or versioning of fitted generators.
* The reference cohort is synthetic by construction. Real clinical tables bring missing values and free text,
  neither of which this version handles (rows with missing values are dropped at import).

## Roadmap

* Save and load fitted generators, with the privacy report attached to the artefact.
* Multi-table (patient and encounter) synthesis with referential integrity.
* A tighter DP conversion (Balle et al.) and per-query budget allocation instead of a shared multiplier.
* Column-level constraints inside the generators (for example monotone age and length-of-stay rules) rather than
  rejection only.
* Missing-value modelling and a leakage check for quasi-identifier combinations.

## Repository layout

```
src/synthia/            package (schema, reference, transforms, generators/, privacy/, evaluation/, report/, api/, bench, cli)
tests/                  118 tests: unit, integration, property-based (hypothesis), adversarial and determinism checks
benchmarks/             results.json and results.md written by make bench
docs/                   ARCHITECTURE.md, EVALUATION.md
.github/workflows/      lint, tests, quick benchmark gate, Docker smoke test
```

## License

MIT.
