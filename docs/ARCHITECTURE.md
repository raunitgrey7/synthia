# Architecture

Synthia is a Python package with one contract at its centre: a `TableSchema` describes the columns, a
`Generator` learns a conforming frame and samples new rows, the `privacy` package supplies the noise and the
accounting, and the `evaluation` package measures what came out. The CLI, the HTTP API and the benchmark are
thin layers over those four pieces.

```
                     +----------------------+
                     |  reference.py        |  seeded patient cohort (documented causal graph), CSV import
                     +----------+-----------+
                                |  DataFrame + TableSchema (schema.py)
                                v
   +--------------------------------------------------------------------------+
   | generators/                                                              |
   |   base.py      Generator ABC: fit(df) / sample(n, seed) / describe()     |
   |   copula.py    Gaussian copula  <----+                                   |
   |   bayesnet.py  Bayesian network <----+-- privacy/ (DPConfig, Accountant, |
   |   ctgan.py     CTGAN-style GAN  <----+   Gaussian mechanism, DP-SGD)     |
   |   ctgan_transform.py  mode-specific normalisation, condition vectors     |
   +--------------------------------------------------------------------------+
                                |  synthetic DataFrame
                                v
   +-------------------------------+      +----------------------------------+
   | constraints.py                |      | evaluation/                      |
   | rejection sampling, CTGAN     |      |   fidelity.py  KS, TVD, assoc.   |
   | condition push-down           |      |   utility.py   TSTR vs TRTR      |
   +-------------------------------+      |   privacy.py   MIA, DCR, NNDR,   |
                                          |                exact matches     |
                                          |   bootstrap.py percentile CIs    |
                                          |   runner.py    EvaluationResult  |
                                          +----------------+-----------------+
                                                           |
            +----------------------+        +--------------v---------------+       +------------------+
            | api/ (FastAPI)       |        | report/ (jinja2 + inline SVG)|       | bench.py         |
            | fit, synthesize,     |------->| self-contained HTML          |       | matrix, gates,   |
            | evaluate, report,    |        +------------------------------+       | results.json/.md |
            | health, metrics      |                                               +------------------+
            +----------------------+
                       ^
                       |  cli.py: data | synthesize | evaluate | report | bench | serve
```

## Modules

| Module | Responsibility | Why it exists separately |
|---|---|---|
| `schema.py` | `ColumnKind`, `ColumnSpec`, `TableSchema`: kinds, category domains, public numeric bounds, inference from data, validation and dtype coercion | Every DP mechanism clips or bins to public bounds; keeping bounds in one typed object (with a `bounds_inferred` flag) makes the "is this epsilon honest" question answerable |
| `reference.py` | The seeded patient cohort with its causal graph, the train/holdout split and CSV import | A known ground truth is what lets the benchmark say whether a dependency was reproduced |
| `transforms.py` | One-hot, `Discretizer` (quantile or public uniform bins plus fine within-bin profiles), `MixedEncoder` | Shared by two generators and all distance-based metrics, so the encodings must agree |
| `privacy/accounting.py` | RDP of the Gaussian and the Poisson-subsampled Gaussian mechanism, composition, conversion to (epsilon, delta), noise calibration by bisection | One accountant for all three generators keeps epsilons comparable |
| `privacy/mechanisms.py` | `DPConfig`, `PrivacyReport`, Gaussian mechanism, noisy histograms, row clipping | Small primitives the generators compose |
| `generators/copula.py` | Marginals (piecewise-linear quantile functions, category probabilities), normal scores, correlation, DP variant through histogram, count and second-moment queries | Fast, exact, strong on marginals and pairwise dependence |
| `generators/bayesnet.py` | Chow-Liu tree with an optional second parent, CPTs, ancestral sampling, DP variant through three count queries | Captures conditional structure the copula flattens; its DP variant is the classic PrivBayes idea in a simpler form |
| `generators/ctgan.py`, `ctgan_transform.py` | CTGAN-style GAN: mode-specific normalisation, condition vectors, training-by-sampling, WGAN-GP; DP-SGD on the discriminator through `torch.func` per-record gradients | The deep baseline the field compares against; on a CPU it is capped by epochs |
| `constraints.py` | `Constraint` (equals, one_of, range), adaptive rejection sampling, CTGAN condition push-down | The product feature: ask for a cohort slice, not just a table |
| `evaluation/*` | Fidelity, utility, privacy metrics with percentile bootstrap intervals | Numbers without intervals cannot be compared |
| `report/*` | jinja2 template, inline SVG chart helpers, table twins for every chart | A single file a reviewer can open anywhere with no network |
| `api/*` | FastAPI app, Pydantic schemas, in-process LRU model store, Prometheus text registry, request ids, API-key guard | Serve fitted models and constrained samples with operational visibility |
| `logging.py` | JSON log lines carrying the request id from a context variable | Correlate a log line with a request without threading ids through signatures |
| `bench.py` | The benchmark matrix, quality gates, `results.json` and `results.md` | The one command behind every number in the README |
| `cli.py` | argparse entry point | Everything is reachable without the server |

## Data flow for one request

1. `POST /v1/fit` builds the reference cohort (or takes uploaded rows), splits it 75/25 with the request seed,
   instantiates the generator with an optional `DPConfig`, fits on the training split and stores the fitted
   model under a hash of the request. Fitting with DP calibrates the noise multiplier first, so the privacy
   report is available immediately after `fit`.
2. `POST /v1/synthesize` loads the model, validates the constraints against the schema and runs adaptive
   rejection sampling (with CTGAN conditioning when a single categorical equality is requested). The response
   carries the rows, the acceptance statistics and the privacy report.
3. `POST /v1/evaluate` samples a table of the training size and runs the three metric families against the
   train and holdout splits that were set aside at fit time. `GET /v1/report/{id}` renders the same as HTML.
4. Every request passes through one middleware that assigns the request id, checks the API key when
   `APP_API_KEY` is set, records a Prometheus counter and latency observation, and emits one JSON log line.

## Determinism

All randomness flows from explicit `numpy.random.Generator` objects seeded by the caller. The GAN additionally
seeds torch, uses a private `torch.Generator` for noise and Gumbel draws, and pins the thread count, because
reduction order depends on it. Sampling with a seed is reproducible independent of earlier calls on the same
model; sampling without a seed advances the model's own stream.

## Threat model and what the epsilon covers

The release is the synthetic table plus the fitted model's statistics. Under DP every statistic that touched a
record passed through a Gaussian mechanism whose RDP cost the accountant composed; the number of rows, the
category domains and the numeric bounds are treated as public. When a schema is inferred from the data, the
bounds are data-dependent and the report says so. Nothing else is claimed: the evaluation attacks (membership
inference, nearest-neighbour distances, exact matches) are empirical checks and complement the accounting
rather than replace it.
