# Evaluation methodology

Every number Synthia publishes comes from `uv run python -m synthia.bench` (the `make bench` target), which
writes `benchmarks/results.json` and `benchmarks/results.md`. This document explains what is measured, how, and
how to reproduce it.

## Dataset: the reference patient cohort

The benchmark uses a seeded synthetic cohort rather than a public dataset, for two reasons: the ground truth
dependencies are known exactly, so a generator can be judged on whether it kept them, and no real patient is
involved. `synthia.reference.generate_patients(n_rows, seed)` draws from this causal graph (topological order):

| Column | Kind, public bounds | Generating rule |
|---|---|---|
| `age` | integer, 18 to 95 | Normal(58, 16), rounded and clipped |
| `sex` | categorical F/M | Bernoulli, 48 percent M |
| `bmi` | continuous, 14 to 60 | log-normal: log(26.5) + 0.004 (age - 55) + 0.02 [M], sd 0.18 |
| `smoker` | never/former/current | softmax of scores 0, -1 + 0.03 (age - 50) + 0.3 [M], -1 - 0.025 (age - 50) + 0.5 [M] |
| `diabetes` | no/yes | logistic: -2.9 + 0.07 (bmi - 27) + 0.035 (age - 50) + 0.4 [M] + 0.3 [current] |
| `systolic_bp` | continuous, 80 to 220 | 95 + 0.45 age + 0.9 (bmi - 27) + 6 [current] + Normal(0, 11) |
| `diastolic_bp` | continuous, 40 to 130 | 0.52 systolic + 12 + 0.3 (bmi - 27) + Normal(0, 7) |
| `heart_rate` | continuous, 40 to 160 | 72 + 7 [current] + 0.35 (bmi - 27) - 0.08 (age - 50) + Normal(0, 9) |
| `glucose` | continuous, 50 to 400 | 92 + 0.9 (bmi - 27) + 0.2 (age - 50) + 60 [diabetes] + Normal(0, 10 + 18 [diabetes]) |
| `cholesterol` | continuous, 100 to 400 | 170 + 0.6 (age - 50) - 0.004 (age - 50)^2 + 1.2 (bmi - 27) + 8 [current] + Normal(0, 30) |
| `hypertension` | no/yes | logistic: -14 + 0.1 systolic + 0.02 (age - 50) |
| `cad` | no/yes | logistic: -4.6 + 0.05 (age - 50) + 0.9 [M] + 0.8 [current] + 0.4 [former] + 0.006 (chol - 180) + 0.9 [HT] + 0.8 [DM] |
| `length_of_stay` | integer, 0 to 60 | negative binomial, mean exp(0.6 + 0.012 (age - 50) + 0.5 [cad] + 0.3 [DM] + 0.2 [HT]), dispersion 3 |
| `readmitted_30d` | no/yes | logistic: -3.6 + 0.03 (age - 50) + 1.2 [cad] + 0.8 [DM] + 0.22 LOS + 0.4 [current] |

Effect sizes are plausible for an adult hospital cohort and chosen to give mixed column types, skewed and count
variables, a rare binary outcome and dependencies of several kinds (linear, logistic, quadratic, mixture-like).
They are not fitted to any real data. The full benchmark draws 12,000 rows and splits them 9,000 train / 3,000
holdout with the benchmark seed; every generator is fitted on the training split only and asked for 9,000 rows.

Any CSV can be substituted (`synthia evaluate --csv file.csv`); the schema is inferred, which means the numeric
bounds are read from the data and the DP report flags that caveat.

## Generators and their DP variants

| Generator | Fit | What DP adds | Queries composed by the accountant |
|---|---|---|---|
| `copula` | empirical marginals (64-knot quantile functions), normal scores, correlation matrix | marginals from 16-bin histograms on the public bounds, noisy row count, second-moment matrix of row-clipped normal scores (clip radius 1.5 sqrt(d)), PSD projection | d + 1 Gaussian queries at 4 sigma, 1 Gaussian query at sigma |
| `bayesnet` | quantile bins (10) with 8 sub-bins, pairwise MI, Chow-Liu tree plus optional second parent, CPTs | equal-width bins (6) on public bounds, one parent, noisy pairwise tables (sensitivity sqrt(d(d-1)/2)), noisy CPT counts (sqrt(d)), noisy within-bin profiles (sqrt(d_numeric)) | 3 Gaussian queries at sigma |
| `ctgan` | mode-specific normalisation, condition vectors, training-by-sampling, residual generator, PAC-10 critic, WGAN-GP, 300 epochs at batch 500 | public min-max scaling, noisy condition frequencies, PAC 1, non-saturating GAN loss, DP-SGD on the critic with Poisson sampling, per-record clipping at L2 norm 1, 60 epochs at batch 1000 | d_cat Gaussian queries at 20 sigma, one subsampled Gaussian query per optimiser step |

Accounting is Renyi DP over orders 1.1 to 63.0 converted with `epsilon = min_alpha rdp(alpha) + log(1/delta) /
(alpha - 1)` at `delta = 1e-5`. The subsampled Gaussian bound is the exact computation of Mironov, Talwar and
Zhang (2019); `tests/test_privacy_accounting.py` checks it against two values printed by TensorFlow Privacy's
`compute_dp_sgd_privacy` (1.18 and 3.01). The noise multiplier is bisected so the composed epsilon lands at or
just below the target; the report shows the value actually reached as `epsilon_spent`. Adjacency is add-or-remove
one record. Row count (copula only, released noisily), category domains and declared numeric bounds are treated
as public.

## Metrics

### Fidelity

* Per-column distance: two-sample Kolmogorov-Smirnov statistic for numeric columns, total variation distance for
  categorical columns. Both are in [0, 1]; 0 is identical. The table reports the mean over numeric columns
  (`mean_ks`) and over categorical columns (`mean_tvd`).
* Association delta: an association matrix is computed for the real and the synthetic table with Pearson
  correlation (numeric pairs), Cramer's V (categorical pairs) and the correlation ratio eta (mixed pairs).
  `correlation_delta_mean` is the mean absolute entry-wise difference over the d(d-1)/2 pairs; the maximum is
  also reported.

### Utility

Train on synthetic, test on real (TSTR) against train on real, test on real (TRTR). Target: `readmitted_30d`.
Two model families, logistic regression (C = 1, standardised features) and histogram gradient boosting (150
iterations, learning rate 0.08), each trained once on the real training split and once on the synthetic table
and scored by ROC AUC on the real holdout. `utility_ratio` is the mean over the two models of TSTR AUC / TRTR AUC.
If the synthetic target has a single class the TSTR model is undefined; it then scores 0.5 and the row is flagged
`degenerate`.

### Privacy

All distances are Euclidean in the `MixedEncoder` space fitted on the real training split (numeric columns
standardised, categoricals one-hot scaled by 1/sqrt(2)).

* `membership_auc`: the full black-box distance attack (Chen et al. 2020). Each real record is scored by minus
  its distance to the nearest synthetic row; the AUC separates training rows (members) from holdout rows
  (non-members). 0.5 means the synthetic table carries no membership signal.
* `dcr_share_train`: fraction of synthetic rows closer to a training row than to a holdout row, with the
  training set subsampled to the holdout size; 0.5 is ideal, higher suggests memorisation.
* `nndr_synthetic`: median over synthetic rows of nearest / second-nearest distance to the training set. Values
  near 0 mean a row sits on one training record. The holdout's own NNDR against the training set is reported as
  the baseline (`nndr_holdout`).
* `exact_match_rate`: share of synthetic rows identical to a training row after rounding numeric columns to two
  decimals, next to the same rate for holdout rows.

### Confidence intervals

Every metric carries a bias-corrected percentile bootstrap interval (200 replicates in the full profile, 95
percent level). Row sets are resampled with replacement, jointly where the statistic needs paired rows (AUC,
ratios, flags) and independently otherwise (two-sample distances). Resampling biases non-negative distances
upward, so intervals are shifted by the bootstrap bias and clipped to the metric's range; unbiased statistics are
practically unaffected. Intervals describe sampling uncertainty at the given table sizes. They do not describe
variation across generator seeds; the benchmark reports a determinism check instead (same seed, identical
samples) and the `fit_seconds` column so the cost of re-running with another seed is known.

## Quality gates

`python -m synthia.bench --check` exits non-zero when any gate fails. Gates (full profile):

| Family | Gate |
|---|---|
| all | membership AUC <= 0.56, DCR share <= 0.56, exact-match rate <= 0.001, NNDR >= 0.5 |
| copula, bayesnet without DP | mean KS <= 0.05, mean TVD <= 0.03, association delta <= 0.10, utility ratio >= 0.85 |
| ctgan without DP | mean KS <= 0.15, mean TVD <= 0.10, utility ratio >= 0.80 |
| any DP fit | utility ratio >= 0.50 and epsilon spent <= epsilon target |
| determinism | copula and bayesnet produce identical samples when refitted with the same seed |

The quick profile used in CI (`--quick`: 3,000 rows, 40 GAN epochs, 50 replicates) keeps the privacy gates and
loosens the fidelity ones; the thresholds are in `src/synthia/bench.py`.

## Reproduce

```bash
make setup
make bench                       # full profile, writes benchmarks/results.json and results.md
uv run python -m synthia.bench --quick --check   # the CI profile
uv run synthia report --generator copula --epsilon 1 --out reports/copula_dp1.html
```

Seeds: cohort seed 0, generator seed 0, sampling seed 1, bootstrap seed 0. Torch is pinned to 4 threads. The GAN
numbers are reproducible on the same machine and library versions; across machines the floating point reduction
order can differ, so expect the GAN rows to move within their intervals while the copula and network rows match
exactly.
