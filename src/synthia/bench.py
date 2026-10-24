"""Benchmark harness: build the reference cohort, fit every generator with and without DP, evaluate, gate.

``python -m synthia.bench`` writes ``benchmarks/results.json`` and ``benchmarks/results.md``. The full profile is
what the README quotes; the quick profile runs the same code on a smaller cohort with fewer epochs and bootstrap
replicates so CI finishes in minutes. Quality gates are explicit thresholds per generator family; ``--check``
turns a failed gate into a non-zero exit status.

Everything is seeded. Torch is pinned to a fixed thread count because reduction order changes with the thread
count and would otherwise make the GAN numbers machine-dependent even with the same seed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from synthia import __version__
from synthia.evaluation import EvaluationResult, evaluate
from synthia.generators import Generator, make_generator
from synthia.privacy import DPConfig
from synthia.reference import generate_patients, patient_schema, train_holdout_split
from synthia.schema import TableSchema

TARGET = "readmitted_30d"
TORCH_THREADS = 4

PROFILES: dict[str, dict[str, Any]] = {
    "full": {
        "n_rows": 12000,
        "n_boot": 200,
        "ctgan_epochs": 300,
        "ctgan_batch": 500,
        "ctgan_dp_epochs": 60,
        "ctgan_dp_batch": 1000,
        "epsilons": (0.5, 1.0, 2.0, 4.0, 8.0),
        "ctgan_epsilons": (1.0, 4.0),
    },
    "quick": {
        "n_rows": 3000,
        "n_boot": 50,
        "ctgan_epochs": 40,
        "ctgan_batch": 250,
        "ctgan_dp_epochs": 10,
        "ctgan_dp_batch": 500,
        "epsilons": (1.0,),
        "ctgan_epsilons": (1.0,),
    },
}

# Gate: (metric, operator, threshold). Privacy gates apply everywhere; fidelity and utility gates depend on the
# family because a CPU-capped GAN and a DP fit are held to looser bounds than the exact copula and network.
PRIVACY_GATES = [
    ("membership_auc", "<=", 0.56),
    ("dcr_share_train", "<=", 0.56),
    ("exact_match_rate", "<=", 0.001),
    ("nndr_synthetic", ">=", 0.5),
]
GATES: dict[str, dict[str, list[tuple[str, str, float]]]] = {
    "full": {
        "exact": [
            ("mean_ks", "<=", 0.05),
            ("mean_tvd", "<=", 0.03),
            ("correlation_delta_mean", "<=", 0.10),
            ("utility_ratio", ">=", 0.85),
        ],
        "ctgan": [("mean_ks", "<=", 0.15), ("mean_tvd", "<=", 0.10), ("utility_ratio", ">=", 0.80)],
        "dp": [("utility_ratio", ">=", 0.50)],
    },
    "quick": {
        "exact": [
            ("mean_ks", "<=", 0.08),
            ("mean_tvd", "<=", 0.05),
            ("correlation_delta_mean", "<=", 0.12),
            ("utility_ratio", ">=", 0.75),
        ],
        "ctgan": [("utility_ratio", ">=", 0.50)],
        "dp": [("utility_ratio", ">=", 0.30)],
    },
}


@dataclass(frozen=True)
class BenchConfig:
    name: str
    generator: str
    epsilon: float | None
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        if self.epsilon is not None:
            return "dp"
        return "ctgan" if self.generator == "ctgan" else "exact"


def build_matrix(profile: dict[str, Any]) -> list[BenchConfig]:
    gan = {
        "epochs": profile["ctgan_epochs"],
        "batch_size": profile["ctgan_batch"],
        "torch_threads": TORCH_THREADS,
    }
    gan_dp = {
        "epochs": profile["ctgan_dp_epochs"],
        "batch_size": profile["ctgan_dp_batch"],
        "torch_threads": TORCH_THREADS,
    }
    matrix = [
        BenchConfig("copula", "copula", None),
        BenchConfig("bayesnet", "bayesnet", None),
        BenchConfig("ctgan", "ctgan", None, gan),
    ]
    for eps in profile["epsilons"]:
        matrix.append(BenchConfig(f"copula-dp{eps:g}", "copula", eps))
        matrix.append(BenchConfig(f"bayesnet-dp{eps:g}", "bayesnet", eps))
    for eps in profile["ctgan_epsilons"]:
        matrix.append(BenchConfig(f"ctgan-dp{eps:g}", "ctgan", eps, gan_dp))
    return matrix


def _compare(value: float, op: str, threshold: float) -> bool:
    if value is None or not np.isfinite(value):
        return False
    return value <= threshold if op == "<=" else value >= threshold


def apply_gates(
    headline: dict[str, float], config: BenchConfig, profile_name: str, privacy_report: dict[str, Any] | None
) -> dict[str, Any]:
    gates = list(PRIVACY_GATES) + GATES[profile_name][config.family]
    out: dict[str, Any] = {}
    for metric, op, threshold in gates:
        value = headline.get(metric)
        out[metric] = {
            "value": value,
            "op": op,
            "threshold": threshold,
            "pass": _compare(value, op, threshold),
        }
    if config.epsilon is not None:
        spent = None if privacy_report is None else privacy_report["epsilon_spent"]
        out["epsilon_spent"] = {
            "value": spent,
            "op": "<=",
            "threshold": config.epsilon,
            "pass": spent is not None and spent <= config.epsilon + 1e-9,
        }
    return out


def _fit(config: BenchConfig, schema: TableSchema, train: pd.DataFrame, seed: int) -> tuple[Generator, float]:
    dp = None if config.epsilon is None else DPConfig(config.epsilon)
    generator = make_generator(config.generator, schema, dp=dp, seed=seed, **config.kwargs)
    started = time.perf_counter()
    generator.fit(train)
    return generator, time.perf_counter() - started


def run_config(
    config: BenchConfig,
    schema: TableSchema,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    profile_name: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    generator, fit_seconds = _fit(config, schema, train, seed)
    synthetic = generator.sample(len(train), seed=seed + 1)
    result: EvaluationResult = evaluate(train, holdout, synthetic, schema, target=TARGET, n_boot=n_boot, seed=seed)
    describe = generator.describe()
    privacy_report = describe.get("privacy")
    headline = result.headline()
    gates = apply_gates(headline, config, profile_name, privacy_report)
    return {
        "name": config.name,
        "generator": config.generator,
        "family": config.family,
        "epsilon_target": config.epsilon,
        "privacy_report": privacy_report,
        "fit_seconds": round(fit_seconds, 3),
        "eval_seconds": round(result.seconds, 3),
        "training": {
            k: v
            for k, v in describe.items()
            if k in {"epochs_completed", "steps_completed", "train_seconds", "batch_size", "pac"}
        },
        "headline": headline,
        "metrics": result.to_dict(),
        "gates": gates,
        "passed": all(g["pass"] for g in gates.values()),
    }


def determinism_check(schema: TableSchema, train: pd.DataFrame, seed: int) -> dict[str, bool]:
    """Same seed, same rows: fit and sample twice for the two cheap generators and compare exactly."""
    out = {}
    for name in ("copula", "bayesnet"):
        frames = [make_generator(name, schema, seed=seed).fit(train).sample(500, seed=seed) for _ in range(2)]
        out[name] = bool(frames[0].equals(frames[1]))
    return out


def _fmt_ci(est: dict[str, Any], digits: int = 3) -> str:
    v, lo, hi = est["value"], est["ci_low"], est["ci_high"]
    if v is None:
        return "n/a"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def render_markdown(results: dict[str, Any]) -> str:
    meta = results["meta"]
    intro = (
        f"Generated {meta['generated_at']} with synthia {meta['synthia_version']} on {meta['platform']} "
        f"({meta['cpu_count']} CPUs, torch {meta['torch_version']} with {meta['torch_threads']} threads). "
        f"Cohort: {meta['n_rows']} rows, {meta['n_train']} train / {meta['n_holdout']} holdout, "
        f"{meta['n_synthetic']} synthetic rows per generator, {meta['n_boot']} bootstrap replicates, "
        f"seed {meta['seed']}. Total wall time {meta['total_seconds']:.0f} s."
    )
    header = (
        "| Generator | epsilon spent | mean KS | mean TVD | assoc. delta | utility ratio | MIA AUC | NNDR | "
        "exact match | fit s | gates |"
    )
    lines = [f"# Benchmark results ({meta['profile']} profile)", "", intro, "", header, "|---" * 11 + "|"]
    for c in results["configs"]:
        lines.append(_markdown_row(c))
    lines += ["", f"Determinism (same seed, identical samples): {results['determinism']}", ""]
    lines.append(f"All gates passed: {results['gate_passed']}")
    return "\n".join(lines) + "\n"


def _markdown_row(c: dict[str, Any]) -> str:
    m, h = c["metrics"], c["headline"]
    eps = "off" if c["privacy_report"] is None else f"{c['privacy_report']['epsilon_spent']:.3f}"
    cells = [
        c["name"],
        eps,
        _fmt_ci(m["fidelity"]["mean_ks"]),
        _fmt_ci(m["fidelity"]["mean_tvd"]),
        _fmt_ci(m["fidelity"]["correlation_delta_mean"]),
        f"{h['utility_ratio']:.3f}",
        _fmt_ci(m["privacy"]["membership_auc"]),
        _fmt_ci(m["privacy"]["nndr_synthetic"]),
        f"{h['exact_match_rate']:.4f}",
        f"{c['fit_seconds']:.1f}",
        "pass" if c["passed"] else "FAIL",
    ]
    return "| " + " | ".join(cells) + " |"


def run_benchmark(quick: bool = False, check: bool = False, out: str = "benchmarks/results.json", seed: int = 0) -> int:
    profile_name = "quick" if quick else "full"
    profile = PROFILES[profile_name]
    torch.set_num_threads(TORCH_THREADS)
    started = time.perf_counter()
    schema = patient_schema()
    frame = generate_patients(profile["n_rows"], seed=seed)
    train, holdout = train_holdout_split(frame, 0.25, seed)
    configs = []
    for config in build_matrix(profile):
        print(f"[bench] {config.name} ...", flush=True)
        record = run_config(config, schema, train, holdout, profile_name, profile["n_boot"], seed)
        h = record["headline"]
        print(
            f"[bench] {config.name}: ks={h['mean_ks']:.4f} tvd={h['mean_tvd']:.4f} "
            f"assoc={h['correlation_delta_mean']:.4f} utility={h['utility_ratio']:.3f} "
            f"mia={h['membership_auc']:.3f} fit={record['fit_seconds']:.1f}s "
            f"eval={record['eval_seconds']:.1f}s {'pass' if record['passed'] else 'FAIL'}",
            flush=True,
        )
        configs.append(record)
    determinism = determinism_check(schema, train, seed)
    results = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "synthia_version": __version__,
            "python": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "cpu_count": os.cpu_count() or 0,
            "torch_threads": TORCH_THREADS,
            "profile": profile_name,
            "seed": seed,
            "n_rows": profile["n_rows"],
            "n_train": len(train),
            "n_holdout": len(holdout),
            "n_synthetic": len(train),
            "n_boot": profile["n_boot"],
            "target": TARGET,
            "command": " ".join(sys.argv),
            "total_seconds": round(time.perf_counter() - started, 1),
        },
        "configs": configs,
        "determinism": determinism,
        "gate_passed": all(c["passed"] for c in configs) and all(determinism.values()),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    out_path.with_suffix(".md").write_text(render_markdown(results))
    print(
        f"[bench] wrote {out_path} and {out_path.with_suffix('.md')} in "
        f"{results['meta']['total_seconds']:.0f}s; gates passed: {results['gate_passed']}"
    )
    if check and not results["gate_passed"]:
        failed = [f"{c['name']}:{m}" for c in configs for m, g in c["gates"].items() if not g["pass"]]
        failed += [f"determinism:{k}" for k, v in determinism.items() if not v]
        print(f"[bench] FAILED gates: {failed}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthia benchmark")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default="benchmarks/results.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run_benchmark(quick=args.quick, check=args.check, out=args.out, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
