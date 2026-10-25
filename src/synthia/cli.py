"""Command line entry point: reference data, synthesis, evaluation, reports, the benchmark and the server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from synthia import __version__
from synthia.constraints import Constraint, sample_with_constraints
from synthia.evaluation import evaluate
from synthia.generators import make_generator
from synthia.privacy import DPConfig
from synthia.reference import generate_patients, load_csv, patient_schema, train_holdout_split
from synthia.report import render_report
from synthia.schema import TableSchema


def _load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, TableSchema, str]:
    if args.csv:
        frame, schema = load_csv(args.csv)
        if args.schema:
            schema = TableSchema.from_dict(json.loads(Path(args.schema).read_text()))
            frame = schema.coerce(frame)
        name = Path(args.csv).name
    else:
        frame, schema, name = (
            generate_patients(args.n_rows, seed=args.seed),
            patient_schema(),
            f"patients (n={args.n_rows})",
        )
    train, holdout = train_holdout_split(frame, args.holdout, args.seed)
    return train, holdout, schema, name


def _generator_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.generator != "ctgan":
        return {}
    return {"epochs": args.epochs, "torch_threads": args.threads}


def _fit(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, pd.DataFrame, TableSchema, str]:
    train, holdout, schema, name = _load_source(args)
    dp = None if args.epsilon is None else DPConfig(args.epsilon, args.delta)
    generator = make_generator(args.generator, schema, dp=dp, seed=args.seed, **_generator_kwargs(args))
    generator.fit(train)
    return generator, train, holdout, schema, name


def cmd_data(args: argparse.Namespace) -> int:
    frame = generate_patients(args.n_rows, seed=args.seed)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} rows to {args.out}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    generator, train, _holdout, _schema, _name = _fit(args)
    constraints = [Constraint(**c) for c in json.loads(args.constraints)] if args.constraints else []
    n = args.n_synthetic or len(train)
    frame, info = sample_with_constraints(generator, n, constraints, seed=args.seed)
    frame.to_csv(args.out, index=False)
    print(
        f"wrote {len(frame)} synthetic rows to {args.out} ({info.rounds} rounds, acceptance {info.acceptance_rate:.3f})"
    )
    if generator.privacy_report is not None:
        print(json.dumps(generator.privacy_report.to_dict(), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    generator, train, holdout, schema, _name = _fit(args)
    synthetic = generator.sample(args.n_synthetic or len(train), seed=args.seed)
    result = evaluate(train, holdout, synthetic, schema, target=args.target, n_boot=args.n_boot, seed=args.seed)
    payload = {
        "generator": generator.describe(),
        "evaluation": result.to_dict(),
        "headline": result.headline(),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    generator, train, holdout, schema, name = _fit(args)
    synthetic = generator.sample(args.n_synthetic or len(train), seed=args.seed)
    result = evaluate(train, holdout, synthetic, schema, target=args.target, n_boot=args.n_boot, seed=args.seed)
    html = render_report(
        train,
        holdout,
        synthetic,
        schema,
        result,
        generator.describe(),
        title=f"Synthia report: {generator.name}",
        dataset_name=name,
        command=" ".join(sys.argv),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from synthia.bench import run_benchmark  # noqa: PLC0415  (torch import is slow; only load for bench)

    return run_benchmark(quick=args.quick, check=args.check, out=args.out, seed=args.seed, readme=args.readme)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415  (server dependency is only needed for this command)

    uvicorn.run("synthia.api.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator", choices=["copula", "ctgan", "bayesnet"], default="copula")
    parser.add_argument("--csv", help="fit on this CSV instead of the reference cohort")
    parser.add_argument("--schema", help="JSON schema file to use with --csv")
    parser.add_argument("--n-rows", type=int, default=4000, help="rows of the reference cohort")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=None, help="enable DP with this target epsilon")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100, help="CTGAN epochs")
    parser.add_argument("--threads", type=int, default=4, help="torch threads for CTGAN")
    parser.add_argument("--n-synthetic", type=int, default=None, help="defaults to the training size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthia", description="Privacy-preserving synthetic tabular data")
    parser.add_argument("--version", action="version", version=f"synthia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="write the reference patient cohort to CSV")
    p.add_argument("--n-rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/patients.csv")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("synthesize", help="fit a generator and write synthetic rows")
    _add_fit_args(p)
    p.add_argument("--constraints", help='JSON list, e.g. [{"column": "age", "minimum": 60}]')
    p.add_argument("--out", default="data/synthetic.csv")
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("evaluate", help="fit, sample and score fidelity, utility and privacy")
    _add_fit_args(p)
    p.add_argument("--target", default=None)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="fit, evaluate and write the HTML report")
    _add_fit_args(p)
    p.add_argument("--target", default=None)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--out", default="reports/report.html")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("bench", help="run the benchmark matrix and write benchmarks/results.json")
    p.add_argument("--quick", action="store_true", help="small profile for CI")
    p.add_argument("--check", action="store_true", help="exit non-zero when a quality gate fails")
    p.add_argument("--out", default="benchmarks/results.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--readme", default=None, help="refresh the results table between the README markers")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
