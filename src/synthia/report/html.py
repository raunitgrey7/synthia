"""Assemble the evaluation of one synthetic table into a single HTML file."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape

from synthia import __version__
from synthia.evaluation.bootstrap import Estimate
from synthia.evaluation.fidelity import association_matrix
from synthia.evaluation.runner import EvaluationResult
from synthia.report import charts
from synthia.schema import ColumnSpec, TableSchema

_TILE_HINTS = {
    "mean_ks": "mean KS over numeric columns; 0 is identical",
    "mean_tvd": "mean TVD over categorical columns; 0 is identical",
    "correlation_delta_mean": "mean absolute association delta; 0 keeps every dependency",
    "utility_ratio": "TSTR / TRTR AUC averaged over models; 1 means no loss",
    "membership_auc": "distance attack; 0.5 means no membership signal",
    "exact_match_rate": "synthetic rows identical to a training row",
}


def _fmt_est(est: dict[str, Any]) -> str:
    value, low, high = est.get("value"), est.get("ci_low"), est.get("ci_high")
    if value is None:
        return "n/a"
    if low is None or high is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def _marginal(real: pd.Series, synthetic: pd.Series, spec: ColumnSpec) -> dict[str, Any]:
    if spec.is_numeric:
        r = real.to_numpy(dtype=np.float64)
        s = synthetic.to_numpy(dtype=np.float64)
        lo = float(min(r.min(), s.min()))
        hi = float(max(r.max(), s.max()))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, 17)
        pr = np.histogram(r, bins=edges)[0] / max(len(r), 1)
        ps = np.histogram(s, bins=edges)[0] / max(len(s), 1)
        labels = [f"{edges[i]:.3g}" for i in range(len(edges) - 1)]
    else:
        cats = list(spec.categories)
        pr = np.array([(real.astype(str) == c).mean() for c in cats])
        ps = np.array([(synthetic.astype(str) == c).mean() for c in cats])
        labels = cats
    svg = charts.grouped_columns(labels, pr.tolist(), ps.tolist(), f"{spec.name} marginal")
    return {"name": spec.name, "svg": svg, "rows": list(zip(labels, pr.tolist(), ps.tolist(), strict=True))}


def _privacy_rows(privacy: dict[str, Any]) -> list[dict[str, Any]]:
    spec = [
        ("membership_auc", "Membership inference AUC", "0.5 means no signal", lambda v: v < 0.55),
        (
            "dcr_share_train",
            "Share of synthetic rows closer to train than holdout",
            "0.5 is ideal",
            lambda v: v < 0.55,
        ),
        (
            "nndr_synthetic",
            "Median NNDR of synthetic rows to train",
            f"holdout baseline {privacy['nndr_holdout']['value']:.3f}",
            lambda v: v > 0.5,
        ),
        (
            "exact_match_rate",
            "Exact-match rate against train",
            f"holdout rate {privacy['exact_match_rate_holdout']['value']:.4f}",
            lambda v: v <= max(privacy["exact_match_rate_holdout"]["value"], 1e-9),
        ),
    ]
    rows = []
    for key, name, reference, ok in spec:
        est = privacy[key]
        value = est["value"] if est["value"] is not None else float("nan")
        rows.append(
            {
                "name": name,
                "value": value,
                "low": est["ci_low"] if est["ci_low"] is not None else float("nan"),
                "high": est["ci_high"] if est["ci_high"] is not None else float("nan"),
                "reference": reference,
                "reading": "no leakage detected" if math.isfinite(value) and ok(value) else "inspect",
            }
        )
    return rows


def _tiles(result: EvaluationResult) -> list[dict[str, str]]:
    f, u, p = result.fidelity, result.utility, result.privacy
    ratio = Estimate(u.mean_ratio, float("nan"), float("nan"), 0, 0.95)
    pairs = [
        ("mean_ks", "Mean KS", f.mean_ks),
        ("mean_tvd", "Mean TVD", f.mean_tvd),
        ("correlation_delta_mean", "Association delta", f.correlation_delta_mean),
        ("utility_ratio", "Utility ratio", ratio),
        ("membership_auc", "Membership AUC", p.membership_auc),
        ("exact_match_rate", "Exact matches", p.exact_match_rate),
    ]
    tiles = []
    for key, label, est in pairs:
        ci = "" if not math.isfinite(est.low) else f"[{est.low:.3f}, {est.high:.3f}]"
        tiles.append({"label": label, "value": f"{est.value:.3f}", "ci": ci, "hint": _TILE_HINTS[key]})
    return tiles


def _dp_summary(privacy_report: dict[str, Any] | None) -> str:
    if privacy_report is None:
        return "off"
    return f"epsilon {privacy_report['epsilon_spent']:.4f}, delta {privacy_report['delta']}"


def render_report(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: TableSchema,
    result: EvaluationResult,
    generator_info: dict[str, Any],
    title: str = "Synthia report",
    dataset_name: str = "reference patients",
    command: str | None = None,
) -> str:
    """Render the full HTML document as a string."""
    env = Environment(autoescape=select_autoescape(["html"]))
    env.globals["fmt_est"] = _fmt_est
    template = env.from_string(resources.files("synthia.report").joinpath("template.html").read_text(encoding="utf-8"))
    res = result.to_dict()
    column_rows = [
        {
            "name": name,
            "metric": info["metric"],
            "value": info["value"],
            "low": info["ci_low"],
            "high": info["ci_high"],
        }
        for name, info in res["fidelity"]["per_column"].items()
    ]
    chart_rows = [(r["name"], r["value"], r["low"], r["high"]) for r in column_rows]
    a_real = association_matrix(real_train, schema)
    a_syn = association_matrix(synthetic, schema)
    delta = np.abs(a_real - a_syn)
    privacy_report = generator_info.get("privacy")
    setup = [
        ("Dataset", dataset_name),
        ("Generator", generator_info.get("generator", "unknown")),
        (
            "Rows: train / holdout / synthetic",
            f"{result.n_train} / {result.n_holdout} / {result.n_synthetic}",
        ),
        (
            "Columns",
            f"{len(schema)} ({len(schema.numeric_names)} numeric, {len(schema.categorical_names)} categorical)",
        ),
        (
            "Differential privacy",
            "off"
            if privacy_report is None
            else f"epsilon {privacy_report['epsilon_spent']:.4f}, delta {privacy_report['delta']}",
        ),
        ("Evaluation time", f"{result.seconds:.1f} s"),
    ]
    for key in ("epochs_completed", "steps_completed", "train_seconds"):
        if key in generator_info:
            setup.append((key.replace("_", " "), str(generator_info[key])))
    return template.render(
        title=title,
        subtitle=(
            f"{generator_info.get('generator', 'generator')} on {dataset_name}: "
            "fidelity, utility and privacy with bootstrap intervals"
        ),
        tiles=_tiles(result),
        setup=setup,
        privacy_report=privacy_report,
        column_chart=charts.bar_chart_with_ci(chart_rows, "Per-column distance"),
        column_rows=column_rows,
        level_pct=round(100 * result.fidelity.mean_ks.level),
        marginals=[_marginal(real_train[c.name], synthetic[c.name], c) for c in schema.columns],
        corr_delta=result.fidelity.correlation_delta_mean,
        corr_delta_max=result.fidelity.correlation_delta_max,
        heat_real=charts.heatmap(a_real, schema.names, "Real associations", diverging=True),
        heat_syn=charts.heatmap(a_syn, schema.names, "Synthetic associations", diverging=True),
        heat_delta=charts.heatmap(
            delta, schema.names, "Absolute difference", diverging=False, vmax=max(float(delta.max()), 0.1)
        ),
        table_real=charts.matrix_table(a_real, schema.names),
        table_syn=charts.matrix_table(a_syn, schema.names),
        table_delta=charts.matrix_table(delta, schema.names),
        utility=res["utility"],
        n_holdout=result.n_holdout,
        privacy_rows=_privacy_rows(res["privacy"]),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
        seeds=generator_info.get("seeds", "see command"),
        n_boot=result.n_boot,
        command=command,
    )
