import json
import logging

import numpy as np
import pytest

from synthia.api.metrics import MetricsRegistry
from synthia.bench import PROFILES, BenchConfig, apply_gates, build_matrix, render_markdown
from synthia.cli import build_parser, main
from synthia.logging import JsonFormatter, request_id_var


def test_matrix_covers_every_generator_with_and_without_dp():
    matrix = build_matrix(PROFILES["quick"])
    names = {c.name for c in matrix}
    assert {"copula", "bayesnet", "ctgan", "copula-dp1", "bayesnet-dp1", "ctgan-dp1"} <= names
    assert {c.family for c in matrix} == {"exact", "ctgan", "dp"}


def test_gates_flag_privacy_failures_and_accounting_overrun():
    good = {
        "membership_auc": 0.5,
        "dcr_share_train": 0.5,
        "exact_match_rate": 0.0,
        "nndr_synthetic": 0.9,
        "utility_ratio": 0.9,
        "mean_ks": 0.01,
        "mean_tvd": 0.01,
        "correlation_delta_mean": 0.02,
    }
    gates = apply_gates(good, BenchConfig("copula", "copula", None), "full", None)
    assert all(g["pass"] for g in gates.values())
    leaky = {**good, "membership_auc": 0.8, "exact_match_rate": 0.2}
    gates = apply_gates(leaky, BenchConfig("copula", "copula", None), "full", None)
    assert not gates["membership_auc"]["pass"] and not gates["exact_match_rate"]["pass"]
    overrun = apply_gates(good, BenchConfig("copula-dp1", "copula", 1.0), "full", {"epsilon_spent": 1.2})
    assert not overrun["epsilon_spent"]["pass"]
    missing = apply_gates({**good, "utility_ratio": float("nan")}, BenchConfig("copula", "copula", None), "quick", None)
    assert not missing["utility_ratio"]["pass"]


def test_markdown_rendering_of_results():
    est = {"value": 0.1, "ci_low": 0.05, "ci_high": 0.15}
    results = {
        "meta": {
            "profile": "quick",
            "generated_at": "now",
            "synthia_version": "x",
            "platform": "p",
            "cpu_count": 1,
            "torch_version": "t",
            "torch_threads": 1,
            "n_rows": 10,
            "n_train": 8,
            "n_holdout": 2,
            "n_synthetic": 8,
            "n_boot": 1,
            "seed": 0,
            "total_seconds": 1.0,
        },
        "configs": [
            {
                "name": "copula",
                "privacy_report": None,
                "fit_seconds": 0.1,
                "passed": True,
                "headline": {"utility_ratio": 0.9, "exact_match_rate": 0.0},
                "metrics": {
                    "fidelity": {"mean_ks": est, "mean_tvd": est, "correlation_delta_mean": est},
                    "privacy": {"membership_auc": est, "nndr_synthetic": est},
                },
            }
        ],
        "determinism": {"copula": True},
        "gate_passed": True,
    }
    text = render_markdown(results)
    assert "| copula | off | 0.100 [0.050, 0.150]" in text
    assert "All gates passed: True" in text


def test_cli_data_and_synthesize(tmp_path):
    out = tmp_path / "patients.csv"
    assert main(["data", "--n-rows", "120", "--seed", "1", "--out", str(out)]) == 0
    assert out.exists()
    syn = tmp_path / "syn.csv"
    code = main(
        [
            "synthesize",
            "--csv",
            str(out),
            "--n-rows",
            "120",
            "--n-synthetic",
            "30",
            "--out",
            str(syn),
            "--constraints",
            '[{"column": "age", "minimum": 40}]',
        ]
    )
    assert code == 0
    import pandas as pd

    frame = pd.read_csv(syn)
    assert len(frame) == 30 and (frame["age"] >= 40).all()


def test_cli_evaluate_and_report(tmp_path):
    out = tmp_path / "eval.json"
    assert main(["evaluate", "--n-rows", "400", "--n-boot", "3", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert "headline" in payload and payload["generator"]["generator"] == "copula"
    html = tmp_path / "r.html"
    assert (
        main(
            [
                "report",
                "--n-rows",
                "400",
                "--n-boot",
                "2",
                "--generator",
                "bayesnet",
                "--epsilon",
                "4",
                "--out",
                str(html),
            ]
        )
        == 0
    )
    assert "Differential privacy accounting" in html.read_text()


def test_cli_parser_rejects_unknown_generator():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["synthesize", "--generator", "vae"])


def test_json_formatter_includes_request_id_and_extras():
    formatter = JsonFormatter()
    record = logging.LogRecord("synthia.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.ms = 1.5
    token = request_id_var.set("rid-1")
    try:
        payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello world" and payload["request_id"] == "rid-1" and payload["ms"] == 1.5
    assert payload["level"] == "INFO"


def test_metrics_registry_renders_all_types():
    reg = MetricsRegistry()
    reg.describe("x_total", "count")
    reg.inc("x_total", {"a": 'b"c'}, 2)
    reg.set("g", 3.5)
    reg.observe({"path": "/p"}, 0.02)
    text = reg.render()
    assert 'x_total{a="b\\"c"} 2.0' in text
    assert "# TYPE g gauge" in text and "g 3.5" in text
    assert 'synthia_http_request_duration_seconds_bucket{le="0.025",path="/p"} 1' in text
    assert 'synthia_http_request_duration_seconds_count{path="/p"} 1' in text
    assert np.isclose(
        float(text.split('synthia_http_request_duration_seconds_sum{path="/p"} ')[1].split("\n")[0]), 0.02
    )
