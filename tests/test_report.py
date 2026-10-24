from html.parser import HTMLParser

from synthia.evaluation import evaluate
from synthia.report import render_report
from synthia.report.charts import (
    bar_chart_with_ci,
    diverging_color,
    grouped_columns,
    heatmap,
    sequential_color,
)


class _Checker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.svg = 0
        self.external = []

    def handle_starttag(self, tag, attrs):
        if tag == "svg":
            self.svg += 1
        for key, value in attrs:
            if key in {"src", "href"} and value and value.startswith(("http:", "https:", "//")):
                self.external.append(value)
        if tag not in {"meta", "br", "line", "path", "rect", "circle"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in {"meta", "br", "line", "path", "rect", "circle"}:
            return
        assert self.stack and self.stack[-1] == tag, f"unbalanced tag {tag}"
        self.stack.pop()


def test_report_is_self_contained_and_well_formed(schema, train, holdout, synthetic, copula):
    result = evaluate(train, holdout, synthetic, schema, target="readmitted_30d", n_boot=5)
    html = render_report(
        train,
        holdout,
        synthetic,
        schema,
        result,
        copula.describe(),
        title="Synthia test",
        command="synthia report",
    )
    checker = _Checker()
    checker.feed(html)
    assert checker.stack == [] or checker.stack == ["html"]
    assert checker.external == []
    assert checker.svg >= len(schema) + 4
    for needle in (
        "Synthia test",
        "Membership inference AUC",
        "readmitted_30d",
        "Per-column distance",
        "synthia report",
    ):
        assert needle in html
    assert chr(0x2014) not in html and chr(0x2013) not in html


def test_report_shows_dp_accounting(schema, train, holdout):
    from synthia.generators.copula import GaussianCopulaGenerator
    from synthia.privacy import DPConfig

    gen = GaussianCopulaGenerator(schema, seed=0, dp=DPConfig(2.0)).fit(train)
    syn = gen.sample(len(train), seed=0)
    result = evaluate(train, holdout, syn, schema, n_boot=0)
    html = render_report(train, holdout, syn, schema, result, gen.describe())
    assert "Differential privacy accounting" in html
    assert "second moment of clipped normal scores" in html
    assert "spent epsilon" in html


def test_chart_helpers_produce_svg():
    bars = bar_chart_with_ci([("age", 0.1, 0.05, 0.15), ("sex", 0.02, 0.0, 0.04)], "t")
    assert bars.startswith("<svg") and bars.count("<path") == 2 and "<title>" in bars
    cols = grouped_columns(["a", "b"], [0.5, 0.5], [0.4, 0.6], "t")
    assert cols.count("<path") == 4
    import numpy as np

    heat = heatmap(np.eye(3), ["x", "y", "z"], "t", diverging=True)
    assert heat.count("<rect") == 9
    assert bar_chart_with_ci([], "t") == "" and grouped_columns([], [], [], "t") == ""


def test_colour_ramps_are_monotone_and_bounded():
    seq = [sequential_color(v, 1.0) for v in (0.0, 0.5, 1.0)]
    assert seq[0] == "#cde2fb" and seq[-1] == "#0d366b" and len(set(seq)) == 3
    assert diverging_color(0.0) == "#f0efec"
    assert diverging_color(-1.0) == "#2a78d6" and diverging_color(1.0) == "#e34948"
    assert diverging_color(float("nan")) == "#f0efec"
