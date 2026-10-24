"""A small Prometheus text-format registry so the service has no runtime dependency on prometheus_client."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    body = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
    return "{" + body + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """Counters, gauges and one latency histogram, rendered in exposition format 0.0.4."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._hist: dict[tuple[tuple[str, str], ...], list[float]] = {}
        self._help: dict[str, str] = {}

    def describe(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._counters[name][key] = self._counters[name].get(key, 0.0) + value

    def set(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._gauges[name][key] = value

    def observe(self, labels: dict[str, str], seconds: float) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._hist.setdefault(key, []).append(seconds)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                lines += self._header(name, "counter")
                lines += [f"{name}{_labels(dict(k))} {v}" for k, v in sorted(series.items())]
            for name, series in sorted(self._gauges.items()):
                lines += self._header(name, "gauge")
                lines += [f"{name}{_labels(dict(k))} {v}" for k, v in sorted(series.items())]
            lines += self._render_histogram("synthia_http_request_duration_seconds")
        return "\n".join(lines) + "\n"

    def _header(self, name: str, kind: str) -> list[str]:
        return [f"# HELP {name} {self._help.get(name, name)}", f"# TYPE {name} {kind}"]

    def _render_histogram(self, name: str) -> Iterable[str]:
        out = self._header(name, "histogram")
        for key, values in sorted(self._hist.items()):
            base = dict(key)
            for bound in _BUCKETS:
                count = sum(1 for v in values if v <= bound)
                out.append(f"{name}_bucket{_labels({**base, 'le': str(bound)})} {count}")
            out.append(f"{name}_bucket{_labels({**base, 'le': '+Inf'})} {len(values)}")
            out.append(f"{name}_sum{_labels(base)} {sum(values)}")
            out.append(f"{name}_count{_labels(base)} {len(values)}")
        return out


registry = MetricsRegistry()
registry.describe("synthia_http_requests_total", "HTTP requests by method, route and status")
registry.describe("synthia_http_request_duration_seconds", "HTTP request latency in seconds")
registry.describe("synthia_rows_generated_total", "Synthetic rows returned by generator")
registry.describe("synthia_models_cached", "Fitted generators held in memory")
registry.describe("synthia_fits_total", "Generator fits by generator and DP flag")
