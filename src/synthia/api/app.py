"""FastAPI application: fit, synthesize under constraints, evaluate and report, with operational endpoints.

Fitted generators live in an in-process LRU cache keyed by a short id so the expensive step (fitting, especially
the GAN) runs once and sampling requests stay cheap. Every request gets an id (taken from ``X-Request-ID`` or
generated), a JSON log line and a Prometheus observation. When ``APP_API_KEY`` is set every route except
``/health`` and ``/metrics`` requires that key in ``X-API-Key``.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from synthia import __version__
from synthia.api.metrics import registry
from synthia.api.schemas import (
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    FitRequest,
    HealthResponse,
    ModelInfo,
    SynthesizeRequest,
    SynthesizeResponse,
)
from synthia.constraints import Constraint, ConstraintError, sample_with_constraints
from synthia.evaluation import evaluate
from synthia.generators import Generator, make_generator
from synthia.logging import configure_logging, get_logger, request_id_var
from synthia.privacy import DPConfig
from synthia.reference import generate_patients, patient_schema, train_holdout_split
from synthia.report import render_report
from synthia.schema import TableSchema

log = get_logger("synthia.api")

PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}
MAX_CTGAN_EPOCHS_API = 300


@dataclass
class FittedModel:
    model_id: str
    generator: Generator
    schema: TableSchema
    train: pd.DataFrame
    holdout: pd.DataFrame
    fit_seconds: float
    source: str


class ModelStore:
    """Thread-safe LRU of fitted models."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, FittedModel] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, model: FittedModel) -> None:
        with self._lock:
            self._items[model.model_id] = model
            self._items.move_to_end(model.model_id)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            registry.set("synthia_models_cached", len(self._items))

    def get(self, model_id: str) -> FittedModel:
        with self._lock:
            model = self._items.get(model_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"unknown model_id {model_id!r}")
            self._items.move_to_end(model_id)
            return model

    def list(self) -> list[FittedModel]:
        with self._lock:
            return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


def _model_id(request: FitRequest) -> str:
    payload = request.model_dump_json(exclude={"rows"}) + (str(len(request.rows)) if request.rows else "")
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _model_info(model: FittedModel) -> ModelInfo:
    details = model.generator.describe()
    return ModelInfo(
        model_id=model.model_id,
        generator=model.generator.name,
        n_train=model.generator.n_train,
        schema_columns=model.schema.names,
        dp=details.get("dp"),
        privacy=details.get("privacy"),
        fit_seconds=round(model.fit_seconds, 3),
        details={k: v for k, v in details.items() if k not in {"dp", "privacy", "generator", "n_train", "fitted"}},
    )


def _frames_for(request: FitRequest) -> tuple[pd.DataFrame, pd.DataFrame, TableSchema, str]:
    if request.rows:
        frame = pd.DataFrame(request.rows)
        if frame.isna().any().any():
            raise HTTPException(status_code=422, detail="rows contain missing values")
        try:
            schema = TableSchema.infer(frame)
            frame = schema.coerce(frame)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(frame) < 50:
            raise HTTPException(status_code=422, detail="at least 50 rows are needed to fit and evaluate")
        train, holdout = train_holdout_split(frame, 0.25, request.seed)
        return train, holdout, schema, "uploaded rows"
    frame = generate_patients(request.n_rows, seed=request.seed)
    train, holdout = train_holdout_split(frame, 0.25, request.seed)
    return train, holdout, patient_schema(), f"patients (n={request.n_rows}, seed={request.seed})"


def _build_generator(request: FitRequest, schema: TableSchema) -> Generator:
    dp = None if request.dp is None else DPConfig(request.dp.epsilon, request.dp.delta)
    kwargs: dict[str, Any] = {}
    if request.generator == "ctgan":
        kwargs["epochs"] = min(request.epochs or 50, MAX_CTGAN_EPOCHS_API)
        kwargs["torch_threads"] = 2
    try:
        return make_generator(request.generator, schema, dp=dp, seed=request.seed, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _to_constraints(specs: list[Any]) -> list[Constraint]:
    return [
        Constraint(
            column=s.column,
            equals=s.equals,
            one_of=tuple(s.one_of) if s.one_of else (),
            minimum=s.minimum,
            maximum=s.maximum,
        )
        for s in specs
    ]


def create_app(store: ModelStore | None = None) -> FastAPI:
    """Build the application; a fresh store per app keeps tests isolated."""
    configure_logging()
    models = store if store is not None else ModelStore()
    app = FastAPI(
        title="Synthia",
        version=__version__,
        description="Privacy-preserving synthetic tabular data: fit, sample under constraints, evaluate, report.",
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    app.middleware("http")(_observer())
    _register_operational(app, models)
    _register_models(app, models)
    _register_synthesis(app, models)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail, "request_id": request_id_var.get()}
        )

    return app


def _observer() -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Middleware: request id, API key guard, Prometheus observation and one JSON log line per request."""

    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id") or secrets.token_hex(8)
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request) if _authorised(request) else _unauthorised(request_id)
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            registry.inc(
                "synthia_http_requests_total",
                {"method": request.method, "path": path, "status": str(response.status_code)},
            )
            registry.observe({"method": request.method, "path": path}, elapsed)
            response.headers["X-Request-ID"] = request_id
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed * 1000, 2),
                },
            )
        finally:
            request_id_var.reset(token)
        return response

    return observe


def _register_operational(app: FastAPI, models: ModelStore) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            models_cached=len(models),
            auth_enabled=bool(os.environ.get("APP_API_KEY")),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return registry.render()

    @app.get("/v1/schema")
    def schema() -> dict[str, Any]:
        return patient_schema().to_dict()


def _register_models(app: FastAPI, models: ModelStore) -> None:
    @app.get("/v1/models", response_model=list[ModelInfo])
    def list_models() -> list[ModelInfo]:
        return [_model_info(m) for m in models.list()]

    @app.post("/v1/fit", response_model=ModelInfo)
    def fit(request: FitRequest) -> ModelInfo:
        model_id = _model_id(request)
        try:
            return _model_info(models.get(model_id))
        except HTTPException:
            pass
        train, holdout, schema, source = _frames_for(request)
        generator = _build_generator(request, schema)
        started = time.perf_counter()
        try:
            generator.fit(train)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        model = FittedModel(model_id, generator, schema, train, holdout, time.perf_counter() - started, source)
        models.put(model)
        registry.inc("synthia_fits_total", {"generator": generator.name, "dp": str(request.dp is not None).lower()})
        log.info(
            "fit", extra={"model_id": model_id, "generator": generator.name, "seconds": round(model.fit_seconds, 3)}
        )
        return _model_info(model)

    @app.get("/v1/models/{model_id}", response_model=ModelInfo)
    def get_model(model_id: str) -> ModelInfo:
        return _model_info(models.get(model_id))


def _register_synthesis(app: FastAPI, models: ModelStore) -> None:
    @app.post("/v1/synthesize", response_model=SynthesizeResponse)
    def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
        model = models.get(request.model_id)
        try:
            frame, info = sample_with_constraints(
                model.generator,
                request.n_rows,
                _to_constraints(request.constraints),
                seed=request.seed,
                max_rounds=request.max_rounds,
            )
        except ConstraintError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        registry.inc("synthia_rows_generated_total", {"generator": model.generator.name}, value=len(frame))
        privacy = model.generator.privacy_report
        return SynthesizeResponse(
            model_id=model.model_id,
            n_rows=len(frame),
            columns=list(frame.columns),
            rows=json.loads(frame.to_json(orient="records")),
            sampling=info.to_dict(),
            privacy=None if privacy is None else privacy.to_dict(),
        )

    @app.post("/v1/evaluate", response_model=EvaluateResponse)
    def evaluate_model(request: EvaluateRequest) -> EvaluateResponse:
        model = models.get(request.model_id)
        synthetic = model.generator.sample(request.n_synthetic or len(model.train), seed=request.seed)
        try:
            result = evaluate(
                model.train,
                model.holdout,
                synthetic,
                model.schema,
                target=request.target,
                n_boot=request.n_boot,
                seed=request.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EvaluateResponse(model_id=model.model_id, headline=result.headline(), result=result.to_dict())

    @app.get("/v1/report/{model_id}", response_class=HTMLResponse)
    def report(model_id: str, n_boot: int = 50, seed: int = 0) -> HTMLResponse:
        model = models.get(model_id)
        n_boot = max(0, min(n_boot, 500))
        synthetic = model.generator.sample(len(model.train), seed=seed)
        result = evaluate(model.train, model.holdout, synthetic, model.schema, n_boot=n_boot, seed=seed)
        html = render_report(
            model.train,
            model.holdout,
            synthetic,
            model.schema,
            result,
            model.generator.describe(),
            title=f"Synthia report: {model.generator.name}",
            dataset_name=model.source,
        )
        return HTMLResponse(html)


def _authorised(request: Request) -> bool:
    key = os.environ.get("APP_API_KEY")
    if not key or request.url.path in PUBLIC_PATHS:
        return True
    provided = request.headers.get("x-api-key", "")
    return secrets.compare_digest(provided.encode(), key.encode())


def _unauthorised(request_id: str) -> Response:
    return JSONResponse(status_code=401, content={"detail": "missing or invalid API key", "request_id": request_id})


app = create_app()
