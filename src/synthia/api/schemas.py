"""Pydantic request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

GeneratorName = Literal["copula", "ctgan", "bayesnet"]


class DPSpec(BaseModel):
    epsilon: float = Field(gt=0, le=1000, description="Target epsilon; the response reports the epsilon spent")
    delta: float = Field(default=1e-5, gt=0, lt=1)


class ConstraintSpec(BaseModel):
    column: str = Field(min_length=1, max_length=128)
    equals: str | float | None = None
    one_of: list[str] | None = Field(default=None, max_length=64)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _restricts_something(self) -> ConstraintSpec:
        if self.equals is None and not self.one_of and self.minimum is None and self.maximum is None:
            raise ValueError("a constraint must set equals, one_of, minimum or maximum")
        return self


class FitRequest(BaseModel):
    source: Literal["patients"] = Field(default="patients", description="Reference dataset to fit on")
    n_rows: int = Field(default=4000, ge=200, le=50_000, description="Rows of the reference cohort to draw")
    generator: GeneratorName = "copula"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    dp: DPSpec | None = None
    epochs: int | None = Field(default=None, ge=1, le=2000, description="CTGAN epochs; ignored by other generators")
    rows: list[dict[str, Any]] | None = Field(
        default=None, max_length=50_000, description="Fit on these rows instead of the reference cohort"
    )


class ModelInfo(BaseModel):
    model_id: str
    generator: str
    n_train: int
    schema_columns: list[str]
    dp: dict[str, float] | None
    privacy: dict[str, Any] | None
    fit_seconds: float
    details: dict[str, Any]


class SynthesizeRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)
    n_rows: int = Field(default=100, ge=1, le=20_000)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    constraints: list[ConstraintSpec] = Field(default_factory=list, max_length=32)
    max_rounds: int = Field(default=10, ge=1, le=50)


class SynthesizeResponse(BaseModel):
    model_id: str
    n_rows: int
    columns: list[str]
    rows: list[dict[str, Any]]
    sampling: dict[str, Any]
    privacy: dict[str, Any] | None


class EvaluateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)
    n_synthetic: int | None = Field(default=None, ge=50, le=20_000, description="Defaults to the training size")
    n_boot: int = Field(default=100, ge=0, le=1000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    target: str | None = None


class EvaluateResponse(BaseModel):
    model_id: str
    headline: dict[str, float | None]
    result: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    models_cached: int
    auth_enabled: bool


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
