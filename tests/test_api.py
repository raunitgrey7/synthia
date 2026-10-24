import pytest
from fastapi.testclient import TestClient

from synthia.api.app import ModelStore, create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(ModelStore(capacity=2)))


@pytest.fixture(scope="module")
def model_id(client):
    response = client.post("/v1/fit", json={"generator": "copula", "n_rows": 600, "seed": 2})
    assert response.status_code == 200
    return response.json()["model_id"]


def test_health_and_request_id_echo(client):
    response = client.get("/health", headers={"X-Request-ID": "req-42"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"] == "req-42"
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 16


def test_schema_endpoint(client):
    body = client.get("/v1/schema").json()
    assert [c["name"] for c in body["columns"]][:2] == ["age", "sex"]


def test_fit_is_cached_by_request_hash(client, model_id):
    again = client.post("/v1/fit", json={"generator": "copula", "n_rows": 600, "seed": 2})
    assert again.json()["model_id"] == model_id
    listed = client.get("/v1/models").json()
    assert any(m["model_id"] == model_id for m in listed)
    assert client.get(f"/v1/models/{model_id}").json()["generator"] == "copula"


def test_fit_validation_errors(client):
    assert client.post("/v1/fit", json={"generator": "nope"}).status_code == 422
    assert client.post("/v1/fit", json={"n_rows": 10}).status_code == 422
    assert client.post("/v1/fit", json={"dp": {"epsilon": -1}}).status_code == 422


def test_fit_on_uploaded_rows(client, cohort):
    rows = cohort.head(120).to_dict(orient="records")
    response = client.post("/v1/fit", json={"generator": "bayesnet", "rows": rows, "seed": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["n_train"] == 90
    sampled = client.post("/v1/synthesize", json={"model_id": body["model_id"], "n_rows": 7, "seed": 0})
    assert sampled.status_code == 200 and sampled.json()["n_rows"] == 7


def test_fit_rejects_rows_with_missing_values(client):
    response = client.post("/v1/fit", json={"rows": [{"a": 1, "b": None}] * 60})
    assert response.status_code == 422
    assert "missing" in response.json()["detail"]


def test_synthesize_honours_constraints(client, model_id):
    payload = {
        "model_id": model_id,
        "n_rows": 25,
        "seed": 5,
        "constraints": [{"column": "age", "minimum": 50, "maximum": 60}, {"column": "sex", "one_of": ["M"]}],
    }
    response = client.post("/v1/synthesize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 25 and len(body["rows"]) == 25
    assert all(50 <= r["age"] <= 60 and r["sex"] == "M" for r in body["rows"])
    assert body["sampling"]["acceptance_rate"] < 1.0
    assert body["privacy"] is None


def test_synthesize_errors(client, model_id):
    assert client.post("/v1/synthesize", json={"model_id": "missing"}).status_code == 404
    assert client.post("/v1/synthesize", json={"model_id": model_id, "n_rows": 0}).status_code == 422
    assert (
        client.post("/v1/synthesize", json={"model_id": model_id, "constraints": [{"column": "age"}]}).status_code
        == 422
    )
    impossible = client.post(
        "/v1/synthesize",
        json={
            "model_id": model_id,
            "n_rows": 3,
            "constraints": [{"column": "age", "minimum": 500}],
            "max_rounds": 2,
        },
    )
    assert impossible.status_code == 422
    assert "too selective" in impossible.json()["detail"]
    assert impossible.json()["request_id"]


def test_dp_fit_returns_privacy_report(client):
    response = client.post("/v1/fit", json={"generator": "copula", "n_rows": 600, "seed": 2, "dp": {"epsilon": 2.0}})
    assert response.status_code == 200
    privacy = response.json()["privacy"]
    assert privacy["epsilon_spent"] <= 2.0
    sampled = client.post("/v1/synthesize", json={"model_id": response.json()["model_id"], "n_rows": 3})
    assert sampled.json()["privacy"]["epsilon_target"] == 2.0


def test_evaluate_and_report(client, model_id):
    response = client.post("/v1/evaluate", json={"model_id": model_id, "n_boot": 5, "seed": 1})
    assert response.status_code == 200
    headline = response.json()["headline"]
    assert 0 <= headline["membership_auc"] <= 1
    assert headline["exact_match_rate"] == 0.0
    report = client.get(f"/v1/report/{model_id}?n_boot=2")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/html")
    assert "<svg" in report.text


def test_metrics_in_prometheus_format(client):
    text = client.get("/metrics").text
    assert "# TYPE synthia_http_requests_total counter" in text
    assert 'synthia_http_requests_total{method="GET",path="/health",status="200"}' in text
    assert "synthia_http_request_duration_seconds_bucket" in text
    assert "synthia_rows_generated_total" in text


def test_api_key_guard(client, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/health").json()["auth_enabled"] is True


def test_store_evicts_least_recently_used(client):
    for seed in (11, 12, 13):
        client.post("/v1/fit", json={"generator": "copula", "n_rows": 300, "seed": seed})
    assert len(client.get("/v1/models").json()) == 2
