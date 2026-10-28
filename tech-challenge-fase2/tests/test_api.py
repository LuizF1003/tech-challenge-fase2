import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data import load_split


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample():
    return load_split().X_test.iloc[0].to_dict()


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"


def test_predict(client, sample):
    r = client.post("/predict", json={"amostras": [sample, sample]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["resultados"]) == 2 and "aviso" in body
    assert "x-request-id" in r.headers


def test_predict_missing_columns(client):
    r = client.post("/predict", json={"amostras": [{"radius_mean": 10}]})
    assert r.status_code == 422
    assert "colunas_ausentes" in r.json()["detail"]


def test_interpret_offline(client, sample):
    r = client.post("/interpret", json={"amostra": sample, "provider": "offline"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid_json"] and body["structured"]["aviso"]


def test_ask_offline(client, sample):
    r = client.post("/ask", json={"amostra": sample, "pergunta": "Qual fator mais pesou?", "provider": "offline"})
    assert r.status_code == 200 and r.json()["structured"]["resposta"]


def test_invalid_provider(client, sample):
    assert client.post("/interpret", json={"amostra": sample, "provider": "nao-existe"}).status_code == 400


def test_metrics_exposed(client, sample):
    client.post("/predict", json={"amostras": [sample]})
    text = client.get("/metrics").text
    assert "http_requests_total" in text and "predictions_total" in text and "model_info" in text


def test_optimize_job(client):
    r = client.post("/optimize", json={"modelo": "logistic_regression", "population_size": 4, "generations": 1})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(120):
        status = client.get(f"/optimize/{job_id}").json()
        if status["status"] != "executando":
            break
        time.sleep(0.5)
    assert status["status"] == "concluido"
    assert status["resultado"]["best_fitness"] > 0.5
