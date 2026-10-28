"""API da Fase 2 — modelo otimizado por AG + interpretação por LLM.

Rodar localmente:
    uvicorn src.api.main:app --reload --port 8000
Documentação interativa: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src import __version__
from src.api import metrics as M
from src.data import PROJECT_DIR, load_split
from src.experiments import RESULTS_DIR, bootstrap_baseline_model
from src.fitness import CVFitness
from src.ga import GAConfig, GeneticAlgorithm
from src.llm import DiagnosisInterpreter, LLMError, get_provider
from src.models import BASELINE_PARAMS, MODEL_NAMES
from src.monitoring import log_event, request_id_var, setup_logging
from src.spaces import SPACES

logger = logging.getLogger("api")
MODELS_DIR = PROJECT_DIR / "models"
DISCLAIMER = "Resultado de apoio à triagem. Não confirma nem descarta câncer e não substitui a avaliação médica."


class State:
    model: Any = None
    card: dict[str, Any] = {}
    jobs: dict[str, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=1)  # 1 otimização por réplica; escala via réplicas/Jobs


def load_model() -> None:
    path = MODELS_DIR / "best_model.joblib"
    if not path.exists():
        logger.warning("modelo otimizado não encontrado; salvando o modelo original da Fase 1")
        bootstrap_baseline_model()
    State.model = joblib.load(path)
    State.card = json.loads((MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))
    M.MODEL_INFO.labels(State.card["model"], State.card["source"]).set(1)
    log_event(logger, "modelo carregado", model=State.card["model"], source=State.card["source"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    load_model()
    yield
    State.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Tech Challenge Fase 2 — Diagnóstico otimizado por AG + LLM",
    description="Serviço educacional de apoio à triagem de câncer de mama. " + DISCLAIMER,
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def observability(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    token = request_id_var.set(rid)
    M.IN_FLIGHT.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        M.IN_FLIGHT.dec()
        if path != "/metrics":
            M.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            M.HTTP_LATENCY.labels(request.method, path).observe(elapsed)
            log_event(logger, "request", method=request.method, path=path, status=status,
                      duration_ms=round(elapsed * 1000, 1))
        request_id_var.reset(token)


# ---------------------------------------------------------------------- #
# Schemas                                                                  #
# ---------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    amostras: list[dict[str, float]] = Field(..., min_length=1, max_length=1000)
    limiar: float | None = Field(None, gt=0, lt=1, description="Sobrescreve o limiar otimizado")


class InterpretRequest(BaseModel):
    amostra: dict[str, float]
    estrategia: Literal["basico", "estruturado", "few_shot"] = "few_shot"
    provider: str | None = Field(None, description="offline | groq | ollama | openai | anthropic")


class AskRequest(BaseModel):
    amostra: dict[str, float]
    pergunta: str = Field(..., min_length=3, max_length=500)
    provider: str | None = None


class OptimizeRequest(BaseModel):
    modelo: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    population_size: int = Field(12, ge=4, le=60)
    generations: int = Field(8, ge=1, le=40)
    mutation_rate: float = Field(0.15, ge=0, le=1)
    crossover_rate: float = Field(0.9, ge=0, le=1)
    selection: Literal["tournament", "roulette", "rank"] = "tournament"
    crossover: Literal["uniform", "one_point", "blx"] = "blx"
    mutation: Literal["gaussian", "uniform"] = "gaussian"
    seed: int = 42


# ---------------------------------------------------------------------- #
def _frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    features = State.card["features"]
    df = pd.DataFrame(rows)
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise HTTPException(422, detail={"mensagem": "Amostra sem os campos esperados pelo modelo.",
                                         "colunas_ausentes": missing})
    return df.loc[:, features]


def _interpreter(provider_name: str | None) -> DiagnosisInterpreter:
    try:
        provider = get_provider(provider_name)
    except (ValueError, LLMError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return DiagnosisInterpreter(State.model, State.card, provider)


def _track_llm(result) -> None:
    status = "fallback" if result.used_fallback else ("ok" if not result.error else "erro")
    M.LLM_REQUESTS.labels(result.provider, status).inc()
    M.LLM_LATENCY.labels(result.provider).observe(result.latency_s)


# ---------------------------------------------------------------------- #
# Rotas                                                                    #
# ---------------------------------------------------------------------- #
@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["infra"])
def ready() -> dict[str, Any]:
    if State.model is None:
        raise HTTPException(503, "modelo não carregado")
    return {"status": "ready", "model": State.card["model"], "source": State.card["source"]}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model", tags=["modelo"])
def model_info() -> dict[str, Any]:
    keys = ["model", "display_name", "source", "params", "test_metrics", "baseline_test_metrics",
            "fitness_weights", "features"]
    return {k: State.card.get(k) for k in keys}


@app.post("/predict", tags=["modelo"])
def predict(req: PredictRequest) -> dict[str, Any]:
    X = _frame(req.amostras)
    proba = State.model.predict_proba(X)[:, 1]
    threshold = req.limiar or State.model.threshold
    out = []
    for p in proba:
        alert = bool(p >= threshold)
        M.PREDICTIONS.labels("alerta" if alert else "sem_alerta").inc()
        M.PREDICTION_PROBA.observe(float(p))
        out.append({"probabilidade_malignidade": round(float(p), 4), "limiar": round(float(threshold), 2),
                    "orientacao": "Encaminhar para avaliação médica" if alert else "Sem alerta pelo limiar técnico"})
    return {"modelo": State.card["display_name"], "origem": State.card["source"], "resultados": out,
            "aviso": DISCLAIMER}


@app.post("/interpret", tags=["llm"])
def interpret(req: InterpretRequest) -> dict[str, Any]:
    result = _interpreter(req.provider).explain_case(_frame([req.amostra]), strategy=req.estrategia)
    _track_llm(result)
    return {k: v for k, v in result.to_dict().items() if k != "raw"}


@app.post("/ask", tags=["llm"])
def ask(req: AskRequest) -> dict[str, Any]:
    result = _interpreter(req.provider).ask(_frame([req.amostra]), req.pergunta)
    _track_llm(result)
    return {k: v for k, v in result.to_dict().items() if k != "raw"}


@app.get("/experiments", tags=["algoritmo genético"])
def experiments() -> dict[str, Any]:
    path = RESULTS_DIR / "experiments.json"
    if not path.exists():
        raise HTTPException(404, "Execute `python -m scripts.run_experiments` primeiro.")
    res = json.loads(path.read_text(encoding="utf-8"))
    table = []
    for model, r in res["models"].items():
        rows = {"baseline": r["baseline"], **r["experiments"]}
        if "random_search" in r:
            rows["random_search"] = r["random_search"]
        for name, rec in rows.items():
            table.append({"modelo": model, "experimento": name, "cv_fitness": round(rec["cv_fitness"], 4),
                          **{f"teste_{k}": round(v, 4) if isinstance(v, float) else v
                             for k, v in rec["test_metrics"].items()}})
    return {"selecionado": res["selected"], "tabela": table}


@app.get("/report/performance", tags=["llm"])
def performance_report(provider: str | None = None) -> dict[str, Any]:
    path = RESULTS_DIR / "experiments.json"
    if not path.exists():
        raise HTTPException(404, "Execute `python -m scripts.run_experiments` primeiro.")
    result = _interpreter(provider).explain_performance(json.loads(path.read_text(encoding="utf-8")))
    _track_llm(result)
    return {k: v for k, v in result.to_dict().items() if k != "raw"}


def _run_job(job_id: str, req: OptimizeRequest) -> None:
    job = State.jobs[job_id]
    M.GA_JOBS_RUNNING.inc()
    try:
        data = load_split()
        fitness = CVFitness(req.modelo, data.X_train, data.y_train)
        cfg = GAConfig(population_size=req.population_size, generations=req.generations,
                       mutation_rate=req.mutation_rate, crossover_rate=req.crossover_rate,
                       selection=req.selection, crossover=req.crossover, mutation=req.mutation, seed=req.seed)

        def progress(stats):
            job["progress"] = {"geracao": stats.generation, "total": req.generations,
                               "melhor_fitness": round(stats.best_fitness, 4)}

        result = GeneticAlgorithm(SPACES[req.modelo], fitness, cfg, on_generation=progress).run()
        baseline_score, _ = fitness(BASELINE_PARAMS[req.modelo])
        job.update(status="concluido", resultado={
            "best_params": result.best_params, "best_fitness": result.best_fitness,
            "baseline_fitness": baseline_score, "cv_metrics": result.best_metrics,
            "evaluations": result.evaluations, "elapsed_s": result.elapsed_s,
            "history": [{"geracao": h.generation, "melhor": h.best_fitness, "media": h.mean_fitness,
                         "diversidade": h.diversity} for h in result.history],
        })
        M.GA_BEST_FITNESS.labels(req.modelo).set(result.best_fitness)
        log_event(logger, "job do AG concluído", job_id=job_id, fitness=round(result.best_fitness, 4))
    except Exception as exc:  # pragma: no cover
        logger.exception("job do AG falhou")
        job.update(status="erro", erro=str(exc))
    finally:
        M.GA_JOBS_RUNNING.dec()


@app.post("/optimize", status_code=202, tags=["algoritmo genético"])
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    if req.modelo not in MODEL_NAMES:
        raise HTTPException(400, "modelo inválido")
    job_id = uuid.uuid4().hex[:10]
    State.jobs[job_id] = {"job_id": job_id, "status": "executando", "parametros": req.model_dump()}
    State.executor.submit(_run_job, job_id, req)
    return {"job_id": job_id, "status_url": f"/optimize/{job_id}"}


@app.get("/optimize/{job_id}", tags=["algoritmo genético"])
def optimize_status(job_id: str) -> dict[str, Any]:
    if job_id not in State.jobs:
        raise HTTPException(404, "job não encontrado")
    return State.jobs[job_id]
