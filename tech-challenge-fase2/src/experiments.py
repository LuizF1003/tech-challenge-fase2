"""Execução dos experimentos: baseline (Fase 1) x AG x busca aleatória.

Protocolo (evita vazamento de dados):
1. Split 80/20 idêntico ao da Fase 1. O teste (114 amostras) não é usado pelo AG.
2. O AG maximiza o fitness de CV 5-fold **no treino**.
3. Cada melhor indivíduo é re-treinado em todo o treino e avaliado:
   (a) no teste hold-out da Fase 1 e
   (b) numa CV repetida 5x3 com outra semente (estimativa mais estável,
       já que o teste tem só 42 casos malignos).
4. O modelo final é escolhido pelo fitness de CV (não pelo teste), salvo em
   ``models/best_model.joblib`` e servido pela API.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

from src.data import PROJECT_DIR, DatasetSplit, load_split
from src.fitness import CVFitness, classification_metrics, composite_score
from src.ga import GAConfig, GeneticAlgorithm, random_search
from src.models import BASELINE_PARAMS, DISPLAY_NAMES, build_model
from src.monitoring import RunTracker, dump_json, log_event
from src.spaces import SPACES

logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_DIR / "configs" / "experiments.yaml"
RESULTS_DIR = PROJECT_DIR / "reports" / "results"
RUNS_DIR = PROJECT_DIR / "reports" / "runs"
MODELS_DIR = PROJECT_DIR / "models"

GA_KEYS = set(GAConfig.__dataclass_fields__)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_metrics(model_name: str, params: dict[str, Any], data: DatasetSplit) -> dict[str, Any]:
    model = build_model(model_name, params).fit(data.X_train, data.y_train)
    proba = model.predict_proba(data.X_test)[:, 1]
    return classification_metrics(data.y_test, proba, model.threshold)


def robust_cv(model_name: str, params: dict[str, Any], data: DatasetSplit, folds: int, repeats: int,
              seed: int, weights: dict[str, float]) -> dict[str, Any]:
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    X, y = data.X_train, data.y_train
    rows = []
    for tr, va in rskf.split(X, y):
        model = build_model(model_name, params).fit(X.iloc[tr], y.iloc[tr])
        m = classification_metrics(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1], model.threshold)
        m["fitness"] = composite_score(m, weights)
        rows.append(m)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "fitness"]
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in keys}


def _ga_config(spec: dict[str, Any], quick: bool) -> GAConfig:
    cfg = GAConfig(**{k: v for k, v in spec.items() if k in GA_KEYS})
    if quick:  # modo rápido para testes/CI
        cfg.population_size = min(cfg.population_size, 6)
        cfg.generations = min(cfg.generations, 2)
        cfg.elitism = min(cfg.elitism, 1)
    return cfg


def run_model(model_name: str, data: DatasetSplit, config: dict[str, Any],
              experiments: list[str] | None = None, quick: bool = False) -> dict[str, Any]:
    fit_cfg = config["fitness"]
    final_cfg = config["final_validation"]
    weights = fit_cfg["weights"]
    space = SPACES[model_name]

    def new_fitness() -> CVFitness:
        return CVFitness(model_name, data.X_train, data.y_train, n_splits=fit_cfg["cv_folds"], weights=weights)

    def evaluate(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "test_metrics": test_metrics(model_name, params, data),
            "robust_cv": robust_cv(model_name, params, data, final_cfg["cv_folds"],
                                   1 if quick else final_cfg["repeats"], final_cfg["seed"], weights),
        }

    out: dict[str, Any] = {"display_name": DISPLAY_NAMES[model_name], "experiments": {}}

    # ---- Baseline (hiperparâmetros da Fase 1) ----
    base_fit = new_fitness()
    score, metrics = base_fit(BASELINE_PARAMS[model_name])
    out["baseline"] = {"params": BASELINE_PARAMS[model_name], "cv_fitness": score, "cv_metrics": metrics,
                       **evaluate(BASELINE_PARAMS[model_name])}
    log_event(logger, "baseline avaliado", model=model_name, cv_fitness=round(score, 4),
              test_recall=out["baseline"]["test_metrics"]["recall"])

    # ---- Experimentos do AG ----
    names = experiments or list(config["experiments"])
    for exp_name in names:
        spec = config["experiments"][exp_name]
        ga_cfg = _ga_config(spec, quick)
        tracker = RunTracker(f"{model_name}-{exp_name}", RUNS_DIR, {"model": model_name, **asdict(ga_cfg)})
        fitness = new_fitness()
        initial = [BASELINE_PARAMS[model_name]] if spec.get("seed_with_baseline") else None
        ga = GeneticAlgorithm(
            space, fitness, ga_cfg, initial=initial,
            on_generation=lambda s, t=tracker: t.log_metrics(
                s.generation, best_fitness=s.best_fitness, mean_fitness=s.mean_fitness,
                diversity=s.diversity, evaluations=s.evaluations),
        )
        log_event(logger, "iniciando experimento", model=model_name, experiment=exp_name)
        result = ga.run()
        record = {
            "description": spec.get("description", ""),
            "config": result.config,
            "best_params": result.best_params,
            "cv_fitness": result.best_fitness,
            "cv_metrics": result.best_metrics,
            "history": [asdict(h) for h in result.history],
            "evaluations": result.evaluations,
            "cache_hits": result.cache_hits,
            "model_fits": fitness.model_fits,
            "elapsed_s": result.elapsed_s,
            "stopped_early": result.stopped_early,
            "run_id": tracker.run_id,
            **evaluate(result.best_params),
        }
        tracker.finish({k: v for k, v in record.items() if k != "history"})
        out["experiments"][exp_name] = record
        log_event(logger, "experimento concluído", model=model_name, experiment=exp_name,
                  cv_fitness=round(result.best_fitness, 4), evaluations=result.evaluations,
                  elapsed_s=result.elapsed_s)

    # ---- Busca aleatória com o mesmo orçamento ----
    rs_cfg = config.get("random_search")
    if rs_cfg:
        ref = rs_cfg["reference_experiment"]
        budget = out["experiments"].get(ref, {}).get("evaluations") or 30
        fitness = new_fitness()
        result = random_search(space, fitness, budget, seed=rs_cfg["seed"])
        out["random_search"] = {
            "budget": budget, "reference_experiment": ref,
            "best_params": result.best_params, "cv_fitness": result.best_fitness,
            "cv_metrics": result.best_metrics, "elapsed_s": result.elapsed_s,
            "curve": [h.best_fitness for h in result.history],
            **evaluate(result.best_params),
        }
    return out


def select_best(results: dict[str, Any]) -> dict[str, Any]:
    """Escolhe o melhor modelo pelo fitness de CV robusta (nunca pelo teste)."""
    best = None
    for model_name, res in results["models"].items():
        candidates = {"baseline": res["baseline"], **res["experiments"]}
        for source, rec in candidates.items():
            score = rec["robust_cv"]["fitness"]["mean"]
            if best is None or score > best["robust_cv_fitness"]:
                best = {"model": model_name, "source": source, "params": rec["best_params"] if source != "baseline" else rec["params"],
                        "robust_cv_fitness": score, "test_metrics": rec["test_metrics"]}
    return best


def save_model(selection: dict[str, Any], data: DatasetSplit, results: dict[str, Any]) -> Path:
    model = build_model(selection["model"], selection["params"]).fit(data.X_train, data.y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "best_model.joblib"
    joblib.dump(model, path)
    card = {
        **deepcopy(selection),
        "display_name": DISPLAY_NAMES[selection["model"]],
        "features": data.feature_names,
        "train_size": int(len(data.X_train)),
        "test_size": int(len(data.X_test)),
        "fitness_weights": results["fitness"]["weights"],
        "train_medians": data.X_train.median().round(6).to_dict(),
        "class_means": {
            "benigno": data.X_train[data.y_train == 0].mean().round(6).to_dict(),
            "maligno": data.X_train[data.y_train == 1].mean().round(6).to_dict(),
        },
        "baseline_test_metrics": results["models"][selection["model"]]["baseline"]["test_metrics"],
    }
    dump_json(card, MODELS_DIR / "model_card.json")
    return path


def bootstrap_baseline_model() -> Path:
    """Salva o modelo original da Fase 1 quando os experimentos ainda não rodaram.

    Permite subir a API num ambiente novo (ex.: contêiner) sem esperar o AG.
    """
    data = load_split()
    config = load_config()
    name = "logistic_regression"
    params = BASELINE_PARAMS[name]
    tm = test_metrics(name, params, data)
    selection = {"model": name, "source": "baseline", "params": params, "robust_cv_fitness": None, "test_metrics": tm}
    results = {"fitness": config["fitness"], "models": {name: {"baseline": {"test_metrics": tm}}}}
    return save_model(selection, data, results)


def run_all(models: list[str] | None = None, experiments: list[str] | None = None,
            quick: bool = False, config_path: Path = CONFIG_PATH,
            results_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    data = load_split()
    results: dict[str, Any] = {
        "dataset": {"train": int(len(data.X_train)), "test": int(len(data.X_test)),
                    "features": len(data.feature_names),
                    "test_malignant": int(data.y_test.sum()), "train_malignant": int(data.y_train.sum())},
        "fitness": config["fitness"],
        "final_validation": config["final_validation"],
        "quick": quick,
        "models": {},
    }
    for model_name in models or config["models"]:
        results["models"][model_name] = run_model(model_name, data, config, experiments, quick)
    results["selected"] = select_best(results)
    save_model(results["selected"], data, results)
    out = results_path or (RESULTS_DIR / ("experiments_quick.json" if quick else "experiments.json"))
    dump_json(results, out)
    log_event(logger, "resultados salvos", path=str(out), selected=results["selected"]["model"],
              source=results["selected"]["source"])
    return results
