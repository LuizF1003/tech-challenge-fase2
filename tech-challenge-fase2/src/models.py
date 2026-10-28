"""Modelos de diagnóstico.

* ``baseline_models()`` reproduz os dois modelos da Fase 1 com os mesmos
  hiperparâmetros (Regressão Logística e Random Forest).
* ``build_model(name, params)`` monta o mesmo pipeline (imputação + padronização
  + classificador) a partir de um dicionário de hiperparâmetros decodificado de
  um cromossomo do algoritmo genético.
* ``ThresholdClassifier`` acrescenta o limiar de decisão como hiperparâmetro —
  em triagem de câncer, mover o limiar é a forma mais direta de trocar falsos
  negativos por falsos positivos, por isso ele também é otimizado pelo AG.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE

MODEL_NAMES = ("logistic_regression", "random_forest")
DISPLAY_NAMES = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
}

_SK_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Envolve um pipeline e aplica um limiar de decisão customizado."""

    def __init__(self, estimator: Any = None, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    @property
    def pipeline(self) -> Pipeline:
        return self.estimator_


def _preprocessor() -> list[tuple[str, Any]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def _logistic(params: dict[str, Any]) -> LogisticRegression:
    l1_ratio = float(params.get("l1_ratio", 0.0))
    kwargs: dict[str, Any] = dict(
        C=float(params.get("C", 1.0)),
        class_weight=params.get("class_weight"),
        max_iter=5_000,
        random_state=RANDOM_STATE,
    )
    if l1_ratio > 0:
        kwargs["solver"] = "saga"
        kwargs["l1_ratio"] = l1_ratio
        kwargs["tol"] = 1e-3  # saga converge devagar; 1e-3 mantém a precisão com custo bem menor
        if _SK_VERSION < (1, 8):  # versões antigas exigem penalty explícito
            kwargs["penalty"] = "elasticnet"
    else:
        kwargs["solver"] = "lbfgs"
    return LogisticRegression(**kwargs)


def _random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth"),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", "sqrt"),
        criterion=params.get("criterion", "gini"),
        class_weight=params.get("class_weight"),
        bootstrap=bool(params.get("bootstrap", True)),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_model(name: str, params: dict[str, Any]) -> ThresholdClassifier:
    """Monta o modelo completo a partir de hiperparâmetros (inclui ``threshold``)."""
    if name == "logistic_regression":
        clf = _logistic(params)
    elif name == "random_forest":
        clf = _random_forest(params)
    else:
        raise ValueError(f"Modelo desconhecido: {name}. Use um de {MODEL_NAMES}.")
    pipeline = Pipeline(_preprocessor() + [("model", clf)])
    return ThresholdClassifier(pipeline, threshold=float(params.get("threshold", 0.5)))


# Hiperparâmetros originais da Fase 1 (src/pipeline.py -> split_and_train)
BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0, "l1_ratio": 0.0, "class_weight": "balanced", "threshold": 0.5,
    },
    "random_forest": {
        "n_estimators": 500, "max_depth": None, "min_samples_split": 2,
        "min_samples_leaf": 2, "max_features": "sqrt", "criterion": "gini",
        "class_weight": "balanced", "bootstrap": True, "threshold": 0.5,
    },
}


def baseline_models() -> dict[str, ThresholdClassifier]:
    return {name: build_model(name, params) for name, params in BASELINE_PARAMS.items()}


def positive_proba(model: Any, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
