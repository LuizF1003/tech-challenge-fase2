"""Função fitness baseada em validação cruzada estratificada.

fitness = w_recall·Recall(maligno) + w_f1·F1(maligno) + w_auc·ROC-AUC

* Calculada **apenas no conjunto de treino** (k-fold estratificado). O conjunto
  de teste fica intocado até a comparação final — caso contrário o AG
  "aprenderia" o teste e a comparação com o modelo original seria enviesada.
* Recall tem o maior peso porque um falso negativo (câncer não sinalizado) é
  o erro mais grave na triagem. F1 entra para impedir a solução trivial
  "tudo é maligno" (recall = 1, precisão baixa) e ROC-AUC mede a qualidade da
  ordenação das probabilidades, independente do limiar.
* As probabilidades out-of-fold são guardadas em cache por hiperparâmetros
  *sem* o limiar: indivíduos que só diferem no ``threshold`` não precisam
  re-treinar o modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.models import build_model

DEFAULT_WEIGHTS = {"recall": 0.45, "f1": 0.35, "roc_auc": 0.20}


def classification_metrics(y_true, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "false_negatives": int(((pred == 0) & (np.asarray(y_true) == 1)).sum()),
        "false_positives": int(((pred == 1) & (np.asarray(y_true) == 0)).sum()),
    }


def composite_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(w * metrics[k] for k, w in weights.items()))


@dataclass
class CVFitness:
    model_name: str
    X: pd.DataFrame
    y: pd.Series
    n_splits: int = 5
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    seed: int = 42

    def __post_init__(self):
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self._proba_cache: dict[tuple, np.ndarray] = {}
        self.model_fits = 0

    def oof_proba(self, params: dict[str, Any]) -> np.ndarray:
        key = tuple(sorted((k, repr(v)) for k, v in params.items() if k != "threshold"))
        if key not in self._proba_cache:
            model = build_model(self.model_name, params).estimator  # pipeline sem o limiar
            self._proba_cache[key] = cross_val_predict(
                model, self.X, self.y, cv=self.cv, method="predict_proba"
            )[:, 1]
            self.model_fits += 1
        return self._proba_cache[key]

    def __call__(self, params: dict[str, Any]) -> tuple[float, dict[str, float]]:
        proba = self.oof_proba(params)
        metrics = classification_metrics(self.y, proba, float(params.get("threshold", 0.5)))
        score = composite_score(metrics, self.weights)
        metrics["fitness"] = score
        return score, metrics
