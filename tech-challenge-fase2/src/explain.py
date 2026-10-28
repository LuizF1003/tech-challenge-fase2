"""Explicações numéricas que servem de contexto (grounding) para a LLM.

A LLM não "vê" o modelo: ela recebe um contexto estruturado com a probabilidade,
o limiar, as métricas do modelo e os fatores que mais pesaram para *esta*
paciente. Assim a explicação gerada fica ancorada em números reais e pode ser
verificada automaticamente (ver ``src/llm/evaluation.py``).

Contribuição local por oclusão (model-agnostic): para cada atributo, troca-se
o valor da paciente pela mediana do treino e mede-se quanto a probabilidade de
malignidade muda. Funciona igual para Regressão Logística e Random Forest e não
depende do SHAP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_BASE_PT = {
    "radius": "raio",
    "texture": "textura",
    "perimeter": "perímetro",
    "area": "área",
    "smoothness": "suavidade",
    "compactness": "compacidade",
    "concavity": "concavidade",
    "concave points": "pontos côncavos",
    "concave_points": "pontos côncavos",
    "symmetry": "simetria",
    "fractal_dimension": "dimensão fractal",
}
_SUFFIX_PT = {"mean": "média", "se": "erro-padrão", "worst": "pior valor"}


def feature_label(name: str) -> str:
    """'concave points_worst' -> 'pontos côncavos (pior valor)'."""
    base, _, suffix = name.rpartition("_")
    if base in _BASE_PT and suffix in _SUFFIX_PT:
        return f"{_BASE_PT[base]} ({_SUFFIX_PT[suffix]})"
    return name


def local_contributions(model: Any, sample: pd.DataFrame, medians: dict[str, float],
                        top_k: int = 5) -> tuple[float, list[dict[str, Any]]]:
    """Devolve (probabilidade, fatores ordenados por impacto absoluto)."""
    sample = sample.iloc[[0]]
    proba = float(model.predict_proba(sample)[:, 1][0])
    columns = list(sample.columns)
    occluded = pd.concat([sample] * len(columns), ignore_index=True)
    for i, col in enumerate(columns):
        occluded.loc[i, col] = medians[col]
    deltas = proba - model.predict_proba(occluded)[:, 1]
    order = np.argsort(-np.abs(deltas))[:top_k]
    factors = []
    for i in order:
        col = columns[i]
        factors.append({
            "atributo": col,
            "nome": feature_label(col),
            "valor": round(float(sample.iloc[0][col]), 4),
            "mediana_treino": round(float(medians[col]), 4),
            "impacto": round(float(deltas[i]), 4),
            "direcao": "aumenta o risco" if deltas[i] > 0 else "reduz o risco",
        })
    return proba, factors


def risk_band(proba: float, threshold: float) -> str:
    if proba >= max(threshold, 0.8):
        return "alto"
    if proba >= threshold:
        return "moderado (acima do limiar)"
    if proba >= threshold * 0.5:
        return "baixo, porém não desprezível"
    return "baixo"


def build_case_context(model: Any, card: dict[str, Any], sample: pd.DataFrame, top_k: int = 5) -> dict[str, Any]:
    """Contexto estruturado de um caso — é o que vai para o prompt da LLM."""
    features = card["features"]
    sample = sample.loc[:, features]
    proba, factors = local_contributions(model, sample, card["train_medians"], top_k)
    threshold = float(getattr(model, "threshold", 0.5))
    for f in factors:
        f["media_benignos"] = round(float(card["class_means"]["benigno"][f["atributo"]]), 4)
        f["media_malignos"] = round(float(card["class_means"]["maligno"][f["atributo"]]), 4)
    tm = card["test_metrics"]
    return {
        "probabilidade_malignidade": round(proba, 4),
        "probabilidade_percentual": round(proba * 100, 1),
        "limiar_decisao": round(threshold, 2),
        "classificacao_modelo": "suspeito de malignidade" if proba >= threshold else "sem alerta",
        "faixa_risco": risk_band(proba, threshold),
        "fatores_principais": factors,
        "modelo": {
            "nome": card["display_name"],
            "origem": "otimizado por algoritmo genético" if card["source"] != "baseline" else "modelo original (Fase 1)",
            "recall_teste_percentual": round(tm["recall"] * 100, 1),
            "precisao_teste_percentual": round(tm["precision"] * 100, 1),
            "roc_auc_teste": round(tm["roc_auc"], 3),
            "falsos_negativos_teste": tm["false_negatives"],
            "falsos_positivos_teste": tm["false_positives"],
            "amostras_teste": card["test_size"],
        },
    }
