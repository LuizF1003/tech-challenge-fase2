"""Interpretação dos diagnósticos em linguagem natural.

``DiagnosisInterpreter`` junta modelo + contexto numérico + prompt + LLM:

    amostra ─► modelo (probabilidade) ─► explain.build_case_context ─► prompt
                                                                      │
                 resposta validada ◄─ parse/validação JSON ◄─ LLM ◄───┘
                         │
                         └─ falhou? usa o provedor offline (fallback)

Também gera o resumo de desempenho "modelo original x otimizado" para médicos
(``explain_performance``) e responde perguntas livres sobre um caso, preparando
a integração com dados textuais do Módulo 3.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.explain import build_case_context
from src.llm.prompts import (CASE_FIELDS, PERFORMANCE_FIELDS, PROMPT_VERSION, SYSTEM_CLINICAL,
                             case_prompt, performance_prompt)
from src.llm.providers import LLMError, LLMProvider, OfflineTemplateProvider

logger = logging.getLogger(__name__)

DISCLAIMER = "Apoio à decisão: não confirma nem descarta câncer e não substitui a avaliação médica."


@dataclass
class Interpretation:
    structured: dict[str, Any] | None
    text: str
    context: dict[str, Any]
    provider: str
    model: str
    strategy: str
    prompt_version: str
    latency_s: float
    valid_json: bool
    used_fallback: bool = False
    error: str | None = None
    raw: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extrai o primeiro objeto JSON do texto (tolera ```json ... ``` e texto extra)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def render_text(structured: dict[str, Any]) -> str:
    titles = {
        "resumo": "Resumo", "fatores_relevantes": "Fatores relevantes", "confiabilidade": "Confiabilidade",
        "recomendacao": "Recomendação", "limitacoes": "Limitações", "aviso": "Aviso",
        "insights": "Insights", "impacto_clinico": "Impacto clínico",
    }
    parts = []
    for key, value in structured.items():
        title = titles.get(key, key)
        if isinstance(value, list):
            parts.append(f"{title}:\n" + "\n".join(f"• {v}" for v in value))
        else:
            parts.append(f"{title}: {value}")
    return "\n\n".join(parts)


class DiagnosisInterpreter:
    def __init__(self, model: Any, card: dict[str, Any], provider: LLMProvider,
                 strategy: str = "few_shot", temperature: float = 0.2, fallback: bool = True):
        self.model = model
        self.card = card
        self.provider = provider
        self.strategy = strategy
        self.temperature = temperature
        self.fallback = fallback
        self._offline = OfflineTemplateProvider()

    # ------------------------------------------------------------------ #
    def _call(self, system: str, user: str, json_mode: bool, context: dict[str, Any],
              fields: list[str], strategy: str) -> Interpretation:
        start = time.perf_counter()
        try:
            resp = self.provider.generate(system, user, temperature=self.temperature, json_mode=json_mode,
                                          context=context)
            structured = extract_json(resp.text) if json_mode else None
            valid = structured is not None and all(structured.get(f) for f in fields)
            if json_mode and not valid:
                raise LLMError("Resposta da LLM fora do formato JSON esperado.")
            if structured is not None and fields == CASE_FIELDS and not structured.get("aviso"):
                structured["aviso"] = DISCLAIMER
            text = render_text(structured) if structured else resp.text
            return Interpretation(structured, text, context, resp.provider, resp.model, strategy, PROMPT_VERSION,
                                  resp.latency_s, valid, raw=resp.text)
        except LLMError as exc:
            logger.warning("LLM falhou (%s): %s", self.provider.name, exc)
            if not self.fallback or self.provider.name == "offline":
                return Interpretation(None, "", context, self.provider.name, self.provider.model, strategy,
                                      PROMPT_VERSION, round(time.perf_counter() - start, 3), False, error=str(exc))
            resp = self._offline.generate(system, user, json_mode=True, context=context)
            structured = json.loads(resp.text)
            return Interpretation(structured, render_text(structured), context, resp.provider, resp.model,
                                  strategy, PROMPT_VERSION, round(time.perf_counter() - start, 3), True,
                                  used_fallback=True, error=str(exc), raw=resp.text)

    # ------------------------------------------------------------------ #
    def explain_case(self, sample: pd.DataFrame | dict[str, Any], strategy: str | None = None) -> Interpretation:
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        strategy = strategy or self.strategy
        context = build_case_context(self.model, self.card, sample)
        system, user, json_mode = case_prompt(context, strategy)
        return self._call(system, user, json_mode, context, CASE_FIELDS, strategy)

    def explain_performance(self, results: dict[str, Any]) -> Interpretation:
        context = performance_context(results)
        system, user, json_mode = performance_prompt(context)
        return self._call(system, user, json_mode, context, PERFORMANCE_FIELDS, "performance")

    def ask(self, sample: pd.DataFrame | dict[str, Any], question: str) -> Interpretation:
        """Pergunta livre de um(a) médico(a) sobre o caso, respondida com base no contexto."""
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        context = build_case_context(self.model, self.card, sample)
        user = (f"CONTEXTO:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\nPERGUNTA: {question}\n\n"
                'Responda em JSON: {"resposta": "...", "aviso": "..."}')
        try:
            resp = self.provider.generate(SYSTEM_CLINICAL, user, temperature=self.temperature, json_mode=True,
                                          context=context)
            structured = extract_json(resp.text) or {"resposta": resp.text, "aviso": DISCLAIMER}
        except LLMError as exc:
            return Interpretation(None, "", context, self.provider.name, self.provider.model, "ask",
                                  PROMPT_VERSION, 0.0, False, error=str(exc))
        if "resumo" in structured and "resposta" not in structured:  # provedor offline
            structured = {"resposta": structured["resumo"], "aviso": structured.get("aviso", DISCLAIMER)}
        structured.setdefault("aviso", DISCLAIMER)
        return Interpretation(structured, render_text(structured), context, resp.provider, resp.model, "ask",
                              PROMPT_VERSION, resp.latency_s, True, raw=resp.text)


def performance_context(results: dict[str, Any]) -> dict[str, Any]:
    """Resumo numérico "original x otimizado" do modelo selecionado."""
    sel = results["selected"]
    res = results["models"][sel["model"]]
    base = res["baseline"]["test_metrics"]
    opt = sel["test_metrics"]

    def fmt(m: dict[str, Any]) -> dict[str, Any]:
        return {"recall_percentual": round(m["recall"] * 100, 1), "precisao_percentual": round(m["precision"] * 100, 1),
                "f1": round(m["f1"], 3), "roc_auc": round(m["roc_auc"], 3), "acuracia_percentual": round(m["accuracy"] * 100, 1),
                "falsos_negativos": m["false_negatives"], "falsos_positivos": m["false_positives"]}

    return {
        "modelo": res["display_name"],
        "experimento_vencedor": sel["source"],
        "amostras_teste": results["dataset"]["test"],
        "malignos_no_teste": results["dataset"]["test_malignant"],
        "original": fmt(base),
        "otimizado": fmt(opt),
        "hiperparametros_otimizados": sel["params"],
        "hiperparametros_originais": res["baseline"]["params"],
    }
