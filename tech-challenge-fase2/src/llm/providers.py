"""Provedores de LLM.

* ``openai``    — qualquer API compatível com OpenAI Chat Completions:
                  OpenAI (GPT), Groq (LLaMA 3, gratuito), Ollama local
                  (``http://localhost:11434/v1``), LM Studio, Together, Gemini etc.
* ``anthropic`` — API Messages da Anthropic (Claude).
* ``offline``   — gerador determinístico baseado em template. Não é uma LLM:
                  existe para a API, os testes e a demonstração funcionarem sem
                  chave/rede, e como *fallback* quando o provedor real falha.

Configuração por variáveis de ambiente (ver ``.env.example``)::

    LLM_PROVIDER=openai|anthropic|offline
    LLM_MODEL=llama-3.3-70b-versatile
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_API_KEY=...
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 900,
                 json_mode: bool = False, context: dict[str, Any] | None = None) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "ollama"  # Ollama ignora a chave
        self.timeout = timeout

    def generate(self, system, user, *, temperature=0.2, max_tokens=900, json_mode=False, context=None):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Falha ao chamar {self.base_url}: {exc}") from exc
        data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"], provider=self.name, model=self.model,
            latency_s=round(time.perf_counter() - start, 3), usage=data.get("usage", {}),
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None, base_url: str = "https://api.anthropic.com",
                 timeout: float = 60.0):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY/LLM_API_KEY não definida.")
        self.model, self.api_key, self.base_url, self.timeout = model, api_key, base_url.rstrip("/"), timeout

    def generate(self, system, user, *, temperature=0.2, max_tokens=900, json_mode=False, context=None):
        if json_mode:
            user += "\n\nResponda somente com o objeto JSON, sem texto antes ou depois."
        start = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/messages", timeout=self.timeout,
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={"model": self.model, "system": system, "max_tokens": max_tokens,
                      "temperature": temperature, "messages": [{"role": "user", "content": user}]},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Falha ao chamar a API da Anthropic: {exc}") from exc
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return LLMResponse(text=text, provider=self.name, model=self.model,
                           latency_s=round(time.perf_counter() - start, 3), usage=data.get("usage", {}))


class OfflineTemplateProvider:
    """Gera a resposta no formato esperado a partir do contexto estruturado.

    Determinístico e sem rede. Usado em testes, na demo sem chave e como
    fallback. Não substitui uma LLM — serve de referência mínima na avaliação.
    """

    name = "offline"
    model = "template-v1"

    def generate(self, system, user, *, temperature=0.2, max_tokens=900, json_mode=False, context=None):
        start = time.perf_counter()
        if context is None:
            raise LLMError("O provedor offline precisa do contexto estruturado.")
        if "fatores_principais" in context:
            payload = self._case(context)
        else:
            payload = self._performance(context)
        text = json.dumps(payload, ensure_ascii=False) if json_mode else self._as_text(payload)
        return LLMResponse(text=text, provider=self.name, model=self.model,
                           latency_s=round(time.perf_counter() - start, 4))

    @staticmethod
    def _case(ctx: dict[str, Any]) -> dict[str, Any]:
        m = ctx["modelo"]
        alerta = ctx["classificacao_modelo"] != "sem alerta"
        fatores = [
            f"{f['nome']}: {f['valor']} (média benignos {f['media_benignos']}, média malignos "
            f"{f['media_malignos']}) — {f['direcao']}"
            for f in ctx["fatores_principais"][:3]
        ]
        return {
            "resumo": (
                f"O modelo estimou probabilidade de malignidade de {ctx['probabilidade_percentual']}%, "
                f"{'acima' if alerta else 'abaixo'} do limiar de decisão de {ctx['limiar_decisao']}. "
                f"Faixa de risco: {ctx['faixa_risco']}."
            ),
            "fatores_relevantes": fatores,
            "confiabilidade": (
                f"No conjunto de teste, o modelo ({m['nome']}, {m['origem']}) identificou "
                f"{m['recall_teste_percentual']}% dos casos malignos (recall), com "
                f"{m['falsos_negativos_teste']} falso(s) negativo(s) em {m['amostras_teste']} amostras."
            ),
            "recomendacao": (
                "Priorizar avaliação médica e considerar confirmação por exame de imagem e biópsia, "
                "conforme julgamento clínico." if alerta else
                "Sem alerta pelo limiar técnico; manter o acompanhamento clínico habitual e correlacionar "
                "com exame físico e imagem."
            ),
            "limitacoes": (
                "Estimativa estatística baseada em medidas morfológicas de núcleos celulares; "
                "não considera histórico clínico, imagem ou outros exames."
            ),
            "aviso": "Apoio à decisão: não confirma nem descarta câncer e não substitui a avaliação médica.",
        }

    @staticmethod
    def _performance(ctx: dict[str, Any]) -> dict[str, Any]:
        b, o = ctx["original"], ctx["otimizado"]
        delta_fn = b["falsos_negativos"] - o["falsos_negativos"]
        return {
            "resumo": (
                f"O modelo {ctx['modelo']} otimizado por algoritmo genético obteve recall de "
                f"{o['recall_percentual']}% contra {b['recall_percentual']}% do modelo original no teste."
            ),
            "insights": [
                f"Falsos negativos no teste: {b['falsos_negativos']} (original) → {o['falsos_negativos']} (otimizado).",
                f"Falsos positivos no teste: {b['falsos_positivos']} (original) → {o['falsos_positivos']} (otimizado).",
                f"ROC-AUC: {b['roc_auc']} (original) → {o['roc_auc']} (otimizado).",
            ],
            "impacto_clinico": (
                f"{'Menos' if delta_fn > 0 else 'Mesmo número de' if delta_fn == 0 else 'Mais'} casos malignos "
                "deixariam de ser sinalizados na triagem."
            ),
            "recomendacao": "Validar o modelo em dados externos antes de qualquer uso assistencial.",
            "aviso": "Métricas obtidas em base pública e amostra pequena; não substituem validação clínica.",
        }

    @staticmethod
    def _as_text(payload: dict[str, Any]) -> str:
        lines = []
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}:\n" + "\n".join(f"- {v}" for v in value))
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


PRESETS = {
    "groq": ("openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "ollama": ("openai", "http://localhost:11434/v1", "llama3.1:8b", None),
    "openai": ("openai", "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "https://api.anthropic.com", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
}


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Cria o provedor a partir do nome (ou de ``LLM_PROVIDER``).

    Aceita os atalhos ``groq``, ``ollama``, ``openai``, ``anthropic`` e ``offline``.
    """
    name = (name or os.getenv("LLM_PROVIDER", "offline")).lower()
    if name == "offline":
        return OfflineTemplateProvider()
    if name not in PRESETS:
        raise ValueError(f"Provedor desconhecido: {name}. Use offline, {', '.join(PRESETS)}.")
    kind, default_url, default_model, key_env = PRESETS[name]
    base_url = os.getenv("LLM_BASE_URL", default_url)
    model = model or os.getenv("LLM_MODEL", default_model)
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else None)
    if kind == "anthropic":
        return AnthropicProvider(model, api_key, base_url)
    provider = OpenAICompatibleProvider(model, base_url, api_key)
    provider.name = name
    return provider
