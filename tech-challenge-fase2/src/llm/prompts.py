"""Prompt engineering.

Três estratégias, comparadas na avaliação (``scripts/evaluate_llm.py``):

``basico``      zero-shot, sem papel nem regras — linha de base.
``estruturado`` papel (persona) + público-alvo + regras de segurança +
                contexto em JSON + formato de saída em JSON com campos fixos.
``few_shot``    tudo do ``estruturado`` + um exemplo completo (few-shot) +
                checklist de autoverificação antes de responder.

Técnicas usadas e por quê:
* **Persona e público**: "você auxilia médicos" ajusta vocabulário e nível técnico.
* **Grounding**: todos os números vêm do contexto JSON; a regra "não invente
  números" é verificada automaticamente na avaliação (fidelidade numérica).
* **Restrições de segurança**: proíbe linguagem de diagnóstico definitivo e
  exige o aviso de apoio à decisão.
* **Saída estruturada (JSON)**: facilita exibir no frontend e validar campos.
* **Few-shot**: um exemplo mostra o tom, o tamanho e como citar os fatores.
* **Temperatura baixa (0,2)**: respostas consistentes entre execuções.
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "2.0"

CASE_FIELDS = ["resumo", "fatores_relevantes", "confiabilidade", "recomendacao", "limitacoes", "aviso"]
PERFORMANCE_FIELDS = ["resumo", "insights", "impacto_clinico", "recomendacao", "aviso"]

SYSTEM_CLINICAL = """Você é um assistente de apoio à decisão clínica em um hospital universitário.
Seu público são médicos(as) mastologistas e patologistas. Você explica, em português do Brasil,
o resultado de um modelo de machine learning que estima a probabilidade de malignidade de uma
massa mamária a partir de medidas morfológicas de núcleos celulares (exame de PAAF digitalizado).

Regras obrigatórias:
1. Use SOMENTE os números presentes no CONTEXTO. Não invente valores, percentuais ou estatísticas.
2. Nunca afirme diagnóstico definitivo. Não use expressões como "confirma câncer", "é maligno",
   "descarta câncer" ou "não há câncer". Fale em probabilidade estimada e sinal de alerta.
3. Cite os fatores pelo nome em português e compare o valor da paciente com as médias de
   casos benignos e malignos quando isso ajudar a interpretação.
4. Seja objetivo: frases curtas, sem jargão de computação (evite "feature", "pipeline", "threshold").
5. Termine sempre com o aviso de que se trata de apoio à decisão e não substitui avaliação médica.
6. Responda apenas com um objeto JSON válido."""

CASE_SCHEMA = """{
  "resumo": "2 a 3 frases com a probabilidade estimada, o limiar e a faixa de risco",
  "fatores_relevantes": ["até 3 itens: fator, valor da paciente, comparação com as médias e efeito no risco"],
  "confiabilidade": "1 a 2 frases sobre o desempenho do modelo no teste (recall, falsos negativos)",
  "recomendacao": "próximo passo sugerido, sempre condicionado ao julgamento clínico",
  "limitacoes": "o que o modelo não considera",
  "aviso": "aviso de apoio à decisão"
}"""

FEW_SHOT_CONTEXT = {
    "probabilidade_malignidade": 0.913, "probabilidade_percentual": 91.3, "limiar_decisao": 0.4,
    "classificacao_modelo": "suspeito de malignidade", "faixa_risco": "alto",
    "fatores_principais": [
        {"nome": "pontos côncavos (pior valor)", "valor": 0.221, "media_benignos": 0.074,
         "media_malignos": 0.183, "direcao": "aumenta o risco"},
        {"nome": "área (pior valor)", "valor": 1750.0, "media_benignos": 557.0,
         "media_malignos": 1430.0, "direcao": "aumenta o risco"},
        {"nome": "textura (média)", "valor": 15.1, "media_benignos": 17.9,
         "media_malignos": 21.6, "direcao": "reduz o risco"},
    ],
    "modelo": {"nome": "Regressão Logística", "origem": "otimizado por algoritmo genético",
               "recall_teste_percentual": 97.6, "falsos_negativos_teste": 1, "amostras_teste": 114},
}

FEW_SHOT_ANSWER = {
    "resumo": ("O modelo estimou probabilidade de malignidade de 91.3%, bem acima do limiar de decisão "
               "de 0.4. A amostra está na faixa de risco alto e foi sinalizada como suspeita."),
    "fatores_relevantes": [
        "Pontos côncavos (pior valor) de 0.221, acima da média de casos malignos (0.183) — aumenta o risco.",
        "Área (pior valor) de 1750.0, maior que a média de malignos (1430.0) — aumenta o risco.",
        "Textura (média) de 15.1, abaixo da média de benignos (17.9) — reduz o risco, mas tem peso menor.",
    ],
    "confiabilidade": ("No teste, o modelo identificou 97.6% dos casos malignos, com 1 falso negativo "
                       "em 114 amostras."),
    "recomendacao": ("Priorizar a avaliação da paciente e considerar confirmação histopatológica, "
                     "conforme julgamento clínico."),
    "limitacoes": "Não considera histórico clínico, exame físico nem achados de imagem.",
    "aviso": "Apoio à decisão: não confirma nem descarta câncer e não substitui a avaliação médica.",
}

SELF_CHECK = """Antes de responder, confira em silêncio:
- todo número citado aparece no CONTEXTO (mesmo valor ou mesmo percentual);
- o texto é coerente com a classificação do modelo (acima/abaixo do limiar);
- não há linguagem de diagnóstico definitivo;
- o campo "aviso" está preenchido.
Mostre apenas o JSON final."""


def _ctx(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, indent=2)


def case_prompt(context: dict[str, Any], strategy: str = "few_shot") -> tuple[str, str, bool]:
    """Devolve (system, user, json_mode) para explicar um caso individual."""
    if strategy == "basico":
        return ("Você é um assistente útil.",
                f"Explique este resultado de um modelo de câncer de mama:\n{_ctx(context)}", False)

    user = f"CONTEXTO:\n{_ctx(context)}\n\nFORMATO DE SAÍDA (JSON):\n{CASE_SCHEMA}"
    if strategy == "few_shot":
        user = (
            f"EXEMPLO DE CONTEXTO:\n{_ctx(FEW_SHOT_CONTEXT)}\n\n"
            f"EXEMPLO DE RESPOSTA:\n{json.dumps(FEW_SHOT_ANSWER, ensure_ascii=False, indent=2)}\n\n"
            f"Agora faça o mesmo para o caso abaixo.\n\n{user}\n\n{SELF_CHECK}"
        )
    elif strategy != "estruturado":
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    return SYSTEM_CLINICAL, user, True


SYSTEM_PERFORMANCE = """Você é um cientista de dados que apresenta resultados para a diretoria clínica
de um hospital universitário. Traduza métricas de modelos de machine learning em insights acionáveis
para médicos, em português do Brasil. Use SOMENTE os números do CONTEXTO, explique o que cada métrica
significa na prática (ex.: falso negativo = caso maligno não sinalizado), aponte riscos e limitações e
não exagere ganhos pequenos. Responda apenas com um objeto JSON válido."""

PERFORMANCE_SCHEMA = """{
  "resumo": "2 a 3 frases comparando o modelo original e o otimizado",
  "insights": ["3 a 5 achados acionáveis, cada um citando números do contexto"],
  "impacto_clinico": "o que a diferença significa para a triagem de pacientes",
  "recomendacao": "próximos passos antes do uso clínico",
  "aviso": "limitações da avaliação"
}"""


def performance_prompt(context: dict[str, Any]) -> tuple[str, str, bool]:
    user = f"CONTEXTO:\n{_ctx(context)}\n\nFORMATO DE SAÍDA (JSON):\n{PERFORMANCE_SCHEMA}"
    return SYSTEM_PERFORMANCE, user, True
