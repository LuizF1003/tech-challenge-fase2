"""Tech Challenge Fase 2 — Otimização de modelos de diagnóstico (Projeto 1).

Pacotes:
    src.data        carregamento/validação do dataset (mesmas regras da Fase 1)
    src.models      modelos baseline da Fase 1 e construção de modelos a partir de hiperparâmetros
    src.ga          algoritmo genético genérico (codificação, operadores, motor)
    src.spaces      espaços de busca de hiperparâmetros (genes) de cada modelo
    src.fitness     função fitness baseada em validação cruzada
    src.experiments execução e comparação dos experimentos
    src.explain     explicações locais/globais usadas como contexto para a LLM
    src.llm         integração com LLMs (provedores, prompts, avaliação)
    src.monitoring  logging estruturado e rastreamento de execuções
    src.api         API FastAPI com métricas Prometheus
"""

__version__ = "2.0.0"
