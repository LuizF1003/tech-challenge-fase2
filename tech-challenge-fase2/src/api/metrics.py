"""Métricas Prometheus expostas em ``GET /metrics``.

Usadas para monitorar a API e como sinal para o autoscaling (HPA por CPU e,
opcionalmente, por requisições/s via Prometheus Adapter).
"""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "Requisições HTTP", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "Latência das requisições HTTP", ["method", "path"],
                         buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
IN_FLIGHT = Gauge("http_requests_in_flight", "Requisições em andamento")

PREDICTIONS = Counter("predictions_total", "Previsões realizadas", ["classe"])
PREDICTION_PROBA = Histogram("prediction_malignancy_probability", "Distribuição da probabilidade prevista",
                             buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95))

LLM_REQUESTS = Counter("llm_requests_total", "Chamadas à LLM", ["provider", "status"])
LLM_LATENCY = Histogram("llm_latency_seconds", "Latência da LLM", ["provider"],
                        buckets=(0.05, 0.25, 0.5, 1, 2, 4, 8, 16, 32))

GA_JOBS_RUNNING = Gauge("ga_jobs_running", "Otimizações do AG em execução")
GA_BEST_FITNESS = Gauge("ga_best_fitness", "Melhor fitness do último job do AG", ["model"])
MODEL_INFO = Gauge("model_info", "Modelo em produção", ["model", "source"])
