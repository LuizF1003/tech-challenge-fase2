"""Monitoramento e logging.

* ``setup_logging`` — logs estruturados em JSON (uma linha por evento), prontos
  para serem coletados por Loki/ELK/CloudWatch. Defina ``LOG_FORMAT=text`` para
  logs legíveis no terminal.
* ``RunTracker`` — rastreamento leve de experimentos (estilo MLflow, sem
  servidor): cada execução grava parâmetros, métricas por geração e resultado
  final em ``reports/runs/<run_id>/``. Se a variável ``MLFLOW_TRACKING_URI``
  estiver definida e o ``mlflow`` instalado, os mesmos dados também são
  enviados ao MLflow.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "extra_fields", {})
        return base + ("  " + " ".join(f"{k}={v}" for k, v in extra.items()) if extra else "")


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.getenv("LOG_FORMAT", "json")
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, **fields: Any) -> None:
    logger.info(msg, extra={"extra_fields": fields})


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


class RunTracker:
    """Registra uma execução (experimento) em disco e, opcionalmente, no MLflow."""

    def __init__(self, name: str, base_dir: Path, params: dict[str, Any] | None = None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{name}-{uuid.uuid4().hex[:6]}"
        self.dir = base_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = params or {}
        self.metrics_file = self.dir / "metrics.jsonl"
        self.start = time.perf_counter()
        dump_json({"run_id": self.run_id, "name": name, "params": self.params}, self.dir / "params.json")
        self._mlflow = None
        if os.getenv("MLFLOW_TRACKING_URI"):
            try:
                import mlflow

                mlflow.set_experiment("tech-challenge-fase2")
                mlflow.start_run(run_name=self.run_id)
                mlflow.log_params({k: str(v) for k, v in self.params.items()})
                self._mlflow = mlflow
            except Exception:  # pragma: no cover - MLflow é opcional
                self._mlflow = None

    def log_metrics(self, step: int, **metrics: float) -> None:
        with self.metrics_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **metrics}, default=_json_default) + "\n")
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def finish(self, result: dict[str, Any]) -> None:
        result = {**result, "duration_s": round(time.perf_counter() - self.start, 2)}
        dump_json(result, self.dir / "result.json")
        if self._mlflow:
            self._mlflow.end_run()
