"""Demonstração ponta a ponta (útil para o vídeo).

    python -m scripts.demo                     # LLM offline (sem chave)
    python -m scripts.demo --provider groq     # LLaMA 3.3 70B via Groq
    python -m scripts.demo --provider ollama   # LLaMA local

1. Mostra o modelo selecionado e o ganho em relação à Fase 1.
2. Faz a previsão de 2 pacientes do teste (1 maligno, 1 benigno).
3. Gera a explicação em linguagem natural para o caso maligno.
4. Responde a uma pergunta livre de um(a) médico(a).
5. Gera o resumo "original x otimizado" para a equipe clínica.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib

from src.data import PROJECT_DIR, load_split
from src.llm import DiagnosisInterpreter, get_provider


def title(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    model = joblib.load(PROJECT_DIR / "models" / "best_model.joblib")
    card = json.loads((PROJECT_DIR / "models" / "model_card.json").read_text(encoding="utf-8"))
    results = json.loads((PROJECT_DIR / "reports" / "results" / "experiments.json").read_text(encoding="utf-8"))
    data = load_split()

    title("1. Modelo em produção")
    b, o = card["baseline_test_metrics"], card["test_metrics"]
    print(f"{card['display_name']} — origem: {card['source']}")
    print(f"Hiperparâmetros: {card['params']}")
    print(f"Teste  original: recall={b['recall']:.3f} F1={b['f1']:.3f} AUC={b['roc_auc']:.3f} FN={b['false_negatives']} FP={b['false_positives']}")
    print(f"Teste otimizado: recall={o['recall']:.3f} F1={o['f1']:.3f} AUC={o['roc_auc']:.3f} FN={o['false_negatives']} FP={o['false_positives']}")

    title("2. Previsão")
    mal = data.X_test[data.y_test == 1].iloc[[0]]
    ben = data.X_test[data.y_test == 0].iloc[[0]]
    for label, s in (("maligno (real)", mal), ("benigno (real)", ben)):
        p = model.predict_proba(s)[:, 1][0]
        print(f"Paciente {s.index[0]} [{label}] → probabilidade {p:.1%} "
              f"({'ALERTA' if p >= model.threshold else 'sem alerta'}, limiar {model.threshold:.2f})")

    interp = DiagnosisInterpreter(model, card, get_provider(args.provider))
    title(f"3. Explicação para o médico ({interp.provider.name})")
    out = interp.explain_case(mal)
    print(out.text)
    print(f"\n[provedor={out.provider} modelo={out.model} latência={out.latency_s}s fallback={out.used_fallback}]")

    title("4. Pergunta livre")
    q = "Quais medidas mais contribuíram para esse resultado e o que devo considerar antes de pedir biópsia?"
    print(f"P: {q}\n")
    print(interp.ask(mal, q).text)

    title("5. Relatório de desempenho para a equipe clínica")
    print(interp.explain_performance(results).text)


if __name__ == "__main__":
    main()
