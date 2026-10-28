"""Avalia a qualidade das interpretações da LLM em casos do conjunto de teste.

Compara as três estratégias de prompt (basico, estruturado, few_shot) com os
critérios automáticos de ``src/llm/evaluation.py`` e, se ``--judge`` for
informado, com notas de um "LLM juiz".

Exemplos:
    python -m scripts.evaluate_llm --provider offline            # sem chave (referência)
    python -m scripts.evaluate_llm --provider groq --judge groq  # LLaMA 3.3 70B via Groq
    python -m scripts.evaluate_llm --provider ollama --cases 8   # LLaMA local via Ollama
"""

from __future__ import annotations

import argparse
import json
import logging
from statistics import mean

import joblib
import numpy as np

from src.data import PROJECT_DIR, load_split
from src.llm import DiagnosisInterpreter, get_provider
from src.llm.evaluation import evaluate_interpretation, llm_judge
from src.monitoring import dump_json, setup_logging

RESULTS = PROJECT_DIR / "reports" / "results"
STRATEGIES = ["basico", "estruturado", "few_shot"]


def pick_cases(model, X, y, n: int, seed: int = 7) -> list[int]:
    """Mistura casos difíceis (erros e perto do limiar) com casos típicos."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= model.threshold).astype(int)
    errors = list(np.where(pred != y.values)[0])
    borderline = list(np.argsort(np.abs(proba - model.threshold))[:n])
    rng = np.random.default_rng(seed)
    typical_m = list(rng.permutation(np.where(y.values == 1)[0]))
    typical_b = list(rng.permutation(np.where(y.values == 0)[0]))
    chosen: list[int] = []
    for pool in (errors, borderline[: n // 4], typical_m, typical_b):
        for i in pool:
            if len(chosen) >= n:
                break
            if i not in chosen and (pool is not typical_m and pool is not typical_b or
                                    sum(1 for c in chosen if (y.values[c] == 1) == (pool is typical_m)) < n // 2):
                chosen.append(int(i))
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default=None, help="offline | groq | ollama | openai | anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge", default=None, help="provedor usado como LLM juiz (opcional)")
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    args = ap.parse_args()
    setup_logging(fmt="text")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    model = joblib.load(PROJECT_DIR / "models" / "best_model.joblib")
    card = json.loads((PROJECT_DIR / "models" / "model_card.json").read_text(encoding="utf-8"))
    data = load_split()
    provider = get_provider(args.provider, args.model)
    judge = get_provider(args.judge) if args.judge else None
    interp = DiagnosisInterpreter(model, card, provider, fallback=False)

    idx = pick_cases(model, data.X_test, data.y_test, args.cases)
    rows, examples = [], []
    for i in idx:
        sample = data.X_test.iloc[[i]]
        for strategy in args.strategies:
            out = interp.explain_case(sample, strategy=strategy)
            scores = evaluate_interpretation(out)
            if judge:
                scores["juiz"] = llm_judge(judge, out)
            rows.append({"caso": int(data.X_test.index[i]), "diagnostico_real": int(data.y_test.iloc[i]),
                         "estrategia": strategy, **scores})
            if len(examples) < 6:
                examples.append({"caso": int(data.X_test.index[i]), "estrategia": strategy, "texto": out.text,
                                 "erro": out.error, "scores": scores})
            print(f"caso {data.X_test.index[i]:>4} {strategy:<12} score={scores['score_geral']:.3f}")

    summary = {}
    for s in args.strategies:
        sub = [r for r in rows if r["estrategia"] == s]
        keys = ["formato", "fidelidade", "cobertura", "coerencia", "seguranca", "legibilidade", "score_geral",
                "latencia_s", "palavras"]
        summary[s] = {k: round(mean(r[k] for r in sub), 4) for k in keys}
        judged = [r["juiz"]["media"] for r in sub if r.get("juiz")]
        if judged:
            summary[s]["juiz_media"] = round(mean(judged), 2)

    perf = interp.explain_performance(json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8")))
    result = {
        "provider": provider.name, "model": provider.model, "judge": judge.name if judge else None,
        "cases": len(idx), "summary": summary, "rows": rows, "examples": examples,
        "performance_report": perf.to_dict(),
    }
    out_path = RESULTS / f"llm_evaluation_{provider.name}.json"
    dump_json(result, out_path)

    print("\n=== Média por estratégia ===")
    for s, m in summary.items():
        print(f"{s:<12} " + " ".join(f"{k}={v}" for k, v in m.items()))
    print(f"\nResultados em {out_path}")


if __name__ == "__main__":
    main()
