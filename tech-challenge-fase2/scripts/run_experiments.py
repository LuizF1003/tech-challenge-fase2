"""Executa todos os experimentos do AG e salva resultados + melhor modelo.

Uso:
    python -m scripts.run_experiments                 # todos os modelos e experimentos
    python -m scripts.run_experiments --models logistic_regression --experiments E1_pop_pequena
    python -m scripts.run_experiments --quick         # versão reduzida (smoke test)
"""

from __future__ import annotations

import argparse
import warnings

from sklearn.exceptions import ConvergenceWarning

from src.experiments import run_all
from src.monitoring import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="logistic_regression random_forest")
    parser.add_argument("--experiments", nargs="*", help="nomes definidos em configs/experiments.yaml")
    parser.add_argument("--quick", action="store_true", help="orçamento mínimo (para testes)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    setup_logging(fmt="text")
    results = run_all(args.models, args.experiments, quick=args.quick)

    print("\n=== Resumo (teste hold-out da Fase 1) ===")
    for model, res in results["models"].items():
        b = res["baseline"]["test_metrics"]
        print(f"\n{res['display_name']}")
        print(f"  {'baseline':<22} recall={b['recall']:.3f} f1={b['f1']:.3f} auc={b['roc_auc']:.3f} FN={b['false_negatives']}")
        for name, exp in res["experiments"].items():
            t = exp["test_metrics"]
            print(f"  {name:<22} recall={t['recall']:.3f} f1={t['f1']:.3f} auc={t['roc_auc']:.3f} FN={t['false_negatives']}"
                  f"  (cv_fitness={exp['cv_fitness']:.4f}, {exp['evaluations']} avaliações)")
    sel = results["selected"]
    print(f"\nModelo selecionado: {sel['model']} / {sel['source']}  params={sel['params']}")


if __name__ == "__main__":
    main()
