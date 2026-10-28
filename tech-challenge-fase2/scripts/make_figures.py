"""Gera os gráficos do relatório a partir de reports/results/*.json.

    python -m scripts.make_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import PROJECT_DIR  # noqa: E402

RESULTS = PROJECT_DIR / "reports" / "results"
FIG = PROJECT_DIR / "reports" / "figures"

PALETTE = ["#2a6f97", "#e07a5f", "#3d9970", "#8e6bbf", "#c9a227", "#6c757d"]
BASE_COLOR, OPT_COLOR = "#9aa5b1", "#2a6f97"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def convergence(res: dict) -> Path:
    models = list(res["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        r = res["models"][model]
        for color, (name, exp) in zip(PALETTE, r["experiments"].items()):
            gens = [h["generation"] for h in exp["history"]]
            ax.plot(gens, [h["best_fitness"] for h in exp["history"]], color=color, lw=1.8, label=name)
            ax.plot(gens, [h["mean_fitness"] for h in exp["history"]], color=color, lw=0.8, ls=":", alpha=0.8)
        ax.axhline(r["baseline"]["cv_fitness"], color="black", ls="--", lw=1, label="Fase 1 (original)")
        ax.set_title(f"Convergência — {r['display_name']}")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness (CV 5-fold no treino)")
        lo = min(r["baseline"]["cv_fitness"], *[e["history"][0]["best_fitness"] for e in r["experiments"].values()])
        ax.set_ylim(lo - 0.006, max(e["cv_fitness"] for e in r["experiments"].values()) + 0.004)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.text(0.5, -0.02, "linha cheia = melhor indivíduo · pontilhada = média da população", ha="center", fontsize=7)
    return _save(fig, "convergencia_ag.png")


def diversity(res: dict) -> Path:
    models = list(res["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3), squeeze=False)
    for ax, model in zip(axes[0], models):
        for color, (name, exp) in zip(PALETTE, res["models"][model]["experiments"].items()):
            ax.plot([h["generation"] for h in exp["history"]], [h["diversity"] for h in exp["history"]],
                    color=color, lw=1.6, label=name)
        ax.set_title(f"Diversidade da população — {res['models'][model]['display_name']}")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Desvio-padrão médio dos genes")
    axes[0][0].legend(fontsize=7)
    return _save(fig, "diversidade_ag.png")


def comparison(res: dict) -> Path:
    sel = res["selected"]
    models = list(res["models"])
    metrics = [("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("accuracy", "Acurácia")]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        r = res["models"][model]
        best_name = max(r["experiments"], key=lambda k: r["experiments"][k]["robust_cv"]["fitness"]["mean"])
        base, opt = r["baseline"]["test_metrics"], r["experiments"][best_name]["test_metrics"]
        x = np.arange(len(metrics))
        b1 = ax.bar(x - 0.18, [base[k] for k, _ in metrics], 0.36, color=BASE_COLOR, label="Original (Fase 1)")
        b2 = ax.bar(x + 0.18, [opt[k] for k, _ in metrics], 0.36, color=OPT_COLOR, label=f"AG ({best_name})")
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, [m for _, m in metrics])
        ax.set_ylim(0.85, 1.01)
        title = r["display_name"] + (" ★ selecionado" if model == sel["model"] else "")
        ax.set_title(f"Teste hold-out — {title}")
        ax.legend(fontsize=7, loc="lower left")
    return _save(fig, "comparativo_teste.png")


def errors(res: dict) -> Path:
    labels, fn, fp = [], [], []
    for model, r in res["models"].items():
        short = "RL" if model == "logistic_regression" else "RF"
        best_name = max(r["experiments"], key=lambda k: r["experiments"][k]["robust_cv"]["fitness"]["mean"])
        for tag, rec in (("original", r["baseline"]), ("AG", r["experiments"][best_name])):
            labels.append(f"{short} {tag}")
            fn.append(rec["test_metrics"]["false_negatives"])
            fp.append(rec["test_metrics"]["false_positives"])
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, fn, 0.36, color="#c0392b", label="Falsos negativos (maligno não sinalizado)")
    ax.bar(x + 0.18, fp, 0.36, color="#e6a23c", label="Falsos positivos (alerta indevido)")
    for i, (a, b) in enumerate(zip(fn, fp)):
        ax.text(i - 0.18, a + 0.05, str(a), ha="center", fontsize=8)
        ax.text(i + 0.18, b + 0.05, str(b), ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Casos no teste (n=114)")
    ax.set_title("Erros no conjunto de teste")
    ax.legend(fontsize=7)
    return _save(fig, "erros_teste.png")


def robust(res: dict) -> Path:
    models = list(res["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3), squeeze=False)
    for ax, model in zip(axes[0], models):
        r = res["models"][model]
        rows = {"Original": r["baseline"], **r["experiments"]}
        if "random_search" in r:
            rows["Busca aleatória"] = r["random_search"]
        names = list(rows)
        means = [rows[n]["robust_cv"]["fitness"]["mean"] for n in names]
        stds = [rows[n]["robust_cv"]["fitness"]["std"] for n in names]
        colors = [BASE_COLOR] + PALETTE[: len(r["experiments"])] + (["#adb5bd"] if "random_search" in r else [])
        ax.barh(names, means, xerr=stds, color=colors, capsize=3)
        ax.set_xlim(min(means) - max(stds) - 0.01, max(means) + max(stds) + 0.005)
        ax.invert_yaxis()
        ax.set_xlabel("Fitness — CV repetida 5×3 (média ± desvio)")
        ax.set_title(f"Validação robusta — {r['display_name']}")
    return _save(fig, "validacao_robusta.png")


def ga_vs_random(res: dict) -> Path:
    models = [m for m in res["models"] if "random_search" in res["models"][m]]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3), squeeze=False)
    for ax, model in zip(axes[0], models):
        r = res["models"][model]
        rs = r["random_search"]
        ref = r["experiments"][rs["reference_experiment"]]
        ax.plot([h["evaluations"] for h in ref["history"]], [h["best_fitness"] for h in ref["history"]],
                color=OPT_COLOR, lw=2, marker="o", ms=3, label=f"AG ({rs['reference_experiment']})")
        ax.plot(np.arange(1, len(rs["curve"]) + 1), rs["curve"], color="#adb5bd", lw=2, label="Busca aleatória")
        ax.axhline(r["baseline"]["cv_fitness"], color="black", ls="--", lw=1, label="Original")
        ax.set_xlabel("Modelos avaliados")
        ax.set_ylabel("Melhor fitness até o momento")
        ax.set_title(f"AG x busca aleatória — {r['display_name']}")
        ax.legend(fontsize=7, loc="lower right")
    return _save(fig, "ag_vs_busca_aleatoria.png")


def llm_eval(path: Path) -> Path | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    crit = ["formato", "fidelidade", "cobertura", "coerencia", "seguranca", "legibilidade", "score_geral"]
    strategies = list(data["summary"])
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(crit))
    w = 0.8 / len(strategies)
    for i, s in enumerate(strategies):
        ax.bar(x + (i - (len(strategies) - 1) / 2) * w, [data["summary"][s][c] for c in crit], w,
               color=PALETTE[i], label=s)
    ax.set_xticks(x, crit, rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Qualidade das interpretações por estratégia de prompt — {data['provider']} ({data['model']})")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    return _save(fig, f"llm_avaliacao_{data['provider']}.png")


def main() -> None:
    res = json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8"))
    paths = [convergence(res), diversity(res), comparison(res), errors(res), robust(res), ga_vs_random(res)]
    for f in sorted(RESULTS.glob("llm_evaluation_*.json")):
        paths.append(llm_eval(f))
    for p in paths:
        if p:
            print(p.relative_to(PROJECT_DIR))


if __name__ == "__main__":
    main()
