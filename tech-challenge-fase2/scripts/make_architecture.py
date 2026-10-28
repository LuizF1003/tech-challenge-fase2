"""Desenha o diagrama de arquitetura (reports/figures/arquitetura.png)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.data import PROJECT_DIR  # noqa: E402

BLUE, GREEN, ORANGE, GREY, PURPLE = "#dbe9f6", "#dff2e6", "#fde8d7", "#eceff2", "#ece4f7"


def box(ax, x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#4a5568", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p1, p2, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9, lw=0.9, color="#4a5568", ls=ls))
    if text:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12, text, ha="center", fontsize=6.3, color="#4a5568")


def polyline(ax, pts, text=None, text_at=None, ls="-"):
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-1][0]], list(ys) + [pts[-1][1]], color="#4a5568", lw=0.9, ls=ls)
    arrow(ax, pts[-2], pts[-1], ls=ls)
    if text:
        ax.text(*text_at, text, ha="center", fontsize=6.3, color="#4a5568",
                bbox=dict(fc="white", ec="none", pad=1))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    # faixas
    ax.add_patch(FancyBboxPatch((0.2, 6.6), 21.6, 5.6, boxstyle="round,pad=0,rounding_size=0.2", fc="#f8fafc", ec="#cbd5e0"))
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 21.6, 6.1, boxstyle="round,pad=0,rounding_size=0.2", fc="#f8fafc", ec="#cbd5e0"))
    ax.text(0.5, 11.85, "OTIMIZAÇÃO EM LOTE (scripts/run_experiments · Job Kubernetes)", fontsize=8, fontweight="bold", color="#2d3748")
    ax.text(0.5, 5.95, "SERVIÇO ONLINE (API FastAPI · réplicas com autoscaling)", fontsize=8, fontweight="bold", color="#2d3748")

    # lote
    box(ax, 0.6, 9.2, 2.6, 1.6, "data.csv\nWisconsin\n569 amostras", GREY)
    box(ax, 4.0, 9.2, 2.6, 1.6, "Split 80/20\n(idêntico à Fase 1)\n455 treino · 114 teste", GREY)
    box(ax, 7.6, 9.6, 3.6, 1.9, "Algoritmo genético\npopulação → seleção →\ncrossover → mutação\n+ elitismo + cache", BLUE, True)
    box(ax, 7.6, 7.0, 3.6, 1.8, "Fitness (CV 5-fold, treino)\n0,45·Recall + 0,35·F1\n+ 0,20·ROC-AUC", BLUE)
    box(ax, 12.4, 9.2, 3.4, 1.8, "Avaliação final\nteste hold-out +\nCV repetida 5×3", BLUE)
    box(ax, 16.8, 10.0, 4.6, 1.3, "models/best_model.joblib\n+ model_card.json", GREEN, True)
    box(ax, 16.8, 8.2, 4.6, 1.3, "reports/results/experiments.json\nfiguras · relatório", GREEN)
    box(ax, 16.8, 6.95, 4.6, 0.95, "RunTracker / MLflow\nmétricas por geração", PURPLE)

    arrow(ax, (3.2, 10.0), (4.0, 10.0))
    arrow(ax, (6.6, 10.3), (7.6, 10.5), "treino")
    arrow(ax, (9.0, 9.6), (9.0, 8.8), None, "<|-|>")
    ax.text(9.2, 9.12, "params ⇄ fitness", fontsize=6.3, color="#4a5568")
    arrow(ax, (11.2, 10.5), (12.4, 10.2), "melhor")
    polyline(ax, [(5.3, 9.2), (5.3, 6.85), (14.1, 6.85), (14.1, 9.2)])
    ax.text(11.5, 6.97, "teste (só no final)", fontsize=6.3, color="#4a5568")
    arrow(ax, (15.8, 10.4), (16.8, 10.6))
    arrow(ax, (15.8, 9.7), (16.8, 8.9))
    arrow(ax, (11.2, 7.42), (16.8, 7.42), ls="--")

    # online
    box(ax, 0.6, 2.6, 2.8, 1.6, "Médico(a) /\nfrontend Angular", GREY)
    box(ax, 4.2, 2.6, 2.6, 1.6, "nginx / Service\nbalanceador", GREY)
    box(ax, 7.6, 1.6, 3.6, 3.0, "API FastAPI\n/predict  /interpret  /ask\n/experiments  /optimize\n/health  /ready  /metrics", BLUE, True)
    box(ax, 12.2, 3.9, 3.6, 1.5, "Contexto numérico\nprobabilidade · limiar ·\nfatores locais (oclusão)", ORANGE)
    box(ax, 12.2, 1.9, 3.6, 1.5, "Prompt engineering\npersona · regras · JSON\nfew-shot · autoverificação", ORANGE)
    box(ax, 16.8, 3.9, 4.6, 1.5, "LLM pré-treinada\nLLaMA 3 (Groq/Ollama) · GPT · Claude", ORANGE, True)
    box(ax, 16.8, 1.9, 4.6, 1.5, "Validação JSON + aviso clínico\nfalhou → fallback offline", ORANGE)
    box(ax, 7.6, 0.4, 3.6, 0.95, "Prometheus → Grafana\nlogs JSON (request_id)", PURPLE)
    box(ax, 4.2, 0.4, 2.6, 1.2, "HPA 2–10 réplicas\nCPU 70% · mem 80%", PURPLE)

    arrow(ax, (3.4, 3.4), (4.2, 3.4))
    arrow(ax, (6.8, 3.4), (7.6, 3.4))
    arrow(ax, (11.2, 3.9), (12.2, 4.5))
    arrow(ax, (14.0, 3.9), (14.0, 3.4))
    arrow(ax, (15.8, 2.9), (16.8, 4.3))
    arrow(ax, (19.1, 3.9), (19.1, 3.4))
    polyline(ax, [(17.6, 1.9), (17.6, 1.75), (11.2, 1.75)], "resposta validada", (14.0, 1.4))
    arrow(ax, (9.4, 1.6), (9.4, 1.35), ls="--")
    arrow(ax, (5.5, 1.6), (5.5, 2.6), ls="--")
    ax.text(5.65, 2.0, "escala", fontsize=6.3, color="#4a5568")
    polyline(ax, [(21.4, 10.65), (21.65, 10.65), (21.65, 6.45), (9.4, 6.45), (9.4, 4.6)],
             "modelo carregado na inicialização", (15.5, 6.33), ls="--")

    out = PROJECT_DIR / "reports" / "figures" / "arquitetura.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
