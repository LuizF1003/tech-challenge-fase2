"""Gera o relatório técnico (Markdown + PDF) a partir dos resultados salvos.

    python -m scripts.make_figures && python -m scripts.generate_report

Tudo que é número vem de reports/results/*.json — se os experimentos ou a
avaliação da LLM forem executados de novo, basta rodar este script.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from src.data import PROJECT_DIR
from src.llm.prompts import SYSTEM_CLINICAL
from src.spaces import SPACES
from src.ga.search_space import CategoricalGene, FloatGene, IntGene

RESULTS = PROJECT_DIR / "reports" / "results"
FIG = PROJECT_DIR / "reports" / "figures"
OUT_MD = PROJECT_DIR / "reports" / "relatorio_tecnico_fase2.md"
OUT_PDF = PROJECT_DIR / "reports" / "relatorio_tecnico_fase2.pdf"

TEAM = [("Gabriel Wesley", "RM376899"), ("Gustavo Leite", "RM375733"), ("Luiz Fellipe", "RM376854")]
REPO = "https://github.com/Gusta2150/desafio_tech"
MODEL_PT = {"logistic_regression": "Regressão Logística", "random_forest": "Random Forest"}


# ----------------------------------------------------------------------------- #
# Helpers                                                                        #
# ----------------------------------------------------------------------------- #
def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def fmt_param(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:g}".replace(".", ",")
    return str(v)


def best_exp(r: dict) -> str:
    return max(r["experiments"], key=lambda k: r["experiments"][k]["robust_cv"]["fitness"]["mean"])


def gene_desc(g) -> tuple[str, str, str]:
    if isinstance(g, FloatGene):
        return g.name, "real" + (" (escala log)" if g.log else ""), f"[{fmt_param(g.low)}, {fmt_param(g.high)}]"
    if isinstance(g, IntGene):
        return g.name, "inteiro", f"[{g.low}, {g.high}]"
    return g.name, "categórico", ", ".join(fmt_param(c) for c in g.choices)


# ----------------------------------------------------------------------------- #
# Conteúdo                                                                       #
# ----------------------------------------------------------------------------- #
def build(res: dict, llm: dict | None, llm_is_real: bool) -> list[tuple]:
    B: list[tuple] = []
    h1 = lambda t: B.append(("h1", t))  # noqa: E731
    h2 = lambda t: B.append(("h2", t))  # noqa: E731
    p = lambda t: B.append(("p", t))  # noqa: E731
    bl = lambda items: B.append(("bullets", items))  # noqa: E731
    tbl = lambda rows, widths=None: B.append(("table", rows, widths))  # noqa: E731
    img = lambda name, w, cap: B.append(("img", FIG / name, w, cap)) if (FIG / name).exists() else None  # noqa: E731

    sel = res["selected"]
    rsel = res["models"][sel["model"]]
    base_sel = rsel["baseline"]["test_metrics"]
    opt_sel = sel["test_metrics"]
    ds = res["dataset"]
    w = res["fitness"]["weights"]
    wsum = sum(w.values())
    w = {k: v / wsum for k, v in w.items()}

    # ---------------- Resumo
    h1("Resumo executivo")
    fn_b, fn_o = base_sel["false_negatives"], opt_sel["false_negatives"]
    p(f"Este relatório descreve a Fase 2 do Tech Challenge (Projeto 1 — otimização de modelos de diagnóstico). "
      f"Partindo dos modelos de classificação de câncer de mama da Fase 1 (Regressão Logística e Random Forest "
      f"sobre o dataset Breast Cancer Wisconsin), implementamos do zero um algoritmo genético para otimizar "
      f"seus hiperparâmetros — incluindo o limiar de decisão — e integramos uma LLM pré-treinada para explicar "
      f"os resultados em linguagem natural para médicos.")
    p(f"Foram executados {len(rsel['experiments'])} experimentos com configurações diferentes do AG para cada "
      f"modelo, além de uma busca aleatória com o mesmo orçamento. O modelo selecionado foi a "
      f"<b>{MODEL_PT[sel['model']]}</b> otimizada no experimento <b>{sel['source']}</b>. No mesmo conjunto de "
      f"teste da Fase 1 ({ds['test']} pacientes, {ds['test_malignant']} malignas), o recall de malignidade foi de "
      f"{pct(base_sel['recall'])} para {pct(opt_sel['recall'])} e os falsos negativos de {fn_b} para {fn_o}, "
      f"com F1 de {num(base_sel['f1'])} → {num(opt_sel['f1'])} e ROC-AUC de {num(base_sel['roc_auc'])} → "
      f"{num(opt_sel['roc_auc'])}.")
    rb, ro = rsel["baseline"]["robust_cv"]["fitness"], rsel["experiments"][sel["source"]]["robust_cv"]["fitness"] \
        if sel["source"] != "baseline" else rsel["baseline"]["robust_cv"]["fitness"]
    p(f"Como o teste tem poucos casos malignos, também medimos o desempenho por validação cruzada repetida (5×3) "
      f"no treino: o fitness passou de {num(rb['mean'], 4)} ± {num(rb['std'], 4)} para "
      f"{num(ro['mean'], 4)} ± {num(ro['std'], 4)}. A seção 6 discute o tamanho real do ganho.")

    # ---------------- Contexto
    h1("1. Contexto e objetivo")
    p("Na Fase 1 o grupo construiu um pipeline de machine learning (com notebook, API FastAPI e frontend Angular) "
      "que classifica massas mamárias como benignas ou malignas a partir de 30 medidas morfológicas de núcleos "
      "celulares (raio, textura, perímetro, área, concavidade etc., em média, erro-padrão e pior valor). Os modelos "
      "usavam hiperparâmetros fixos escolhidos manualmente e limiar de 0,5.")
    p("Objetivos desta fase:")
    bl(["Otimizar os hiperparâmetros dos modelos da Fase 1 com um algoritmo genético (codificação, seleção, "
        "cruzamento, mutação e fitness baseada em métricas clínicas).",
        "Comparar os modelos otimizados com os originais e realizar pelo menos 3 experimentos com configurações "
        "diferentes do AG.",
        "Integrar uma LLM pré-treinada para gerar explicações dos diagnósticos e transformar métricas em insights "
        "para médicos, com prompt engineering e avaliação de qualidade.",
        "Preparar a solução para produção: API, monitoramento, logging e escalabilidade automática."])

    # ---------------- Dados
    h1("2. Dados e protocolo experimental")
    p(f"Usamos o mesmo arquivo e as mesmas regras de limpeza da Fase 1 (remoção de <i>id</i> e da coluna vazia, "
      f"rótulos M/B → 1/0) e o mesmo split estratificado 80/20 com <i>random_state=42</i>: {ds['train']} amostras "
      f"de treino ({ds['train_malignant']} malignas) e {ds['test']} de teste ({ds['test_malignant']} malignas). "
      f"Manter o split idêntico garante que a comparação com a Fase 1 seja justa.")
    p("Para evitar vazamento de informação, o protocolo separa claramente busca e avaliação:")
    bl(["O AG só enxerga o conjunto de treino: cada indivíduo é avaliado por validação cruzada estratificada 5-fold.",
        "O melhor indivíduo de cada experimento é re-treinado em todo o treino e avaliado (a) no teste hold-out da "
        "Fase 1 e (b) numa validação cruzada repetida 5×3 com outra semente (2026).",
        "O modelo publicado na API é escolhido pela validação repetida — nunca pelo teste."])

    # ---------------- AG
    h1("3. Algoritmo genético")
    h2("3.1 Codificação (representação dos genes)")
    p("Cada indivíduo é um vetor real x ∈ [0, 1]^d. Cada posição é um gene, decodificado conforme o tipo do "
      "hiperparâmetro: genes reais (com escala logarítmica para parâmetros multiplicativos como C), inteiros "
      "(arredondamento) e categóricos (o intervalo [0, 1) é dividido em faixas iguais, uma por categoria). Com essa "
      "representação os mesmos operadores servem para qualquer modelo, e podemos usar operadores próprios de "
      "codificação real, como BLX-α e mutação gaussiana.")
    for model, space in SPACES.items():
        rows = [["Gene", "Tipo", "Domínio"]] + [list(gene_desc(g)) for g in space.genes]
        p(f"<b>{MODEL_PT[model]}</b> — {space.size} genes:")
        tbl(rows, [0.25, 0.25, 0.5])
    p("O <b>limiar de decisão</b> é um gene nos dois modelos. Na triagem de câncer um falso negativo (caso maligno "
      "não sinalizado) é muito mais grave que um falso positivo, e o limiar é o controle mais direto desse "
      "equilíbrio — na Fase 1 ele estava fixo em 0,5. No Random Forest, n_estimators vai até 300 (a Fase 1 usava "
      "500) para caber no orçamento computacional; a curva de ganho de árvores adicionais é bem plana nessa faixa.")

    h2("3.2 Operadores")
    tbl([["Etapa", "Operadores implementados", "Papel"],
         ["Seleção", "Torneio (k=3), roleta (com deslocamento pelo mínimo), ranking linear",
          "Escolher pais; k e o tipo controlam a pressão seletiva"],
         ["Cruzamento", "Uniforme, um ponto, BLX-α (α=0,5)",
          "Combinar boas soluções; BLX-α explora além dos pais"],
         ["Mutação", "Gaussiana (σ configurável), uniforme (reinicialização)",
          "Manter diversidade; gaussiana faz ajuste fino, uniforme explora"],
         ["Elitismo", "Os n melhores passam intactos", "Garante que o melhor fitness nunca piora"],
         ["Outros", "Cache de fitness, parada antecipada, semeadura com o modelo original",
          "Economia de treinos e convergência controlada"]], [0.15, 0.45, 0.40])
    p("A roleta usa o fitness deslocado pelo mínimo da população: como os valores ficam muito próximos "
      "(0,95–0,98), a roleta pura seria quase aleatória.")

    h2("3.3 Função fitness")
    p(f"fitness = {num(w['recall'], 2)} · Recall(maligno) + {num(w['f1'], 2)} · F1(maligno) + "
      f"{num(w['roc_auc'], 2)} · ROC-AUC, calculado sobre as probabilidades out-of-fold da CV 5-fold no treino.")
    bl(["<b>Recall</b> tem o maior peso: é a métrica que a Fase 1 já priorizava (não deixar passar casos malignos).",
        "<b>F1</b> impede a solução degenerada \"tudo é maligno\" (recall 1,0 com precisão baixa), que um limiar "
        "muito baixo produziria.",
        "<b>ROC-AUC</b> mede a qualidade da ordenação das probabilidades, independente do limiar.",
        "Indivíduos que só diferem no limiar reaproveitam as probabilidades já calculadas (sem re-treinar) e "
        "indivíduos repetidos vêm do cache — o que reduz bastante o custo, principalmente no Random Forest."])

    # ---------------- Experimentos
    h1("4. Experimentos")
    cfg_rows = [["Experimento", "Pop.", "Ger.", "Seleção", "Crossover", "Mutação", "Elite", "Observação"]]
    first = next(iter(res["models"].values()))
    for name, exp in first["experiments"].items():
        c = exp["config"]
        mut = f"{c['mutation']} ({num(c['mutation_rate'], 2)})"
        cross = f"{c['crossover']} ({num(c['crossover_rate'], 1)})"
        obs = "semeado c/ original; parada em 6" if c.get("patience") else ""
        cfg_rows.append([name, str(c["population_size"]), str(c["generations"]), c["selection"], cross, mut,
                         str(c["elitism"]), obs])
    tbl(cfg_rows, [0.2, 0.06, 0.06, 0.12, 0.15, 0.15, 0.06, 0.2])
    p("Todos os experimentos usam a mesma semente (42), a mesma fitness e as mesmas dobras de CV, para que as "
      "diferenças venham apenas da configuração do AG.")

    for model, r in res["models"].items():
        h2(f"4.{list(res['models']).index(model) + 1} Resultados — {r['display_name']}")
        rows = [["Configuração", "Fitness CV", "CV 5×3 (média ± dp)", "Recall teste", "F1 teste", "AUC teste",
                 "FN", "FP", "Avaliações", "Tempo (s)"]]
        allrows = {"Original (Fase 1)": r["baseline"], **r["experiments"]}
        if "random_search" in r:
            allrows["Busca aleatória"] = r["random_search"]
        for name, rec in allrows.items():
            t, rc = rec["test_metrics"], rec["robust_cv"]["fitness"]
            rows.append([name, num(rec["cv_fitness"], 4), f"{num(rc['mean'], 4)} ± {num(rc['std'], 4)}",
                         pct(t["recall"]), num(t["f1"]), num(t["roc_auc"]), str(t["false_negatives"]),
                         str(t["false_positives"]), str(rec.get("evaluations", rec.get("budget", "—"))),
                         str(round(rec.get("elapsed_s", 0))) if name != "Original (Fase 1)" else "—"])
        tbl(rows, [0.19, 0.09, 0.15, 0.09, 0.07, 0.07, 0.05, 0.05, 0.1, 0.08])
        be = best_exp(r)
        bp = r["experiments"][be]["best_params"]
        p(f"Melhor configuração pela validação repetida: <b>{be}</b>. Hiperparâmetros encontrados: "
          + "; ".join(f"{k} = {fmt_param(v)}" for k, v in bp.items()) + ". Originais: "
          + "; ".join(f"{k} = {fmt_param(v)}" for k, v in r["baseline"]["params"].items()) + ".")

    img("convergencia_ag.png", 1.0, "Figura 1 — Convergência: melhor indivíduo (linha cheia) e média da população "
        "(pontilhada) por geração. A linha tracejada é o fitness do modelo original.")
    img("diversidade_ag.png", 1.0, "Figura 2 — Diversidade da população (desvio-padrão médio dos genes).")

    h2("4.3 Análise dos experimentos")
    lr = res["models"].get("logistic_regression")
    analysis = []
    for model, r in res["models"].items():
        exps = r["experiments"]
        best_cv = max(exps, key=lambda k: exps[k]["cv_fitness"])
        fewest = min(exps, key=lambda k: exps[k]["evaluations"])
        analysis.append(
            f"<b>{r['display_name']}</b>: todos os experimentos superaram o fitness do modelo original "
            f"({num(r['baseline']['cv_fitness'], 4)}) já nas primeiras gerações. O maior fitness de CV foi do "
            f"{best_cv} ({num(exps[best_cv]['cv_fitness'], 4)}); o {fewest} chegou a "
            f"{num(exps[fewest]['cv_fitness'], 4)} com apenas {exps[fewest]['evaluations']} avaliações."
            if all(e["cv_fitness"] > r["baseline"]["cv_fitness"] for e in exps.values()) else
            f"<b>{r['display_name']}</b>: o maior fitness de CV foi do {best_cv} "
            f"({num(exps[best_cv]['cv_fitness'], 4)}), contra {num(r['baseline']['cv_fitness'], 4)} do original."
        )
    def final_div(exp_name: str) -> str:
        vals = [r["experiments"][exp_name]["history"][-1]["diversity"] for r in res["models"].values()
                if exp_name in r["experiments"]]
        return "–".join(num(v, 2) for v in sorted(vals))

    names = list(first["experiments"])
    tour = [n for n in names if first["experiments"][n]["config"]["selection"] == "tournament"]
    roul = [n for n in names if first["experiments"][n]["config"]["selection"] == "roulette"]
    rank = [n for n in names if first["experiments"][n]["config"]["selection"] == "rank"]
    small = min(names, key=lambda n: first["experiments"][n]["config"]["population_size"])
    evals = sum(e["evaluations"] for r in res["models"].values() for e in r["experiments"].values())
    fits = sum(e["model_fits"] for r in res["models"].values() for e in r["experiments"].values())
    seeded = [n for n in names if n.startswith("E5")]
    seeded_txt = ""
    if seeded and "E2_blx_torneio" in names:
        s, e2 = seeded[0], "E2_blx_torneio"
        same = [r["display_name"] for r in res["models"].values()
                if abs(r["experiments"][s]["cv_fitness"] - r["experiments"][e2]["cv_fitness"]) < 1e-9]
        lower = [r["display_name"] for r in res["models"].values()
                 if r["experiments"][s]["cv_fitness"] < r["experiments"][e2]["cv_fitness"] - 1e-9]
        seeded_txt = ("<b>Semeadura (E5)</b>: colocar o modelo original na população inicial não trouxe vantagem. "
                      + (f"Na {', '.join(same)} o resultado foi idêntico ao E2 (mesma configuração sem semente): o "
                         f"original é superado logo na primeira geração e a busca segue o mesmo caminho. " if same else "")
                      + (f"No {', '.join(lower)} ficou abaixo do E2 — a semente pode atrair a busca para a vizinhança "
                         f"da solução manual." if lower else ""))
    bl(analysis + [
        f"<b>Pressão seletiva e diversidade</b> (Figura 2): com torneio ({', '.join(tour)}) a população converge "
        f"rápido — no {tour[0]} a diversidade final ficou em {final_div(tour[0])}. A roleta ({', '.join(roul)}), mesmo "
        f"com mutação baixa, manteve diversidade {final_div(roul[0])} no fim: como os fitness são muito próximos, ela "
        f"seleciona quase uniformemente. A mutação alta e uniforme com ranking ({', '.join(rank)}) manteve a maior "
        f"diversidade ({final_div(rank[0])}), o que evita convergência prematura, mas deixa a média da população "
        f"mais baixa e ruidosa (Figura 1).",
        f"<b>Tamanho da população e orçamento</b>: o {small}, com a menor população e o menor número de avaliações, "
        f"chegou a resultados comparáveis aos experimentos maiores. Com a fitness quase saturada (≈0,97) e espaços de "
        f"busca pequenos, mais avaliações rendem ganhos marginais — as diferenças entre experimentos ficam dentro do "
        f"desvio-padrão da validação repetida.",
    ] + ([seeded_txt] if seeded_txt else []) + [
        f"<b>Cache</b>: somando todos os experimentos, foram {evals} avaliações de fitness, mas só {fits} combinações "
        f"de hiperparâmetros precisaram ser treinadas (CV 5-fold) — indivíduos repetidos e variações apenas do "
        f"limiar reaproveitam resultados.",
    ])

    # ---------------- Comparativo
    h1("5. Comparativo: modelos originais × otimizados")
    rows = [["Modelo", "Versão", "Acurácia", "Precisão", "Recall", "F1", "ROC-AUC", "FN", "FP"]]
    for model, r in res["models"].items():
        be = best_exp(r)
        for tag, rec in (("Original", r["baseline"]), (f"AG ({be})", r["experiments"][be])):
            t = rec["test_metrics"]
            rows.append([r["display_name"], tag, pct(t["accuracy"]), pct(t["precision"]), pct(t["recall"]),
                         num(t["f1"]), num(t["roc_auc"]), str(t["false_negatives"]), str(t["false_positives"])])
    tbl(rows, [0.18, 0.2, 0.09, 0.09, 0.09, 0.08, 0.09, 0.06, 0.06])
    img("comparativo_teste.png", 1.0, f"Figura 3 — Métricas no teste hold-out da Fase 1 ({ds['test']} amostras).")
    img("erros_teste.png", 0.62, "Figura 4 — Falsos negativos e falsos positivos no teste (RL = Regressão "
        "Logística, RF = Random Forest).")
    img("validacao_robusta.png", 1.0, "Figura 5 — Fitness na validação cruzada repetida 5×3 (média ± desvio).")

    h2("5.1 Comparação com outra abordagem: busca aleatória")
    if lr and "random_search" in lr:
        txt = []
        for model, r in res["models"].items():
            rs = r["random_search"]
            ref = r["experiments"][rs["reference_experiment"]]
            txt.append(f"{r['display_name']}: AG ({rs['reference_experiment']}) {num(ref['cv_fitness'], 4)} × busca "
                       f"aleatória {num(rs['cv_fitness'], 4)} com {rs['budget']} avaliações.")
        p("Para verificar se o AG agrega valor além de testar muitas combinações, comparamos cada modelo com uma "
          "busca aleatória que recebe o mesmo número de avaliações do experimento E2. " + " ".join(txt))
        img("ag_vs_busca_aleatoria.png", 1.0, "Figura 6 — Melhor fitness encontrado em função do número de modelos "
            "avaliados: AG × busca aleatória.")
        p("Neste espaço de busca relativamente pequeno (4 e 9 genes) e com fitness quase saturada (≈0,97), a "
          "vantagem do AG sobre a busca aleatória é pequena; ela tende a crescer com espaços maiores (mais modelos, "
          "seleção de atributos), que são o próximo passo natural.")

    # ---------------- Discussão
    h1("6. Discussão do ganho")
    p(f"O ganho veio principalmente de dois fatores: (1) o <b>limiar de decisão</b> — o AG escolheu "
      f"{fmt_param(sel['params'].get('threshold'))} em vez de 0,5, deslocando o ponto de operação para "
      f"capturar mais casos malignos — e (2) a <b>regularização</b>/estrutura do modelo. A diferença de "
      f"falsos negativos no teste ({fn_b} → {fn_o}) é clinicamente relevante, mas o teste tem apenas "
      f"{ds['test_malignant']} casos malignos: cada caso equivale a {pct(1 / ds['test_malignant'])} de recall. "
      f"Por isso reportamos também a CV repetida, em que as diferenças são menores e os desvios-padrão se "
      f"sobrepõem parcialmente — o ganho é consistente na direção, mas modesto em magnitude, o que é esperado "
      f"num dataset em que o modelo original já tinha ROC-AUC ≈ 0,99.")
    p(f"O custo do ganho em recall é visível nos falsos positivos ({base_sel['false_positives']} → "
      f"{opt_sel['false_positives']} no teste). Em triagem, um alerta indevido gera um exame adicional; um falso "
      "negativo pode atrasar o diagnóstico de um câncer — a fitness foi desenhada para refletir essa assimetria. "
      "Os pesos da fitness são configuráveis em configs/experiments.yaml e devem ser validados com a equipe clínica.")

    # ---------------- LLM
    h1("7. Integração com LLM")
    h2("7.1 Abordagem")
    p("A LLM não acessa o modelo nem os dados brutos. Para cada paciente, a API monta um <b>contexto estruturado "
      "(grounding)</b> com: probabilidade de malignidade, limiar, faixa de risco, os 5 fatores que mais pesaram "
      "<i>para aquela paciente</i> (com o valor dela e as médias de benignos e malignos no treino) e o desempenho "
      "do modelo no teste. Os fatores locais são calculados por <b>oclusão</b>: cada atributo é substituído pela "
      "mediana do treino e mede-se quanto a probabilidade muda — método agnóstico ao modelo, sem depender do SHAP.")
    bl(["<b>Explicação de casos</b> (/interpret): resumo, fatores relevantes, confiabilidade, recomendação, "
        "limitações e aviso, em JSON.",
        "<b>Insights de desempenho</b> (/report/performance): traduz a comparação original × otimizado (recall, "
        "falsos negativos etc.) em impacto clínico para a equipe.",
        "<b>Perguntas livres</b> (/ask): o(a) médico(a) pergunta em linguagem natural sobre o caso — base para a "
        "integração com dados textuais do Módulo 3.",
        "<b>Provedores plugáveis</b>: qualquer API compatível com OpenAI (LLaMA 3 via Groq ou Ollama local, GPT, "
        "LM Studio) e a API da Anthropic, configurados por variáveis de ambiente.",
        "<b>Fallback</b>: se a LLM falha ou responde fora do formato, a API devolve uma explicação por template e "
        "registra o evento na métrica llm_requests_total{status=\"fallback\"}."])

    h2("7.2 Prompt engineering")
    tbl([["Técnica", "Como foi aplicada", "Objetivo"],
         ["Persona e público", "Assistente de apoio à decisão para mastologistas e patologistas",
          "Vocabulário e nível técnico adequados"],
         ["Grounding", "Todos os números vêm do CONTEXTO em JSON", "Evitar números inventados"],
         ["Restrições de segurança", "Proibido diagnóstico definitivo; aviso obrigatório",
          "Uso responsável em saúde"],
         ["Saída estruturada", "Esquema JSON com campos fixos", "Exibir no frontend e validar"],
         ["Few-shot", "Um par contexto → resposta completo", "Tom, tamanho e forma de citar fatores"],
         ["Autoverificação", "Checklist silencioso antes de responder", "Coerência com o limiar e números"],
         ["Temperatura 0,2", "Parâmetro da chamada", "Respostas estáveis"]], [0.2, 0.45, 0.35])
    p("Três estratégias foram comparadas: <b>basico</b> (zero-shot, sem regras), <b>estruturado</b> "
      "(persona + regras + JSON) e <b>few_shot</b> (estruturado + exemplo + autoverificação). Trecho do prompt "
      "de sistema usado nas estratégias estruturado e few_shot:")
    B.append(("code", SYSTEM_CLINICAL))

    h2("7.3 Avaliação da qualidade")
    tbl([["Critério", "Como é medido"],
         ["formato", "JSON válido com todos os campos esperados"],
         ["fidelidade", "Fração dos números citados que existem no contexto (detecta alucinação numérica)"],
         ["cobertura", "Fração dos 3 fatores principais mencionados"],
         ["coerencia", "Texto descreve o risco na mesma direção da classificação (acima/abaixo do limiar)"],
         ["seguranca", "Tem aviso de apoio à decisão e não usa linguagem de diagnóstico definitivo"],
         ["legibilidade", "Tamanho médio das frases (≤ 25 palavras = 1)"],
         ["juiz (opcional)", "Outra LLM dá notas 1–5 para clareza, utilidade clínica, fidelidade e segurança"]],
        [0.2, 0.8])
    if llm:
        crit = ["formato", "fidelidade", "cobertura", "coerencia", "seguranca", "legibilidade", "score_geral"]
        has_judge = any("juiz_media" in v for v in llm["summary"].values())
        rows = [["Estratégia"] + crit + (["juiz (1–5)"] if has_judge else []) + ["latência (s)"]]
        for s, m in llm["summary"].items():
            rows.append([s] + [num(m[c], 2) for c in crit] + ([num(m.get("juiz_media", 0), 2)] if has_judge else [])
                        + [num(m["latencia_s"], 2)])
        p(f"Resultados com <b>{llm['provider']} / {llm['model']}</b> em {llm['cases']} casos do teste "
          f"(erros do modelo, casos próximos ao limiar e casos típicos de cada classe):")
        tbl(rows)
        img(f"llm_avaliacao_{llm['provider']}.png", 0.85, "Figura 7 — Qualidade das interpretações por estratégia "
            "de prompt.")
        if not llm_is_real:
            p("<b>Atenção:</b> estes números foram gerados com o provedor <i>offline</i> (template determinístico), "
              "que serve de referência e de fallback. Para a avaliação com uma LLM real execute "
              "<i>python -m scripts.evaluate_llm --provider groq --judge groq</i> (ou ollama) e gere o relatório "
              "novamente.")
        else:
            s = llm["summary"]
            if "basico" in s and "few_shot" in s:
                p(f"O prompt básico teve score geral {num(s['basico']['score_geral'], 2)}, contra "
                  f"{num(s['few_shot']['score_geral'], 2)} do few-shot: sem regras e sem formato, a LLM tende a "
                  f"omitir o aviso, a não produzir JSON e a usar números que não estão no contexto.")
        ex = next((e for e in llm["examples"] if e["estrategia"] == "few_shot"), None)
        if ex:
            p("Exemplo de explicação gerada (estratégia few_shot):")
            B.append(("code", ex["texto"]))
        perf = llm.get("performance_report", {})
        if perf.get("text"):
            p("Exemplo de insights de desempenho para a equipe clínica (/report/performance):")
            B.append(("code", perf["text"]))

    # ---------------- Escalabilidade
    h1("8. Arquitetura, escalabilidade e monitoramento")
    img("arquitetura.png", 1.0, "Figura 8 — Arquitetura da solução.")
    bl(["<b>API sem estado</b> (FastAPI): o modelo é lido de models/best_model.joblib na inicialização; qualquer "
        "réplica atende qualquer requisição.",
        "<b>Escala local</b>: docker compose up --scale api=N, com nginx re-resolvendo o DNS das réplicas a cada 10 s.",
        "<b>Escala automática</b> (deploy/k8s/api.yaml): HorizontalPodAutoscaler de 2 a 10 réplicas por CPU (70%) "
        "e memória (80%), subida rápida (30 s) e descida lenta (5 min) para evitar oscilação; readiness/liveness "
        "probes, rolling update sem indisponibilidade e PodDisruptionBudget.",
        "<b>Otimização em lote</b> (deploy/k8s/ga-job.yaml): o AG roda como Job indexado, um pod por modelo, sem "
        "competir com a API. A rota /optimize existe para demonstração (no máximo 1 job por réplica).",
        "<b>Monitoramento</b>: métricas Prometheus em /metrics (volume e latência por rota, requisições em andamento, "
        "distribuição das probabilidades previstas para detectar drift, chamadas/latência/fallback da LLM, jobs do "
        "AG e melhor fitness) e Grafana no docker-compose.",
        "<b>Logging</b>: logs JSON estruturados com request_id propagado (cabeçalho x-request-id).",
        "<b>Rastreamento de experimentos</b>: cada execução do AG grava parâmetros, métricas por geração e resultado "
        "em reports/runs/ e, se MLFLOW_TRACKING_URI estiver definida, também no MLflow.",
        "A implantação em nuvem é opcional no enunciado; os manifestos Kubernetes funcionam em qualquer cluster "
        "gerenciado (EKS, GKE, AKS)."])

    # ---------------- Testes
    h1("9. Organização do código e testes")
    p("Projeto Python com Poetry (pyproject.toml) ou venv (requirements.txt), código em src/, scripts de execução "
      "em scripts/, configuração em configs/ e notebook de demonstração em notebooks/02_demo_fase2.ipynb. "
      "Os testes automatizados (pytest, pasta tests/) cobrem:")
    bl(["codificação e decodificação dos genes, operadores de seleção, cruzamento e mutação;",
        "o motor do AG: convergência num problema de referência, elitismo monotônico, reprodutibilidade por semente, "
        "cache, parada antecipada, semeadura e tolerância a falhas da fitness;",
        "dados (split idêntico à Fase 1), modelos, limiar e fitness (inclusive o reaproveitamento sem re-treino);",
        "LLM: contexto, prompts, extração de JSON, fallback, e os critérios de avaliação (números inventados, "
        "linguagem proibida, cobertura e coerência);",
        "API: health/ready, previsão, validação de entrada, interpretação, perguntas, métricas e job do AG."])

    # ---------------- Desafios
    h1("10. Desafios enfrentados e soluções")
    tbl([["Desafio", "Solução"],
         ["Hiperparâmetros de tipos diferentes no mesmo cromossomo", "Codificação real normalizada com genes tipados"],
         ["Custo computacional do Random Forest com CV", "Cache de fitness, reaproveitamento das probabilidades "
          "quando só o limiar muda e faixa de n_estimators ajustada"],
         ["Risco de sobreajustar ao teste", "AG só no treino; seleção pela CV repetida; teste usado uma vez"],
         ["Fitness quase saturada (≈0,97) e roleta pouco seletiva", "Roleta com deslocamento pelo mínimo e "
          "comparação com torneio e ranking"],
         ["Solução trivial \"tudo maligno\" ao otimizar recall", "F1 e AUC na fitness"],
         ["API depreciou penalty na Regressão Logística (scikit-learn ≥ 1.8)", "Uso de l1_ratio com verificação "
          "de versão"],
         ["LLM pode inventar números ou dar diagnóstico definitivo", "Grounding, regras no prompt e avaliação "
          "automática de fidelidade e segurança"],
         ["LLM indisponível ou resposta fora do formato", "Validação do JSON e fallback por template, monitorado"],
         ["Atributos correlacionados geram contribuições contraintuitivas na oclusão", "Contexto inclui as médias "
          "por classe para o médico interpretar; documentado como limitação"]], [0.45, 0.55])

    h1("11. Limitações e próximos passos")
    bl(["Dataset público e pequeno (569 amostras); antes de qualquer uso assistencial é necessária validação "
        "externa e prospectiva com dados do hospital.",
        "Os pesos da fitness refletem uma preferência por recall que deve ser definida com a equipe clínica.",
        "Explicações locais por oclusão não tratam a correlação entre atributos (ex.: raio, perímetro e área).",
        "Próximos passos (Módulo 3): ampliar o espaço de busca (seleção de atributos, outros algoritmos), "
        "otimização multiobjetivo (NSGA-II) para a fronteira recall × falsos positivos, integrar laudos textuais "
        "ao contexto da LLM e expor as explicações no frontend Angular da Fase 1."])

    h1("12. Como reproduzir")
    B.append(("code", "cd tech-challenge-fase2\n"
              "python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate\n"
              "pip install -r requirements.txt                     # ou: poetry install\n"
              "python -m scripts.run_experiments                   # AG: todos os experimentos\n"
              "python -m scripts.evaluate_llm --provider groq      # avaliação da LLM (GROQ_API_KEY no .env)\n"
              "python -m scripts.make_figures && python -m scripts.generate_report\n"
              "pytest                                              # testes\n"
              "uvicorn src.api.main:app --port 8000                # API em http://localhost:8000/docs"))
    p(f"Repositório: {REPO}")
    return B


# ----------------------------------------------------------------------------- #
# Renderização                                                                   #
# ----------------------------------------------------------------------------- #
def _md_inline(text: str) -> str:
    text = re.sub(r"</?b>", "**", text)
    return re.sub(r"</?i>", "*", text)


def to_markdown(blocks: list[tuple], title: str, subtitle: str) -> str:
    out = [f"# {title}", "", f"**{subtitle}**", "",
           "Integrantes: " + " · ".join(f"{n} ({rm})" for n, rm in TEAM), ""]
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            out += [f"## {b[1]}", ""]
        elif kind == "h2":
            out += [f"### {b[1]}", ""]
        elif kind == "p":
            out += [_md_inline(b[1]), ""]
        elif kind == "bullets":
            out += [f"- {_md_inline(i)}" for i in b[1]] + [""]
        elif kind == "table":
            rows = b[1]
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * len(rows[0]))
            out += ["| " + " | ".join(_md_inline(str(c)).replace("|", "\\|") for c in r) + " |" for r in rows[1:]]
            out.append("")
        elif kind == "img":
            out += [f"![{b[3]}](figures/{b[1].name})", "", f"*{b[3]}*", ""]
        elif kind == "code":
            out += ["```text", b[1], "```", ""]
    return "\n".join(out)


def to_pdf(blocks: list[tuple], title: str, subtitle: str, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    font, bold, mono = "Helvetica", "Helvetica-Bold", "Courier"
    candidates = [Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts"), Path("/Library/Fonts")]
    for folder in candidates:
        if (folder / "DejaVuSans.ttf").exists():
            pdfmetrics.registerFont(TTFont("DV", str(folder / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DVB", str(folder / "DejaVuSans-Bold.ttf")))
            pdfmetrics.registerFont(TTFont("DVM", str(folder / "DejaVuSansMono.ttf")))
            from reportlab.lib.fonts import addMapping
            addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DV"); addMapping("DV", 1, 1, "DVB")  # noqa: E702
            font, bold, mono = "DV", "DVB", "DVM"
            break

    def safe(t: str) -> str:
        if font != "Helvetica":
            return t
        for a, b in {"→": "->", "×": "x", "α": "alfa", "≤": "<=", "≈": "~", "⇄": "<->", "★": "*", "·": "-",
                     "∈": "em", "±": "+/-", "–": "-", "—": "-"}.items():
            t = t.replace(a, b)
        return t

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName=font, fontSize=9.2, leading=13.2, spaceAfter=5)
    H1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=6,
                        textColor=colors.HexColor("#1f4e79"))
    H2 = ParagraphStyle("h2", parent=body, fontName=bold, fontSize=10.8, leading=14, spaceBefore=8, spaceAfter=4,
                        textColor=colors.HexColor("#2a6f97"))
    cap = ParagraphStyle("cap", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor("#555555"),
                         alignment=TA_CENTER, spaceAfter=8)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=9.2, spaceAfter=0)
    cellh = ParagraphStyle("cellh", parent=cell, fontName=bold, textColor=colors.white)
    code = ParagraphStyle("code", parent=body, fontName=mono, fontSize=7, leading=9, backColor=colors.HexColor("#f4f6f8"),
                          borderPadding=5, leftIndent=4, rightIndent=4, spaceBefore=3, spaceAfter=8)

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.7 * cm, title=title, author="Grupo — Tech Challenge")
    width = A4[0] - 3.6 * cm

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.0 * cm, safe("Tech Challenge — Fase 2 · Projeto 1"))
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"{doc_.page}")
        canvas.restoreState()

    story: list = []
    # capa
    story += [Spacer(1, 4.5 * cm),
              Paragraph(safe("TECH CHALLENGE — FASE 2"), ParagraphStyle("k", parent=H2, alignment=TA_CENTER)),
              Paragraph(safe(title), ParagraphStyle("t", parent=H1, fontSize=20, leading=25, alignment=TA_CENTER)),
              Spacer(1, 0.3 * cm),
              Paragraph(safe(subtitle), ParagraphStyle("s", parent=body, fontSize=11, alignment=TA_CENTER)),
              Spacer(1, 2.5 * cm)]
    for n, rm in TEAM:
        story.append(Paragraph(f"{n} — {rm}", ParagraphStyle("m", parent=body, fontSize=11, alignment=TA_CENTER)))
    story += [Spacer(1, 1 * cm), Paragraph(safe(f"Repositório: {REPO}"), ParagraphStyle("r", parent=body, alignment=TA_CENTER)),
              Paragraph(date.today().strftime("%d/%m/%Y"), ParagraphStyle("d", parent=body, alignment=TA_CENTER)),
              PageBreak()]

    for b in blocks:
        kind = b[0]
        if kind == "h1":
            story.append(Paragraph(safe(b[1]), H1))
        elif kind == "h2":
            story.append(Paragraph(safe(b[1]), H2))
        elif kind == "p":
            story.append(Paragraph(safe(b[1]), body))
        elif kind == "bullets":
            story.append(ListFlowable([ListItem(Paragraph(safe(i), body), leftIndent=12) for i in b[1]],
                                      bulletType="bullet", start="•", leftIndent=10, bulletFontSize=7))
        elif kind == "table":
            rows, ws = b[1], b[2]
            ncol = len(rows[0])
            ws = ws or [1 / ncol] * ncol
            data = [[Paragraph(safe(str(c)), cellh if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
            t = Table(data, colWidths=[width * x / sum(ws) for x in ws], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a6f97")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story += [t, Spacer(1, 6)]
        elif kind == "img":
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(str(b[1])).getSize()
            wpx = width * b[2]
            story.append(KeepTogether([Image(str(b[1]), width=wpx, height=wpx * ih / iw),
                                       Paragraph(safe(b[3]), cap)]))
        elif kind == "code":
            story.append(Preformatted(safe(b[1]), code, maxLineLength=118))
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)


def main() -> None:
    res = json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8"))
    llm, llm_is_real = None, False
    real = [f for f in sorted(RESULTS.glob("llm_evaluation_*.json")) if not f.name.endswith("_offline.json")]
    if real:
        llm, llm_is_real = json.loads(real[0].read_text(encoding="utf-8")), True
    elif (RESULTS / "llm_evaluation_offline.json").exists():
        llm = json.loads((RESULTS / "llm_evaluation_offline.json").read_text(encoding="utf-8"))
    title = "Otimização de modelos de diagnóstico de câncer de mama com algoritmos genéticos e LLMs"
    subtitle = "Relatório técnico — Projeto 1: otimização de modelos de diagnóstico"
    blocks = build(res, llm, llm_is_real)
    OUT_MD.write_text(to_markdown(blocks, title, subtitle), encoding="utf-8")
    to_pdf(blocks, title, subtitle, OUT_PDF)
    print(OUT_MD)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
