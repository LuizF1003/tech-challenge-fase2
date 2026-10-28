"""Cria notebooks/02_demo_fase2.ipynb (rode depois com nbconvert --execute para incluir as saídas)."""

import nbformat as nbf

from src.data import PROJECT_DIR

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Tech Challenge — Fase 2 · Demonstração\n\n"
       "Otimização dos modelos de diagnóstico da Fase 1 com **algoritmo genético** e interpretação dos "
       "resultados com **LLM**.\n\n"
       "> Projeto educacional de apoio à triagem. Não substitui a avaliação de profissionais de saúde."),
    code("import os, sys, json, warnings\n"
         "sys.path.insert(0, os.path.abspath('..'))\n"
         "os.chdir(os.path.abspath('..'))\n"
         "warnings.filterwarnings('ignore')\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "pd.set_option('display.width', 160)"),
    md("## 1. Dados — mesmo split da Fase 1"),
    code("from src.data import load_split\n"
         "data = load_split()\n"
         "print('treino:', data.X_train.shape, '| teste:', data.X_test.shape, '| malignos no teste:', int(data.y_test.sum()))"),
    md("## 2. Codificação: cromossomo → hiperparâmetros\n\n"
       "Cada indivíduo é um vetor em [0,1]^d; cada gene é decodificado conforme o seu tipo."),
    code("import numpy as np\n"
         "from src.spaces import SPACES\n"
         "space = SPACES['random_forest']\n"
         "chrom = np.random.default_rng(0).random(space.size)\n"
         "print('cromossomo:', chrom.round(3))\n"
         "space.decode(chrom)"),
    md("## 3. Rodando um AG pequeno ao vivo (Regressão Logística)"),
    code("from src.fitness import CVFitness\n"
         "from src.ga import GAConfig, GeneticAlgorithm\n"
         "from src.models import BASELINE_PARAMS\n\n"
         "fitness = CVFitness('logistic_regression', data.X_train, data.y_train)\n"
         "print('fitness do modelo original:', round(fitness(BASELINE_PARAMS['logistic_regression'])[0], 4))\n"
         "cfg = GAConfig(population_size=10, generations=6, crossover='blx', mutation_rate=0.15, seed=1)\n"
         "result = GeneticAlgorithm(SPACES['logistic_regression'], fitness, cfg).run()\n"
         "pd.DataFrame([{'geração': h.generation, 'melhor': round(h.best_fitness, 4), 'média': round(h.mean_fitness, 4),\n"
         "               'diversidade': round(h.diversity, 3)} for h in result.history])"),
    code("result.best_params, round(result.best_fitness, 4)"),
    md("## 4. Resultados completos dos experimentos\n\n"
       "Gerados por `python -m scripts.run_experiments` (5 configurações do AG + busca aleatória por modelo)."),
    code("res = json.load(open('reports/results/experiments.json'))\n"
         "rows = []\n"
         "for model, r in res['models'].items():\n"
         "    allr = {'original': r['baseline'], **r['experiments'], 'busca_aleatoria': r['random_search']}\n"
         "    for name, rec in allr.items():\n"
         "        t = rec['test_metrics']\n"
         "        rows.append({'modelo': model, 'config': name, 'cv_fitness': round(rec['cv_fitness'], 4),\n"
         "                     'cv5x3': round(rec['robust_cv']['fitness']['mean'], 4), 'recall': round(t['recall'], 3),\n"
         "                     'f1': round(t['f1'], 3), 'auc': round(t['roc_auc'], 3), 'FN': t['false_negatives'],\n"
         "                     'FP': t['false_positives']})\n"
         "pd.DataFrame(rows)"),
    code("print('Selecionado:', res['selected']['model'], '/', res['selected']['source'])\n"
         "res['selected']['params']"),
    code("for f in ['convergencia_ag.png', 'comparativo_teste.png', 'erros_teste.png', 'ag_vs_busca_aleatoria.png']:\n"
         "    display(Image(f'reports/figures/{f}'))"),
    md("## 5. Interpretação com LLM\n\n"
       "Defina `LLM_PROVIDER` (groq, ollama, openai, anthropic) no `.env`. Sem chave, o provedor `offline` é usado."),
    code("import joblib\n"
         "from src.llm import DiagnosisInterpreter, get_provider\n"
         "model = joblib.load('models/best_model.joblib')\n"
         "card = json.load(open('models/model_card.json'))\n"
         "provider = get_provider(os.getenv('LLM_PROVIDER', 'offline'))\n"
         "interp = DiagnosisInterpreter(model, card, provider)\n"
         "caso = data.X_test[data.y_test == 1].iloc[[0]]\n"
         "out = interp.explain_case(caso)\n"
         "print(out.text)\n"
         "print('\\n[provedor:', out.provider, '| fallback:', out.used_fallback, ']')"),
    md("### Contexto numérico enviado à LLM (grounding)"),
    code("pd.DataFrame(out.context['fatores_principais'])"),
    md("### Avaliação automática da resposta"),
    code("from src.llm.evaluation import evaluate_interpretation\n"
         "evaluate_interpretation(out)"),
    md("### Insights de desempenho para a equipe clínica"),
    code("print(interp.explain_performance(res).text)"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
path = PROJECT_DIR / "notebooks" / "02_demo_fase2.ipynb"
nbf.write(nb, path)
print(path)
