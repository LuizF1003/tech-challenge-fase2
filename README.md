# Sistema de apoio à classificação de câncer de mama

Projeto educacional de Machine Learning para análise de dados relacionados a massas mamárias, com notebook de análise exploratória e interface web em Angular para visualização interativa dos resultados.

## Integrantes

- Gabriel Wesley — RM376899
- Gustavo Leite — RM375733
- Luiz Fellipe — RM376854

---

## Fase 2 — Otimização com algoritmo genético e LLM

A Fase 2 (Projeto 1) está em [`tech-challenge-fase2/`](tech-challenge-fase2/README.md): os modelos da
Fase 1 têm os hiperparâmetros otimizados por um algoritmo genético e uma LLM explica os resultados
para médicos. Relatório técnico: [`tech-challenge-fase2/reports/relatorio_tecnico_fase2.pdf`](tech-challenge-fase2/reports/relatorio_tecnico_fase2.pdf).

---

## Visão geral

O sistema analisa dados estruturados de exames relacionados a massas mamárias e auxilia a classificação técnica dos registros como:

- `B` — benigno;
- `M` — maligno.

A solução utiliza atributos numéricos como raio, textura, perímetro, área, concavidade, simetria e outras medidas morfológicas para identificar padrões nos dados.

> **Importante:** este projeto possui finalidade educacional e de apoio à triagem. Ele não substitui avaliação médica, exames clínicos, mamografia, biópsia ou decisões de profissionais da saúde.

---

## Objetivo

Construir uma solução inicial de Inteligência Artificial aplicada à saúde feminina, utilizando Machine Learning para identificar padrões associados a diagnósticos benignos e malignos de câncer de mama.

O projeto contempla:

- validação de compatibilidade de arquivos CSV;
- análise exploratória dos dados;
- tratamento e preparação dos dados;
- treinamento de modelos de classificação;
- comparação entre algoritmos;
- avaliação utilizando métricas de desempenho;
- explicabilidade das previsões;
- testes com novos dados;
- geração de relatório em PDF;
- visualização do pipeline por meio de uma interface web.

---

## Notebook de análise

O notebook principal está disponível em:

```text
notebooks/01_eda.ipynb
```

Ele apresenta o fluxo de análise de dados e Machine Learning, incluindo:

1. carregamento e inspeção do dataset;
2. validação das colunas e dos valores;
3. análise exploratória dos dados;
4. tratamento e preparação das variáveis;
5. treinamento dos modelos;
6. comparação entre Regressão Logística e Random Forest;
7. avaliação com métricas de classificação;
8. interpretação da importância das variáveis;
9. realização de previsões para novas amostras.

Para executar o notebook, instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

Em seguida, inicie o Jupyter:

```bash
jupyter notebook
```

Abra o arquivo:

```text
notebooks/01_eda.ipynb
```

---

## Frontend — Angular 16

O frontend disponibiliza uma interface web para executar e visualizar o fluxo do notebook de forma interativa.

A aplicação consome a API do backend e apresenta resultados de validação, análise exploratória, modelagem, avaliação, explicabilidade e previsão.

Tecnologias utilizadas:

- Angular 16;
- componentes standalone;
- TypeScript;
- `ng2-charts`;
- Chart.js.

### Como rodar o frontend

Pré-requisito: o backend deve estar em execução em:

```text
http://localhost:8000
```

A URL da API pode ser alterada no arquivo:

```text
frontend/src/environments/environment.ts
```

Execute os comandos:

```bash
cd frontend
npm install --legacy-peer-deps
npm start
```

Depois, abra no navegador:

```text
http://localhost:4200
```

> O parâmetro `--legacy-peer-deps` é utilizado devido a um possível conflito entre dependências peer do `ng2-charts`, `@angular/cdk` e Angular 16 no resolvedor estrito do npm.

### Fluxo da aplicação

#### 1. Upload — `/upload`

Permite:

- enviar um arquivo CSV para análise;
- utilizar o dataset padrão do projeto.

#### 2. Dashboard — `/dashboard`

O dashboard organiza os resultados do pipeline em abas:

1. **Validação**
   - Verifica se o CSV possui estrutura compatível com os dados de câncer de mama.

2. **EDA**
   - Exibe distribuição dos dados;
   - Correlações entre variáveis;
   - Estatísticas descritivas.

3. **Modelagem**
   - Compara os modelos de Regressão Logística e Random Forest.

4. **Avaliação**
   - Apresenta classification report;
   - Matriz de confusão;
   - Curva ROC;
   - Métricas de desempenho.

5. **Explicabilidade**
   - Mostra permutation importance;
   - Apresenta análises com SHAP.

6. **Prever amostra**
   - Permite preenchimento manual de dados;
   - Upload de dados em lote;
   - Download de relatório diagnóstico em PDF.

### Build de produção

```bash
cd frontend
npm run build
```

---

## Estrutura do projeto

```text
tech-challenge-fase1/
├── backend/                               # API e processamento da aplicação
├── frontend/                              # Interface web Angular 16
│   └── src/
│       └── app/
│           ├── core/                      # Tipos TS e AnalysisService
│           ├── shared/                    # Componentes reutilizáveis
│           └── pages/
│               ├── upload/                # Página de envio de CSV
│               └── dashboard/             # Dashboard com abas de análise
├── data/
│   ├── data.csv
│   ├── cancer_mama_1_diagnostico.csv
│   ├── breast_cancer_data.csv
│   └── outros_datasets.csv
├── notebooks/
│   └── 01_eda.ipynb                       # Notebook principal de análise
├── models/
│   └── modelo_cancer_mama.joblib          # Modelo treinado
├── reports/
│   ├── relatorio_tecnico.md
│   └── analise_diagnostica_cancer_mama.pdf
├── scripts/
│   └── generate_report_pdf.py             # Geração de relatório PDF
├── src/
│   └── data_utils.py                      # Funções auxiliares
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Modelos utilizados

O projeto compara dois algoritmos de classificação:

- **Regressão Logística**
  - Modelo estatístico utilizado como referência;
  - Adequado para interpretação de relações entre variáveis e classes.

- **Random Forest**
  - Modelo baseado em múltiplas árvores de decisão;
  - Capaz de identificar relações não lineares e interações entre atributos.

A escolha do modelo deve considerar métricas como acurácia, precisão, recall, F1-score e AUC-ROC, especialmente porque erros de classificação em contextos de saúde devem ser interpretados com cuidado.

---

## Dados

Os datasets utilizados ficam no diretório:

```text
data/
```

Os arquivos devem conter informações estruturadas referentes às características morfológicas das massas mamárias e uma coluna de diagnóstico, normalmente identificada pelas classes:

```text
B
M
```

---

## Relatórios

Os relatórios técnicos e diagnósticos gerados ficam no diretório:

```text
reports/
```

O sistema pode gerar um PDF com os resultados das análises e previsões realizadas.

---

## Aviso ético e clínico

Os resultados gerados pelo sistema representam uma análise computacional baseada em dados e modelos de Machine Learning.

Eles não devem ser utilizados isoladamente para determinar diagnósticos médicos. A interpretação clínica deve ser realizada exclusivamente por profissionais qualificados, considerando histórico médico, exames complementares e avaliação especializada.
