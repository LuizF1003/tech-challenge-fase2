"""Espaços de busca (genes) dos modelos da Fase 1.

O limiar de decisão (``threshold``) entra como gene nos dois modelos: num
problema de triagem o custo de um falso negativo (câncer não sinalizado) é
muito maior que o de um falso positivo, e o limiar é o controle mais direto
desse equilíbrio.
"""

from src.ga.search_space import CategoricalGene, FloatGene, IntGene, SearchSpace

LOGISTIC_SPACE = SearchSpace(
    [
        FloatGene("C", 1e-3, 1e2, log=True),        # força da regularização (inverso)
        FloatGene("l1_ratio", 0.0, 1.0),            # 0 = L2 (Ridge) ... 1 = L1 (Lasso)
        CategoricalGene("class_weight", (None, "balanced")),
        FloatGene("threshold", 0.20, 0.80),
    ],
    rounding={"C": 4, "l1_ratio": 2, "threshold": 2},
)

RANDOM_FOREST_SPACE = SearchSpace(
    [
        IntGene("n_estimators", 50, 300),
        CategoricalGene("max_depth", (3, 5, 8, 12, 20, None)),
        IntGene("min_samples_split", 2, 20),
        IntGene("min_samples_leaf", 1, 10),
        CategoricalGene("max_features", ("sqrt", "log2", 0.3, 0.5, 0.8)),
        CategoricalGene("criterion", ("gini", "entropy")),
        CategoricalGene("class_weight", (None, "balanced", "balanced_subsample")),
        CategoricalGene("bootstrap", (True, False)),
        FloatGene("threshold", 0.20, 0.80),
    ],
    rounding={"threshold": 2},
)

SPACES = {
    "logistic_regression": LOGISTIC_SPACE,
    "random_forest": RANDOM_FOREST_SPACE,
}
