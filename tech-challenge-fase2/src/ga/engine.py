"""Motor do algoritmo genético (geracional, com elitismo).

Fluxo de cada geração::

    avaliar população ─► registrar estatísticas ─► elitismo (copia os melhores)
          ▲                                              │
          │                                              ▼
    nova população ◄─ mutação ◄─ cruzamento ◄─ seleção dos pais

O motor é independente do problema: recebe um ``SearchSpace`` e uma função
``fitness(params) -> (score, metricas)``. Avaliações repetidas do mesmo conjunto
de hiperparâmetros são servidas por um cache (economiza treinos).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from src.ga.operators import CROSSOVER, MUTATION, SELECTION, population_diversity
from src.ga.search_space import SearchSpace

logger = logging.getLogger(__name__)

FitnessFn = Callable[[dict[str, Any]], tuple[float, dict[str, float]]]


@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 15
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    selection: str = "tournament"      # tournament | roulette | rank
    tournament_k: int = 3
    crossover: str = "uniform"         # uniform | one_point | blx
    blx_alpha: float = 0.5
    mutation: str = "gaussian"         # gaussian | uniform
    mutation_sigma: float = 0.1
    elitism: int = 2
    patience: int | None = None        # parada antecipada (gerações sem melhora)
    seed: int = 42

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size deve ser >= 2")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism deve estar em [0, population_size)")
        for value, name in ((self.crossover_rate, "crossover_rate"), (self.mutation_rate, "mutation_rate")):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} deve estar em [0, 1]")
        if self.selection not in SELECTION:
            raise ValueError(f"selection inválida: {self.selection}")
        if self.crossover not in CROSSOVER:
            raise ValueError(f"crossover inválido: {self.crossover}")
        if self.mutation not in MUTATION:
            raise ValueError(f"mutation inválida: {self.mutation}")


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    diversity: float
    evaluations: int
    cache_hits: int
    elapsed_s: float
    best_params: dict[str, Any]


@dataclass
class GAResult:
    best_params: dict[str, Any]
    best_fitness: float
    best_metrics: dict[str, float]
    history: list[GenerationStats] = field(default_factory=list)
    evaluations: int = 0
    cache_hits: int = 0
    elapsed_s: float = 0.0
    stopped_early: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out


class GeneticAlgorithm:
    def __init__(
        self,
        space: SearchSpace,
        fitness: FitnessFn,
        config: GAConfig | None = None,
        on_generation: Callable[[GenerationStats], None] | None = None,
        initial: list[dict[str, Any]] | None = None,
    ):
        self.space = space
        self.fitness_fn = fitness
        self.config = config or GAConfig()
        self.config.validate()
        self.on_generation = on_generation
        self.initial = initial or []
        self.rng = np.random.default_rng(self.config.seed)
        self._cache: dict[tuple, tuple[float, dict[str, float]]] = {}
        self.evaluations = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ #
    def _evaluate(self, chromosome: np.ndarray) -> tuple[float, dict[str, float]]:
        key = self.space.key(chromosome)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        params = self.space.decode(chromosome)
        try:
            result = self.fitness_fn(params)
        except Exception as exc:  # combinação inválida não derruba a busca
            logger.warning("fitness falhou para %s: %s", params, exc)
            result = (0.0, {"error": 1.0})
        self.evaluations += 1
        self._cache[key] = result
        return result

    def _initial_population(self) -> np.ndarray:
        pop = self.space.random(self.rng, self.config.population_size)
        # "sementes": permite incluir o modelo original na população inicial
        for i, params in enumerate(self.initial[: self.config.population_size]):
            pop[i] = self.space.encode(params)
        return pop

    def _next_generation(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        cfg = self.config
        n = cfg.population_size
        order = np.argsort(fitness)[::-1]
        elite = pop[order[: cfg.elitism]].copy()

        n_children = n - cfg.elitism
        n_parents = n_children + (n_children % 2)
        select = SELECTION[cfg.selection]
        parents_idx = select(fitness, n_parents, self.rng, k=cfg.tournament_k)
        parents = pop[parents_idx]

        children = []
        cross = CROSSOVER[cfg.crossover]
        mutate = MUTATION[cfg.mutation]
        for i in range(0, n_parents, 2):
            a, b = parents[i], parents[i + 1]
            if self.rng.random() < cfg.crossover_rate:
                c1, c2 = cross(a, b, self.rng, alpha=cfg.blx_alpha)
            else:
                c1, c2 = a.copy(), b.copy()
            children.append(mutate(c1, cfg.mutation_rate, self.rng, sigma=cfg.mutation_sigma))
            children.append(mutate(c2, cfg.mutation_rate, self.rng, sigma=cfg.mutation_sigma))
        children = np.array(children[:n_children])
        return np.vstack([elite, children]) if cfg.elitism else children

    # ------------------------------------------------------------------ #
    def run(self) -> GAResult:
        cfg = self.config
        start = time.perf_counter()
        pop = self._initial_population()

        best_fit, best_chrom, best_metrics = -np.inf, None, {}
        history: list[GenerationStats] = []
        stale, stopped_early = 0, False

        for gen in range(cfg.generations + 1):  # geração 0 = população inicial
            results = [self._evaluate(ind) for ind in pop]
            fitness = np.array([r[0] for r in results])

            idx = int(np.argmax(fitness))
            if fitness[idx] > best_fit + 1e-12:
                best_fit, best_chrom, best_metrics = float(fitness[idx]), pop[idx].copy(), results[idx][1]
                stale = 0
            else:
                stale += 1

            stats = GenerationStats(
                generation=gen,
                best_fitness=best_fit,
                mean_fitness=float(fitness.mean()),
                std_fitness=float(fitness.std()),
                worst_fitness=float(fitness.min()),
                diversity=population_diversity(pop),
                evaluations=self.evaluations,
                cache_hits=self.cache_hits,
                elapsed_s=round(time.perf_counter() - start, 3),
                best_params=self.space.decode(best_chrom),
            )
            history.append(stats)
            logger.info(
                "geração %d | melhor=%.4f média=%.4f diversidade=%.3f avaliações=%d",
                gen, stats.best_fitness, stats.mean_fitness, stats.diversity, stats.evaluations,
            )
            if self.on_generation:
                self.on_generation(stats)

            if cfg.patience is not None and stale >= cfg.patience:
                stopped_early = True
                logger.info("parada antecipada: %d gerações sem melhora", stale)
                break
            if gen < cfg.generations:
                pop = self._next_generation(pop, fitness)

        return GAResult(
            best_params=self.space.decode(best_chrom),
            best_fitness=best_fit,
            best_metrics=best_metrics,
            history=history,
            evaluations=self.evaluations,
            cache_hits=self.cache_hits,
            elapsed_s=round(time.perf_counter() - start, 3),
            stopped_early=stopped_early,
            config=asdict(cfg),
        )


def random_search(space: SearchSpace, fitness: FitnessFn, budget: int, seed: int = 42) -> GAResult:
    """Busca aleatória com o mesmo orçamento de avaliações (linha de base de comparação)."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    best = (-np.inf, None, {})
    history = []
    for i, chrom in enumerate(space.random(rng, budget)):
        score, metrics = fitness(space.decode(chrom))
        if score > best[0]:
            best = (score, chrom, metrics)
        history.append(GenerationStats(
            generation=i, best_fitness=float(best[0]), mean_fitness=float(score), std_fitness=0.0,
            worst_fitness=float(score), diversity=0.0, evaluations=i + 1, cache_hits=0,
            elapsed_s=round(time.perf_counter() - start, 3), best_params=space.decode(best[1]),
        ))
    return GAResult(
        best_params=space.decode(best[1]), best_fitness=float(best[0]), best_metrics=best[2],
        history=history, evaluations=budget, elapsed_s=round(time.perf_counter() - start, 3),
        config={"method": "random_search", "budget": budget, "seed": seed},
    )
