import numpy as np
import pytest

from src.ga import CategoricalGene, FloatGene, GAConfig, GeneticAlgorithm, IntGene, SearchSpace, random_search
from src.ga.operators import (blx_alpha_crossover, gaussian_mutation, one_point_crossover, population_diversity,
                              rank_selection, roulette_selection, tournament_selection, uniform_crossover,
                              uniform_mutation)
from src.spaces import SPACES


# ----------------------------- codificação --------------------------------- #
def test_float_gene_log_scale_bounds_and_roundtrip():
    g = FloatGene("C", 1e-3, 1e2, log=True)
    assert g.decode(0.0) == pytest.approx(1e-3)
    assert g.decode(1.0) == pytest.approx(1e2)
    assert g.decode(0.6) == pytest.approx(1.0)  # meio da escala log entre 1e-3 e 1e2
    assert g.decode(g.encode(3.5)) == pytest.approx(3.5)


def test_int_and_categorical_genes():
    i = IntGene("n", 2, 20)
    assert i.decode(0) == 2 and i.decode(1) == 20 and isinstance(i.decode(0.5), int)
    c = CategoricalGene("x", ("a", "b", "c"))
    assert [c.decode(u) for u in (0.0, 0.34, 0.99, 1.0)] == ["a", "b", "c", "c"]
    assert c.decode(c.encode("b")) == "b"


def test_search_spaces_decode_valid_params():
    rng = np.random.default_rng(0)
    for space in SPACES.values():
        for chrom in space.random(rng, 20):
            params = space.decode(chrom)
            assert set(params) == set(space.names)
            assert 0.2 <= params["threshold"] <= 0.8


def test_duplicate_gene_names_rejected():
    with pytest.raises(ValueError):
        SearchSpace([FloatGene("a", 0, 1), IntGene("a", 0, 3)])


# ------------------------------ operadores --------------------------------- #
def test_tournament_with_full_size_always_picks_best():
    fit = np.array([0.1, 0.9, 0.5, 0.3])
    idx = tournament_selection(fit, 50, np.random.default_rng(0), k=40)
    assert (idx == 1).mean() > 0.95


@pytest.mark.parametrize("selector", [roulette_selection, rank_selection])
def test_proportional_selectors_favor_fitter(selector):
    fit = np.array([0.90, 0.91, 0.99])
    idx = selector(fit, 3000, np.random.default_rng(1))
    counts = np.bincount(idx, minlength=3)
    assert counts[2] > counts[0]


def test_crossovers_keep_shape_and_bounds():
    rng = np.random.default_rng(0)
    a, b = rng.random(8), rng.random(8)
    for op in (uniform_crossover, one_point_crossover, blx_alpha_crossover):
        c1, c2 = op(a, b, rng)
        assert c1.shape == a.shape and c2.shape == a.shape
        assert ((0 <= c1) & (c1 <= 1)).all() and ((0 <= c2) & (c2 <= 1)).all()
    c1, c2 = uniform_crossover(a, b, rng)
    assert np.allclose(np.sort(np.concatenate([c1, c2])), np.sort(np.concatenate([a, b])))


def test_mutation_rate_zero_is_identity_and_one_changes():
    rng = np.random.default_rng(0)
    x = rng.random(10)
    assert np.allclose(gaussian_mutation(x, 0.0, rng), x)
    assert np.allclose(uniform_mutation(x, 0.0, rng), x)
    assert not np.allclose(uniform_mutation(x, 1.0, rng), x)


def test_diversity_zero_for_identical_population():
    assert population_diversity(np.ones((5, 3)) * 0.4) == 0


# -------------------------------- motor ------------------------------------ #
TOY = SearchSpace([FloatGene(f"x{i}", 0, 1) for i in range(5)])


def toy_fitness(params):
    x = np.array(list(params.values()))
    return float(-((x - 0.7) ** 2).sum()), {}


def test_ga_converges_on_toy_problem_and_beats_random():
    cfg = GAConfig(population_size=30, generations=30, crossover="blx", mutation_rate=0.2, seed=1)
    ga = GeneticAlgorithm(TOY, toy_fitness, cfg).run()
    rs = random_search(TOY, toy_fitness, budget=ga.evaluations, seed=1)
    assert ga.best_fitness > -0.01
    assert ga.best_fitness > rs.best_fitness


def test_elitism_makes_best_fitness_monotonic():
    cfg = GAConfig(population_size=12, generations=15, elitism=2, seed=3)
    hist = GeneticAlgorithm(TOY, toy_fitness, cfg).run().history
    bests = [h.best_fitness for h in hist]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))


def test_same_seed_is_reproducible():
    cfg = GAConfig(population_size=10, generations=5, seed=7)
    r1 = GeneticAlgorithm(TOY, toy_fitness, cfg).run()
    r2 = GeneticAlgorithm(TOY, toy_fitness, GAConfig(population_size=10, generations=5, seed=7)).run()
    assert r1.best_params == r2.best_params


def test_cache_avoids_reevaluating_same_individual():
    calls = []

    def fit(p):
        calls.append(p)
        return 1.0, {}

    space = SearchSpace([CategoricalGene("c", ("a", "b"))])
    res = GeneticAlgorithm(space, fit, GAConfig(population_size=10, generations=5)).run()
    assert len(calls) <= 2 and res.cache_hits > 0


def test_patience_stops_early():
    res = GeneticAlgorithm(TOY, lambda p: (1.0, {}), GAConfig(population_size=6, generations=50, patience=3)).run()
    assert res.stopped_early and len(res.history) == 4


def test_seeding_puts_initial_individual_in_population():
    space = SearchSpace([FloatGene("a", 0, 1)])
    res = GeneticAlgorithm(space, lambda p: (-abs(p["a"] - 0.123), {}),
                           GAConfig(population_size=4, generations=0), initial=[{"a": 0.123}]).run()
    assert res.best_params["a"] == pytest.approx(0.123)


def test_failing_fitness_does_not_crash():
    def fit(p):
        if p["x0"] > 0.5:
            raise ValueError("combinação inválida")
        return p["x0"], {}

    res = GeneticAlgorithm(TOY, fit, GAConfig(population_size=8, generations=3)).run()
    assert res.best_fitness <= 0.5


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        GAConfig(selection="xyz").validate()
    with pytest.raises(ValueError):
        GAConfig(population_size=4, elitism=4).validate()
