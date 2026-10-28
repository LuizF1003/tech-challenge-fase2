"""Operadores genéticos: seleção, cruzamento e mutação.

Todos trabalham sobre cromossomos reais em [0, 1]^d (ver ``search_space.py``)
e recebem um ``np.random.Generator`` explícito para garantir reprodutibilidade.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Rng = np.random.Generator

# --------------------------------------------------------------------------- #
# Seleção — devolve os ÍNDICES dos pais escolhidos                             #
# --------------------------------------------------------------------------- #


def tournament_selection(fitness: np.ndarray, n: int, rng: Rng, k: int = 3) -> np.ndarray:
    """Torneio: sorteia ``k`` indivíduos e escolhe o de maior fitness.

    ``k`` controla a pressão seletiva (k maior → convergência mais rápida).
    """
    contenders = rng.integers(0, len(fitness), size=(n, k))
    winners = contenders[np.arange(n), np.argmax(fitness[contenders], axis=1)]
    return winners


def roulette_selection(fitness: np.ndarray, n: int, rng: Rng, **_) -> np.ndarray:
    """Roleta (proporcional ao fitness).

    O fitness é deslocado pelo mínimo da população; como os valores ficam muito
    próximos (ex.: 0,95–0,98), sem o deslocamento a roleta seria quase aleatória.
    """
    shifted = fitness - fitness.min() + 1e-9
    probs = shifted / shifted.sum()
    return rng.choice(len(fitness), size=n, p=probs)


def rank_selection(fitness: np.ndarray, n: int, rng: Rng, **_) -> np.ndarray:
    """Seleção por ranking linear: probabilidade proporcional à posição."""
    ranks = np.empty(len(fitness))
    ranks[np.argsort(fitness)] = np.arange(1, len(fitness) + 1)
    probs = ranks / ranks.sum()
    return rng.choice(len(fitness), size=n, p=probs)


SELECTION: dict[str, Callable[..., np.ndarray]] = {
    "tournament": tournament_selection,
    "roulette": roulette_selection,
    "rank": rank_selection,
}

# --------------------------------------------------------------------------- #
# Cruzamento — recebe dois pais e devolve dois filhos                          #
# --------------------------------------------------------------------------- #


def uniform_crossover(a: np.ndarray, b: np.ndarray, rng: Rng, **_) -> tuple[np.ndarray, np.ndarray]:
    """Cada gene vem de um dos pais com probabilidade 0,5."""
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: Rng, **_) -> tuple[np.ndarray, np.ndarray]:
    """Corta os cromossomos em um ponto e troca as caudas."""
    if a.size < 2:
        return a.copy(), b.copy()
    point = rng.integers(1, a.size)
    return (np.concatenate([a[:point], b[point:]]),
            np.concatenate([b[:point], a[point:]]))


def blx_alpha_crossover(a: np.ndarray, b: np.ndarray, rng: Rng, alpha: float = 0.5,
                        **_) -> tuple[np.ndarray, np.ndarray]:
    """BLX-α (Blend crossover) para codificação real.

    Cada filho é sorteado no intervalo [min - α·d, max + α·d], d = |a - b|,
    o que permite explorar um pouco além dos pais. Resultado é recortado em [0, 1].
    """
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    d = hi - lo
    lo, hi = lo - alpha * d, hi + alpha * d
    c1 = rng.uniform(lo, hi)
    c2 = rng.uniform(lo, hi)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


CROSSOVER: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "uniform": uniform_crossover,
    "one_point": one_point_crossover,
    "blx": blx_alpha_crossover,
}

# --------------------------------------------------------------------------- #
# Mutação — altera cada gene com probabilidade ``rate``                        #
# --------------------------------------------------------------------------- #


def gaussian_mutation(x: np.ndarray, rate: float, rng: Rng, sigma: float = 0.1, **_) -> np.ndarray:
    """Soma ruído N(0, σ) aos genes sorteados (ajuste fino ao redor da solução)."""
    mask = rng.random(x.shape) < rate
    noise = rng.normal(0.0, sigma, size=x.shape)
    return np.clip(np.where(mask, x + noise, x), 0, 1)


def uniform_mutation(x: np.ndarray, rate: float, rng: Rng, **_) -> np.ndarray:
    """Substitui os genes sorteados por um valor aleatório (exploração ampla)."""
    mask = rng.random(x.shape) < rate
    return np.where(mask, rng.random(x.shape), x)


MUTATION: dict[str, Callable[..., np.ndarray]] = {
    "gaussian": gaussian_mutation,
    "uniform": uniform_mutation,
}


def population_diversity(pop: np.ndarray) -> float:
    """Diversidade = desvio-padrão médio por gene (0 = população convergida)."""
    return float(pop.std(axis=0).mean())
