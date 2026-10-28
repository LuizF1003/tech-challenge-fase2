"""Codificação dos hiperparâmetros em genes.

Representação escolhida: **cromossomo real normalizado**. Cada indivíduo é um
vetor ``x ∈ [0, 1]^d`` e cada posição (gene) é decodificada para um
hiperparâmetro pelo seu tipo:

* ``FloatGene``  — contínuo, em escala linear ou logarítmica (ex.: ``C`` em
  [1e-3, 1e2] usa escala log para que cada ordem de grandeza tenha a mesma
  chance de ser explorada);
* ``IntGene``    — inteiro, por arredondamento;
* ``CategoricalGene`` — o intervalo [0, 1) é dividido em ``k`` faixas iguais,
  uma por categoria.

Vantagem: todos os operadores genéticos (crossover BLX-α, mutação gaussiana etc.)
funcionam sobre o mesmo vetor real, independentemente do tipo de hiperparâmetro,
e o espaço de busca de um novo modelo é só uma lista de genes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class FloatGene:
    name: str
    low: float
    high: float
    log: bool = False

    def decode(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            lo, hi = math.log10(self.low), math.log10(self.high)
            return float(10 ** (lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))

    def encode(self, value: float) -> float:
        if self.log:
            lo, hi = math.log10(self.low), math.log10(self.high)
            return float(np.clip((math.log10(value) - lo) / (hi - lo), 0, 1))
        return float(np.clip((value - self.low) / (self.high - self.low), 0, 1))


@dataclass(frozen=True)
class IntGene:
    name: str
    low: int
    high: int

    def decode(self, u: float) -> int:
        u = float(np.clip(u, 0.0, 1.0))
        return int(round(self.low + u * (self.high - self.low)))

    def encode(self, value: int) -> float:
        return float(np.clip((value - self.low) / (self.high - self.low), 0, 1))


@dataclass(frozen=True)
class CategoricalGene:
    name: str
    choices: tuple[Any, ...] = field(default_factory=tuple)

    def decode(self, u: float) -> Any:
        u = float(np.clip(u, 0.0, 1.0 - 1e-12))
        return self.choices[int(u * len(self.choices))]

    def encode(self, value: Any) -> float:
        idx = self.choices.index(value)
        return (idx + 0.5) / len(self.choices)


Gene = FloatGene | IntGene | CategoricalGene


class SearchSpace:
    """Lista ordenada de genes que define o cromossomo de um modelo."""

    def __init__(self, genes: Sequence[Gene], rounding: dict[str, int] | None = None):
        names = [g.name for g in genes]
        if len(names) != len(set(names)):
            raise ValueError("Genes com nomes repetidos no espaço de busca.")
        self.genes = list(genes)
        # casas decimais usadas para deixar parâmetros legíveis e melhorar o cache
        self.rounding = rounding or {}

    @property
    def size(self) -> int:
        return len(self.genes)

    @property
    def names(self) -> list[str]:
        return [g.name for g in self.genes]

    def random(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.random((n, self.size))

    def decode(self, chromosome: Sequence[float]) -> dict[str, Any]:
        params = {}
        for gene, u in zip(self.genes, chromosome):
            value = gene.decode(u)
            if isinstance(value, float) and gene.name in self.rounding:
                value = round(value, self.rounding[gene.name])
            params[gene.name] = value
        return params

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        return np.array([g.encode(params[g.name]) for g in self.genes], dtype=float)

    def key(self, chromosome: Sequence[float]) -> tuple:
        """Chave hashável dos parâmetros decodificados (usada no cache de fitness)."""
        return tuple(sorted((k, repr(v)) for k, v in self.decode(chromosome).items()))
