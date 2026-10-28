"""Carregamento e preparação dos dados.

Mantém exatamente as regras da Fase 1 (``tech-challenge-fase1/src/pipeline.py``):
remove colunas vazias e ``id``, normaliza os rótulos de ``diagnosis`` (M/B) e
usa o mesmo split estratificado 80/20 com ``random_state=42``. Assim o conjunto
de teste é idêntico ao da Fase 1 e a comparação "original vs otimizado" é justa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "diagnosis"
TEST_SIZE = 0.20

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PHASE1_DATA_DIR = PROJECT_DIR.parent / "tech-challenge-fase1" / "data"

LABEL_MAP = {
    "m": "M", "malignant": "M", "maligno": "M",
    "b": "B", "benign": "B", "benigno": "B",
}


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return self.X_train.columns.tolist()


def resolve_data_path(filename: str = "data.csv") -> Path:
    """Procura o CSV na pasta da Fase 2 e, se não existir, na da Fase 1."""
    for folder in (DATA_DIR, PHASE1_DATA_DIR):
        candidate = folder / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset '{filename}' não encontrado em {DATA_DIR} nem em {PHASE1_DATA_DIR}. "
        "Extraia o data.7z da raiz do repositório para tech-challenge-fase2/data/."
    )


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica as mesmas regras de limpeza da Fase 1 e devolve (X, y)."""
    data = raw.dropna(axis=1, how="all").drop(columns=["id"], errors="ignore").copy()
    if TARGET not in data.columns:
        raise ValueError("A base precisa conter a coluna 'diagnosis' (M/B).")
    labels = data[TARGET].astype("string").str.strip().str.lower()
    invalid = set(labels.dropna().unique()) - set(LABEL_MAP)
    if invalid:
        raise ValueError(f"Rótulos inesperados em diagnosis: {sorted(invalid)}")
    y = labels.map(LABEL_MAP).map({"M": 1, "B": 0}).astype(int).rename(TARGET)
    X = data.drop(columns=TARGET)
    X = X.select_dtypes("number")
    return X, y


def load_dataset(filename: str = "data.csv") -> tuple[pd.DataFrame, pd.Series]:
    return prepare(pd.read_csv(resolve_data_path(filename)))


def split(X: pd.DataFrame, y: pd.Series) -> DatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def load_split(filename: str = "data.csv") -> DatasetSplit:
    X, y = load_dataset(filename)
    return split(X, y)
