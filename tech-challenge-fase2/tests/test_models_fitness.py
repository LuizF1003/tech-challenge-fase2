import numpy as np
import pandas as pd
import pytest

from src.data import load_split, prepare
from src.fitness import CVFitness, classification_metrics, composite_score
from src.models import BASELINE_PARAMS, ThresholdClassifier, build_model


@pytest.fixture(scope="module")
def split():
    return load_split()


def test_split_matches_phase1(split):
    assert len(split.X_train) == 455 and len(split.X_test) == 114
    assert int(split.y_test.sum()) == 42
    assert "id" not in split.X_train.columns and "Unnamed: 32" not in split.X_train.columns
    assert split.X_train.shape[1] == 30


def test_prepare_normalizes_labels_and_rejects_unknown():
    df = pd.DataFrame({"id": [1, 2], "diagnosis": ["Maligno", " b "], "radius_mean": [1.0, 2.0]})
    X, y = prepare(df)
    assert y.tolist() == [1, 0] and list(X.columns) == ["radius_mean"]
    with pytest.raises(ValueError):
        prepare(df.assign(diagnosis=["x", "B"]))


@pytest.mark.parametrize("name", list(BASELINE_PARAMS))
def test_build_model_with_baseline_params(name, split):
    model = build_model(name, BASELINE_PARAMS[name]).fit(split.X_train.iloc[:150], split.y_train.iloc[:150])
    proba = model.predict_proba(split.X_test)[:, 1]
    assert ((0 <= proba) & (proba <= 1)).all()


def test_threshold_controls_prediction(split):
    model = build_model("logistic_regression", {"C": 1.0, "threshold": 0.5}).fit(split.X_train, split.y_train)
    proba = model.predict_proba(split.X_test)[:, 1]
    model.threshold = 0.2
    assert (model.predict(split.X_test) == (proba >= 0.2)).all()
    assert isinstance(model, ThresholdClassifier)


def test_metrics_and_composite():
    y = np.array([1, 1, 0, 0])
    m = classification_metrics(y, np.array([0.9, 0.3, 0.2, 0.6]), 0.5)
    assert m["recall"] == 0.5 and m["false_negatives"] == 1 and m["false_positives"] == 1
    assert composite_score({"recall": 1, "f1": 0, "roc_auc": 0}, {"recall": 1, "f1": 0, "roc_auc": 0}) == 1


def test_cv_fitness_reuses_fit_when_only_threshold_changes(split):
    fit = CVFitness("logistic_regression", split.X_train, split.y_train, n_splits=3)
    s1, _ = fit({"C": 1.0, "l1_ratio": 0.0, "class_weight": None, "threshold": 0.5})
    s2, m2 = fit({"C": 1.0, "l1_ratio": 0.0, "class_weight": None, "threshold": 0.3})
    assert fit.model_fits == 1
    assert 0.8 < s1 <= 1 and 0.8 < s2 <= 1
    assert sum(fit.weights.values()) == pytest.approx(1.0)
    assert m2["fitness"] == pytest.approx(s2)
