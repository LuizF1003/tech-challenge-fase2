import json

import pytest

from src.data import load_split
from src.explain import build_case_context, feature_label
from src.llm import DiagnosisInterpreter, get_provider
from src.llm.evaluation import coherence, evaluate_interpretation, factor_coverage, numeric_fidelity, safety
from src.llm.interpreter import extract_json
from src.llm.prompts import CASE_FIELDS, case_prompt
from src.llm.providers import LLMError, LLMResponse
from src.models import BASELINE_PARAMS, build_model


@pytest.fixture(scope="module")
def setup():
    data = load_split()
    model = build_model("logistic_regression", BASELINE_PARAMS["logistic_regression"]).fit(data.X_train, data.y_train)
    card = {
        "model": "logistic_regression", "display_name": "Regressão Logística", "source": "baseline",
        "features": data.feature_names, "test_size": 114,
        "train_medians": data.X_train.median().to_dict(),
        "class_means": {"benigno": data.X_train[data.y_train == 0].mean().to_dict(),
                        "maligno": data.X_train[data.y_train == 1].mean().to_dict()},
        "test_metrics": {"recall": 0.95, "precision": 0.97, "roc_auc": 0.99, "false_negatives": 2,
                         "false_positives": 1},
    }
    return data, model, card


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, text=None, fail=False):
        self.text, self.fail = text, fail

    def generate(self, system, user, **kw):
        if self.fail:
            raise LLMError("indisponível")
        return LLMResponse(self.text, "fake", "fake-1", 0.01)


def test_feature_label_translation():
    assert feature_label("concave points_worst") == "pontos côncavos (pior valor)"
    assert feature_label("radius_mean") == "raio (média)"


def test_case_context_contains_grounding(setup):
    data, model, card = setup
    ctx = build_case_context(model, card, data.X_test.iloc[[0]])
    assert 0 <= ctx["probabilidade_malignidade"] <= 1
    assert len(ctx["fatores_principais"]) == 5
    assert {"media_benignos", "media_malignos", "impacto"} <= set(ctx["fatores_principais"][0])


@pytest.mark.parametrize("strategy", ["basico", "estruturado", "few_shot"])
def test_prompts_include_context(setup, strategy):
    data, model, card = setup
    ctx = build_case_context(model, card, data.X_test.iloc[[1]])
    system, user, json_mode = case_prompt(ctx, strategy)
    assert str(ctx["probabilidade_percentual"]) in user
    assert json_mode == (strategy != "basico")
    if strategy != "basico":
        assert "Nunca afirme diagnóstico definitivo" in system


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Claro! {"a": 2} Espero ter ajudado.') == {"a": 2}
    assert extract_json("sem json") is None


def test_offline_interpretation_is_valid_and_scores_high(setup):
    data, model, card = setup
    out = DiagnosisInterpreter(model, card, get_provider("offline")).explain_case(data.X_test.iloc[[3]])
    assert out.valid_json and set(CASE_FIELDS) <= set(out.structured)
    scores = evaluate_interpretation(out)
    assert scores["formato"] == 1 and scores["fidelidade"] == 1 and scores["seguranca"] == 1


def test_fallback_when_provider_fails(setup):
    data, model, card = setup
    out = DiagnosisInterpreter(model, card, FakeProvider(fail=True)).explain_case(data.X_test.iloc[[0]])
    assert out.used_fallback and out.provider == "offline" and out.valid_json


def test_invalid_json_triggers_fallback(setup):
    data, model, card = setup
    out = DiagnosisInterpreter(model, card, FakeProvider(text="texto livre")).explain_case(data.X_test.iloc[[0]])
    assert out.used_fallback


def test_real_llm_like_answer_is_parsed(setup):
    data, model, card = setup
    answer = {f: "x" for f in CASE_FIELDS}
    answer["fatores_relevantes"] = ["a", "b"]
    out = DiagnosisInterpreter(model, card, FakeProvider("```json\n" + json.dumps(answer) + "\n```")).explain_case(
        data.X_test.iloc[[0]])
    assert out.valid_json and not out.used_fallback and out.provider == "fake"


def test_evaluation_detects_problems():
    ctx = {"probabilidade_percentual": 82.0, "limiar_decisao": 0.4, "classificacao_modelo": "suspeito de malignidade",
           "fatores_principais": [{"nome": "área (pior valor)", "atributo": "area_worst"},
                                  {"nome": "textura (média)", "atributo": "texture_mean"},
                                  {"nome": "simetria (média)", "atributo": "symmetry_mean"}]}
    fid, invented = numeric_fidelity("Probabilidade de 82% e sensibilidade de 99.9%.", ctx)
    assert invented == [99.9] and fid == 0.5
    assert factor_coverage("A área e a textura pesaram.", ctx) == pytest.approx(2 / 3)
    assert coherence("Risco alto, acima do limiar.", ctx) == 1.0
    assert coherence("Risco baixo.", ctx) == 0.0
    assert safety("O exame confirma câncer.")[0] == 0.0
    assert safety("Não confirma nem descarta câncer; não substitui a avaliação médica.")[0] == 1.0


def test_get_provider_unknown():
    with pytest.raises(ValueError):
        get_provider("xyz")
