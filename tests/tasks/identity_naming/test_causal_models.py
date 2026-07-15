"""Tests for the identity_naming task."""

import pytest

from causalab.tasks.identity_naming.config import (
    IdentityNamingConfig,
    DOMAIN_PRESETS,
    _note_to_midi,  # pyright: ignore[reportPrivateUsage]
)
from causalab.tasks.identity_naming.causal_models import (
    create_causal_model,
    GET_VARIABLE_VALUES,
    GET_CYCLIC_VARIABLES,
    GET_EMBEDDINGS,
    GET_PERIODIC_INFO,
    GET_TEMPLATE,
)


pytestmark = pytest.mark.unit


DOMAIN_TYPES = list(DOMAIN_PRESETS.keys())


@pytest.fixture(params=DOMAIN_TYPES)
def config(request):
    return IdentityNamingConfig(domain_type=request.param)


@pytest.fixture
def model_and_config(config):
    return create_causal_model(config), config


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_invalid_domain():
    with pytest.raises(ValueError, match="domain_type must be one of"):
        IdentityNamingConfig(domain_type="nonexistent")


def test_preset_fills_entities(config):
    assert len(config.entities) > 0
    assert len(config.entity_to_result) == len(config.entities)
    assert len(config.templates) >= 1


# ---------------------------------------------------------------------------
# MIDI conversion tests
# ---------------------------------------------------------------------------


def test_midi_c4():
    assert _note_to_midi("C4") == 60


def test_midi_a0():
    assert _note_to_midi("A0") == 21


def test_midi_csharp4():
    assert _note_to_midi("C#4") == 61


def test_midi_boundary():
    """B3 = 59, C4 = 60 — octave boundary."""
    assert _note_to_midi("B3") == 59
    assert _note_to_midi("C4") == 60


# ---------------------------------------------------------------------------
# Causal model tests
# ---------------------------------------------------------------------------


def test_pitch_midi_result():
    cfg = IdentityNamingConfig(domain_type="pitch_midi")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "C4"})
    assert trace["result"] == "60"


def test_pitch_midi_result_sharp():
    cfg = IdentityNamingConfig(domain_type="pitch_midi")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "C#4"})
    assert trace["result"] == "61"


def test_pitch_midi_raw_input():
    cfg = IdentityNamingConfig(domain_type="pitch_midi")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "C4"})
    assert trace["raw_input"] == "The MIDI number for C4 is "
    assert trace["raw_output"] == "60"


def test_all_entities_produce_valid_results(model_and_config):
    model, config = model_and_config
    result_set = set(model.values["result"])
    for entity in config.entities:
        trace = model.new_trace({"entity": entity})
        assert trace["result"] in result_set


# ---------------------------------------------------------------------------
# Dynamic getter tests
# ---------------------------------------------------------------------------


def test_get_variable_values(model_and_config):
    model, cfg = model_and_config
    vv = GET_VARIABLE_VALUES(model)
    assert set(vv.keys()) == {"entity", "result"}
    assert vv["entity"] == cfg.entities


def test_get_cyclic_variables(model_and_config):
    model, _ = model_and_config
    assert GET_CYCLIC_VARIABLES(model) == set()


def test_get_embeddings(model_and_config):
    model, cfg = model_and_config
    emb = GET_EMBEDDINGS(model)
    assert "entity" in emb
    assert "result" in emb
    # Entity embedding should return a list of floats
    vec = emb["entity"](cfg.entities[0])
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)


def test_get_periodic_info(model_and_config):
    model, _ = model_and_config
    assert GET_PERIODIC_INFO(model) is None


def test_get_template(model_and_config):
    model, cfg = model_and_config
    template = GET_TEMPLATE(model)
    assert "{entity}" in template
    assert template == cfg.templates[0]


def test_output_tokens_declare_prefixed_result_forms(model_and_config):
    """``output_tokens["result"]`` declares ``output_prefix + name`` per value (#296)."""
    model, cfg = model_and_config
    ot = model.output_tokens["result"]
    assert set(ot) == set(model.values["result"])
    for v in model.values["result"]:
        assert ot[v] == [cfg.output_prefix + str(v)]


# ---------------------------------------------------------------------------
# Counterfactual tests
# ---------------------------------------------------------------------------


def test_generate_dataset(model_and_config):
    model, config = model_and_config
    from causalab.tasks.identity_naming.counterfactuals import generate_dataset

    examples = generate_dataset(model, n=10, seed=42)
    assert len(examples) == 10

    # Check that templates cycle
    templates = config.templates
    for i, ex in enumerate(examples):
        expected_template = templates[i % len(templates)]
        raw_input = ex["input"]["raw_input"]
        entity = ex["input"]["entity"]
        assert raw_input == expected_template.format(entity=entity)


def test_generate_dataset_deterministic(model_and_config):
    model, _ = model_and_config
    from causalab.tasks.identity_naming.counterfactuals import generate_dataset

    ex1 = generate_dataset(model, n=5, seed=123)
    ex2 = generate_dataset(model, n=5, seed=123)
    for a, b in zip(ex1, ex2):
        assert a["input"]["entity"] == b["input"]["entity"]
        assert a["input"]["raw_input"] == b["input"]["raw_input"]
