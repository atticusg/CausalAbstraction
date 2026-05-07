"""Tests for the natural_domains_arithmetic factory task."""

import pytest

from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.natural_domains_arithmetic.causal_models import (
    create_causal_model,
    create_random_causal_model,
    GET_VARIABLE_VALUES,
    GET_CYCLIC_VARIABLES,
    GET_EMBEDDINGS,
    GET_PERIODIC_INFO,
    GET_RESULT_TOKEN_PATTERN,
)
from causalab.tasks.loader import load_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOMAIN_TYPES = ["weekdays", "months", "hours"]


@pytest.fixture(params=DOMAIN_TYPES)
def config(request):
    return NaturalDomainConfig(domain_type=request.param)


@pytest.fixture(params=DOMAIN_TYPES)
def model_and_config(request):
    cfg = NaturalDomainConfig(domain_type=request.param)
    model = create_causal_model(cfg)
    return model, cfg


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_presets_fill():
    """Preset data auto-fills when entities list is empty."""
    for dt in DOMAIN_TYPES:
        cfg = NaturalDomainConfig(domain_type=dt)
        assert len(cfg.entities) > 0
        assert len(cfg.numbers) > 0
        assert len(cfg.number_to_int) > 0
        assert cfg.template != ""


def test_config_invalid_domain():
    with pytest.raises(ValueError, match="domain_type must be"):
        NaturalDomainConfig(domain_type="invalid_domain")


# ---------------------------------------------------------------------------
# Model creation tests
# ---------------------------------------------------------------------------


def test_model_creation(model_and_config):
    model, cfg = model_and_config
    assert model.id == f"natural_domains_arithmetic_{cfg.domain_type}"
    # All models have the same variable names
    assert "entity" in model.values
    assert "number" in model.values
    assert "result" in model.values
    assert "raw_input" in model.values
    assert "raw_output" in model.values


def test_model_sample_input(model_and_config):
    model, cfg = model_and_config
    trace = model.sample_input()
    assert trace["entity"] in cfg.entities
    assert trace["number"] in cfg.numbers


# ---------------------------------------------------------------------------
# Result computation tests
# ---------------------------------------------------------------------------


def test_weekdays_result():
    cfg = NaturalDomainConfig(domain_type="weekdays")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "Thursday", "number": "three"})
    assert trace["result"] == "Sunday"


def test_months_result():
    cfg = NaturalDomainConfig(domain_type="months")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "October", "number": "three"})
    assert trace["result"] == "January"


def test_hours_result():
    cfg = NaturalDomainConfig(domain_type="hours")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "23", "number": "four"})
    assert trace["result"] == "3"


# ---------------------------------------------------------------------------
# raw_input / raw_output tests
# ---------------------------------------------------------------------------


def test_weekdays_raw_input():
    cfg = NaturalDomainConfig(domain_type="weekdays")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "Thursday", "number": "three"})
    assert trace["raw_input"] == "Q: What day is three days after Thursday?\nA:"
    assert trace["raw_output"] == " Sunday"


def test_months_raw_input():
    cfg = NaturalDomainConfig(domain_type="months")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "October", "number": "three"})
    assert trace["raw_input"] == "Q: What month is three months after October?\nA:"
    assert trace["raw_output"] == " January"


def test_hours_raw_input():
    cfg = NaturalDomainConfig(domain_type="hours")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "23", "number": "four"})
    assert (
        trace["raw_input"] == "Q: What hour comes four hours after 23 on a clock?\nA: "
    )
    assert trace["raw_output"] == "3"


# ---------------------------------------------------------------------------
# Random baseline tests
# ---------------------------------------------------------------------------


def test_random_baseline(config):
    model = create_random_causal_model(config)
    assert model.id.endswith("_random")
    trace = model.sample_input()
    # Random entities are words, not in the original entity list
    assert trace["entity"] not in config.entities
    # Result should also be a random word
    assert trace["result"] in model.values["result"]


# ---------------------------------------------------------------------------
# Dynamic getter tests
# ---------------------------------------------------------------------------


def test_get_variable_values(model_and_config):
    model, cfg = model_and_config
    vv = GET_VARIABLE_VALUES(model)
    assert set(vv.keys()) == {"entity", "number", "result"}
    assert vv["entity"] == cfg.entities
    assert vv["number"] == cfg.numbers


def test_get_cyclic_variables_weekdays():
    cfg = NaturalDomainConfig(domain_type="weekdays")
    model = create_causal_model(cfg)
    cv = GET_CYCLIC_VARIABLES(model)
    assert cv == {"entity", "number", "result"}


def test_get_cyclic_variables_months():
    cfg = NaturalDomainConfig(domain_type="months")
    model = create_causal_model(cfg)
    cv = GET_CYCLIC_VARIABLES(model)
    assert cv == {"entity", "result"}


def test_get_embeddings(model_and_config):
    model, cfg = model_and_config
    emb = GET_EMBEDDINGS(model)
    assert "entity" in emb
    assert "result" in emb
    # Check embedding returns a list of floats
    val = emb["entity"](cfg.entities[0])
    assert isinstance(val, list)
    assert isinstance(val[0], float)


def test_get_periodic_info_weekdays():
    cfg = NaturalDomainConfig(domain_type="weekdays")
    model = create_causal_model(cfg)
    pi = GET_PERIODIC_INFO(model)
    assert pi == {"entity": 7, "number": 7, "result": 7}


def test_get_periodic_info_months():
    cfg = NaturalDomainConfig(domain_type="months")
    model = create_causal_model(cfg)
    pi = GET_PERIODIC_INFO(model)
    assert pi == {"entity": 12, "result": 12}


def test_get_result_token_pattern(model_and_config):
    model, cfg = model_and_config
    pattern = GET_RESULT_TOKEN_PATTERN(model)
    if cfg.output_prefix:
        assert any([p.startswith(cfg.output_prefix) for p in pattern("Monday")])
    else:
        assert "3" in pattern("3")


# ---------------------------------------------------------------------------
# Loader integration test
# ---------------------------------------------------------------------------


def test_load_task_integration():
    """load_task with factory config works end-to-end."""
    for dt in DOMAIN_TYPES:
        cfg = NaturalDomainConfig(domain_type=dt)
        task = load_task("natural_domains_arithmetic", task_cfg=cfg)
        assert task.name == "natural_domains_arithmetic"
        assert "entity" in task.causal_model.values
        assert "number" in task.causal_model.values
        assert "result" in task.causal_model.values
        # Verify model produces traces
        trace = task.causal_model.sample_input()
        assert "raw_input" in trace
        assert "raw_output" in trace


def test_load_task_random():
    """load_task with random=True works."""
    for dt in DOMAIN_TYPES:
        cfg = NaturalDomainConfig(domain_type=dt)
        task = load_task("natural_domains_arithmetic", task_cfg=cfg, random=True)
        assert task.name == "natural_domains_arithmetic"
        trace = task.causal_model.sample_input()
        assert "raw_input" in trace


# ---------------------------------------------------------------------------
# Integer domain tests
# ---------------------------------------------------------------------------


def test_integer_result():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "five", "number": "three"})
    assert trace["result"] == "8"


def test_integer_addend_identity():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "one", "number": "seven"})
    assert trace["result"] == "8"


def test_integer_max_result():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "fifteen", "number": "nine"})
    assert trace["result"] == "24"


def test_integer_raw_input_output():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "five", "number": "three"})
    assert trace["raw_input"] == "Q: What is three added to five?\nA:"
    assert trace["raw_output"] == " 8"


def test_integer_result_range_exhaustive():
    """All entity+number combinations produce digit-form results in [0, 100].

    This verifies the single-token constraint: Llama-class tokenizers encode
    integers 0–999 as single tokens, so all results here are single-token.
    """
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    valid_results = {str(i) for i in range(2, 26)}
    for entity in cfg.entities:
        for number in cfg.numbers:
            trace = model.new_trace({"entity": entity, "number": number})
            assert trace["result"] in valid_results, (
                f"result '{trace['result']}' out of range for "
                f"entity='{entity}', number='{number}'"
            )


def test_integer_non_cyclic():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    cv = GET_CYCLIC_VARIABLES(model)
    assert cv == set()


def test_integer_no_periodic_info():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    pi = GET_PERIODIC_INFO(model)
    assert pi is None


def test_integer_embeddings():
    cfg = NaturalDomainConfig(domain_type="integer")
    model = create_causal_model(cfg)
    emb = GET_EMBEDDINGS(model)
    assert emb["entity"]("one") == [1.0]
    assert emb["entity"]("ten") == [10.0]
    assert emb["result"]("15") == [15.0]
    assert emb["result"]("25") == [25.0]


def test_load_task_integer_integration():
    cfg = NaturalDomainConfig(domain_type="integer")
    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    assert task.name == "natural_domains_arithmetic"
    trace = task.causal_model.sample_input()
    assert "raw_input" in trace
    assert "raw_output" in trace
    result_int = int(trace["result"])
    assert 2 <= result_int <= 25


# ---------------------------------------------------------------------------
# Age domain tests
# ---------------------------------------------------------------------------


def test_age_result():
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "30", "number": "5"})
    assert trace["result"] == "35"


def test_age_max_result():
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "90", "number": "10"})
    assert trace["result"] == "100"


def test_age_raw_input_output():
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    trace = model.new_trace({"entity": "30", "number": "5"})
    assert trace["raw_input"] == (
        "Alice is 30 years old. Bob is 5 years older than Alice. "
        "Q: How old is Bob?\nA: Bob is "
    )
    assert trace["raw_output"] == "35"


def test_age_result_range_exhaustive():
    """After input_filter, every enumerated (entity, number) pair lands in {"2", …, "100"}.

    Entities 1–99 plus numbers 1–10 produce some out-of-range sums (e.g. 95+10=105);
    those pairs are dropped by the model's input_filter so enumerate_inputs only
    yields valid in-range results.
    """
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    valid_results = {str(i) for i in range(2, 101)}
    for trace in model.enumerate_inputs():
        assert trace["result"] in valid_results, (
            f"result '{trace['result']}' out of range for "
            f"entity='{trace['entity']}', number='{trace['number']}'"
        )


def test_age_non_cyclic():
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    cv = GET_CYCLIC_VARIABLES(model)
    assert cv == set()


def test_age_no_periodic_info():
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    pi = GET_PERIODIC_INFO(model)
    assert pi is None


def test_age_embeddings():
    cfg = NaturalDomainConfig(domain_type="age")
    model = create_causal_model(cfg)
    emb = GET_EMBEDDINGS(model)
    assert emb["entity"]("90") == [90.0]
    assert emb["entity"]("1") == [1.0]
    assert emb["result"]("100") == [100.0]
    assert emb["number"]("7") == [7.0]


def test_load_task_age_integration():
    cfg = NaturalDomainConfig(domain_type="age")
    task = load_task("natural_domains_arithmetic", task_cfg=cfg)
    assert task.name == "natural_domains_arithmetic"
    trace = task.causal_model.sample_input()
    assert "raw_input" in trace
    assert "raw_output" in trace
    result_int = int(trace["result"])
    assert 2 <= result_int <= 100
