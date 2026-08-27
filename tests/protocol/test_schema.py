"""Strict-parse behavior: sugar, aliases, wrapper shapes, raw loading."""

from __future__ import annotations

import pytest

from causalab.protocol.errors import ParseError, ValidationError
from causalab.protocol.schema import PositionSpec, Sweep, load_raw, parse_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def test_load_raw_rejects_duplicate_keys():
    with pytest.raises(ParseError) as err:
        load_raw('{"version": "1", "version": "1"}')
    assert err.value.code == "P2"


def test_load_raw_rejects_non_finite():
    with pytest.raises(ParseError):
        load_raw('{"version": NaN}')


def test_load_raw_rejects_non_object_top_level():
    with pytest.raises(ParseError) as err:
        load_raw("[1, 2]")
    assert err.value.code == "P1"


def test_int_pos_sugar_expands():
    doc = parse_document(base_doc())
    pos = doc.reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.index == -1


def test_neural_model_alias():
    raw = base_doc()
    raw["neural_model"] = raw.pop("model")
    # dict insertion order puts the alias last; rebuild in legal order
    ordered = {"version": raw.pop("version"), "neural_model": raw.pop("neural_model")}
    ordered.update(raw)
    doc = parse_document(ordered)
    assert doc.model.key == "gpt2"


def test_both_model_spellings_rejected():
    raw = base_doc()
    raw["neural_model"] = dict(raw["model"])
    with pytest.raises(ParseError):
        parse_document(raw)


def test_missing_required_section():
    raw = base_doc()
    del raw["sites"]
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P2"


def test_unsupported_version():
    raw = base_doc()
    raw["version"] = "2"
    with pytest.raises(ParseError):
        parse_document(raw)


def test_position_spec_needs_exactly_one_anchor():
    raw = base_doc()
    raw["positions"] = {"p": {"index": -1, "variable": "x"}}
    raw["reads"]["v_cf"]["pos"] = "p"
    with pytest.raises(ParseError):
        parse_document(in_order(raw))


def test_all_pos_sugar_expands():
    """The bare string ``"all"`` is the all-positions sugar, not a lookup in
    the positions table (§2.3)."""
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = "all"
    pos = parse_document(raw).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.all is True


def test_all_anchor_parses_inline():
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"all": True}
    pos = parse_document(raw).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.all is True and pos.index is None


def test_all_is_exclusive_with_the_other_anchors():
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"all": True, "index": -1}
    with pytest.raises(ParseError):
        parse_document(raw)


@pytest.mark.parametrize("modifier", ["scope", "relative_to"])
def test_all_takes_no_modifier(modifier):
    """``scope``/``relative_to`` narrow an index or span; there is nothing
    left to narrow inside "every token"."""
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"all": True, modifier: {"variable": "subject"}}
    with pytest.raises(ParseError):
        parse_document(raw)


def test_all_is_a_flag_not_a_selection():
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"all": [0, 1]}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "all" in str(err.value)


def test_layerless_component_rejects_layer():
    raw = base_doc()
    raw["sites"]["lm_head"]["layer"] = 0
    with pytest.raises(ParseError):
        parse_document(raw)


def test_do_has_exactly_one_mechanism():
    raw = base_doc()
    raw["writes"]["patch"]["do"] = {"swap": "v_cf", "renormalize": True}
    with pytest.raises(ParseError):
        parse_document(raw)


def test_sweep_wrapper_parses_to_axis_values():
    raw = base_doc()
    raw["sites"]["tgt"]["layer"] = {"sweep": [1, 2, 3]}
    doc = parse_document(raw)
    layer = doc.sites["tgt"].layer
    assert isinstance(layer, Sweep) and layer.values == (1, 2, 3)


def test_entry_level_sweep_on_positions():
    raw = base_doc()
    raw["positions"] = {"tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}}
    raw["reads"]["v_cf"]["pos"] = "tap"
    raw["writes"]["patch"]["pos"] = "tap"
    doc = parse_document(in_order(raw))
    tap = doc.positions["tap"]
    assert isinstance(tap, Sweep) and len(tap.values) == 2


# --------------------------------------------------------------------------- #
# §2.10 token_form — the metric-level answer-tokenization knob
# --------------------------------------------------------------------------- #


def test_metric_token_form_defaults_to_auto():
    """Every pre-``token_form`` document keeps the historical resolver."""
    doc = parse_document(base_doc())
    assert doc.metrics["ld"].token_form == "auto"


@pytest.mark.parametrize("form", ["auto", "bare", "space_prefixed"])
def test_metric_token_form_parses_each_form(form: str):
    raw = base_doc()
    raw["metrics"]["ld"]["token_form"] = form
    assert parse_document(in_order(raw)).metrics["ld"].token_form == form


def test_metric_token_form_rejects_an_unknown_form():
    raw = base_doc()
    raw["metrics"]["ld"]["token_form"] = "spaced"
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P4"


def test_metric_token_form_is_refused_on_kinds_that_resolve_no_string():
    """``kl`` compares two reads and ``top_k`` decodes ids it found — neither
    turns an authored string into a token id, so the knob is meaningless."""
    raw = base_doc()
    raw["metrics"]["ld"] = {
        "kind": "top_k",
        "of": "logits",
        "k": 3,
        "token_form": "bare",
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"


def test_metric_token_form_is_not_sweepable():
    """A sweep over token_form would fork a campaign on a tokenizer detail
    rather than a research variable; §3 wrappers stay off this field."""
    raw = base_doc()
    raw["metrics"]["ld"]["token_form"] = {"sweep": ["bare", "space_prefixed"]}
    with pytest.raises(ValidationError) as err:
        parse_document(in_order(raw))
    assert err.value.rule == 14


# the continuation frame (§2.3) ---------------------------------------------- #


def _generated(anchor: dict, budget: int = 8) -> dict:
    return {"generated": {"max_new_tokens": budget}, **anchor}


@pytest.mark.parametrize(
    "anchor,attr,expected",
    [
        ({"all": True}, "all", True),
        ({"index": -1}, "index", -1),
        ({"span": [0, 3]}, "span", (0, 3)),
        ({"variable": "expected"}, "variable", "expected"),
    ],
)
def test_generated_frame_takes_every_anchor(anchor, attr, expected):
    """``generated`` is a frame selector, not an anchor: the anchor
    vocabulary is unchanged inside the continuation."""
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = _generated(anchor)
    pos = parse_document(raw).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec)
    assert getattr(pos, attr) == expected
    assert pos.generated == {"max_new_tokens": 8}


def test_generated_needs_an_anchor():
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"generated": {"max_new_tokens": 8}}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "exactly one" in str(err.value)


@pytest.mark.parametrize(
    "extra",
    [
        {"column": "entity"},
        {"index": 0, "scope": {"variable": "subject"}},
        {"index": 1, "relative_to": {"variable": "subject"}},
    ],
)
def test_generated_refuses_prompt_frame_notions(extra):
    """A ``column`` holds a substring of the *input* text and
    ``scope``/``relative_to`` anchor on a prompt variable's token run —
    neither exists in a frame the prompt does not contain."""
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"generated": {"max_new_tokens": 8}, **extra}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert "generated" in str(err.value)


@pytest.mark.parametrize(
    "budget,fragment",
    [
        ({"max_new_tokens": 0}, "at least"),
        ({"max_new_tokens": -3}, "at least"),
        ({}, "max_new_tokens"),
        ({"max_new_tokens": 4, "temperature": 0.7}, "temperature"),
        (8, "expected an object"),
    ],
)
def test_generated_budget_shapes(budget, fragment):
    """The budget is a mapping so stopping conditions can join it later —
    a bare int, a missing budget, and sampling knobs all refuse."""
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = {"generated": budget, "index": -1}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert fragment in str(err.value)


def test_generated_budget_is_sweepable():
    raw = base_doc()
    raw["positions"] = {
        "tail": {"generated": {"max_new_tokens": {"sweep": [4, 16]}}, "index": -1}
    }
    raw["reads"]["v_cf"]["pos"] = "tail"
    pos = parse_document(in_order(raw)).positions["tail"]
    assert isinstance(pos, PositionSpec)
    assert isinstance(pos.generated["max_new_tokens"], Sweep)


def test_prompt_frame_positions_carry_no_generated():
    """Every pre-existing position stays prompt-frame: the field is absent,
    not defaulted, so existing canonical forms are untouched."""
    pos = parse_document(base_doc()).reads["v_cf"].pos
    assert isinstance(pos, PositionSpec) and pos.generated is None


def test_decode_metric_takes_no_value_fields():
    raw = base_doc()
    raw["metrics"]["ld"] = {"kind": "decode", "of": "logits"}
    metric = parse_document(in_order(raw)).metrics["ld"]
    assert str(metric.kind) == "decode"
    assert dict(metric.fields) == {}


def test_decode_metric_rejects_a_stray_field():
    """The kind reduces the tokens a decode produced; there is nothing to
    parametrize, so an extra key is a typo, not an option."""
    raw = base_doc()
    raw["metrics"]["ld"] = {"kind": "decode", "of": "logits", "k": 1}
    with pytest.raises(ParseError):
        parse_document(in_order(raw))
