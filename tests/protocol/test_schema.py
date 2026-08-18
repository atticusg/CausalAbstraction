"""Strict-parse behavior: sugar, aliases, wrapper shapes, raw loading."""

from __future__ import annotations

import pytest

from causalab.protocol.errors import ParseError
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
    pos = doc.reads["v_src"].pos
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
    raw["reads"]["v_src"]["pos"] = "p"
    with pytest.raises(ParseError):
        parse_document(in_order(raw))


def test_layerless_component_rejects_layer():
    raw = base_doc()
    raw["sites"]["lm_head"]["layer"] = 0
    with pytest.raises(ParseError):
        parse_document(raw)


def test_do_has_exactly_one_mechanism():
    raw = base_doc()
    raw["edits"]["patch"]["do"] = {"swap": "v_src", "renormalize": True}
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
    raw["reads"]["v_src"]["pos"] = "tap"
    raw["edits"]["patch"]["pos"] = "tap"
    doc = parse_document(in_order(raw))
    tap = doc.positions["tap"]
    assert isinstance(tap, Sweep) and len(tap.values) == 2
