"""Sweep discovery and deterministic expansion (spec §3)."""

from __future__ import annotations

import pytest

from causalab.protocol.errors import ValidationError
from causalab.protocol.sweep import coordinate_label, expand, find_axes

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def swept_doc():
    raw = base_doc()
    raw["positions"] = {"tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}}
    raw["sites"]["tgt"]["layer"] = {"sweep": {"range": [0, 4]}}
    raw["reads"]["v_cf"]["pos"] = "tap"
    raw["writes"]["patch"]["pos"] = "tap"
    return in_order(raw)


def test_axes_in_document_order():
    axes = find_axes(swept_doc())
    assert [a.id for a in axes] == ["positions.tap", "sites.tgt.layer"]
    assert axes[1].values == (0, 1, 2, 3)


def test_cross_product_last_axis_fastest():
    expansion = expand(swept_doc())
    assert len(expansion.points) == 8
    coords = [p.coords for p in expansion.points]
    assert coords[0] == {"positions.tap": {"index": -1}, "sites.tgt.layer": 0}
    assert coords[1] == {"positions.tap": {"index": -1}, "sites.tgt.layer": 1}
    assert coords[4] == {"positions.tap": {"variable": "subject"}, "sites.tgt.layer": 0}


def test_substitution_produces_concrete_points():
    expansion = expand(swept_doc())
    point = expansion.points[5]
    assert point.raw["sites"]["tgt"]["layer"] == 1
    assert point.raw["positions"]["tap"] == {"variable": "subject"}
    # entities off the axes are untouched
    assert point.raw["reads"]["logits"] == swept_doc()["reads"]["logits"]


def test_unswept_document_expands_to_itself():
    raw = base_doc()
    expansion = expand(raw)
    assert not expansion.is_swept
    assert len(expansion.points) == 1
    assert expansion.points[0].raw == raw


def test_range_step():
    raw = base_doc()
    raw["sites"]["tgt"]["layer"] = {"sweep": {"range": [0, 10, 3]}}
    axes = find_axes(raw)
    assert axes[0].values == (0, 3, 6, 9)


def test_wrapper_inside_list_rejected():
    raw = base_doc()
    raw["reads"]["v_cf"]["dims"] = [0, {"sweep": [1, 2]}]
    with pytest.raises(ValidationError) as err:
        find_axes(raw)
    assert err.value.rule == 14


def test_nested_wrapper_rejected():
    raw = base_doc()
    raw["sites"]["tgt"]["layer"] = {"sweep": [1, {"sweep": [2, 3]}]}
    with pytest.raises(ValidationError) as err:
        find_axes(raw)
    assert err.value.rule == 14


def test_empty_axis_rejected():
    raw = base_doc()
    raw["sites"]["tgt"]["layer"] = {"sweep": []}
    with pytest.raises(ValidationError) as err:
        find_axes(raw)
    assert err.value.rule == 14


def test_coordinate_labels():
    assert coordinate_label({"featurizers.rot.k": 8}, entry="rot") == "[k=8]"
    assert coordinate_label({"sites.target.layer": 5}) == "[target.layer=5]"
    assert coordinate_label({"train.seed": 0}) == "[seed=0]"
