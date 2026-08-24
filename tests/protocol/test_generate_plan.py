"""The decode's derived shape: depth, materialization, capability (§4, §6, §8).

The point of these tests is that the *cost* of a generate document is a
value the planner emits, not a backend heuristic: a document that wants one
token's distribution must not oblige the same work as one that wants every
token's. So they assert the plan, never memory.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.backend import requires
from causalab.protocol.plan import plan_point
from causalab.protocol.schema import parse_document

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def probe_doc(anchor: dict[str, Any], budget: int = 8) -> dict[str, Any]:
    """A minimal document whose saved metric reduces a continuation read."""
    raw = base_doc()
    raw["positions"] = {"cont": {"generated": {"max_new_tokens": budget}, **anchor}}
    raw["reads"]["logits"]["pos"] = "cont"
    return in_order(raw)


def group_for(raw: dict[str, Any], model: str = "patched"):
    plan = plan_point(parse_document(raw))
    return next(g for g in plan.groups if g.model == model)


def test_decode_depth_is_the_budget():
    group = group_for(probe_doc({"index": -1}, budget=12))
    assert group.decode_depth == 12


def test_prompt_frame_documents_do_not_decode():
    for group in plan_point(parse_document(base_doc())).groups:
        assert group.decode_depth == 0
        assert group.materialize == ()


def test_depth_is_the_max_over_the_groups_reads():
    """Two reads of one model at different budgets share one decode: the run
    goes as deep as the deepest, each read windows its own."""
    raw = probe_doc({"index": -1}, budget=4)
    raw["positions"]["long"] = {"generated": {"max_new_tokens": 16}, "all": True}
    raw["reads"]["tail"] = {
        "site": "lm_head",
        "pos": "long",
        "model": "patched",
        "input": "base",
    }
    raw["save"].append(
        {
            "value": "tail",
            "model": "patched",
            "input": "base",
            "file_path": "tail.safetensors",
        }
    )
    group = group_for(in_order(raw))
    assert group.decode_depth == 16
    assert {m.read for m in group.materialize} == {"logits", "tail"}


def test_a_metric_input_needs_a_distribution():
    """`logits` feeds a logit_diff, so its addressed position has to exist as
    a full vocabulary vector somewhere."""
    group = group_for(probe_doc({"index": -1}))
    (item,) = group.materialize
    assert (item.read, item.site) == ("logits", "lm_head")
    assert item.needs_distribution is True


def test_saving_a_read_obliges_building_it():
    """A continuation harvest at a non-head site: saved, so the backend owes
    the tensor whatever the site is. The *false* branch of
    ``needs_distribution`` is unreachable in v1 — every metric kind reduces
    logits — and becomes reachable when metric kinds declare a domain and an
    ids-only kind (a text probe) stops counting as a consumer."""
    raw = base_doc()
    raw["positions"] = {"cont": {"generated": {"max_new_tokens": 8}, "all": True}}
    raw["reads"] = {
        "v_cf": raw["reads"]["v_cf"],
        "acts": {
            "site": "tgt",
            "pos": "cont",
            "model": "patched",
            "input": "base",
        },
    }
    raw["metrics"] = {}
    raw["save"] = [
        {
            "value": "acts",
            "model": "patched",
            "input": "base",
            "file_path": "acts.safetensors",
        }
    ]
    group = group_for(in_order(raw))
    (item,) = group.materialize
    assert (item.read, item.site) == ("acts", "tgt")
    assert item.needs_distribution is True


def test_a_generating_group_does_not_elide():
    """Elision ends the forward at the deepest tap; a decode needs the head
    on every step, so there is nothing left to skip."""
    group = group_for(probe_doc({"index": -1}))
    assert group.stop_after is None


def test_generate_is_a_required_capability():
    doc = parse_document(probe_doc({"index": -1}))
    assert "generate" in requires(doc)
    assert "generate" not in requires(parse_document(base_doc()))


def test_decode_depth_is_not_in_the_group_digest():
    """A decode changes what a group *produces*, not what its prefill
    computes — so two points differing only in depth still share a prefill,
    the same reason taps are not in the digest."""
    short = group_for(probe_doc({"index": -1}, budget=4))
    long = group_for(probe_doc({"index": -1}, budget=32))
    assert short.decode_depth != long.decode_depth
    assert short.digest == long.digest
