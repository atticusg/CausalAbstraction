"""Shared document builders for the protocol tests."""

from __future__ import annotations

from typing import Any

from causalab.protocol.schema import SECTION_ORDER


def base_doc() -> dict[str, Any]:
    """A minimal valid interchange document on gpt2 (layer 3 < 12)."""
    return {
        "version": "1",
        "model": {"key": "gpt2", "revision": "main"},
        "data": {
            "base": {"dataset": "weekdays/train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "sites": {
            "tgt": {"component": "block_output", "layer": 3},
            "lm_head": {"component": "lm_head"},
        },
        "reads": {
            "v_cf": {
                "site": "tgt",
                "pos": -1,
                "model": "original",
                "input": "counterfactual",
            },
            "logits": {
                "site": "lm_head",
                "pos": -1,
                "model": "patched",
                "input": "base",
            },
        },
        "writes": {"patch": {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}},
        "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
        "metrics": {
            "ld": {
                "kind": "logit_diff",
                "of": "logits",
                "a": "cf_answer",
                "b": "base_answer",
                "token_form": "space_prefixed",
            }
        },
        "save": [
            {
                "value": "ld",
                "model": "patched",
                "input": "base",
                "file_path": "ld.json",
            }
        ],
    }


def in_order(raw: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a mutated document with its top-level sections in the §1
    order — test mutations append sections at the dict end, which would
    (correctly) trip the section-order rule they aren't testing."""
    return {key: raw[key] for key in SECTION_ORDER if key in raw}
