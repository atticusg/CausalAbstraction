"""Regression tests for ``methods/interchange/single_pair.run_single_pair_trace``.

The single-pair trace patches the counterfactual (source) residual into the
base at the *same* token index. ``get_list_of_each_token`` builds one fixed
index per *base* token, so when the counterfactual tokenizes shorter than the
base those indices run off the end of the source — an out-of-bounds gather that
on GPU is a CUDA scatter/gather assertion poisoning the context (#176). The
trace must drop those positions instead, and the analysis wrapper must label its
heatmap from the *effective* (post-drop) positions so axes match the cells.

These run on CPU against the tiny-random Llama pipeline (``mock_tiny_lm``); the
failing condition (length-mismatched base/counterfactual) reproduces without a
GPU — on CPU the same out-of-bounds index raises rather than asserting.
"""

from __future__ import annotations

import json

import pytest

from causalab.analyses.locate.single_pair_trace import save_single_pair_trace
from causalab.methods.interchange.single_pair import run_single_pair_trace
from causalab.neural.token_positions import get_list_of_each_token

pytestmark = pytest.mark.unit

# A base prompt with many tokens vs. a single-token counterfactual: the fixed
# base positions beyond the counterfactual's length have no source token to
# patch and previously triggered the out-of-bounds gather.
_LONG_BASE = "the quick brown fox jumps over the lazy dog"
_SHORT_CF = "cat"


def _tok_len(pipeline, text: str) -> int:
    return int(pipeline.load([{"raw_input": text}])["attention_mask"][0].sum())


def test_drops_positions_beyond_counterfactual(mock_tiny_lm) -> None:
    """Length-mismatched pair: positions past the counterfactual are dropped,
    the trace completes, and only the surviving positions are traced."""
    base_len = _tok_len(mock_tiny_lm, _LONG_BASE)
    cf_len = _tok_len(mock_tiny_lm, _SHORT_CF)
    assert base_len > cf_len, (
        "precondition: base must be longer than the counterfactual"
    )

    token_positions = get_list_of_each_token(_LONG_BASE, mock_tiny_lm)
    assert len(token_positions) == base_len  # one position per base token

    result = run_single_pair_trace(
        pipeline=mock_tiny_lm,
        prompt=_LONG_BASE,
        counterfactual_prompt=_SHORT_CF,
        token_positions=token_positions,
        layers=[0],
        verbose=False,
    )

    # Only positions the counterfactual can supply survive.
    effective = result["token_positions"]
    assert len(effective) == cf_len

    # Every traced cell corresponds to a surviving position (no orphan/extra
    # cells), and the count is layers × surviving-positions.
    traced_ids = {pos_id for (_layer, pos_id) in result["intervention_results"]}
    assert traced_ids == {tp.id for tp in effective}
    assert len(result["intervention_results"]) == 1 * len(effective)


def test_keeps_all_when_lengths_match(mock_tiny_lm) -> None:
    """Equal-length base/counterfactual: nothing is dropped (the common case,
    e.g. the shipped weekdays golden's equal-length swaps)."""
    prompt = "alpha beta gamma delta"
    token_positions = get_list_of_each_token(prompt, mock_tiny_lm)

    result = run_single_pair_trace(
        pipeline=mock_tiny_lm,
        prompt=prompt,
        counterfactual_prompt=prompt,  # identical ⇒ same length ⇒ no drops
        token_positions=token_positions,
        layers=[0],
        verbose=False,
    )

    assert len(result["token_positions"]) == len(token_positions)
    traced_ids = {pos_id for (_layer, pos_id) in result["intervention_results"]}
    assert traced_ids == {tp.id for tp in token_positions}


def test_save_single_pair_trace_labels_match_cells(mock_tiny_lm, tmp_path) -> None:
    """The analysis wrapper labels its heatmap from the effective positions, so
    every traced cell's position-id has a matching label and id (#176)."""
    token_positions = get_list_of_each_token(_LONG_BASE, mock_tiny_lm)

    trace_data = save_single_pair_trace(
        pipeline=mock_tiny_lm,
        prompt=_LONG_BASE,
        counterfactual_prompt=_SHORT_CF,
        token_positions=token_positions,
        layers=[0],
        out_dir=str(tmp_path),
    )

    # The written artifact is internally consistent: labels/ids line up, and
    # every cell's position-id is one of the (effective) position-ids.
    assert len(trace_data["token_labels"]) == len(trace_data["token_position_ids"])
    cell_pos_ids = {key.split("|", 1)[1] for key in trace_data["cells"]}
    assert cell_pos_ids <= set(trace_data["token_position_ids"])

    # And the JSON on disk matches what was returned (re-plottable artifact).
    with open(tmp_path / "single_pair_trace.json") as f:
        on_disk = json.load(f)
    assert on_disk["token_position_ids"] == trace_data["token_position_ids"]
    assert set(on_disk["cells"]) == set(trace_data["cells"])
