"""Tests for ``methods/interchange/attribution`` — the attribution-patching
pre-scan (CAP3, #456).

* ``numerical_unit`` — the gradient × Δactivation math on the tiny-random
  Llama pipeline (CPU, frozen params — the leaf-trick branch), pinned against
  an explicit autograd reference: raw HF hooks make each layer's block output
  a grad leaf, backward the same logit difference, and compute
  ``grad · (a_source − a_base)`` per cell by hand, one pair at a time.
* ``unit`` — ranking/agreement helpers and the method's refusals, no forward
  pass beyond the tokenizer.

The end-to-end gate (locate / DAS grids pruned to the pre-scan's top-k) is
covered by the smoke configs ``weekdays_locate_prescan`` /
``weekdays_subspace_das_prescan`` and the golden ``weekdays_locate_prescan``.
"""

from __future__ import annotations

import math

import pytest
import torch

from causalab.methods.interchange.attribution import (
    _readout_token_id,
    run_attribution_prescan,
    select_top_k,
    spearman_rank_correlation,
    top_k_agreement,
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec

_DATASET = [
    {
        "input": {"raw_input": "The quick brown fox jumps"},
        "counterfactual_inputs": [{"raw_input": "A lazy old dog sleeps"}],
    },
    {
        "input": {"raw_input": "Paris is the capital of"},
        "counterfactual_inputs": [{"raw_input": "Rome is the capital of"}],
    },
    {
        "input": {"raw_input": "Two plus two equals"},
        "counterfactual_inputs": [{"raw_input": "Three plus three equals"}],
    },
]
_PAIR_IDS = [(7, 3), (11, 5), (2, 9)]
_LAYERS = (0, 1)
_POSITIONS = (2, 1)  # static, in every example's unpadded frame


def _grid_targets() -> dict[tuple, list[list[SiteSpec]]]:
    """A (layer × position) grid of single-spec residual-stream cells."""
    return {
        (layer, f"p{pos}"): [
            [
                SiteSpec(
                    fsite=FeaturizedSite(Site("block_output", layer)),
                    positions=(pos,),
                    key=f"L{layer}P{pos}",
                )
            ]
        ]
        for layer in _LAYERS
        for pos in _POSITIONS
    }


def _autograd_reference(pipeline) -> dict[tuple, float]:
    """Explicit per-pair grad × Δ: raw HF forward+backward with leaf hooks.

    Runs one example per forward (no padding, so a static index addresses the
    same token in both frames), makes each layer's block output an autograd
    leaf, backwards the identical logit difference, and dots the gradient
    with the source − base activation difference by hand.
    """
    hf = pipeline.hf_model
    scores = {key: 0.0 for key in _grid_targets()}
    for example, (cf_id, base_id) in zip(_DATASET, _PAIR_IDS):
        base_enc = pipeline.load([example["input"]])
        cf_enc = pipeline.load(example["counterfactual_inputs"])

        stash: dict[int, torch.Tensor] = {}

        def leaf_hook_for(layer: int):
            def hook(_m, _i, out):
                hidden = out[0] if isinstance(out, tuple) else out
                if hidden.requires_grad:
                    hidden.retain_grad()
                else:
                    hidden.requires_grad_(True)
                stash[layer] = hidden
                return out

            return hook

        handles = [
            hf.model.layers[layer].register_forward_hook(leaf_hook_for(layer))
            for layer in _LAYERS
        ]
        try:
            out = hf(
                input_ids=base_enc["input_ids"].cpu(),
                attention_mask=base_enc["attention_mask"].cpu(),
            )
        finally:
            for handle in handles:
                handle.remove()
        (out.logits[0, -1, cf_id] - out.logits[0, -1, base_id]).backward()

        cf_stash: dict[int, torch.Tensor] = {}

        def capture_hook_for(layer: int):
            def hook(_m, _i, out):
                cf_stash[layer] = (out[0] if isinstance(out, tuple) else out).detach()
                return out

            return hook

        handles = [
            hf.model.layers[layer].register_forward_hook(capture_hook_for(layer))
            for layer in _LAYERS
        ]
        try:
            with torch.no_grad():
                hf(
                    input_ids=cf_enc["input_ids"].cpu(),
                    attention_mask=cf_enc["attention_mask"].cpu(),
                )
        finally:
            for handle in handles:
                handle.remove()

        for layer in _LAYERS:
            grad = stash[layer].grad
            assert grad is not None
            base_act = stash[layer].detach()[0]
            cf_act = cf_stash[layer][0]
            for pos in _POSITIONS:
                delta = cf_act[pos].float() - base_act[pos].float()
                scores[(layer, f"p{pos}")] += float(
                    (delta * grad[0, pos].float()).sum()
                )
    return {key: value / len(_DATASET) for key, value in scores.items()}


class TestAttributionPrescanNumericalUnit:
    pytestmark = pytest.mark.numerical_unit

    def test_gradient_times_delta_matches_autograd_reference(
        self, mock_tiny_lm
    ) -> None:
        """The batched two-forward pre-scan reproduces the per-pair explicit
        autograd computation cell for cell, and ranks cells identically."""
        targets = _grid_targets()
        scores = run_attribution_prescan(
            targets, _DATASET, mock_tiny_lm, batch_size=2, pair_token_ids=_PAIR_IDS
        )
        reference = _autograd_reference(mock_tiny_lm)
        assert set(scores) == set(reference)
        for key in targets:
            assert scores[key] == pytest.approx(reference[key], abs=1e-4), key
        assert select_top_k(scores, 2) == select_top_k(reference, 2)

    def test_last_layer_mid_positions_get_zero_gradient(self, mock_tiny_lm) -> None:
        """Causal-structure sanity: on a 2-layer model the last layer's
        non-final positions cannot reach the final-position logits, so their
        attribution is exactly zero."""
        targets = _grid_targets()
        scores = run_attribution_prescan(
            targets, _DATASET, mock_tiny_lm, batch_size=3, pair_token_ids=_PAIR_IDS
        )
        for pos in _POSITIONS:
            assert scores[(1, f"p{pos}")] == 0.0


class TestAttributionPrescanProperty:
    pytestmark = pytest.mark.property

    def test_deterministic_and_covers_every_cell(self, mock_tiny_lm) -> None:
        """Two identical runs agree exactly, and the score dict covers
        exactly the target keys (one float per grid cell)."""
        targets = _grid_targets()
        first = run_attribution_prescan(
            targets, _DATASET, mock_tiny_lm, batch_size=2, pair_token_ids=_PAIR_IDS
        )
        second = run_attribution_prescan(
            targets, _DATASET, mock_tiny_lm, batch_size=2, pair_token_ids=_PAIR_IDS
        )
        assert first == second
        assert set(first) == set(targets)
        assert all(isinstance(value, float) for value in first.values())

    def test_batch_size_invariant(self, mock_tiny_lm) -> None:
        """Scores are a per-example mean, so batching must not change them
        (up to float re-association across the padded frames)."""
        targets = _grid_targets()
        one = run_attribution_prescan(
            targets, _DATASET, mock_tiny_lm, batch_size=1, pair_token_ids=_PAIR_IDS
        )
        three = run_attribution_prescan(
            targets, _DATASET, mock_tiny_lm, batch_size=3, pair_token_ids=_PAIR_IDS
        )
        for key in targets:
            assert one[key] == pytest.approx(three[key], abs=1e-4), key


class TestAttributionPrescanUnit:
    pytestmark = pytest.mark.unit

    def test_select_top_k_orders_signed_descending(self) -> None:
        scores = {("a",): 1.0, ("b",): -3.0, ("c",): 2.0}
        assert select_top_k(scores, 2) == [("c",), ("a",)]
        assert select_top_k(scores, 2, by_abs=True) == [("b",), ("c",)]
        with pytest.raises(ValueError, match="positive"):
            select_top_k(scores, 0)

    def test_top_k_agreement_counts_shared_top_cells(self) -> None:
        approx = {("a",): 3.0, ("b",): 2.0, ("c",): 1.0, ("d",): 0.0}
        exact = {("a",): 1.0, ("b",): 3.0, ("c",): 0.0, ("d",): 2.0}
        # top-2: approx {a, b}, exact {b, d} → overlap 1/2
        assert top_k_agreement(approx, exact, 2) == pytest.approx(0.5)
        # k is capped at half the shared keys (here 2), so the overlap can
        # actually miss even when k covers the whole domain
        assert top_k_agreement(approx, exact, 4) == pytest.approx(0.5)
        # by_abs ranks the approx side by magnitude
        signed = {("a",): -3.0, ("b",): 2.0, ("c",): 1.0, ("d",): 0.0}
        assert top_k_agreement(signed, exact, 2, by_abs=True) == pytest.approx(0.5)
        assert math.isnan(top_k_agreement({}, exact, 2))  # no shared keys

    def test_spearman_rank_correlation_bounds_and_signs(self) -> None:
        approx = {("a",): 0.1, ("b",): 0.5, ("c",): 0.9}
        assert spearman_rank_correlation(approx, approx) == pytest.approx(1.0)
        flipped = {key: -value for key, value in approx.items()}
        assert spearman_rank_correlation(approx, flipped) == pytest.approx(-1.0)
        assert math.isnan(spearman_rank_correlation(approx, {("a",): 0.0}))  # <2 keys

    def test_pair_ids_must_align_with_dataset(self, mock_tiny_lm) -> None:
        with pytest.raises(ValueError, match="pair_token_ids"):
            run_attribution_prescan(
                _grid_targets(), _DATASET, mock_tiny_lm, 2, pair_token_ids=[(1, 2)]
            )

    def test_multi_unit_target_refused(self, mock_tiny_lm) -> None:
        spec = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 0)), positions=(1,), key="u0"
        )
        other = SiteSpec(
            fsite=FeaturizedSite(Site("block_output", 1)), positions=(1,), key="u1"
        )
        targets = {("joint",): [[spec, other]]}
        with pytest.raises(ValueError, match="one unit per grid cell"):
            run_attribution_prescan(targets, _DATASET, mock_tiny_lm, 2, _PAIR_IDS)

    def test_persistent_edits_refused(self, mock_tiny_lm) -> None:
        """The pre-scan approximates interchange on the UNEDITED model; an
        installed persistent edit (causalab.neural.persistent) is refused
        loudly instead of silently ranking cells against a different model."""
        from causalab.neural.edit import Edit
        from causalab.neural.featurized_site import FeaturizedSite
        from causalab.neural.persistent import persistent_edits
        from causalab.neural.site import Site

        edit = Edit(FeaturizedSite(Site("block_output", 0)), g=lambda f: f)
        with persistent_edits(mock_tiny_lm.model, edit):
            with pytest.raises(ValueError, match="persistent edits"):
                run_attribution_prescan(
                    _grid_targets(), _DATASET, mock_tiny_lm, 2, _PAIR_IDS
                )
        # uninstalled on exit: the pre-scan runs again
        scores = run_attribution_prescan(
            _grid_targets(), _DATASET, mock_tiny_lm, 2, _PAIR_IDS
        )
        assert set(scores) == set(_grid_targets())

    def test_featurized_unit_refused(self, mock_tiny_lm) -> None:
        from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer

        torch.manual_seed(0)
        spec = SiteSpec(
            fsite=FeaturizedSite(
                Site("block_output", 0),
                SubspaceFeaturizer(shape=(16, 4), trainable=False),
            ),
            positions=(1,),
            key="rotated",
        )
        targets = {("rot",): [[spec]]}
        with pytest.raises(NotImplementedError, match="raw activation space"):
            run_attribution_prescan(targets, _DATASET, mock_tiny_lm, 2, _PAIR_IDS)

    def test_readout_token_id_falls_back_to_first_token(self, mock_tiny_lm) -> None:
        """Single-token answers resolve exactly; multi-token answers read
        their first emitted token (never raise)."""
        tokenizer = mock_tiny_lm.tokenizer
        single = _readout_token_id(mock_tiny_lm, "cat")
        assert isinstance(single, int)
        multi_answer = "unquestionably extraordinary"
        assert len(tokenizer.encode(" " + multi_answer, add_special_tokens=False)) > 1
        first = _readout_token_id(mock_tiny_lm, multi_answer)
        assert (
            first == tokenizer.encode(" " + multi_answer, add_special_tokens=False)[0]
        )
