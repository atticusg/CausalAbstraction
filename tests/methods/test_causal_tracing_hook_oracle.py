"""Hook oracle for the mixed noise+replace causal-tracing model (GH #380).

ROME-style causal tracing builds a *mixed* intervenable model: a ``noise``
intervention corrupts the entry site and a ``replace`` intervention restores a
later site to its clean value — both in one forward, relying on forward-order
hook firing (the corruption is written before the restored site is read).

The noise draw is a backbone internal, so this pins the RNG-independent
*behaviour* of the mixed model against hand-rolled hooks:

* corruption alone moves the output off clean (the noise actually fires);
* corruption **plus** restoring the final-layer last-token residual to its clean
  value recovers the clean output exactly (the restore mediates, and ``replace``
  overwrites whatever the corruption produced);
* the mixed run is reproducible at a fixed seed.

See ``docs/PYVENE_HOOK_COVERAGE.md``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.LM_units import ResidualStream
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index
from causalab.neural.units import InterchangeTarget
from causalab.methods.steer.steer import run_steering_interventions

from tests.neural.activations.hook_oracle import (
    capture_residual,
    make_trace,
    next_token_logits,
)
from tests._helpers.tiny import tiny_random_model, tiny_random_gpt2_model

_ENTRY_LAYER = 0
_BASE = "the quick brown fox jumps"


@pytest.fixture(scope="module")
def _llama_pipe() -> LMPipeline:
    return LMPipeline(tiny_random_model(), max_new_tokens=1, padding_side="left")


@pytest.fixture(scope="module")
def _gpt2_pipe() -> LMPipeline:
    return LMPipeline(tiny_random_gpt2_model(), max_new_tokens=1, padding_side="left")


@pytest.fixture(params=["llama", "gpt2"])
def oracle_pipeline(request, _llama_pipe, _gpt2_pipe) -> LMPipeline:
    """Tiny-random Llama and GPT-2 pipelines (``max_new_tokens=1``); the hook-oracle
    resolvers dispatch on the module tree, so the mixed noise+replace model is
    asserted on both families."""
    return _llama_pipe if request.param == "llama" else _gpt2_pipe


def _final_layer(pipeline: LMPipeline) -> int:
    """Index of the last decoder layer — the residual site that fully mediates the
    next-token logits at the last position (restoring it to clean recovers clean)."""
    return pipeline.model.config.num_hidden_layers - 1


def _last_unit(pipeline: LMPipeline, layer: int) -> ResidualStream:
    tp = TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last"
    )
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        target_output=True,
        shape=(pipeline.model.config.hidden_size,),
    )


def _run_mixed(
    pipeline: LMPipeline,
    target: InterchangeTarget,
    steering_vectors: dict,
    type_by_unit: dict,
    *,
    seed: int = 0,
) -> torch.Tensor:
    result = run_steering_interventions(
        pipeline,
        [{"input": make_trace(_BASE), "counterfactual_inputs": []}],
        target,
        steering_vectors,
        mode="replace",
        type_by_unit=type_by_unit,
        noise_seed=seed,
        output_scores=True,
    )
    return result["scores"][0][0]


class TestCausalTracingMixedModel:
    pytestmark = pytest.mark.property

    def test_corruption_alone_is_non_trivial(self, oracle_pipeline: LMPipeline) -> None:
        """Noise at the entry site alone moves the next-token logits off clean."""
        hidden = oracle_pipeline.model.config.hidden_size
        entry = _last_unit(oracle_pipeline, _ENTRY_LAYER)
        target = InterchangeTarget([[entry]])
        clean = next_token_logits(
            oracle_pipeline, oracle_pipeline.load([make_trace(_BASE)])
        )
        corrupted = _run_mixed(
            oracle_pipeline,
            target,
            {entry.id: torch.full((hidden,), 5.0)},
            {entry.id: "noise"},
        )
        assert not torch.allclose(corrupted, clean, atol=1e-4)

    def test_restore_recovers_clean(self, oracle_pipeline: LMPipeline) -> None:
        """Noise at the entry + replacing the final-layer last-token residual with
        its clean value (the mediating site) recovers the clean output exactly —
        the mixed model runs both interventions in one forward, and ``replace``
        overwrites the corrupted final residual with the clean one."""
        hidden = oracle_pipeline.model.config.hidden_size
        restore_layer = _final_layer(oracle_pipeline)
        entry = _last_unit(oracle_pipeline, _ENTRY_LAYER)
        restore = _last_unit(oracle_pipeline, restore_layer)
        # One unit per group: the steering path supplies one source vector per
        # intervention group, so the corrupted entry and the restored site sit in
        # separate groups.
        target = InterchangeTarget([[entry], [restore]])

        base_inputs = oracle_pipeline.load([make_trace(_BASE)])
        clean = next_token_logits(oracle_pipeline, base_inputs)
        clean_restore = capture_residual(oracle_pipeline, restore_layer, base_inputs)[
            :, -1, :
        ].squeeze(0)

        recovered = _run_mixed(
            oracle_pipeline,
            target,
            {
                entry.id: torch.full((hidden,), 5.0),  # noise scale
                restore.id: clean_restore,  # clean value to restore
            },
            {entry.id: "noise", restore.id: "replace"},
        )
        torch.testing.assert_close(recovered, clean, atol=1e-4, rtol=1e-3)

    def test_mixed_run_is_reproducible(self, oracle_pipeline: LMPipeline) -> None:
        """Same seed → byte-identical mixed-model output across runs."""
        hidden = oracle_pipeline.model.config.hidden_size
        entry = _last_unit(oracle_pipeline, _ENTRY_LAYER)
        target = InterchangeTarget([[entry]])
        vecs = {entry.id: torch.full((hidden,), 5.0)}
        types = {entry.id: "noise"}
        a = _run_mixed(oracle_pipeline, target, vecs, types, seed=3)
        b = _run_mixed(oracle_pipeline, target, vecs, types, seed=3)
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
