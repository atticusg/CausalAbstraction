"""Ragged spans: a batch may select a different number of tokens per example.

A span whose width varies across the batch used to be rejected outright —
positions were stacked into one rectangular ``[batch, n_positions]`` tensor and
``torch.as_tensor`` raises on a ragged nested list. Positions are now flat
``(example, position)`` pairs, so nothing has to be rectangular.

"It runs" is not the contract. The contract is that batching changes nothing:
one ragged batch must produce exactly what you would get by handling each
example on its own. That is what these tests assert, against a ground truth
built with raw forward hooks — no engine — so a shared bug in the batched path
cannot make both sides agree. See ``docs/HOOK_ORACLES.md``.

The uniform case is covered elsewhere; what is specific here is that example 0
selecting two tokens and example 1 selecting one must not bleed into each other,
which is exactly what a wrong flattening or a mis-aligned per-example source
would do.
"""

from __future__ import annotations

import pytest
import torch

from causalab.causal.trace import CausalTrace
from causalab.neural.LM_units import ResidualStream
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.interchange_mode import (
    batched_interchange_intervention,
    prepare_interchange_batch,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget

from tests.neural.activations.hook_oracle import (
    capture_residual,
    make_trace,
)

pytestmark = pytest.mark.property

_LAYER = 1
# Deliberately different lengths: the first prompt's span is two tokens wide,
# the second's is one. Base and counterfactual agree within each example, which
# is what interchange's pairing requires; the batch as a whole is ragged.
_LONG_BASE = "the quick brown fox jumps"
_LONG_SOURCE = "a slow lazy old dog sits"
_SHORT_BASE = "hi there"
_SHORT_SOURCE = "yo there"


def _ragged_position(pipeline: LMPipeline) -> TokenPosition:
    """Two tokens for a long prompt, one for a short one — keyed on the prompt so
    an example's base and counterfactual always agree with each other."""

    def index(trace: CausalTrace) -> list[int]:
        return [1, 2] if len(trace["raw_input"].split()) >= 3 else [1]

    return TokenPosition(index, pipeline, id="ragged")


def _unit(pipeline: LMPipeline) -> ResidualStream:
    return ResidualStream(
        layer=_LAYER,
        token_indices=_ragged_position(pipeline),
        target_output=True,
        shape=(pipeline.model.config.hidden_size,),
    )


def _dataset() -> list[dict]:
    return [
        {
            "input": make_trace(_LONG_BASE),
            "counterfactual_inputs": [make_trace(_LONG_SOURCE)],
        },
        {
            "input": make_trace(_SHORT_BASE),
            "counterfactual_inputs": [make_trace(_SHORT_SOURCE)],
        },
    ]


class TestRaggedCollect:
    def test_ragged_collect_matches_per_example_hooks(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """Collecting a ragged span yields one row per selected token, in
        example-then-position order, each equal to what a hook sees for that
        example alone."""
        unit = _unit(tiny_pipeline)
        collected = collect_features(_dataset(), tiny_pipeline, [unit], batch_size=2)[
            unit.id
        ]
        # 2 tokens for the long example + 1 for the short one.
        assert collected.shape[0] == 3

        expected = []
        for prompt, wanted in ((_LONG_BASE, [1, 2]), (_SHORT_BASE, [1])):
            # One example on its own: no padding, no batching, plain hook.
            activation = capture_residual(
                tiny_pipeline, _LAYER, tiny_pipeline.load([make_trace(prompt)])
            )
            for position in wanted:
                expected.append(activation[0, position])
        expected_rows = torch.stack(expected)

        torch.testing.assert_close(
            collected.cpu().float(), expected_rows.cpu().float(), atol=1e-4, rtol=1e-3
        )

    def test_examples_do_not_bleed_into_each_other(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """The short example's single row must be its own token, not a
        neighbour's. A flattening that mixed up row boundaries would still
        produce the right *number* of rows, so pin the identity of the last one.
        """
        unit = _unit(tiny_pipeline)
        collected = collect_features(_dataset(), tiny_pipeline, [unit], batch_size=2)[
            unit.id
        ]
        short = capture_residual(
            tiny_pipeline, _LAYER, tiny_pipeline.load([make_trace(_SHORT_BASE)])
        )
        torch.testing.assert_close(
            collected[-1].cpu().float(),
            short[0, 1].cpu().float(),
            atol=1e-4,
            rtol=1e-3,
        )
        # ...and is genuinely a different vector from the long example's rows,
        # so the check above cannot pass by everything being equal.
        assert not torch.allclose(
            collected[-1].cpu().float(), collected[0].cpu().float(), atol=1e-4
        )


class TestRaggedInterchange:
    def test_ragged_interchange_matches_unbatched(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A ragged batch must give each example the logits it would get alone.

        This is the assertion that matters: if the flattening paired the wrong
        source row with the wrong base row, the batched run would still complete
        and still produce plausible logits — they would just be wrong.
        """
        target = InterchangeTarget([[_unit(tiny_pipeline)]])
        batched = batched_interchange_intervention(
            tiny_pipeline, _dataset(), target, output_scores=True
        )

        # ``scores`` is indexed by generation step, each entry ``[batch, vocab]``.
        for i, example in enumerate(_dataset()):
            alone = batched_interchange_intervention(
                tiny_pipeline,
                [example],
                InterchangeTarget([[_unit(tiny_pipeline)]]),
                output_scores=True,
            )
            for step, batched_step in enumerate(batched["scores"]):
                torch.testing.assert_close(
                    batched_step[i].cpu().float(),
                    alone["scores"][step][0].cpu().float(),
                    atol=1e-4,
                    rtol=1e-3,
                )

    def test_ragged_positions_survive_preparation(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """The resolved positions keep their per-example widths rather than being
        padded or truncated to a common one."""
        batch = prepare_interchange_batch(
            tiny_pipeline, _dataset(), InterchangeTarget([[_unit(tiny_pipeline)]])
        )
        assert [len(ex) for ex in batch.base_positions[0]] == [2, 1]
        assert [len(ex) for ex in batch.source_positions[0]] == [2, 1]
