"""Tests for the Plan-logits → generate-shape adapter
(``methods.path_patching.outputs``).

``plan_outputs`` adapts one batch's Plan-saved prefill logits to a
:class:`~causalab.neural.pipeline.GenerationResult` (EU5b, #487); at the
path-patching contract (``max_new_tokens == 1``) greedy generation's single
step *is* the prefill. The numerical tier pins the adapter's numbers at
handcrafted logits — the last-position slice, its argmax, the decode — with
decoy peaks at non-last positions so a wrong sequence-axis slice cannot pass.
The property tier pins the output shape/key contract and the single-step
refusal (``check_single_step``).
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.path_patching.outputs import check_single_step, plan_outputs
from causalab.neural.pipeline import GenerationResult, LMPipeline


def _known_logits(vocab: int) -> torch.Tensor:
    """``(2, 4, vocab)`` zeros with a distinct argmax at the *last* position of
    each row, plus **larger** decoy peaks at non-last positions — a wrong
    sequence-axis slice would surface as the decoy's token id."""
    logits = torch.zeros(2, 4, vocab)
    logits[0, -1, 7] = 5.0
    logits[1, -1, 11] = 2.0
    logits[0, 0, 23] = 10.0  # decoys: the argmax of the WRONG position
    logits[1, 2, 29] = 9.0
    return logits


class TestPlanOutputsNumerical:
    pytestmark = pytest.mark.numerical_unit

    def test_sequences_scores_and_strings_from_known_logits(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        vocab = mock_tiny_lm.model.config.vocab_size
        logits = _known_logits(vocab)
        out = plan_outputs(mock_tiny_lm, logits)
        # Greedy single step == last-position argmax, per row.
        assert out.sequences.tolist() == [[7], [11]]
        # ``scores[0]`` IS the last-position slice, untouched (full vocab).
        assert out.scores is not None
        assert torch.equal(out.scores[0], logits[:, -1, :])
        # Decoded through the tokenizer directly — independent of ``dump``.
        expected = mock_tiny_lm.tokenizer.batch_decode(
            torch.tensor([[7], [11]]), skip_special_tokens=True
        )
        assert out.strings == expected


class TestPlanOutputsContract:
    pytestmark = pytest.mark.property

    def test_output_shapes_and_fields(self, mock_tiny_lm: LMPipeline) -> None:
        vocab = mock_tiny_lm.model.config.vocab_size
        out = plan_outputs(mock_tiny_lm, _known_logits(vocab))
        assert isinstance(out, GenerationResult)
        assert out.sequences.shape == (2, 1)
        assert out.sequences.dtype == torch.long
        assert out.scores is not None
        assert len(out.scores) == 1
        assert out.scores[0].shape == (2, vocab)
        assert isinstance(out.strings, list) and len(out.strings) == 2

    def test_output_scores_false_gives_none_scores(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        vocab = mock_tiny_lm.model.config.vocab_size
        out = plan_outputs(mock_tiny_lm, _known_logits(vocab), output_scores=False)
        assert out.scores is None
        assert out.scores_top_k is None

    def test_integer_output_scores_keeps_full_vocab(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        # Top-k compression is the runner's job (compress_scores_top_k over
        # the flat result); the adapter must hand over full-vocab scores.
        vocab = mock_tiny_lm.model.config.vocab_size
        out = plan_outputs(mock_tiny_lm, _known_logits(vocab), output_scores=5)
        assert out.scores is not None
        assert out.scores[0].shape == (2, vocab)

    def test_single_step_pipeline_accepted(self, mock_tiny_lm: LMPipeline) -> None:
        assert mock_tiny_lm.max_new_tokens == 1
        check_single_step(mock_tiny_lm)  # must not raise

    def test_multi_token_pipeline_refused(
        self, mock_tiny_lm: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mock_tiny_lm, "max_new_tokens", 3)
        with pytest.raises(NotImplementedError, match="max_new_tokens=3"):
            check_single_step(mock_tiny_lm)
