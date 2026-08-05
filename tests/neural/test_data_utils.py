"""Direct tests for ``causalab.neural.activations.data_utils``.

These helpers run *after* every intervention pass — i.e. after
``run_interchange_interventions`` (``interchange_mode.py``), ``interpolate.py``,
and ``methods/steer/steer.py`` produce batched HuggingFace outputs on GPU. The
public surface is two functions:

* :func:`convert_to_top_k` — compresses per-position full-vocab logit tensors
  into a ``{top_k_logits, top_k_indices, top_k_tokens}`` struct so that
  artifact writes don't blow out memory (vocab_size is typically ~256k for
  modern LMs).
* :func:`move_outputs_to_cpu` — detaches every nested tensor and moves it to
  CPU before serialization, recursing into ``dict``/``list``/``tuple``
  containers and preserving structure.

If either helper drops a key, mis-shapes a tensor, or fails to detach, the
intervention artifacts on disk are corrupt and every downstream analysis
(next-token distributions, IIA scoring) reads garbage. The classes below
cover the ``unit-direct`` and ``property-direct`` tiers that
``standards.yaml`` requires for ``causalab/neural/``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.activations.data_utils import (
    convert_to_top_k,
    move_outputs_to_cpu,
)


# --------------------------------------------------------------------------- #
#  Local RNG / helpers                                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


def _make_outputs(
    scores: list[torch.Tensor] | None,
    *,
    sequences: torch.Tensor | None = None,
    extras: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a single-batch outputs list mirroring HF ``generate`` shape.

    The real callers pass a list with one dict per HF ``model.generate``
    invocation; each dict carries a ``scores`` tuple-of-tensors plus optional
    ``sequences``/metadata keys. The helpers must not assume the dict has any
    particular non-``scores`` shape — only that ``scores``, if truthy, is an
    iterable of ``(batch, vocab)`` tensors.
    """
    batch_dict: dict[str, Any] = {}
    if sequences is not None:
        batch_dict["sequences"] = sequences
    if scores is not None:
        batch_dict["scores"] = scores
    if extras:
        batch_dict.update(extras)
    return [batch_dict]


# --------------------------------------------------------------------------- #
#  convert_to_top_k                                                            #
# --------------------------------------------------------------------------- #
class TestConvertToTopKUnit:
    """``convert_to_top_k`` compresses per-position vocab logits to top-k structs.

    Invoked by the interchange / interpolate / steer runners between the raw
    HF ``generate`` output and artifact serialization, so its contract on
    keys, shapes, and tokenizer dispatch is what every downstream analysis
    reads back from disk.
    """

    pytestmark = pytest.mark.unit

    def test_raises_on_non_positive_k(self, tiny_pipeline) -> None:
        outputs = _make_outputs(scores=[torch.zeros(2, 8)])
        with pytest.raises(ValueError, match="k must be positive"):
            convert_to_top_k(outputs, tiny_pipeline, k=0)
        with pytest.raises(ValueError, match="k must be positive"):
            convert_to_top_k(outputs, tiny_pipeline, k=-3)

    def test_preserves_non_scores_keys_verbatim(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        sequences = torch.tensor([[1, 2, 3], [4, 5, 6]])
        outputs = _make_outputs(
            scores=[randn((2, 8), rng)],
            sequences=sequences,
            extras={"metadata": {"trial": 7}, "label": "alpha"},
        )
        result = convert_to_top_k(outputs, tiny_pipeline, k=3)
        assert len(result) == 1
        # Non-scores values pass through by object identity.
        assert result[0]["sequences"] is sequences
        assert result[0]["metadata"] == {"trial": 7}
        assert result[0]["label"] == "alpha"

    def test_top_k_shapes_match_contract(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        batch_size, vocab_size, k = 3, 16, 4
        scores = [randn((batch_size, vocab_size), rng) for _ in range(2)]
        outputs = _make_outputs(scores=scores)

        result = convert_to_top_k(outputs, tiny_pipeline, k=k)
        assert len(result[0]["scores"]) == 2  # one struct per generated position
        for position in result[0]["scores"]:
            assert position["top_k_logits"].shape == (batch_size, k)
            assert position["top_k_indices"].shape == (batch_size, k)
            # top_k_tokens is List[batch][k] of decoded strings.
            assert len(position["top_k_tokens"]) == batch_size
            for row in position["top_k_tokens"]:
                assert len(row) == k
                assert all(isinstance(tok, str) for tok in row)

    def test_top_k_indices_match_torch_topk(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        scores = [randn((2, 32), rng)]
        outputs = _make_outputs(scores=scores)

        result = convert_to_top_k(outputs, tiny_pipeline, k=5)
        expected_values, expected_indices = torch.topk(scores[0], k=5, dim=1)
        torch.testing.assert_close(
            result[0]["scores"][0]["top_k_logits"], expected_values
        )
        torch.testing.assert_close(
            result[0]["scores"][0]["top_k_indices"], expected_indices
        )

    def test_tokenizer_called_with_singleton_lists_no_special_skip(
        self,
        tiny_pipeline,
        rng: torch.Generator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``batch_decode`` is dispatched on a flattened list of singleton
        ``[idx]`` lists with ``skip_special_tokens=False`` — the contract that
        keeps token strings independent (no cross-token merging) and lets
        special tokens like ``<|endoftext|>`` survive into the artifact.
        """
        scores = [randn((2, 16), rng)]
        outputs = _make_outputs(scores=scores)
        captured: dict[str, Any] = {}

        real_batch_decode = tiny_pipeline.tokenizer.batch_decode

        def spy(token_ids, **kwargs):
            captured["token_ids"] = token_ids
            captured["kwargs"] = kwargs
            return real_batch_decode(token_ids, **kwargs)

        monkeypatch.setattr(tiny_pipeline.tokenizer, "batch_decode", spy)
        convert_to_top_k(outputs, tiny_pipeline, k=3)

        assert captured["kwargs"].get("skip_special_tokens") is False
        # Each element is a singleton list wrapping an int index.
        assert all(
            isinstance(item, list) and len(item) == 1 and isinstance(item[0], int)
            for item in captured["token_ids"]
        )
        # Flattened shape: batch_size (2) * k (3) = 6.
        assert len(captured["token_ids"]) == 6

    def test_missing_scores_key_passes_through(self, tiny_pipeline) -> None:
        outputs = [{"sequences": torch.tensor([[1, 2, 3]])}]
        result = convert_to_top_k(outputs, tiny_pipeline, k=3)
        assert "scores" not in result[0]
        assert torch.equal(result[0]["sequences"], torch.tensor([[1, 2, 3]]))

    def test_empty_scores_list_passes_through(self, tiny_pipeline) -> None:
        """``if scores:`` treats an empty list as falsy and leaves the key off
        the returned dict (see Risks/open-question 2 in the refactor plan).
        """
        outputs = _make_outputs(scores=[])
        result = convert_to_top_k(outputs, tiny_pipeline, k=3)
        # Empty list is falsy → branch skipped → no 'scores' carried over.
        assert "scores" not in result[0]


class TestConvertToTopKProperty:
    """Algebraic invariants of ``convert_to_top_k`` over the scores struct."""

    pytestmark = pytest.mark.property

    def test_k_greater_than_vocab_clamps_to_vocab_size(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        batch_size, vocab_size = 2, 7
        scores = [randn((batch_size, vocab_size), rng)]
        outputs = _make_outputs(scores=scores)

        # No IndexError; k_actual = min(k, vocab_size) → vocab_size.
        result = convert_to_top_k(outputs, tiny_pipeline, k=vocab_size * 5)
        position = result[0]["scores"][0]
        assert position["top_k_logits"].shape == (batch_size, vocab_size)
        assert position["top_k_indices"].shape == (batch_size, vocab_size)
        # All vocab indices are present in some order, per row.
        for row in position["top_k_indices"]:
            assert sorted(row.tolist()) == list(range(vocab_size))

    def test_logits_sorted_descending_per_row(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        scores = [randn((4, 64), rng)]
        outputs = _make_outputs(scores=scores)
        result = convert_to_top_k(outputs, tiny_pipeline, k=10)
        logits = result[0]["scores"][0]["top_k_logits"]
        # Each row monotonic non-increasing.
        diffs = logits[:, 1:] - logits[:, :-1]
        assert torch.all(diffs <= 0)

    def test_indices_within_vocab_range_and_unique_per_row(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        vocab_size = 50
        scores = [randn((3, vocab_size), rng)]
        outputs = _make_outputs(scores=scores)
        result = convert_to_top_k(outputs, tiny_pipeline, k=8)
        indices = result[0]["scores"][0]["top_k_indices"]
        assert (indices >= 0).all()
        assert (indices < vocab_size).all()
        for row in indices:
            assert len(set(row.tolist())) == row.numel()  # all distinct

    def test_output_key_set_equals_input_key_set(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        outputs = _make_outputs(
            scores=[randn((2, 16), rng)],
            sequences=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            extras={"meta": "x"},
        )
        result = convert_to_top_k(outputs, tiny_pipeline, k=3)
        assert set(result[0].keys()) == set(outputs[0].keys())

    def test_double_application_raises_on_topk_struct(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        """``convert_to_top_k`` is *not* idempotent — re-running on already-top-k
        outputs trips because the per-position entries are dicts, not tensors,
        so ``position_logits.shape`` raises ``AttributeError``. The refactor
        plan asks us to pin this contract (assert it raises or document) so a
        future caller doesn't accidentally double-convert.
        """
        scores = [randn((2, 16), rng)]
        outputs = _make_outputs(scores=scores)
        once = convert_to_top_k(outputs, tiny_pipeline, k=3)
        with pytest.raises(AttributeError):
            convert_to_top_k(once, tiny_pipeline, k=3)


# --------------------------------------------------------------------------- #
#  move_outputs_to_cpu                                                         #
# --------------------------------------------------------------------------- #
class TestMoveOutputsToCpuUnit:
    """``move_outputs_to_cpu`` detaches + moves every nested tensor to CPU.

    Runs immediately before artifact serialization; if any tensor stays on
    GPU or keeps requires_grad, ``torch.save`` either OOMs or pickles a
    needless autograd graph into the artifact.
    """

    pytestmark = pytest.mark.unit

    def test_tensor_at_top_level_is_detached_and_on_cpu(self) -> None:
        x = torch.randn(3, 4, requires_grad=True)
        outputs = [{"scores": x}]
        result = move_outputs_to_cpu(outputs)
        assert result[0]["scores"].device.type == "cpu"
        assert result[0]["scores"].requires_grad is False
        # Detach returns a new tensor; the original should be unchanged.
        assert outputs[0]["scores"].requires_grad is True

    def test_dict_structure_recursed_into(self) -> None:
        inner = torch.randn(2, 2)
        outputs = [{"scores": [{"top_k_logits": inner, "label": "k"}]}]
        result = move_outputs_to_cpu(outputs)
        moved = result[0]["scores"][0]["top_k_logits"]
        assert moved.device.type == "cpu"
        # Non-tensor primitives ride along untouched.
        assert result[0]["scores"][0]["label"] == "k"

    def test_list_and_tuple_types_preserved(self) -> None:
        outputs = [
            {
                "as_list": [torch.zeros(2), torch.ones(2)],
                "as_tuple": (torch.zeros(2), torch.ones(2)),
            }
        ]
        result = move_outputs_to_cpu(outputs)
        assert isinstance(result[0]["as_list"], list)
        assert isinstance(result[0]["as_tuple"], tuple)

    def test_none_and_primitives_pass_through(self) -> None:
        outputs = [
            {
                "none": None,
                "int": 7,
                "float": 1.5,
                "str": "hello",
                "bool": True,
            }
        ]
        result = move_outputs_to_cpu(outputs)
        assert result[0]["none"] is None
        assert result[0]["int"] == 7
        assert result[0]["float"] == 1.5
        assert result[0]["str"] == "hello"
        assert result[0]["bool"] is True

    def test_handles_topk_payload_from_convert_to_top_k(
        self, tiny_pipeline, rng: torch.Generator
    ) -> None:
        """End-to-end shape: feed ``convert_to_top_k``'s output through
        ``move_outputs_to_cpu`` and confirm every nested tensor lands on CPU
        without losing structure. This is the production call sequence.
        """
        scores = [randn((2, 16), rng) for _ in range(2)]
        outputs = _make_outputs(scores=scores)
        topk = convert_to_top_k(outputs, tiny_pipeline, k=3)
        moved = move_outputs_to_cpu(topk)
        for position in moved[0]["scores"]:
            assert position["top_k_logits"].device.type == "cpu"
            assert position["top_k_indices"].device.type == "cpu"
            assert position["top_k_logits"].requires_grad is False
            assert isinstance(position["top_k_tokens"], list)

    def test_custom_object_with_hidden_tensor_passes_through(self) -> None:
        """The recursion only handles ``dict``/``list``/``tuple`` containers
        (per Risk 3 in the refactor plan). Tensors hidden inside a custom
        object are an explicit non-guarantee.
        """

        class Wrapper:
            def __init__(self, t: torch.Tensor) -> None:
                self.t = t

        wrapped = Wrapper(torch.randn(2, requires_grad=True))
        outputs = [{"wrapped": wrapped}]
        result = move_outputs_to_cpu(outputs)
        # Same object handed back; the hidden tensor is *not* detached.
        assert result[0]["wrapped"] is wrapped
        assert result[0]["wrapped"].t.requires_grad is True


class TestMoveOutputsToCpuProperty:
    """Idempotence and structural invariants of ``move_outputs_to_cpu``."""

    pytestmark = pytest.mark.property

    def test_idempotent_under_double_application(self) -> None:
        outputs = [{"scores": [torch.randn(2, 4, requires_grad=True)]}]
        once = move_outputs_to_cpu(outputs)
        twice = move_outputs_to_cpu(once)
        torch.testing.assert_close(once[0]["scores"][0], twice[0]["scores"][0])
        for batch in twice:
            for score in batch["scores"]:
                assert score.device.type == "cpu"
                assert score.requires_grad is False

    def test_structural_isomorphism_preserved(self) -> None:
        outputs = [
            {
                "a": torch.zeros(1),
                "b": [torch.zeros(1), {"c": torch.zeros(1)}],
                "d": (torch.zeros(1),),
            }
        ]
        result = move_outputs_to_cpu(outputs)
        assert isinstance(result, list) and len(result) == 1
        assert set(result[0].keys()) == {"a", "b", "d"}
        assert isinstance(result[0]["b"], list) and len(result[0]["b"]) == 2
        assert isinstance(result[0]["b"][1], dict)
        assert set(result[0]["b"][1].keys()) == {"c"}
        assert isinstance(result[0]["d"], tuple) and len(result[0]["d"]) == 1

    def test_input_not_mutated_in_place(self) -> None:
        original_tensor = torch.randn(3, 3, requires_grad=True)
        outputs = [{"scores": original_tensor}]
        # Snapshot pre-call identities.
        pre_id_list = id(outputs)
        pre_id_dict = id(outputs[0])
        pre_id_tensor = id(outputs[0]["scores"])

        result = move_outputs_to_cpu(outputs)

        # Original container references unchanged.
        assert id(outputs) == pre_id_list
        assert id(outputs[0]) == pre_id_dict
        assert id(outputs[0]["scores"]) == pre_id_tensor
        # Original tensor still tracks grad.
        assert outputs[0]["scores"].requires_grad is True
        # Returned tensor is a distinct, detached, on-CPU view.
        assert result[0]["scores"] is not original_tensor
        assert result[0]["scores"].requires_grad is False
