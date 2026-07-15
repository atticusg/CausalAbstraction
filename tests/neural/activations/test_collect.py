"""Direct tests for ``causalab.neural.activations.collect``.

This module hosts the dataset-wide and per-batch activation harvesting
primitives that feed every featurizer learner (``methods/pca``,
``methods/spline``, ``methods/flow``) and every interchange / path-steering
analysis. ``collect_features`` produces a ``{unit_id -> (n_samples, hidden)}``
dict on CPU; ``collect_source_representations`` / ``collect_batch_representations``
produce the per-location source tensors that ``neural/activations/interchange_mode.py``
and ``methods/metric.py`` pass to pyvene as ``source_representations`` during
cross-model patching; ``collect_class_centroids`` packages a ``collect_features``
call plus a per-value reduction used by path-steering visualization.

If any of these returns the wrong shape, ordering, or device, every
downstream interchange / featurizer training run silently consumes corrupted
activations — so this module is the canonical place to pin those contracts.

The *unit* classes keep the historical mocked-pyvene scaffolding (they exist
to assert envelope reshape / dict-keying / log-message behaviour, not numerics).
The *property* classes use the real ``tiny_pipeline`` fixture from
``tests/neural/conftest.py`` so the pyvene envelope shape is exercised
end-to-end against a real (tiny) ``IntervenableModel``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.LM_units import ResidualStream
from causalab.neural.activations.collect import (
    _collect_activations_single_batch,  # pyright: ignore[reportPrivateUsage]
    collect_batch_representations,
    collect_class_centroids,
    collect_features,
    collect_source_representations,
)
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget


# --------------------------------------------------------------------------- #
#  Local RNG helper                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


# --------------------------------------------------------------------------- #
#  Module-level fixtures shared across the mocked unit-tier classes           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_dataset() -> list[dict]:
    """Tiny mock dataset matching the ``CounterfactualExample`` schema."""
    return [
        {"input": "input_1"},
        {"input": "input_2"},
        {"input": "input_3"},
    ]


@pytest.fixture
def mock_loaded_inputs() -> dict[str, torch.Tensor]:
    """Mock output of ``pipeline.load`` — a batch of tokenized inputs."""
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }


@pytest.fixture
def residual_units() -> list[MagicMock]:
    """Two distinct ``ResidualStream``-shaped mocks (distinct ``id``s)."""
    unit1 = MagicMock()
    unit1.id = "ResidualStream(Layer:0,Token:last_token)"
    unit1.index_component.return_value = [[0, 1], [0, 1]]
    unit2 = MagicMock()
    unit2.id = "ResidualStream(Layer:2,Token:last_token)"
    unit2.index_component.return_value = [[0, 1], [0, 1]]
    return [unit1, unit2]


@pytest.fixture
def mock_pipeline(mock_loaded_inputs) -> MagicMock:
    """Lightweight ``Pipeline`` stand-in: just needs ``model.eval()`` + ``load()``.

    Avoids ``mock_pipeline`` (deprecated; tries to ``AutoTokenizer.from_pretrained``
    the literal string ``"mock_model"``). ``collect_features`` only touches
    ``pipeline.model.eval()`` and ``pipeline.load(...)`` on the pipeline
    itself — everything else flows through the mocked ``IntervenableModel``.
    """
    pipeline = MagicMock()
    pipeline.load = MagicMock(return_value=mock_loaded_inputs)
    pipeline.model = MagicMock()
    return pipeline


def _patch_pyvene_returning(activations_factory):
    """Patch the pyvene helpers in ``collect`` to return a fabricated tuple.

    The returned ``activations_factory`` is invoked on every call so each batch
    gets a fresh list of tensors (matching pyvene 0.1.8+'s
    ``((base_outputs, collected_activations), cf_outputs)`` envelope).
    """
    mock_model = MagicMock()
    mock_model.side_effect = lambda *args, **kwargs: (
        (MagicMock(), activations_factory()),
        None,
    )
    return (
        patch(
            "causalab.neural.activations.collect.prepare_intervenable_model",
            return_value=mock_model,
        ),
        patch("causalab.neural.activations.collect.delete_intervenable_model"),
        mock_model,
    )


# --------------------------------------------------------------------------- #
#  collect_features                                                           #
# --------------------------------------------------------------------------- #
class TestCollectFeaturesUnit:
    """Dataset-wide activation harvest feeding ``methods/`` featurizer learners.

    ``collect_features`` is the entry point for every featurizer training
    run: ``methods/pca``, ``methods/spline``, ``methods/flow``,
    ``methods/interchange/layer_scan``, and the
    ``analyses/{activation_manifold, subspace, path_steering}`` analyses all
    consume its ``{unit.id -> (n_samples, hidden)}`` output on CPU. This class
    asserts the post-pyvene reshape / dict-keying logic via mocked
    ``IntervenableModel``s (numerics-free).
    """

    pytestmark = pytest.mark.unit

    def test_returns_dict_keyed_by_unit_id(
        self, mock_pipeline, residual_units, mock_dataset, mock_loaded_inputs
    ) -> None:
        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [torch.randn(2, 32), torch.randn(2, 32)]
        )
        with (
            prep_p as mock_prepare,
            del_p,
        ):
            result = collect_features(
                mock_dataset, mock_pipeline, residual_units, batch_size=2
            )

        mock_prepare.assert_called_once_with(
            mock_pipeline, residual_units, intervention_type="collect"
        )
        assert isinstance(result, dict)
        assert set(result) == {u.id for u in residual_units}
        for u in residual_units:
            assert isinstance(result[u.id], torch.Tensor)

    def test_emits_debug_log_with_unit_count_and_shape(
        self,
        mock_pipeline,
        residual_units,
        mock_dataset,
        mock_loaded_inputs,
        caplog,
    ) -> None:
        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [torch.randn(2, 32), torch.randn(2, 32)]
        )
        with (
            prep_p,
            del_p,
            caplog.at_level(
                logging.DEBUG, logger="causalab.neural.activations.collect"
            ),
        ):
            collect_features(mock_dataset, mock_pipeline, residual_units, batch_size=2)

        assert "Collected features for" in caplog.text
        assert "Feature tensor shape:" in caplog.text

    def test_returns_tensors_on_cpu_when_source_is_gpu_like(
        self, mock_pipeline, residual_units, mock_dataset, mock_loaded_inputs
    ) -> None:
        """Even if pyvene returns tensors on a non-CPU device, the dict is CPU."""
        from causalab.neural.pipeline import resolve_device

        device = torch.device(resolve_device())
        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [
                torch.randn(2, 32, device=device),
                torch.randn(2, 32, device=device),
            ]
        )
        with (
            prep_p,
            del_p,
            patch("torch.cuda.empty_cache"),
        ):
            result = collect_features(
                mock_dataset, mock_pipeline, residual_units, batch_size=2
            )

        for tensor in result.values():
            assert tensor.device.type == "cpu"

    def test_result_is_2d_with_hidden_last(
        self, mock_pipeline, residual_units, mock_dataset, mock_loaded_inputs
    ) -> None:
        hidden_size = 32
        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [torch.randn(2, hidden_size), torch.randn(2, hidden_size)]
        )
        with (
            prep_p,
            del_p,
        ):
            result = collect_features(
                mock_dataset, mock_pipeline, residual_units, batch_size=2
            )

        for tensor in result.values():
            assert tensor.ndim == 2
            assert tensor.shape[1] == hidden_size

    def test_attention_head_4d_reshapes_to_head_dim(
        self, mock_pipeline, mock_dataset, mock_loaded_inputs
    ) -> None:
        """4-D ``(batch, seq, n_heads, head_dim)`` flattens onto ``head_dim``."""
        unit1 = MagicMock()
        unit1.id = "AttentionHead(Layer:0,Head:0)"
        unit1.index_component.return_value = [[0], [0]]
        unit2 = MagicMock()
        unit2.id = "AttentionHead(Layer:0,Head:1)"
        unit2.index_component.return_value = [[0], [0]]
        attention_units = [unit1, unit2]

        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [torch.randn(2, 1, 4, 8), torch.randn(2, 1, 4, 8)]
        )
        with (
            prep_p,
            del_p,
        ):
            result = collect_features(
                mock_dataset, mock_pipeline, attention_units, batch_size=2
            )

        for u in attention_units:
            assert result[u.id].shape[1] == 8

    def test_residual_3d_reshapes_to_hidden_dim(
        self, mock_pipeline, mock_dataset, mock_loaded_inputs
    ) -> None:
        """3-D ``(batch, seq, hidden)`` flattens onto ``hidden``."""
        unit1 = MagicMock()
        unit1.id = "ResidualStream(Layer:0)"
        unit1.index_component.return_value = [[0], [0]]
        unit2 = MagicMock()
        unit2.id = "ResidualStream(Layer:1)"
        unit2.index_component.return_value = [[0], [0]]
        rs_units = [unit1, unit2]

        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [torch.randn(2, 1, 32), torch.randn(2, 1, 32)]
        )
        with (
            prep_p,
            del_p,
        ):
            result = collect_features(
                mock_dataset, mock_pipeline, rs_units, batch_size=2
            )

        for u in rs_units:
            assert result[u.id].shape[1] == 32

    def test_mixed_2d_and_3d_shapes_are_supported(
        self, mock_pipeline, mock_dataset, mock_loaded_inputs
    ) -> None:
        unit1 = MagicMock()
        unit1.id = "Unit1"
        unit1.index_component.return_value = [[0], [0]]
        unit2 = MagicMock()
        unit2.id = "Unit2"
        unit2.index_component.return_value = [[0], [0]]
        mixed_units = [unit1, unit2]

        prep_p, del_p, _ = _patch_pyvene_returning(
            lambda: [torch.randn(2, 64), torch.randn(2, 1, 32)]
        )
        with (
            prep_p,
            del_p,
        ):
            result = collect_features(
                mock_dataset,
                mock_pipeline,
                mixed_units,  # pyright: ignore[reportArgumentType]
                batch_size=2,
            )

        assert result["Unit1"].shape[1] == 64
        assert result["Unit2"].shape[1] == 32

    def test_raises_when_pyvene_returns_wrong_unit_count(
        self, mock_pipeline, residual_units, mock_dataset, mock_loaded_inputs
    ) -> None:
        """Defends against a future pyvene refactor changing the envelope contract."""
        # One activation tensor returned but two units requested.
        prep_p, del_p, _ = _patch_pyvene_returning(lambda: [torch.randn(2, 32)])
        with (
            prep_p,
            del_p,
            pytest.raises(ValueError, match="Unexpected activations format"),
        ):
            collect_features(mock_dataset, mock_pipeline, residual_units, batch_size=2)

    def test_collect_output_logits_returns_tuple_dict_and_list(
        self, mock_pipeline, residual_units, mock_dataset, mock_loaded_inputs
    ) -> None:
        """When ``collect_output_logits=True``, return ``(dict, list[Tensor])``."""
        # Two batches: first carries 2 examples, second carries 1.
        batch_sizes = iter([2, 1])

        def model_side_effect(*args, **kwargs):
            n = next(batch_sizes)
            fake_model_output = MagicMock()
            fake_model_output.logits = torch.randn(n, 3, 100)
            return (
                (fake_model_output, [torch.randn(n, 32), torch.randn(n, 32)]),
                None,
            )

        mock_model = MagicMock()
        mock_model.side_effect = model_side_effect
        with (
            patch(
                "causalab.neural.activations.collect.prepare_intervenable_model",
                return_value=mock_model,
            ),
            patch("causalab.neural.activations.collect.delete_intervenable_model"),
        ):
            result = collect_features(
                mock_dataset,
                mock_pipeline,
                residual_units,
                batch_size=2,
                collect_output_logits=True,
            )

        assert isinstance(result, tuple)
        features_dict, logits_list = result
        assert isinstance(features_dict, dict)
        assert isinstance(logits_list, list)
        # 3 examples / batch_size=2 → 2 batches → 2 + 1 = 3 per-example logits.
        assert len(logits_list) == 3
        for lg in logits_list:
            assert isinstance(lg, torch.Tensor)

    def test_duplicate_unit_id_raises(
        self, mock_pipeline, mock_dataset, mock_loaded_inputs
    ) -> None:
        """Two units sharing an ``id`` would silently merge — must raise instead."""
        u1 = MagicMock()
        u1.id = "dupe"
        u1.index_component.return_value = [[0, 1], [0, 1]]
        u2 = MagicMock()
        u2.id = "dupe"
        u2.index_component.return_value = [[0, 1], [0, 1]]
        with pytest.raises(ValueError, match="Duplicate model_unit.id"):
            collect_features(mock_dataset, mock_pipeline, [u1, u2], batch_size=2)


class TestCollectActivationsPositionIds:
    """The collection forward must receive left-pad-correct ``position_ids``.

    Without them the plain (non-generate) forward numbers positions from the pad
    tokens, corrupting padded-row activations on absolute-position models. RoPE
    models are immune, so the RoPE ``tiny_pipeline`` fixture can't catch this
    numerically — we assert the contract directly via a mock that captures the
    inputs dict handed to the forward (the primitive now routes it through
    ``pipeline.ensure_position_ids``).

    Scope: this pins what *this function* hands to the model; it does not prove
    pyvene forwards ``position_ids`` on to the underlying HF model. The
    ``ensure_position_ids`` formula itself is pinned in
    ``tests/neural/test_pipeline.py``.
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _capture_forward_inputs(loaded_inputs: dict[str, torch.Tensor]) -> dict:
        """Run the primitive against a mock model; return the captured inputs dict."""
        captured: dict = {}

        def side_effect(inputs, unit_locations=None):
            captured["inputs"] = inputs
            # pyvene envelope: ((base_outputs, collected_activations), cf_outputs)
            return ((MagicMock(), [torch.zeros(2, 1, 4)]), None)

        mock_model = MagicMock(side_effect=side_effect)
        # indices: one unit, batch of 2, one position each — shape only needs to
        # be well-formed; the mock ignores it.
        _collect_activations_single_batch(
            mock_model, loaded_inputs, indices=[[[0], [0]]]
        )
        assert "inputs" in captured, "forward was never called"
        return captured["inputs"]

    def test_derives_leftpad_position_ids_for_padded_batch(self) -> None:
        """A left-padded mask → HF left-pad ``position_ids`` (pads pinned to 1)."""
        loaded = {
            "input_ids": torch.tensor([[0, 0, 5, 6], [7, 8, 9, 10]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        }
        inputs = self._capture_forward_inputs(loaded)
        assert "position_ids" in inputs, "collection forward got no position_ids"
        # cumsum(mask)-1, with pad slots (mask==0) pinned to 1 — matches HF's
        # prepare_inputs_for_generation for left padding.
        expected = torch.tensor([[1, 1, 0, 1], [0, 1, 2, 3]])
        assert torch.equal(inputs["position_ids"], expected)

    def test_unpadded_batch_reduces_to_arange(self) -> None:
        """An all-ones mask → plain ``arange`` (the previously-correct behavior)."""
        loaded = {
            "input_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
        inputs = self._capture_forward_inputs(loaded)
        expected = torch.arange(3).unsqueeze(0).expand(2, -1)
        assert torch.equal(inputs["position_ids"], expected)

    def test_existing_position_ids_are_not_overwritten(self) -> None:
        """If the pipeline already emitted position_ids, leave them untouched."""
        preset = torch.tensor([[3, 4, 5], [6, 7, 8]])
        loaded = {
            "input_ids": torch.tensor([[5, 6, 7], [8, 9, 10]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "position_ids": preset,
        }
        inputs = self._capture_forward_inputs(loaded)
        assert torch.equal(inputs["position_ids"], preset)


def _trace(text: str) -> CausalTrace:
    """Helper: build a minimal ``CausalTrace`` carrying just ``raw_input``."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _make_residual_unit(pipeline, layer: int) -> ResidualStream:
    """Helper: build a real ``ResidualStream`` unit at ``layer``, last token."""

    def last_token(trace):
        return [len(pipeline.load([trace])["input_ids"][0]) - 1]

    tp = TokenPosition(last_token, pipeline, id="last_token")
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        shape=(pipeline.model.config.hidden_size,),
        target_output=True,
    )


def _make_residual_unit_at_token(
    pipeline, layer: int, token_index: int
) -> ResidualStream:
    """``ResidualStream`` reading a FIXED token index (unpadded-frame; the left-pad
    shift in ``index_component`` rebases it into the padded batch). Reading a
    *non-final* token is what surfaces the left-pad ``position_ids`` bug."""
    tp = TokenPosition(lambda trace: [token_index], pipeline, id=f"tok{token_index}")
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        shape=(pipeline.model.config.hidden_size,),
        target_output=True,
    )


class TestCollectFeaturesProperty:
    """Tier-property invariants for ``collect_features`` on a real (tiny) ``IntervenableModel``.

    These tests exercise pyvene's
    ``((base_outputs, collected_activations), cf_outputs)`` envelope
    end-to-end against ``tests/neural/conftest.py::tiny_pipeline`` — the same
    shape contract the production runners hit. ``collect_features`` now pins
    ``model.eval()`` + ``torch.no_grad()`` for the duration of collection;
    these tests assert the resulting tensors have no autograd graph.
    """

    pytestmark = pytest.mark.property

    def test_sample_count_matches_dataset_length(self, tiny_pipeline) -> None:
        dataset = [{"input": _trace(f"hello {i}")} for i in range(5)]
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [unit], batch_size=2)
        # One scalar position per example × 5 examples → (5, hidden).
        assert result[unit.id].shape[0] == len(dataset)

    def test_tensors_are_on_cpu(self, tiny_pipeline) -> None:
        dataset = [{"input": _trace("hello world")}]
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [unit], batch_size=1)
        for t in result.values():
            assert t.device.type == "cpu"

    def test_deterministic_under_fixed_seed(self, tiny_pipeline) -> None:
        """Calling twice on the same dataset → byte-identical tensors."""
        dataset = [{"input": _trace(f"hello {i}")} for i in range(3)]
        unit_a = _make_residual_unit(tiny_pipeline, layer=0)
        out_a = collect_features(dataset, tiny_pipeline, [unit_a], batch_size=2)
        unit_b = _make_residual_unit(tiny_pipeline, layer=0)
        out_b = collect_features(dataset, tiny_pipeline, [unit_b], batch_size=2)
        # Same id-template since both units target layer 0 / last_token.
        (key_a,) = out_a.keys()
        (key_b,) = out_b.keys()
        assert torch.equal(out_a[key_a], out_b[key_b])

    def test_no_grad_is_pinned_during_collection(self, tiny_pipeline) -> None:
        """Collected tensors must not carry an autograd graph (forward-only)."""
        dataset = [{"input": _trace("hello world")}]
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        result = collect_features(dataset, tiny_pipeline, [unit], batch_size=1)
        for t in result.values():
            assert t.requires_grad is False
            assert t.grad_fn is None

    def test_dict_keys_equal_unit_ids(self, tiny_pipeline) -> None:
        unit0 = _make_residual_unit(tiny_pipeline, layer=0)
        unit1 = _make_residual_unit(tiny_pipeline, layer=1)
        dataset = [{"input": _trace("hello world")}]
        result = collect_features(dataset, tiny_pipeline, [unit0, unit1], batch_size=1)
        assert set(result.keys()) == {unit0.id, unit1.id}

    @pytest.mark.parametrize(
        "pipeline_fixture", ["tiny_pipeline", "tiny_gpt2_pipeline"]
    )
    def test_left_padded_collection_matches_unpadded(
        self, pipeline_fixture, request
    ) -> None:
        """End-to-end guard for the position_ids fix on a real model.

        Collecting a non-final token must be identical whether each example is
        processed alone (``batch_size=1`` → no padding, the reference) or together
        in one left-padded batch. On an **absolute-position** model
        (``tiny_gpt2_pipeline``) this fails without ``position_ids`` on the
        collection forward — the padded row is mis-encoded (the ROME-replication
        bug). On a **RoPE** model (``tiny_pipeline``) it holds either way, since
        relative positions are invariant to a uniform left-pad shift.
        Parametrizing both pins "GPT-2 is fixed" and "Llama is unchanged" together.
        """
        pipeline = request.getfixturevalue(pipeline_fixture)
        dataset = [
            {"input": _trace("the cat sat quietly on the warm windowsill")},
            {
                "input": _trace(
                    "the quick brown fox jumps over the lazy dog again and again today"
                )
            },
        ]
        # Guards so the test actually exercises the bug: the short row must be
        # left-padded in the batch, and long enough to read a non-final token.
        short_len = int(pipeline.load([dataset[0]["input"]])["input_ids"].shape[1])
        batch_mask = pipeline.load([ex["input"] for ex in dataset])["attention_mask"]
        assert (batch_mask == 0).any(), "batch did not introduce left padding"
        assert short_len >= 3, f"short prompt too short ({short_len} tokens)"

        unit_solo = _make_residual_unit_at_token(pipeline, layer=1, token_index=1)
        unit_batched = _make_residual_unit_at_token(pipeline, layer=1, token_index=1)
        solo = collect_features(dataset, pipeline, [unit_solo], batch_size=1)
        batched = collect_features(dataset, pipeline, [unit_batched], batch_size=2)

        assert torch.allclose(
            solo[unit_solo.id], batched[unit_batched.id], atol=1e-4
        ), (
            f"[{pipeline_fixture}] left-padded collection diverged from unpadded at "
            "a non-final token — position_ids on the collection forward is what "
            "makes these equal on an absolute-position model"
        )

    def test_duplicate_unit_id_raises_on_real_pipeline(self, tiny_pipeline) -> None:
        """Duplicate-id contract must hold against the real pipeline too."""
        unit_a = _make_residual_unit(tiny_pipeline, layer=0)
        unit_b = _make_residual_unit(tiny_pipeline, layer=0)
        # Both targets are layer 0 + last_token → identical ``id``.
        assert unit_a.id == unit_b.id
        dataset = [{"input": _trace("hello")}]
        with pytest.raises(ValueError, match="Duplicate model_unit.id"):
            collect_features(dataset, tiny_pipeline, [unit_a, unit_b], batch_size=1)

    def test_multitoken_span_collects_each_position_independently(
        self, tiny_pipeline
    ) -> None:
        """A uniform multi-token span collects one d-vector PER position.

        With ``keep_last_dim=True`` (PR #334) the collect intervention gathers
        ``(b, num_pos, d)`` and ``collect_features`` flattens it to
        ``(b*num_pos, d)`` — so the span's per-position rows must equal collecting
        each position on its own. The old folded ``(b, num_pos*d)`` reshape
        interleaved positions and features and would not match. This is the one
        consumer of the collect change (review obs #2) the single-token path
        leaves byte-identical; here we lock the *multi-token* path.
        """
        hidden = tiny_pipeline.model.config.hidden_size
        dataset = [{"input": _trace(f"hello world {i}")} for i in range(3)]

        def _unit(positions: list[int], unit_id: str) -> ResidualStream:
            tp = TokenPosition(lambda _x, p=positions: p, tiny_pipeline, id=unit_id)
            return ResidualStream(
                layer=0,
                token_indices=tp,
                target_output=True,
                shape=(hidden,),
            )

        span = _unit([0, 1], "span01")
        pos0 = _unit([0], "pos0")
        pos1 = _unit([1], "pos1")

        out_span = collect_features(dataset, tiny_pipeline, [span], batch_size=2)[
            span.id
        ]
        out_p0 = collect_features(dataset, tiny_pipeline, [pos0], batch_size=2)[pos0.id]
        out_p1 = collect_features(dataset, tiny_pipeline, [pos1], batch_size=2)[pos1.id]

        # (b, num_pos, d) flattened row-major: rows are [ex0-pos0, ex0-pos1, ...].
        assert out_span.shape == (len(dataset) * 2, hidden)
        assert torch.allclose(out_span[0::2], out_p0, atol=1e-5)
        assert torch.allclose(out_span[1::2], out_p1, atol=1e-5)


# --------------------------------------------------------------------------- #
#  collect_batch_representations                                              #
# --------------------------------------------------------------------------- #
class TestCollectBatchRepresentationsUnit:
    """Per-batch primitive producing pyvene-compatible ``source_representations``.

    Returns ``list[Tensor]`` ordered to match the flat-unit traversal across
    ``interchange_target`` groups. ``neural/activations/interchange_mode.py``
    and ``methods/metric.py`` then feed that list to pyvene as
    ``source_representations`` during cross-model patching, so any drift in
    ordering or length silently corrupts the patched outputs.
    """

    pytestmark = pytest.mark.unit

    def test_returns_one_tensor_per_unit_across_groups(self) -> None:
        """Flat output length equals ``sum(len(group) for group in target)``."""
        groups = [
            [MagicMock(id=f"u{i}") for i in range(2)],
            [MagicMock(id="u2")],
        ]
        target = InterchangeTarget(groups)
        pipeline = MagicMock()
        call_idx = {"n": 0}

        def model_side_effect(*args, **kwargs):
            call_idx["n"] += 1
            # Group 0 has 2 units, group 1 has 1 unit.
            num_units = [2, 1][call_idx["n"] - 1]
            acts = [torch.randn(1, 16) for _ in range(num_units)]
            return ((MagicMock(), acts), None)

        mock_im = MagicMock()
        mock_im.side_effect = model_side_effect

        batched_cf = [
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            },
            {
                "input_ids": torch.tensor([[3, 4]]),
                "attention_mask": torch.tensor([[1, 1]]),
            },
        ]
        cf_indices = [[[0]], [[0]], [[0]]]  # one per flat unit
        out = collect_batch_representations(
            pipeline, batched_cf, target, cf_indices, intervenable_model=mock_im
        )
        assert len(out) == 3

    def test_does_not_delete_externally_provided_model(self) -> None:
        """When caller passes ``intervenable_model``, function must NOT delete it."""
        groups = [[MagicMock(id="u0")]]
        target = InterchangeTarget(groups)
        pipeline = MagicMock()
        mock_im = MagicMock()
        mock_im.side_effect = lambda *a, **k: ((MagicMock(), [torch.randn(1, 8)]), None)
        batched_cf = [
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        ]
        cf_indices = [[[0]]]

        with patch(
            "causalab.neural.activations.collect.delete_intervenable_model"
        ) as mock_del:
            _ = collect_batch_representations(
                pipeline,
                batched_cf,
                target,
                cf_indices,
                intervenable_model=mock_im,
            )
        mock_del.assert_not_called()

    def test_owns_and_deletes_model_when_none_passed(self) -> None:
        """When caller omits ``intervenable_model``, function creates AND tears down."""
        groups = [[MagicMock(id="u0", is_static=lambda: True)]]
        target = InterchangeTarget(groups)
        pipeline = MagicMock()
        mock_im = MagicMock()
        mock_im.side_effect = lambda *a, **k: ((MagicMock(), [torch.randn(1, 8)]), None)
        batched_cf = [
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        ]
        cf_indices = [[[0]]]

        with (
            patch(
                "causalab.neural.activations.collect.prepare_intervenable_model",
                return_value=mock_im,
            ) as mock_prep,
            patch(
                "causalab.neural.activations.collect.delete_intervenable_model"
            ) as mock_del,
        ):
            _ = collect_batch_representations(
                pipeline,
                batched_cf,
                target,
                cf_indices,
                intervenable_model=None,
            )
        mock_prep.assert_called_once()
        mock_del.assert_called_once_with(mock_im)

    def test_per_group_indices_are_sliced_by_unit_cursor(self) -> None:
        """``counterfactual_indices`` is sliced ``[unit_idx : unit_idx + n]`` per group."""
        groups = [
            [MagicMock(id="u0"), MagicMock(id="u1")],
            [MagicMock(id="u2")],
        ]
        target = InterchangeTarget(groups)
        pipeline = MagicMock()
        seen_indices: list[list] = []

        def model_side_effect(_inp, unit_locations=None, **kwargs):
            sources, _base = unit_locations["sources->base"]
            seen_indices.append(sources)
            n = len(sources)
            return ((MagicMock(), [torch.randn(1, 8) for _ in range(n)]), None)

        mock_im = MagicMock()
        mock_im.side_effect = model_side_effect

        cf_indices = [
            ["GROUP0_UNIT0"],
            ["GROUP0_UNIT1"],
            ["GROUP1_UNIT0"],
        ]
        batched_cf = [
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])},
            {"input_ids": torch.tensor([[2]]), "attention_mask": torch.tensor([[1]])},
        ]
        _ = collect_batch_representations(
            pipeline, batched_cf, target, cf_indices, intervenable_model=mock_im
        )
        # First call (group 0) saw indices for [u0, u1]; second call (group 1) saw [u2].
        assert seen_indices[0] == [["GROUP0_UNIT0"], ["GROUP0_UNIT1"]]
        assert seen_indices[1] == [["GROUP1_UNIT0"]]


class TestCollectBatchRepresentationsProperty:
    """Length-and-ordering invariants for ``collect_batch_representations``.

    The returned ``list[Tensor]`` is consumed positionally by pyvene against
    ``intervenable_model.sorted_keys``. The two property assertions below pin
    that contract end-to-end on the real (tiny) pipeline.
    """

    pytestmark = pytest.mark.property

    def test_output_length_equals_total_flat_units(self, tiny_pipeline) -> None:
        """Single-group case: 2 units → output list of length 2.

        Multi-group / cross-model patching cases are exercised transitively
        via ``methods/metric.py`` callers.
        """
        unit0 = _make_residual_unit(tiny_pipeline, layer=0)
        unit1 = _make_residual_unit(tiny_pipeline, layer=1)
        target = InterchangeTarget([[unit0, unit1]])

        cf_inputs = [_trace("alpha")]
        batched_cf = [tiny_pipeline.load(cf_inputs)]
        cf_indices = [
            unit0.index_component(cf_inputs, batch=True, is_original=False),
            unit1.index_component(cf_inputs, batch=True, is_original=False),
        ]
        out = collect_batch_representations(
            tiny_pipeline, batched_cf, target, cf_indices
        )
        assert len(out) == len(target.flatten())

    def test_owns_vs_borrows_model_returns_same_lengths(self, tiny_pipeline) -> None:
        """Hoisting the ``intervenable_model`` outside the call must not change output shape."""
        from causalab.neural.activations.intervenable_model import (
            delete_intervenable_model,
            prepare_intervenable_model,
        )

        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        cf_inputs = [_trace("alpha")]
        batched_cf = [tiny_pipeline.load(cf_inputs)]
        cf_indices = [unit.index_component(cf_inputs, batch=True, is_original=False)]

        out_owned = collect_batch_representations(
            tiny_pipeline, batched_cf, target, cf_indices
        )
        im = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="collect"
        )
        try:
            out_borrowed = collect_batch_representations(
                tiny_pipeline,
                batched_cf,
                target,
                cf_indices,
                intervenable_model=im,
            )
        finally:
            delete_intervenable_model(im)

        assert len(out_owned) == len(out_borrowed)
        for a, b in zip(out_owned, out_borrowed):
            assert a.shape == b.shape


# --------------------------------------------------------------------------- #
#  collect_source_representations                                             #
# --------------------------------------------------------------------------- #
class TestCollectSourceRepresentationsUnit:
    """Convenience wrapper: tokenize counterfactuals + delegate to batch primitive.

    Used by ``neural/activations/interchange_mode.py`` and ``methods/metric.py``
    to harvest source-pipeline activations before patching them into a target
    pipeline. The wrapper composes ``pipeline.load`` over zipped
    ``counterfactual_inputs`` and computes per-unit indices with
    ``is_original=False``.
    """

    pytestmark = pytest.mark.unit

    def test_forwards_to_collect_batch_representations(self) -> None:
        pipeline = MagicMock()
        pipeline.load = MagicMock(
            side_effect=lambda group: {
                "_loaded": group,
                "attention_mask": torch.ones(len(group), 1, dtype=torch.long),
            }
        )

        unit = MagicMock(id="u0")
        unit.index_component = MagicMock(return_value=[[0]])
        target = InterchangeTarget([[unit]])
        examples = [
            {"input": "base", "counterfactual_inputs": ["cf_g0_e0"]},
            {"input": "base2", "counterfactual_inputs": ["cf_g0_e1"]},
        ]
        with patch(
            "causalab.neural.activations.collect.collect_batch_representations",
            return_value=[torch.randn(1, 8)],
        ) as mock_delegate:
            out = collect_source_representations(
                pipeline,
                examples,
                target,
                source_intervenable_model=None,
            )

        # Exactly one delegation: the wrapper does not loop.
        assert mock_delegate.call_count == 1
        _, kwargs = mock_delegate.call_args
        assert kwargs["intervenable_model"] is None
        assert out is mock_delegate.return_value

    def test_passes_through_prebuilt_intervenable_model(self) -> None:
        """A caller-provided ``source_intervenable_model`` reaches the delegate."""
        pipeline = MagicMock()
        pipeline.load = MagicMock(
            side_effect=lambda group: {
                "_loaded": group,
                "attention_mask": torch.ones(len(group), 1, dtype=torch.long),
            }
        )

        unit = MagicMock(id="u0")
        unit.index_component = MagicMock(return_value=[[0]])
        target = InterchangeTarget([[unit]])
        examples = [{"input": "b", "counterfactual_inputs": ["cf"]}]
        sentinel = object()

        with patch(
            "causalab.neural.activations.collect.collect_batch_representations",
            return_value=[],
        ) as mock_delegate:
            _ = collect_source_representations(
                pipeline,
                examples,
                target,
                source_intervenable_model=sentinel,  # type: ignore[arg-type]
            )

        _, kwargs = mock_delegate.call_args
        assert kwargs["intervenable_model"] is sentinel

    def test_index_component_called_with_is_original_false(self) -> None:
        """Counterfactual indices use ``is_original=False`` (vs. base inputs)."""
        pipeline = MagicMock()
        pipeline.load = MagicMock(
            side_effect=lambda group: {
                "_loaded": group,
                "attention_mask": torch.ones(len(group), 1, dtype=torch.long),
            }
        )

        unit = MagicMock(id="u0")
        unit.index_component = MagicMock(return_value=[[0]])
        target = InterchangeTarget([[unit]])
        examples = [{"input": "b", "counterfactual_inputs": ["cf_a"]}]

        with patch(
            "causalab.neural.activations.collect.collect_batch_representations",
            return_value=[],
        ):
            _ = collect_source_representations(pipeline, examples, target)

        called_kwargs = unit.index_component.call_args.kwargs
        assert called_kwargs["is_original"] is False
        assert called_kwargs["batch"] is True


class TestCollectSourceRepresentationsProperty:
    """End-to-end invariants for the tokenize-and-collect wrapper on real pipelines."""

    pytestmark = pytest.mark.property

    def test_output_length_equals_total_flat_units(self, tiny_pipeline) -> None:
        """Single-group case: 2 units → output list of length 2.

        Multi-group counterfactual setups are exercised transitively via
        ``neural/activations/interchange_mode.py``'s cross-model patching path.
        """
        unit0 = _make_residual_unit(tiny_pipeline, layer=0)
        unit1 = _make_residual_unit(tiny_pipeline, layer=1)
        target = InterchangeTarget([[unit0, unit1]])
        examples = [
            {
                "input": _trace("base_a"),
                "counterfactual_inputs": [_trace("cf_a")],
            },
            {
                "input": _trace("base_b"),
                "counterfactual_inputs": [_trace("cf_b")],
            },
        ]
        out = collect_source_representations(tiny_pipeline, examples, target)
        assert len(out) == len(target.flatten())

    def test_prebuilt_model_yields_same_shapes(self, tiny_pipeline) -> None:
        """Wrapper output is invariant to caller pre-building the intervenable model."""
        from causalab.neural.activations.intervenable_model import (
            delete_intervenable_model,
            prepare_intervenable_model,
        )

        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        examples = [
            {"input": _trace("base"), "counterfactual_inputs": [_trace("cf_a")]},
        ]

        out_no_model = collect_source_representations(tiny_pipeline, examples, target)
        im = prepare_intervenable_model(
            tiny_pipeline, target, intervention_type="collect"
        )
        try:
            out_with_model = collect_source_representations(
                tiny_pipeline,
                examples,
                target,
                source_intervenable_model=im,
            )
        finally:
            delete_intervenable_model(im)

        assert len(out_no_model) == len(out_with_model)
        for a, b in zip(out_no_model, out_with_model):
            assert a.shape == b.shape


# --------------------------------------------------------------------------- #
#  collect_class_centroids                                                    #
# --------------------------------------------------------------------------- #
class TestCollectClassCentroidsUnit:
    """Per-variable-value centroid reducer feeding path-steering visualisation.

    Wraps ``collect_features`` on a single unit then averages the per-sample
    feature vectors by class index (``task.intervention_values``-aligned),
    returning ``(centroids[n_classes, k], mask[n_classes])``. Consumed by
    ``analyses/path_steering/path_visualization.py``.
    """

    pytestmark = pytest.mark.unit

    def test_returns_centroids_and_mask_with_correct_shapes(self) -> None:
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
            {"input": {"v": "C"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B", "C"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids, mask = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        assert centroids.shape == (3, 3)
        assert mask.shape == (3,)
        assert mask.dtype == torch.bool

    def test_centroids_average_only_matching_samples(self) -> None:
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        # Class A has samples 0, 1; class B has sample 2; class C absent.
        features = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 5.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B", "C"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids, mask = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        # Class A: mean of rows 0 and 1 = [2.0, 0.0].
        assert torch.allclose(centroids[0], torch.tensor([2.0, 0.0]))
        # Class B: just row 2 = [0.0, 5.0].
        assert torch.allclose(centroids[1], torch.tensor([0.0, 5.0]))
        # Class C: no samples → mask False, zero row.
        assert not mask[2]
        assert torch.equal(centroids[2], torch.zeros(2))

    def test_mask_true_only_for_classes_with_at_least_one_sample(self) -> None:
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor([[1.0, 2.0]])
        samples = [{"input": {"v": "A"}}]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            _, mask = collect_class_centroids(samples, MagicMock(), target, task)

        assert mask.tolist() == [True, False]


class TestCollectClassCentroidsProperty:
    """Algebraic invariants of the centroid reducer."""

    pytestmark = pytest.mark.property

    def test_permutation_invariance_in_sample_order(self) -> None:
        """Reordering ``filtered_samples`` (and ``features`` to match) preserves centroids."""
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 3.0],
                [0.0, 4.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
            {"input": {"v": "B"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids_orig, mask_orig = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        # Reverse both ``samples`` and ``features`` together.
        perm = list(reversed(range(len(samples))))
        samples_perm = [samples[i] for i in perm]
        features_perm = features[perm]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features_perm},
        ):
            centroids_perm, mask_perm = collect_class_centroids(
                samples_perm, MagicMock(), target, task
            )

        assert torch.allclose(centroids_orig, centroids_perm)
        assert torch.equal(mask_orig, mask_perm)

    def test_single_sample_centroid_equals_that_sample(self) -> None:
        """With one sample per class, each centroid row equals that sample's features."""
        unit = MagicMock(id="unit0")
        target = InterchangeTarget([[unit]])
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        samples = [
            {"input": {"v": "A"}},
            {"input": {"v": "B"}},
        ]
        task = MagicMock()
        task.intervention_variable = "v"
        task.intervention_values = ["A", "B"]

        with patch(
            "causalab.neural.activations.collect.collect_features",
            return_value={"unit0": features},
        ):
            centroids, mask = collect_class_centroids(
                samples, MagicMock(), target, task
            )

        assert torch.allclose(centroids[0], features[0])
        assert torch.allclose(centroids[1], features[1])
        assert bool(mask.all().item()) is True
