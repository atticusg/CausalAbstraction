"""Tests for ``causalab.neural.activations.intervenable_model``.

This module wraps a :class:`~causalab.neural.pipeline.Pipeline`'s base HF
model into a :class:`pyvene.IntervenableModel` configured from a flat list
of :class:`~causalab.neural.units.AtomicModelUnit` (or an
:class:`~causalab.neural.units.InterchangeTarget`), and keeps hook devices
coherent across single-device and sharded (``hf_device_map``) models. It is
the entry point used by every activation-collection / interchange /
interpolation path
(``causalab/neural/activations/{collect,interchange_mode,interpolate}.py``)
and the intervention-style methods
(``methods/interchange/layer_scan.py``, ``methods/steer/{steer,collect}.py``,
``methods/trained_subspace/train.py``, ``analyses/pullback/main.py``). A
wrong intervention config or mis-targeted device means every downstream
interchange / steering analysis silently produces cross-device errors or
incorrect interventions.
"""

from __future__ import annotations

from typing import Any

import pyvene as pv  # type: ignore[import-untyped]
import pytest
import torch

from causalab.neural.LM_units import AttentionHeadValue, ResidualStream
from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
    device_for_layer,
    prepare_intervenable_model,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import ComponentIndexer, InterchangeTarget


# --------------------------------------------------------------------------- #
#  Local helpers                                                              #
# --------------------------------------------------------------------------- #
def _hidden_size(pipeline: LMPipeline) -> int:
    """Hidden size of the tiny pipeline's backing model."""
    return int(pipeline.model.config.hidden_size)


def _make_static_unit(pipeline: LMPipeline, layer: int) -> ResidualStream:
    """Build a single static-index ``ResidualStream`` at ``layer``."""
    return ResidualStream(
        layer=layer,
        token_indices=[0],
        shape=(_hidden_size(pipeline),),
        target_output=True,
    )


def _make_dynamic_unit(pipeline: LMPipeline, layer: int) -> ResidualStream:
    """Build a single dynamic-index ``ResidualStream`` at ``layer``.

    The ``ComponentIndexer`` makes ``is_static()`` return ``False``, which is
    what flips ``IntervenableModel.use_fast`` off in
    :func:`prepare_intervenable_model`.
    """

    def _indices(_: Any) -> list[int]:
        return [0]

    indexer = ComponentIndexer(_indices, id="last_token_dyn")
    return ResidualStream(
        layer=layer,
        token_indices=indexer,
        shape=(_hidden_size(pipeline),),
        target_output=True,
    )


def _intervention_buffer_devices(
    intervenable_model: pv.IntervenableModel,
) -> list[torch.device]:
    """Collect the devices of every buffer carried by every intervention.

    Identity featurizers carry no nn.Parameter; their `to(device)` only
    moves buffers (e.g. the mask tensor on ``FeatureInterchangeIntervention``).
    Including buffers makes the device assertion non-vacuous.
    """
    devices: list[torch.device] = []
    for intervention in intervenable_model.interventions.values():
        for buf in intervention.buffers():
            devices.append(buf.device)
        for param in intervention.parameters():
            devices.append(param.device)
    return devices


# --------------------------------------------------------------------------- #
#  prepare_intervenable_model — unit tier                                     #
# --------------------------------------------------------------------------- #
class TestPrepareIntervenableModelUnit:
    """Construct a real ``pv.IntervenableModel`` from a tiny LM pipeline.

    Asserts the construction surface that downstream collect / interchange
    paths rely on: return type, ``use_fast`` selection, and one config per
    model unit. Uses real pyvene against the tiny-random Llama stub — no
    ``pv.*`` mocking — because docs/TESTS.md's mocking policy forbids mocking
    between two numerical pieces we own.
    """

    pytestmark = pytest.mark.unit

    def test_returns_pyvene_intervenable_model(self, tiny_pipeline: LMPipeline) -> None:
        unit = _make_static_unit(tiny_pipeline, layer=0)
        result = prepare_intervenable_model(tiny_pipeline, [unit])
        assert isinstance(result, pv.IntervenableModel)

    def test_use_fast_true_for_all_static_units(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        units = [_make_static_unit(tiny_pipeline, layer=i) for i in range(2)]
        im = prepare_intervenable_model(tiny_pipeline, units)
        assert im.use_fast is True

    def test_one_intervention_per_unit(self, tiny_pipeline: LMPipeline) -> None:
        units = [_make_static_unit(tiny_pipeline, layer=i) for i in range(2)]
        im = prepare_intervenable_model(tiny_pipeline, units)
        assert len(im.interventions) == len(units)

    def test_accepts_interchange_target_wrapping(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        """A single-group ``InterchangeTarget`` and a flat list must produce
        the same set of intervention keys (shared counterfactual group)."""
        units = [_make_static_unit(tiny_pipeline, layer=i) for i in range(2)]
        target = InterchangeTarget([units])
        im_flat = prepare_intervenable_model(tiny_pipeline, units)
        im_target = prepare_intervenable_model(tiny_pipeline, target)
        assert set(im_flat.interventions.keys()) == set(im_target.interventions.keys())

    @pytest.mark.parametrize("intervention_type", ["interchange", "collect"])
    def test_intervention_type_dispatch(
        self, tiny_pipeline: LMPipeline, intervention_type: str
    ) -> None:
        """Pin the dispatch surface for the two intervention types shipped
        baselines actually exercise.

        The plan calls out that the full set is six (``interchange, collect,
        mask, add, replace, interpolation``); only these two are wired into
        the runner pipeline today.
        """
        unit = _make_static_unit(tiny_pipeline, layer=0)
        im = prepare_intervenable_model(
            tiny_pipeline, [unit], intervention_type=intervention_type
        )
        assert isinstance(im, pv.IntervenableModel)
        assert len(im.interventions) == 1


# --------------------------------------------------------------------------- #
#  prepare_intervenable_model — property tier                                 #
# --------------------------------------------------------------------------- #
class TestPrepareIntervenableModelProperty:
    """Invariants on ``prepare_intervenable_model``'s configuration logic.

    * ``use_fast`` tracks ``is_static()`` of every unit.
    * Flat-list and single-group ``InterchangeTarget`` inputs are
      configuration-equivalent.
    * Sharded (``hf_device_map``) path lands interventions on the mapped
      device and disables pyvene's ``get_device()`` so gather/scatter fall
      back to the layer-tensor device.
    """

    pytestmark = pytest.mark.property

    def test_use_fast_flips_false_with_any_dynamic_unit(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        static_unit = _make_static_unit(tiny_pipeline, layer=0)
        dynamic_unit = _make_dynamic_unit(tiny_pipeline, layer=1)
        im = prepare_intervenable_model(tiny_pipeline, [static_unit, dynamic_unit])
        assert im.use_fast is False

    def test_flat_list_matches_single_group_target_length(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        units = [_make_static_unit(tiny_pipeline, layer=i) for i in range(2)]
        target = InterchangeTarget([units])
        im_flat = prepare_intervenable_model(tiny_pipeline, units)
        im_target = prepare_intervenable_model(tiny_pipeline, target)
        assert len(im_flat.interventions) == len(im_target.interventions) == len(units)

    def test_sharded_model_sets_get_device_to_none(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``pipeline.model`` carries an ``hf_device_map`` attribute,
        the wrapper installs ``get_device = lambda: None`` so pyvene's
        gather/scatter falls back to the layer tensor's actual device
        instead of the embedding GPU.
        """
        synthetic_map = {
            "model.layers.0": "cpu",
            "model.layers.1": "cpu",
        }
        monkeypatch.setattr(
            tiny_pipeline.model, "hf_device_map", synthetic_map, raising=False
        )
        units = [_make_static_unit(tiny_pipeline, layer=i) for i in range(2)]
        im = prepare_intervenable_model(tiny_pipeline, units)
        assert im.get_device() is None

    def test_sharded_model_moves_interventions_to_mapped_device(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sharded path's observable effect: every intervention's buffers /
        parameters land on the mapped device for the layer it hooks. We
        synthesize an ``hf_device_map`` pointing at CPU so the test runs
        on any host — the device-resolution contract is independent of
        whether multiple GPUs exist.
        """
        synthetic_map = {
            "model.layers.0": "cpu",
            "model.layers.1": "cpu",
        }
        monkeypatch.setattr(
            tiny_pipeline.model, "hf_device_map", synthetic_map, raising=False
        )
        units = [_make_static_unit(tiny_pipeline, layer=i) for i in range(2)]
        im = prepare_intervenable_model(tiny_pipeline, units)

        devices = _intervention_buffer_devices(im)
        # Tiny ResidualStream + identity featurizer carries buffers (the mask
        # tensor on FeatureInterchangeIntervention). The sharded branch must
        # have moved them onto CPU per the synthetic map.
        assert devices, "expected interventions to carry at least one buffer"
        for d in devices:
            assert d.type == "cpu", f"intervention buffer on {d}, expected cpu"


# --------------------------------------------------------------------------- #
#  device_for_layer — unit tier                                               #
# --------------------------------------------------------------------------- #
class TestDeviceForLayerUnit:
    """Direct tests for ``device_for_layer``.

    On a single-device pipeline this returns ``pipeline.model.device``; on a
    sharded pipeline it returns the mapped device for the requested layer.
    Consumers (steering vectors, featurizers) rely on this to keep tensors
    on the same device as the layer's residual stream.
    """

    pytestmark = pytest.mark.unit

    def test_single_device_returns_model_device(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        result = device_for_layer(tiny_pipeline, 0)
        assert isinstance(result, torch.device)
        assert result == tiny_pipeline.model.device

    def test_sharded_returns_mapped_device_per_layer(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synthetic_map = {
            "model.layers.3": "cpu",
            "model.layers.7": "meta",
        }
        monkeypatch.setattr(
            tiny_pipeline.model, "hf_device_map", synthetic_map, raising=False
        )
        assert device_for_layer(tiny_pipeline, 3) == torch.device("cpu")
        assert device_for_layer(tiny_pipeline, 7) == torch.device("meta")


# --------------------------------------------------------------------------- #
#  device_for_layer — property tier                                           #
# --------------------------------------------------------------------------- #
class TestDeviceForLayerProperty:
    """Invariants for ``device_for_layer``.

    * Return type is always ``torch.device`` (never ``str``).
    * Ancestor-path fallback succeeds for any layer when only a coarser key
      (``"model"``) is present in ``hf_device_map`` — the dotted-path walk
      must never raise.
    """

    pytestmark = pytest.mark.property

    def test_return_type_is_torch_device(self, tiny_pipeline: LMPipeline) -> None:
        assert isinstance(device_for_layer(tiny_pipeline, 0), torch.device)

    def test_return_type_is_torch_device_sharded(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tiny_pipeline.model,
            "hf_device_map",
            {"model.layers.0": "cpu"},
            raising=False,
        )
        result = device_for_layer(tiny_pipeline, 0)
        assert isinstance(result, torch.device)

    @pytest.mark.parametrize("layer", [0, 3, 31, 77])
    def test_ancestor_fallback_returns_coarse_key_device(
        self,
        tiny_pipeline: LMPipeline,
        monkeypatch: pytest.MonkeyPatch,
        layer: int,
    ) -> None:
        """When ``hf_device_map`` only has the ancestor key ``"model"``
        (no per-layer keys), the function must walk up the dotted path and
        return the ancestor's device for any layer index, without raising.
        """
        monkeypatch.setattr(
            tiny_pipeline.model, "hf_device_map", {"model": "cpu"}, raising=False
        )
        assert device_for_layer(tiny_pipeline, layer) == torch.device("cpu")


# --------------------------------------------------------------------------- #
#  delete_intervenable_model — unit tier                                      #
# --------------------------------------------------------------------------- #
class TestDeleteIntervenableModelUnit:
    """Direct tests for ``delete_intervenable_model``.

    Memory cleanup helper used by every interchange / steering analysis
    once it finishes a sweep. Contract:
      * returns ``None``
      * moves intervention tensors to CPU before dropping the reference
      * never raises when CUDA is unavailable.
    """

    pytestmark = pytest.mark.unit

    def test_returns_none(self, tiny_pipeline: LMPipeline) -> None:
        unit = _make_static_unit(tiny_pipeline, layer=0)
        im = prepare_intervenable_model(tiny_pipeline, [unit])
        assert delete_intervenable_model(im) is None

    def test_interventions_moved_to_cpu(self, tiny_pipeline: LMPipeline) -> None:
        """After the call, every intervention buffer / parameter lives on
        CPU, regardless of the pipeline's original device.
        """
        unit = _make_static_unit(tiny_pipeline, layer=0)
        im = prepare_intervenable_model(tiny_pipeline, [unit])

        delete_intervenable_model(im)

        devices = _intervention_buffer_devices(im)
        assert devices, "expected interventions to carry at least one buffer"
        for d in devices:
            assert d.type == "cpu", f"intervention buffer on {d}, expected cpu"

    def test_no_exception_without_cuda(
        self, tiny_pipeline: LMPipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The empty-cache branch is skipped when CUDA is unavailable — the
        function must still return cleanly. This guards against a refactor
        that drops the ``torch.cuda.is_available()`` guard.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        unit = _make_static_unit(tiny_pipeline, layer=0)
        im = prepare_intervenable_model(tiny_pipeline, [unit])
        delete_intervenable_model(im)


# --------------------------------------------------------------------------- #
#  delete_intervenable_model — property tier                                  #
# --------------------------------------------------------------------------- #
class TestDeleteIntervenableModelProperty:
    """Invariants for ``delete_intervenable_model``.

    The helper calls ``set_device("cpu", set_model=False)`` — the
    ``set_model=False`` is load-bearing: the wrapped base model must not be
    moved off its device, otherwise the next pipeline forward would fail or
    silently relocate the weights. Asserted indirectly by checking that the
    pipeline can still produce logits after cleanup.
    """

    pytestmark = pytest.mark.property

    def test_pipeline_forward_still_works_after_cleanup(
        self, tiny_pipeline: LMPipeline
    ) -> None:
        device_before = tiny_pipeline.model.device

        unit = _make_static_unit(tiny_pipeline, layer=0)
        im = prepare_intervenable_model(tiny_pipeline, [unit])
        delete_intervenable_model(im)

        # The wrapped base model must not have moved.
        assert tiny_pipeline.model.device == device_before

        # And a forward pass through the bare model still works.
        enc = tiny_pipeline.tokenizer(["hello"], return_tensors="pt")
        input_ids = enc["input_ids"].to(device_before)
        attention_mask = enc["attention_mask"].to(device_before)
        with torch.no_grad():
            out = tiny_pipeline.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        assert out.logits.shape[0] == 1


# --------------------------------------------------------------------------- #
#  Decoupled-head_dim guard (#386) — property tier                            #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def decoupled_head_dim_pipeline() -> LMPipeline:
    """A tiny-random Llama whose ``head_dim`` is *decoupled* from
    ``hidden_size // num_attention_heads`` (like Qwen3), for exercising the
    per-head guard without a GPU / a 4B model."""
    from transformers import AutoConfig, LlamaForCausalLM

    from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME

    config = AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME)
    config.head_dim = (config.hidden_size // config.num_attention_heads) + 2
    assert config.head_dim != config.hidden_size // config.num_attention_heads
    model = LlamaForCausalLM(config)  # random init — CPU, guard only
    model.eval()
    return LMPipeline(model_or_name=model, max_new_tokens=1, padding_side="left")


class TestHeadDimGuardProperty:
    """`prepare_intervenable_model` must fail loudly (not silently mis-slice) when
    a per-head unit targets a decoupled-`head_dim` model — pyvene 0.1.8 slices
    per-head vectors at `hidden // n_head`, so the returned activation would be the
    wrong width on e.g. Qwen3 (#386). Guard is head-specific and no-ops on the
    common coupled case."""

    pytestmark = pytest.mark.property

    def test_head_unit_on_decoupled_head_dim_raises(
        self, decoupled_head_dim_pipeline: LMPipeline
    ) -> None:
        unit = AttentionHeadValue(layer=0, head=0, token_indices=[0])
        with pytest.raises(NotImplementedError, match="decoupled head_dim"):
            prepare_intervenable_model(
                decoupled_head_dim_pipeline, [unit], intervention_type="collect"
            )

    def test_residual_unit_on_decoupled_head_dim_ok(
        self, decoupled_head_dim_pipeline: LMPipeline
    ) -> None:
        """The guard is head-specific: a non-head (residual) unit builds fine even
        on a decoupled-`head_dim` model."""
        unit = ResidualStream(
            layer=0,
            token_indices=[0],
            shape=(_hidden_size(decoupled_head_dim_pipeline),),
            target_output=True,
        )
        model = prepare_intervenable_model(
            decoupled_head_dim_pipeline, [unit], intervention_type="collect"
        )
        delete_intervenable_model(model)

    def test_head_unit_on_coupled_model_ok(self, tiny_pipeline: LMPipeline) -> None:
        """Regression: the guard must NOT fire on the common coupled case
        (`head_dim == hidden // n_head`), so head units keep working there."""
        unit = AttentionHeadValue(layer=0, head=0, token_indices=[0])
        model = prepare_intervenable_model(
            tiny_pipeline, [unit], intervention_type="collect"
        )
        delete_intervenable_model(model)
