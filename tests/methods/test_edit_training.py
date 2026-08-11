"""Tests for :mod:`causalab.methods.edit_training` — the ED3 outer
optimization loop, moved from ``causalab/neural/trainable.py`` per
docs/CODEBASE.md §3 invariant 1 (#449 finding 3).

Tiers (``causalab/methods`` owes ``numerical_unit`` + ``property`` direct):

* ``numerical_unit`` — the temperature schedule's pinned shape/endpoint
  values on fixed inputs.
* ``unit`` — loop input validation.
* ``property`` — mini DBM/DAS training runs on a fresh tiny Llama where the
  loss actually decreases, the anneal reaches its end temperature, and the
  trained state moves. (The per-step *grad contract* itself stays pinned in
  ``tests/neural/test_trainable.py`` against the grad-enabled hook oracle.)
* ``golden`` — a real DBM training step on the coherent Qwen3-4B backbone
  (GPU): placement puts the gate on the site's device and a step produces
  finite loss + grads there; nightly only.

Hyperparameters are passed explicitly at every callsite — the signatures
carry no defaults (invariant 5; canonical values live in
``configs/train_config.py``'s ``DEFAULT_CONFIG``).
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

from causalab.methods.edit_training import (
    TrainBatch,
    temperature_schedule,
    train_edits,
)
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import MaskGate
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.trainable import (
    concat_label_inputs,
    das_edit,
    dbm_edit,
    selected_feature_ids,
)

from tests._helpers.tiny import fresh_tiny_random_llama


def _subspace(
    width: int, k: int, *, trainable: bool, seed: int = 0
) -> SubspaceFeaturizer:
    torch.manual_seed(seed)
    return SubspaceFeaturizer(shape=(width, k), trainable=trainable)


def _trainable_params(feat: SubspaceFeaturizer) -> list[torch.nn.Parameter]:
    return [p for p in feat.featurizer.parameters() if p.requires_grad]


def _trace(text: str) -> Any:
    from causalab.causal.trace import CausalTrace, Mechanism

    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


# --------------------------------------------------------------------------- #
#  numerical_unit — the schedule's pinned values                               #
# --------------------------------------------------------------------------- #
class TestTemperatureScheduleNumericalUnit:
    pytestmark = pytest.mark.numerical_unit

    def test_temperature_schedule_shape_and_endpoints(self) -> None:
        temps = temperature_schedule(1.0, 0.001, 10, annealing_fraction=0.5)
        assert temps.shape == (10,)
        assert temps[0] == pytest.approx(1.0)
        torch.testing.assert_close(temps[5:], torch.full((5,), 0.001))

    def test_temperature_schedule_edge_fractions(self) -> None:
        torch.testing.assert_close(
            temperature_schedule(1.0, 0.1, 4, annealing_fraction=0.0),
            torch.full((4,), 0.1),
        )
        assert temperature_schedule(1.0, 0.1, 4, annealing_fraction=1.0).shape == (4,)
        with pytest.raises(ValueError, match="total_steps"):
            temperature_schedule(1.0, 0.1, 0, annealing_fraction=0.5)


# --------------------------------------------------------------------------- #
#  unit — loop input validation                                                #
# --------------------------------------------------------------------------- #
class TestTrainEditsValidationUnit:
    pytestmark = pytest.mark.unit

    def test_train_edits_input_validation(self) -> None:
        with pytest.raises(ValueError, match="at least one batch"):
            train_edits(
                None,
                [],
                pad_token_id=0,
                epochs=1,
                lr=1e-3,
                temperature=(1.0, 0.001),
                annealing_fraction=0.5,
                regularization_coefficient=0.0,
            )


# --------------------------------------------------------------------------- #
#  property — the loop actually trains                                         #
# --------------------------------------------------------------------------- #
class TestTrainingRuns:
    pytestmark = pytest.mark.property

    @pytest.fixture(scope="class")
    def pipe(self) -> LMPipeline:
        raw, _tok = fresh_tiny_random_llama()
        return LMPipeline(raw, max_new_tokens=1, position_ids=True)

    def _batches(
        self, pipe: LMPipeline, make_edit: Callable[[torch.Tensor], Edit]
    ) -> list[TrainBatch]:
        """Self-consistent objective: each label is the model's *own* greedy
        next token, so the clean forward already minimizes the CE — an edit
        perturbs it, and training has a true descent direction (a DBM gate can
        reach the all-base no-op exactly; a DAS rotation moves toward the
        least-damaging subspace). Arbitrary labels on a random-weight model
        carry almost no learnable signal."""
        texts = [
            ["the quick brown fox", "a cat sat on"],
            ["every valley shall", "few things are"],
        ]
        batches = []
        for t in texts:
            base = pipe.load([_trace(x) for x in t])
            with torch.no_grad():
                clean = pipe.hf_model(
                    input_ids=base["input_ids"],
                    attention_mask=base["attention_mask"],
                ).logits[:, -1, :]
            labels = [pipe.tokenizer.decode(i) for i in clean.argmax(dim=-1).tolist()]
            joint, label_ids = concat_label_inputs(pipe, dict(base), labels)
            raw_src = Site("block_output", 0).collect(
                pipe.model, {k: joint[k] for k in ("input_ids", "attention_mask")}
            )[:, -2:-1]
            batches.append(
                TrainBatch(
                    inputs=joint, label_ids=label_ids, edits=(make_edit(raw_src),)
                )
            )
        return batches

    def test_dbm_training_decreases_loss_and_anneals(self, pipe: LMPipeline) -> None:
        d = int(pipe.hf_model.config.hidden_size)
        feat = _subspace(d, 4, trainable=False)
        gate = MaskGate(4).train()
        fsite = FeaturizedSite(Site("block_output", 1), feat)
        batches = self._batches(
            pipe, lambda raw: dbm_edit(fsite, raw, gate, positions=[-2])
        )
        history = train_edits(
            pipe.model,
            batches,
            pad_token_id=pipe.tokenizer.pad_token_id,
            epochs=8,
            lr=0.05,
            gates=[gate],
            temperature=(1.0, 0.01),
            annealing_fraction=0.5,
            regularization_coefficient=0.01,
        )
        assert len(history) == 16
        assert history[-1]["loss"] < history[0]["loss"]
        assert gate.temperature is not None
        assert gate.temperature.item() == pytest.approx(0.01)  # annealed to the end
        ids = selected_feature_ids(gate)
        assert ids is None or isinstance(ids, list)

    def test_das_training_decreases_loss_and_moves_rotation(
        self, pipe: LMPipeline
    ) -> None:
        d = int(pipe.hf_model.config.hidden_size)
        feat = _subspace(d, 2, trainable=True)
        fsite = FeaturizedSite(Site("block_output", 1), feat)
        before = [p.detach().clone() for p in _trainable_params(feat)]
        batches = self._batches(pipe, lambda raw: das_edit(fsite, raw, positions=[-2]))
        history = train_edits(
            pipe.model,
            batches,
            pad_token_id=pipe.tokenizer.pad_token_id,
            epochs=8,
            lr=0.05,
            temperature=(1.0, 0.001),
            annealing_fraction=0.5,
            regularization_coefficient=0.0,
        )
        assert history[-1]["loss"] < history[0]["loss"]
        after = _trainable_params(feat)
        assert any(
            not torch.allclose(b, a.detach()) for b, a in zip(before, after)
        )  # the rotation actually moved


# --------------------------------------------------------------------------- #
#  golden — a real training step on the coherent GPU backbone                  #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    pytestmark = pytest.mark.golden

    def test_dbm_step_places_and_trains_on_coherent_model(self) -> None:
        """Freeze + place + two DBM steps on Qwen3-4B: the gate lands on the
        site's layer device, the loss is finite, and its grads arrive there."""
        pipe = LMPipeline(
            "Qwen/Qwen3-4B-Instruct-2507",
            max_new_tokens=2,
            position_ids=True,
            device_map="auto",
        )
        assert all(not p.requires_grad for p in pipe.hf_model.parameters())
        model = pipe.model
        layer = int(model.num_layers) // 2
        d = int(pipe.hf_model.config.hidden_size)
        feat = _subspace(d, 8, trainable=False)
        gate = MaskGate(8).train()
        fsite = FeaturizedSite(Site("block_output", layer), feat)

        base = pipe.load([_trace("The capital of France is")])
        joint, label_ids = concat_label_inputs(pipe, dict(base), [" Paris"])
        raw_src = Site("block_output", layer - 1).collect(
            model, {k: joint[k] for k in ("input_ids", "attention_mask")}
        )[:, -3:-2]
        batch = TrainBatch(
            inputs=joint,
            label_ids=label_ids,
            edits=(dbm_edit(fsite, raw_src, gate, positions=[-3]),),
        )
        history = train_edits(
            model,
            [batch],
            pad_token_id=pipe.tokenizer.pad_token_id,
            epochs=2,
            lr=0.01,
            gates=[gate],
            temperature=(1.0, 0.01),
            annealing_fraction=0.5,
            regularization_coefficient=0.0,
        )
        layer_device = next(model.model.layers[layer].parameters()).device
        assert gate.mask.device == layer_device  # explicit placement, no monkeypatch
        assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)
