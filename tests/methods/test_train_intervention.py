"""Training-loop orchestration on the ED3 stack (MX2, #409).

``_run_training_loop`` composes :mod:`causalab.neural.trainable`'s primitives
(``das_edit`` / ``dbm_edit`` around shared featurizer/gate modules,
``traced_label_loss`` under the saved-logits grad contract,
``score_label_predictions`` for the in-loop accuracy) and owns only the
orchestration around them. These tests pin that orchestration on the spec
surface (WU4, #506 — initialization and the trained readout are functional):

* ``property`` — real tiny-model runs: DAS actually moves the rotation, DBM
  *returns* the hard-threshold ``feature_indices`` readout and trained specs
  (per-feature list untied, ``None``/``[]`` tied; an all-off mask drops its
  spec — the no-op-by-omission contract).
* ``unit`` — scripted-boundary pins (``traced_label_loss`` patched at the
  module seam): early stopping counts epochs-without-improvement, memory
  cleanup fires per ``memory_cleanup_freq``.

The pinned numerics of the primitives themselves (grad parity vs the raw-hook
oracle, loss-slice faithfulness, anneal-to-end) live in
``tests/neural/test_trainable.py``; the value-pinned end-to-end contract is
the ``weekdays_subspace`` golden.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.configs.train_config import merge_with_defaults
from causalab.methods.trained_subspace.train import (
    _initialize_featurizers,  # pyright: ignore[reportPrivateUsage]
    _run_training_loop,  # pyright: ignore[reportPrivateUsage]
)
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec

from tests._helpers.tiny import fresh_tiny_random_llama

_HIDDEN = 16  # tiny-random Llama hidden size
_KEY = "residual.L0.p0"


def _sample(x: str) -> CausalTrace:
    text = f"The next number after {x} is "
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _labeled_dataset(n: int = 4) -> list[dict[str, Any]]:
    """Labeled counterfactual examples — same-length single-digit prompts so
    the batch has no padding variance, with the ``label`` the loss slice
    scores against."""
    pairs = [("3", "7"), ("5", "2"), ("8", "4"), ("1", "9")]
    return [
        {
            "input": _sample(base),
            "counterfactual_inputs": [_sample(cf)],
            "label": " 4",
        }
        for base, cf in pairs[:n]
    ]


def _checker(neural_output: dict, expected: Any) -> bool:
    expected_value = (
        expected.get("string", expected) if isinstance(expected, dict) else expected
    )
    return neural_output["string"].strip() == str(expected_value).strip()


def _groups() -> list[list[SiteSpec]]:
    # Layer 0 of the 2-layer tiny model: its block output feeds layer 1's
    # attention, so an edit at position 0 reaches the label-position logits
    # (an edit at the LAST layer's position 0 could not — block outputs feed
    # the unembed per-position — and the rotation would see zero gradient).
    spec = SiteSpec(
        fsite=FeaturizedSite(Site("block_output", 0)),
        positions=(0,),
        key=_KEY,
        width=_HIDDEN,
    )
    return [[spec]]


def _config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "train_batch_size": 2,
        "training_epoch": 2,
        "init_lr": 1e-2,
        "DAS": {"n_features": 4},
        "memory_cleanup_freq": 50,
    }
    base.update(overrides)
    return dict(merge_with_defaults(base))  # pyright: ignore[reportArgumentType]


@pytest.fixture(scope="module")
def pipe() -> LMPipeline:
    raw, _tok = fresh_tiny_random_llama()
    return LMPipeline(raw, max_new_tokens=1, padding_side="left")


def _train(
    pipe: LMPipeline, groups: list[list[SiteSpec]], config: dict[str, Any]
) -> tuple[list[list[SiteSpec]], dict[str, list[int] | None], str]:
    initialized = _initialize_featurizers({("single",): groups}, config)[("single",)]
    return _run_training_loop(
        pipeline=pipe,
        groups=initialized,
        counterfactual_dataset=_labeled_dataset(),  # pyright: ignore[reportArgumentType]
        intervention_type=config["intervention_type"],
        config=config,
        checker=_checker,
    )


class TestTrainingLoopRuns:
    pytestmark = pytest.mark.property

    def test_initialize_featurizers_is_functional(self, pipe: LMPipeline) -> None:
        """Initialization returns new specs; the caller's specs are untouched
        (the ``set_featurizer`` mutation is gone — WU4)."""
        groups = _groups()
        config = _config(intervention_type="interchange")
        initialized = _initialize_featurizers({("single",): groups}, config)
        original = groups[0][0]
        new = initialized[("single",)][0][0]
        assert original.fsite.featurizer.id == "null"  # untouched
        assert new.fsite.featurizer.id == f"DAS_{_KEY}"
        assert new.key == original.key

    def test_das_training_moves_the_rotation(self, pipe: LMPipeline) -> None:
        config = _config(intervention_type="interchange")
        groups = _initialize_featurizers({("single",): _groups()}, config)[("single",)]
        spec = groups[0][0]
        before = spec.fsite.featurizer.featurizer.rotate.weight.detach().clone()  # type: ignore[attr-defined]

        trained_groups, _indices, summary = _run_training_loop(
            pipeline=pipe,
            groups=groups,
            counterfactual_dataset=_labeled_dataset(),  # pyright: ignore[reportArgumentType]
            intervention_type="interchange",
            config=config,
            checker=_checker,
        )

        after = spec.fsite.featurizer.featurizer.rotate.weight.detach()  # type: ignore[attr-defined]
        assert isinstance(summary, str) and "Trained intervention" in summary
        assert not torch.allclose(before, after), "rotation never received a step"
        # DAS returns the (shared-featurizer) specs unchanged in shape.
        assert [[s.key for s in g] for g in trained_groups] == [[_KEY]]

    def test_dbm_training_returns_per_feature_readout(self, pipe: LMPipeline) -> None:
        config = _config(
            intervention_type="mask", featurizer_kwargs={"tie_masks": False}
        )
        trained_groups, feature_indices, _ = _train(pipe, _groups(), config)
        indices = feature_indices[_KEY]
        # Untied hard-threshold readout: an explicit (possibly empty) index list.
        assert isinstance(indices, list)
        assert all(0 <= i < _HIDDEN for i in indices)
        if indices:
            # Trained spec carries the selection functionally.
            assert trained_groups[0][0].fsite.feature_ids == tuple(indices)
        else:
            # All-off mask: the spec is dropped (no-op by omission) — an empty
            # selection is not constructible as a spec.
            assert trained_groups[0] == []

    def test_dbm_tied_readout_is_all_or_nothing(self, pipe: LMPipeline) -> None:
        config = _config(
            intervention_type="mask", featurizer_kwargs={"tie_masks": True}
        )
        trained_groups, feature_indices, _ = _train(pipe, _groups(), config)
        indices = feature_indices[_KEY]
        # Tied gate keeps the legacy convention: None (= all features) or [].
        assert indices is None or indices == []
        if indices is None:
            assert trained_groups[0][0].fsite.feature_ids is None
        else:
            assert trained_groups[0] == []  # all-off → dropped edit


class TestOrchestration:
    """Scripted-boundary pins: ``traced_label_loss`` is patched at the module
    seam so the loop's control flow is observable without real forwards (the
    raw-source collection still runs the real tiny model)."""

    pytestmark = pytest.mark.unit

    def _scripted(self, losses: list[float]):
        calls = {"n": 0}

        def fake_traced_label_loss(model, inputs, label_ids, edits, pad_token_id):
            value = losses[min(calls["n"], len(losses) - 1)]
            calls["n"] += 1
            return (
                torch.tensor(value, requires_grad=True),
                label_ids.detach().cpu(),
            )

        return calls, fake_traced_label_loss

    def test_early_stopping_counts_epochs_without_improvement(
        self, pipe: LMPipeline
    ) -> None:
        # 1 batch/epoch; losses strictly increase, so with patience=1 the loop
        # must stop after epoch 2 (one epoch without improvement).
        calls, fake = self._scripted([0.5, 0.6, 0.7, 0.8, 0.9])
        config = _config(
            intervention_type="interchange",
            train_batch_size=4,
            training_epoch=10,
            patience=1,
        )
        with patch(
            "causalab.methods.trained_subspace.train.traced_label_loss",
            side_effect=fake,
        ):
            _train(pipe, _groups(), config)
        assert calls["n"] == 2, "early stopping should end training after 2 epochs"

    def test_memory_cleanup_fires_per_frequency(self, pipe: LMPipeline) -> None:
        # is_available must read True for the cleanup branch, which would
        # poison every real CUDA-checking path on a CPU box (nnsight traces,
        # AdamW's graph-capture probe) — so everything that computes is
        # scripted out at its seam and only the loop's control flow runs.
        calls, fake = self._scripted([0.5])
        config = _config(
            intervention_type="interchange",
            train_batch_size=4,
            training_epoch=3,
            memory_cleanup_freq=1,
        )
        # Initialize first: the functional initializer returns NEW specs, and
        # the scripted raw-source dict must key on the ids the loop will see.
        groups = _initialize_featurizers({("single",): _groups()}, config)[("single",)]
        raw = {
            id(spec): torch.zeros(4, 1, _HIDDEN) for group in groups for spec in group
        }
        with (
            patch(
                "causalab.methods.trained_subspace.train.traced_label_loss",
                side_effect=fake,
            ),
            patch(
                "causalab.methods.trained_subspace.train._collect_raw_sources",
                return_value=raw,
            ),
            patch("torch.optim.AdamW", return_value=MagicMock()),
            patch(
                "causalab.methods.trained_subspace.train.transformers.get_scheduler",
                return_value=MagicMock(),
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            _run_training_loop(
                pipeline=pipe,
                groups=groups,
                counterfactual_dataset=_labeled_dataset(),  # pyright: ignore[reportArgumentType]
                intervention_type="interchange",
                config=config,
                checker=_checker,
            )
        # One cleanup per epoch (1 batch/epoch, freq=1).
        assert mock_empty_cache.call_count >= 3
