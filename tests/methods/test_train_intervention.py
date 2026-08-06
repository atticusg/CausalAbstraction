"""The intervention training loop (DAS and DBM).

``_run_training_loop`` optimizes an intervention while the base model stays
frozen. It used to be tested entirely through mocks — a mocked
``IntervenableModel`` whose ``interventions`` dict held a ``MagicMock`` — a mocked
``AdamW``, a mocked scheduler, and a mocked loss function — so it asserted that
the backbone was wired up, never that anything was learned. There is no such
object to mock any more, and the loop is cheap enough to just run: a tiny random
Llama, a rank-2 rotation, a handful of steps.

So these assert the properties that make the loop a training loop:

* the trainable parameters actually move, and the base model does not;
* the same intervention objects are optimized on every step;
* DBM ends with a binary mask written back as the units' feature indices;
* the mask temperature anneals on the configured schedule;
* early stopping fires when the loss stops improving.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.methods.trained_subspace.train import (
    _check_rotations_survived,  # pyright: ignore[reportPrivateUsage]
    _run_training_loop as run_training_loop,  # pyright: ignore[reportPrivateUsage]
)
from causalab.neural.LM_units import ResidualStream
from causalab.neural.activations.engine import build_interventions
from causalab.neural.featurizer import Featurizer
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget

pytestmark = pytest.mark.unit

RANK = 2


def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


@pytest.fixture
def dataset() -> list[dict[str, Any]]:
    """Labeled counterfactual examples — the shape the loss function consumes."""
    return [
        {
            "input": _trace("the quick brown fox"),
            "counterfactual_inputs": [_trace("a slow lazy dog")],
            "label": "cat",
        },
        {
            "input": _trace("one two three four"),
            "counterfactual_inputs": [_trace("five six seven eight")],
            "label": "dog",
        },
    ]


def _last_token(pipeline) -> TokenPosition:
    def index(trace):
        return [len(pipeline.tokenizer(trace["raw_input"])["input_ids"]) - 1]

    return TokenPosition(index, pipeline, id="last_token")


def _target(pipeline, featurizer_for) -> InterchangeTarget:
    """One unit per layer, each with a freshly built featurizer."""
    hidden = pipeline.model.config.hidden_size
    position = _last_token(pipeline)
    units = [
        ResidualStream(
            layer=layer,
            token_indices=position,
            target_output=True,
            shape=(hidden,),
            featurizer=featurizer_for(hidden),
        )
        for layer in (0, 1)
    ]
    # One group: both units read the same counterfactual, so each example needs
    # exactly one counterfactual input.
    return InterchangeTarget([units])


def _das_target(pipeline) -> InterchangeTarget:
    return _target(
        pipeline,
        lambda hidden: SubspaceFeaturizer(shape=(hidden, RANK), trainable=True),
    )


def _mask_target(pipeline) -> InterchangeTarget:
    return _target(
        pipeline,
        lambda hidden: Featurizer(n_features=hidden, id="mask", tie_masks=False),
    )


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "train_batch_size": 2,
        "training_epoch": 2,
        "init_lr": 1e-2,
        "memory_cleanup_freq": 1,
        "patience": None,
        "scheduler_type": "constant",
        "shuffle": False,
        "masking": {
            "regularization_coefficient": 0.1,
            "temperature_schedule": (1.0, 0.01),
            "temperature_annealing_fraction": 0.5,
        },
        "featurizer_kwargs": {"tie_masks": False},
    }
    config.update(overrides)
    return config


def _loss_fn():
    """The real loss function, with a permissive checker so no answer must match."""
    from causalab.methods.metric import LM_loss_and_metric_fn

    def loss_and_metric(pipeline, examples, target, interventions, source_pipeline):
        return LM_loss_and_metric_fn(
            pipeline,
            examples,
            target,
            lambda *_args: 1.0,
            interventions=interventions,
            source_pipeline=source_pipeline,
        )

    return loss_and_metric


def _constant_loss(pipeline, examples, target, interventions, source_pipeline):
    """A loss that never improves, for the early-stopping tests.

    Kept differentiable (it multiplies a real parameter by zero) so ``backward``
    has a graph to walk — a bare constant would raise instead of exercising the
    loop.
    """
    parameter = next(iter(interventions[0].parameters()))
    return parameter.sum() * 0.0 + 1.0, {"accuracy": 0.5}, {}


def _rotations(target: InterchangeTarget) -> list[torch.Tensor]:
    return [
        unit.featurizer.featurizer.rotate.weight.detach().clone()
        for unit in target.flatten()
    ]


class TestDASTraining:
    """``intervention_type="interchange"`` trains each unit's rotation."""

    def test_rotation_is_updated(self, mock_tiny_lm, dataset) -> None:
        """The whole point: the subspace moves. A loop that ran but optimized
        nothing would still return cleanly, so this is the load-bearing check."""
        target = _das_target(mock_tiny_lm)
        before = _rotations(target)

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=target,
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(),
            loss_and_metric_fn=_loss_fn(),
        )

        for old, new in zip(before, _rotations(target)):
            assert not torch.allclose(old, new), "rotation did not train"

    def test_rotation_stays_orthonormal(self, mock_tiny_lm, dataset) -> None:
        """The orthogonal parametrization must survive optimization — otherwise
        the 'subspace' drifts into an arbitrary linear map and the featurizer's
        error term stops being the orthogonal complement."""
        target = _das_target(mock_tiny_lm)
        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=target,
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(),
            loss_and_metric_fn=_loss_fn(),
        )
        for rotation in _rotations(target):
            gram = rotation.T @ rotation
            torch.testing.assert_close(
                gram, torch.eye(RANK, dtype=gram.dtype), atol=1e-4, rtol=1e-3
            )

    def test_base_model_is_not_trained(self, mock_tiny_lm, dataset) -> None:
        """Only the intervention learns; the model under study must be frozen."""
        target = _das_target(mock_tiny_lm)
        before = {
            name: parameter.detach().clone()
            for name, parameter in mock_tiny_lm.model.named_parameters()
        }

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=target,
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(),
            loss_and_metric_fn=_loss_fn(),
        )

        for name, parameter in mock_tiny_lm.model.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            assert parameter.grad is None or not parameter.grad.any()

    def test_loss_function_is_called_once_per_batch(
        self, mock_tiny_lm, dataset
    ) -> None:
        calls: list[int] = []
        inner = _loss_fn()

        def counting(pipeline, examples, target, interventions, source_pipeline):
            calls.append(len(examples))
            return inner(pipeline, examples, target, interventions, source_pipeline)

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=_das_target(mock_tiny_lm),
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(training_epoch=3, train_batch_size=1),
            loss_and_metric_fn=counting,
        )
        # 3 epochs x 2 batches of 1
        assert calls == [1] * 6

    def test_interventions_persist_across_steps(self, mock_tiny_lm, dataset) -> None:
        """Every step must see the *same* intervention objects — rebuilding them
        per batch would hand the optimizer parameters that are then discarded, so
        training would silently do nothing."""
        seen: list[tuple[int, ...]] = []
        inner = _loss_fn()

        def capture(pipeline, examples, target, interventions, source_pipeline):
            seen.append(tuple(id(i) for i in interventions))
            return inner(pipeline, examples, target, interventions, source_pipeline)

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=_das_target(mock_tiny_lm),
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(training_epoch=2, train_batch_size=1),
            loss_and_metric_fn=capture,
        )
        assert len(set(seen)) == 1


class TestDBMTraining:
    """``intervention_type="mask"`` learns a binary mask over features."""

    def test_mask_is_written_back_as_feature_indices(
        self, mock_tiny_lm, dataset
    ) -> None:
        """After training, each unit carries the selected feature indices — the
        artifact every downstream consumer reads."""
        target = _mask_target(mock_tiny_lm)
        hidden = mock_tiny_lm.model.config.hidden_size
        for unit in target.flatten():
            assert unit.get_feature_indices() is None

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=target,
            counterfactual_dataset=dataset,
            intervention_type="mask",
            config=_config(),
            loss_and_metric_fn=_loss_fn(),
        )

        for unit in target.flatten():
            indices = unit.get_feature_indices()
            assert isinstance(indices, list)
            assert all(0 <= i < hidden for i in indices)

    def test_tied_masks_select_all_or_nothing(self, mock_tiny_lm, dataset) -> None:
        """A tied mask is one gate for the whole unit, so the unit is either fully
        selected (``None`` = no restriction) or fully dropped (``[]``)."""
        target = _target(
            mock_tiny_lm,
            lambda hidden: Featurizer(n_features=hidden, id="mask", tie_masks=True),
        )

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=target,
            counterfactual_dataset=dataset,
            intervention_type="mask",
            config=_config(featurizer_kwargs={"tie_masks": True}),
            loss_and_metric_fn=_loss_fn(),
        )

        for unit in target.flatten():
            assert unit.get_feature_indices() in (None, [])

    def test_temperature_anneals(self, mock_tiny_lm, dataset) -> None:
        """The gate sharpens over training: the temperature must fall from the
        schedule's start toward its end, or the mask never becomes binary."""
        temperatures: list[float] = []
        inner = _loss_fn()

        def observing(pipeline, examples, target, interventions, source_pipeline):
            temperatures.append(float(interventions[0].get_temperature()))
            return inner(pipeline, examples, target, interventions, source_pipeline)

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=_mask_target(mock_tiny_lm),
            counterfactual_dataset=dataset,
            intervention_type="mask",
            config=_config(training_epoch=4, train_batch_size=1),
            loss_and_metric_fn=observing,
        )

        assert temperatures[0] == pytest.approx(1.0)
        assert temperatures[-1] < temperatures[0]
        assert temperatures[-1] == pytest.approx(0.01, abs=1e-6)


class TestEarlyStopping:
    def test_stops_when_loss_stops_improving(self, mock_tiny_lm, dataset) -> None:
        """With a constant loss no epoch improves, so training must stop after
        ``patience`` epochs rather than running the full budget."""
        epochs: list[int] = []

        def flat(pipeline, examples, target, interventions, source_pipeline):
            epochs.append(1)
            return _constant_loss(
                pipeline, examples, target, interventions, source_pipeline
            )

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=_das_target(mock_tiny_lm),
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(training_epoch=50, patience=2, train_batch_size=2),
            loss_and_metric_fn=flat,
        )
        # 1 batch/epoch: the first epoch sets the best loss, then 2 epochs
        # without improvement trip the patience counter.
        assert len(epochs) == 3

    def test_runs_every_epoch_when_patience_is_none(
        self, mock_tiny_lm, dataset
    ) -> None:
        epochs: list[int] = []

        def flat(pipeline, examples, target, interventions, source_pipeline):
            epochs.append(1)
            return _constant_loss(
                pipeline, examples, target, interventions, source_pipeline
            )

        run_training_loop(
            pipeline=mock_tiny_lm,
            interchange_target=_das_target(mock_tiny_lm),
            counterfactual_dataset=dataset,
            intervention_type="interchange",
            config=_config(training_epoch=4, patience=None, train_batch_size=2),
            loss_and_metric_fn=flat,
        )
        assert len(epochs) == 4


class TestNoTrainableParameters:
    def test_identity_featurizer_for_das_is_rejected(
        self, mock_tiny_lm, dataset
    ) -> None:
        """An interchange run over identity featurizers has nothing to optimize.
        Silently 'training' it would report a score with no learned subspace
        behind it, so it must fail loudly."""
        with pytest.raises(ValueError, match="No trainable parameters"):
            run_training_loop(
                pipeline=mock_tiny_lm,
                interchange_target=_target(mock_tiny_lm, lambda _h: Featurizer()),
                counterfactual_dataset=dataset,
                intervention_type="interchange",
                config=_config(),
                loss_and_metric_fn=_loss_fn(),
            )


class TestSharedFeaturizerIsNotTrainedTwice:
    """Units may share one featurizer; the optimizer must still see it once.

    Tying a single subspace across several layers is a deliberate pattern —
    ``Featurizer.tie_masks`` says as much for masks. Each unit gets its own
    intervention object wrapping that shared featurizer, and ``parameters()``
    deduplicates within a module but cannot across two, so collecting naively
    yields the same tensor twice.

    Torch does not reject that: it warns about duplicates in a parameter group
    and then steps the tensor once per occurrence, so the rotation trains at
    double the configured learning rate. Silent, and easy to lose in a training
    log — hence a test rather than a comment.
    """

    pytestmark = pytest.mark.unit

    def test_duplicate_parameters_would_double_the_step(self) -> None:
        """Pin the torch behaviour the deduplication exists to avoid, so this
        test explains itself if the fix is ever removed."""
        once = torch.nn.Parameter(torch.zeros(3))
        twice = torch.nn.Parameter(torch.zeros(3))
        once.grad, twice.grad = torch.ones(3), torch.ones(3)
        torch.optim.AdamW([once], lr=0.1).step()
        with pytest.warns(UserWarning, match="duplicate parameters"):
            opt = torch.optim.AdamW([twice, twice], lr=0.1)
        opt.step()
        # Not an exact doubling — the second step sees updated Adam moments — but
        # unmistakably a second step rather than rounding.
        ratio = (twice.data.abs() / once.data.abs()).mean().item()
        assert 1.5 < ratio < 2.5, ratio

    def test_shared_featurizer_yields_one_parameter_set(
        self, mock_tiny_lm: Any
    ) -> None:
        hidden = mock_tiny_lm.model.config.hidden_size
        shared = SubspaceFeaturizer(shape=(hidden, 3), trainable=True, id="shared")
        position = _last_token(mock_tiny_lm)
        units = [
            ResidualStream(
                layer=layer,
                token_indices=position,
                target_output=True,
                shape=(hidden,),
                featurizer=shared,
            )
            for layer in (0, 1)
        ]
        interventions = build_interventions(units, "interchange")

        naive = [p for i in interventions for p in i.parameters() if p.requires_grad]
        seen: set[int] = set()
        deduped = [
            p
            for i in interventions
            for p in i.parameters()
            if p.requires_grad and not (id(p) in seen or seen.add(id(p)))
        ]
        # The naive collection really does repeat it — otherwise this proves nothing.
        assert len(naive) > len(deduped)
        assert len({id(p) for p in deduped}) == len(deduped)

    def test_training_loop_does_not_hand_the_optimizer_a_duplicate(
        self, mock_tiny_lm: Any, dataset: list[dict[str, Any]]
    ) -> None:
        """The production path, not just the pattern.

        The two tests above establish that duplicates exist and that torch
        double-steps them; this one drives the real loop with a shared featurizer
        and asserts the duplicate never reaches the optimizer. Torch's own
        ``UserWarning`` is the oracle, so removing the deduplication fails this
        test even though the loop would still run and still appear to train.
        """
        hidden = mock_tiny_lm.model.config.hidden_size
        shared = SubspaceFeaturizer(shape=(hidden, RANK), trainable=True, id="shared")
        position = _last_token(mock_tiny_lm)
        target = InterchangeTarget(
            [
                [
                    ResidualStream(
                        layer=layer,
                        token_indices=position,
                        target_output=True,
                        shape=(hidden,),
                        featurizer=shared,
                    )
                    for layer in (0, 1)
                ]
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            run_training_loop(
                pipeline=mock_tiny_lm,
                interchange_target=target,
                counterfactual_dataset=dataset,
                intervention_type="interchange",
                config=_config(),
                loss_and_metric_fn=_loss_fn(),
            )


class TestRotationCollapseIsCaught:
    """A DAS rotation that degenerates must fail the run, not return a null result.

    The parametrization holding the rotation orthonormal ends in
    ``Q * X.diagonal().int()``. That truncation only produces an orthonormal
    matrix while the underlying diagonal keeps magnitude >= 1; the diagonal gets
    no gradient, so nothing pushes it, but *any* weight decay pulls it under 1,
    where ``.int()`` rounds it to zero and the rotation's columns become exactly
    zero.

    Nothing raises when that happens. The featurizer maps every activation to
    zero, the whole activation lands in the reconstruction error, and each
    interchange becomes a no-op — so the run completes and reports that the
    subspace carries no causal signal. A false negative shaped exactly like a
    finding, which is why this is checked rather than left to a comment on the
    optimizer.
    """

    pytestmark = pytest.mark.unit

    def test_weight_decay_really_does_collapse_the_rotation(self) -> None:
        """Pin the torch behaviour the guard exists for, so the guard explains
        itself if anyone doubts it is needed."""
        torch.manual_seed(0)
        featurizer = SubspaceFeaturizer(shape=(16, RANK), trainable=True)
        params = [p for p in featurizer.featurizer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01)
        for _ in range(50):
            features, _ = featurizer.featurize(torch.randn(8, 16))
            loss = (features**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        weight = featurizer.featurizer.rotate.weight.detach()
        assert float(weight.abs().max()) == 0.0, "expected the collapse to reproduce"

    def test_a_collapsed_rotation_raises_rather_than_reporting_nothing(
        self, mock_tiny_lm: Any
    ) -> None:
        target = _das_target(mock_tiny_lm)
        # Collapse it the way decay does: shrink the *underlying* parameter's
        # diagonal below 1, where the parametrization's `.int()` truncates it to
        # zero. Zeroing `rotate.weight` directly would not persist — it is a
        # parametrized property recomputed from `original` on every access, which
        # is exactly why the collapse is invisible until you look at the matrix
        # the featurizer actually uses.
        for unit in target.flatten():
            with torch.no_grad():
                unit.featurizer.featurizer.rotate.parametrizations.weight.original.mul_(
                    0.01
                )
        assert (
            float(target.flatten()[0].featurizer.featurizer.rotate.weight.abs().max())
            == 0.0
        ), "the collapse should have zeroed the effective rotation"
        with pytest.raises(RuntimeError, match="no longer orthonormal"):
            _check_rotations_survived(target)

    def test_a_healthy_rotation_passes(self, mock_tiny_lm: Any) -> None:
        """The guard must not fire on an ordinary trained subspace."""
        _check_rotations_survived(_das_target(mock_tiny_lm))
