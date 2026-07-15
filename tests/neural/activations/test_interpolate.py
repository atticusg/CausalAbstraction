"""Direct tests for ``causalab.neural.activations.interpolate``.

This module drives **interpolation-style** pyvene interventions: given a
featurizer-aware :class:`IntervenableModel`, it pushes a user-supplied
``fn(f_base, f_src, **params)`` onto every intervention instance and runs
batched generation, replacing activations with
``inverse_featurizer(fn(...), base_err)``. At ``alpha=1`` this collapses to
interchange; at ``alpha=0`` it is the identity. Direct consumers:
``causalab.methods.steer.collect`` (path-steering grids and DAS-style sweeps)
and ``causalab.methods.pullback.optimization`` (``replace_fn`` patching for
geodesic recapitulation).

If these helpers misroute ``fn``, mishandle ``output_scores``, or fail to
move outputs to CPU, every downstream path-steering / pullback analysis
writes corrupted activation/logit artifacts.

Tests are laid out as one unit class and one property class per public
symbol. The *unit* classes keep mocked
pyvene scaffolding (they assert wiring / dispatch, not numerics). The
*property* class uses the real ``tiny_pipeline`` fixture from
``tests/neural/conftest.py`` so the pyvene envelope is exercised end-to-end
against a real (tiny) :class:`IntervenableModel` — mirroring the canonical
pattern in ``tests/neural/activations/test_collect.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.LM_units import ResidualStream
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.neural.activations.interpolate import (
    batched_interpolation_intervention,
    run_interpolation_interventions,
    set_interventions_interpolation,
    sweep_interpolation_interventions,
)
from causalab.neural.featurizer import Featurizer
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget

# Module under test — patched paths for boundary fakes.
MODULE = "causalab.neural.activations.interpolate"


# --------------------------------------------------------------------------- #
#  Local helpers (unit tier)                                                  #
# --------------------------------------------------------------------------- #
class _FakeIntervention:
    """Minimal stand-in for a pyvene intervention exposing ``set_interpolation``.

    Mirrors the contract of
    :class:`causalab.neural.featurizer.FeatureInterpolateIntervention`'s
    ``set_interpolation(fn, **params)`` hook used by the source under test:
    the unit test for :func:`set_interventions_interpolation` only needs the
    hook to be observable, not a real pyvene module.
    """

    def __init__(self) -> None:
        self.fn: Any = None
        self.params: dict[str, Any] = {}
        self.set_calls = 0

    def set_interpolation(self, fn: Any, **params: Any) -> None:
        self.fn = fn
        self.params = params
        self.set_calls += 1


class _BarePyveneStub:
    """A pyvene intervention without ``set_interpolation`` — must be skipped silently."""

    def __init__(self) -> None:
        self.touched = False


def _make_fake_intervenable_model(interventions: dict[str, Any]) -> Any:
    """Return an object exposing ``.interventions`` mirroring pyvene's surface."""
    model = MagicMock()
    model.interventions = interventions
    return model


def _make_test_dataset(n: int = 3) -> list[Any]:
    """Build a minimal counterfactual dataset (``CounterfactualExample`` is a TypedDict)."""
    return [
        {"input": f"input{i}", "counterfactual_inputs": [f"cf{i}"]} for i in range(n)
    ]


# --------------------------------------------------------------------------- #
#  Helpers for the property tier (tiny-real pipeline)                         #
# --------------------------------------------------------------------------- #
def _trace(text: str) -> CausalTrace:
    """Build a minimal ``CausalTrace`` carrying just ``raw_input``."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _make_residual_unit(pipeline, layer: int) -> ResidualStream:
    """Build a real ``ResidualStream`` unit at ``layer``, last-token position.

    Mirrors the helper in ``tests/neural/activations/test_collect.py`` — the
    canonical tiny-real pattern for this subdir.
    """

    def last_token(trace):
        return [len(pipeline.load([trace])["input_ids"][0]) - 1]

    tp = TokenPosition(last_token, pipeline, id="last_token")
    return ResidualStream(
        layer=layer,
        token_indices=tp,
        shape=(pipeline.model.config.hidden_size,),
        target_output=True,
    )


def _make_cf_dataset(n: int) -> list[dict[str, Any]]:
    """Build a tiny counterfactual dataset of ``CausalTrace`` examples.

    Each example has one base input and one counterfactual input — the
    minimum shape :func:`prepare_intervenable_inputs` accepts.
    """
    return [
        {
            "input": _trace(f"base_{i}"),
            "counterfactual_inputs": [_trace(f"cf_{i}")],
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
#  set_interventions_interpolation                                            #
# --------------------------------------------------------------------------- #
class TestSetInterventionsInterpolationUnit:
    """``set_interventions_interpolation`` pushes ``fn`` + ``params`` onto every
    intervention exposing ``set_interpolation``.

    This is the single entry point by which the batched runner gets a user's
    interpolation function onto pyvene's intervention instances. If it
    silently drops the kwargs or skips the tuple-shape branch, the model
    runs with stale or no interpolation and produces silently-wrong
    activations.
    """

    pytestmark = pytest.mark.unit

    def test_pushes_fn_and_params_onto_plain_intervention(self) -> None:
        inter = _FakeIntervention()
        model = _make_fake_intervenable_model({"k0": inter})

        def linear(
            *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return (1 - alpha) * f_base + alpha * f_src

        set_interventions_interpolation(model, linear, alpha=0.3)

        assert inter.fn is linear
        assert inter.params == {"alpha": 0.3}
        assert inter.set_calls == 1

    def test_unwraps_tuple_intervention_shape(self) -> None:
        """pyvene sometimes stores interventions as ``(module, ...)`` tuples;
        the source explicitly handles ``isinstance(v, tuple)``.
        """
        inter = _FakeIntervention()
        # Tuple form: first element is the live intervention, rest are metadata.
        model = _make_fake_intervenable_model({"k0": (inter, "metadata")})

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        set_interventions_interpolation(model, fn)

        assert inter.fn is fn
        assert inter.set_calls == 1

    def test_skips_intervention_without_set_interpolation(self) -> None:
        """Interventions lacking the hook (e.g. a CollectIntervention placed on
        the same intervenable model) are silently passed over — no
        ``AttributeError`` propagates.
        """
        bare = _BarePyveneStub()
        live = _FakeIntervention()
        model = _make_fake_intervenable_model({"k_bare": bare, "k_live": live})

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        # Must not raise.
        set_interventions_interpolation(model, fn, alpha=0.5)

        # bare untouched; live got the call.
        assert bare.touched is False
        assert live.fn is fn
        assert live.params == {"alpha": 0.5}

    def test_calling_twice_replaces_rather_than_stacks(self) -> None:
        """Each call overwrites the previous fn/params — last write wins."""
        inter = _FakeIntervention()
        model = _make_fake_intervenable_model({"k0": inter})

        def fn_a(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        def fn_b(
            *, f_base: torch.Tensor, f_src: torch.Tensor, beta: float
        ) -> torch.Tensor:
            return beta * f_src

        set_interventions_interpolation(model, fn_a, alpha=0.0)
        set_interventions_interpolation(model, fn_b, beta=0.7)

        assert inter.fn is fn_b
        assert inter.params == {"beta": 0.7}
        assert "alpha" not in inter.params
        assert inter.set_calls == 2

    def test_visits_every_intervention(self) -> None:
        """All values in ``intervenable_model.interventions`` are visited."""
        inters = [_FakeIntervention() for _ in range(4)]
        model = _make_fake_intervenable_model(
            {f"k{i}": inter for i, inter in enumerate(inters)}
        )

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        set_interventions_interpolation(model, fn)

        for inter in inters:
            assert inter.fn is fn
            assert inter.set_calls == 1


# --------------------------------------------------------------------------- #
#  batched_interpolation_intervention                                         #
# --------------------------------------------------------------------------- #
class TestBatchedInterpolationInterventionUnit:
    """``batched_interpolation_intervention`` performs one batched forward.

    Wires ``prepare_intervenable_inputs`` → ``set_interventions_interpolation``
    → ``pipeline.intervenable_generate`` and moves the batched tensor inputs
    back to CPU. Returns the raw output dict produced by the pipeline.
    """

    pytestmark = pytest.mark.unit

    def test_dispatches_fn_through_set_interventions(self) -> None:
        """The user-supplied ``fn`` must reach the interventions via
        :func:`set_interventions_interpolation` exactly once.
        """
        pipeline = MagicMock()
        intervenable_model = MagicMock()
        interchange_target = MagicMock()

        # Stub prepared inputs as plain dicts so the CPU move loop is a no-op
        # on tensor-less batches.
        with (
            patch(
                f"{MODULE}.prepare_intervenable_inputs",
                return_value=({}, [{}], {}, []),
            ),
            patch(f"{MODULE}.set_interventions_interpolation") as mock_set,
        ):
            pipeline.intervenable_generate.return_value = {
                "sequences": torch.tensor([[1, 2]])
            }

            def fn(
                *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
            ) -> torch.Tensor:
                return f_base

            out = batched_interpolation_intervention(
                pipeline,
                intervenable_model,
                examples=_make_test_dataset(1),
                interchange_target=interchange_target,
                fn=fn,
                params={"alpha": 0.5},
            )

            mock_set.assert_called_once_with(intervenable_model, fn, alpha=0.5)
            assert "sequences" in out

    def test_forwards_output_scores_kwarg(self) -> None:
        """``output_scores`` is plumbed through to ``intervenable_generate``."""
        pipeline = MagicMock()
        with (
            patch(
                f"{MODULE}.prepare_intervenable_inputs",
                return_value=({}, [{}], {}, []),
            ),
            patch(f"{MODULE}.set_interventions_interpolation"),
        ):
            pipeline.intervenable_generate.return_value = {
                "sequences": torch.tensor([[1, 2]]),
                "scores": [torch.tensor([[0.1, 0.2]])],
            }

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            batched_interpolation_intervention(
                pipeline,
                MagicMock(),
                examples=_make_test_dataset(1),
                interchange_target=MagicMock(),
                fn=fn,
                params={},
                output_scores=10,
            )

            kwargs = pipeline.intervenable_generate.call_args.kwargs
            assert kwargs["output_scores"] == 10

    def test_moves_prepared_inputs_to_cpu(self) -> None:
        """After the generate call, every prepared-input tensor is on CPU."""
        pipeline = MagicMock()
        # Prepared inputs carry tensors that we will assert are CPU-resident
        # after the call. We start them on CPU (no GPU in CI), and assert the
        # method invoked is ``.cpu()`` on each, which is what the source uses.
        base_t = MagicMock(spec=torch.Tensor)
        base_t.cpu.return_value = torch.tensor([1, 2])
        cf_t = MagicMock(spec=torch.Tensor)
        cf_t.cpu.return_value = torch.tensor([3, 4])
        batched_base = {"input_ids": base_t}
        batched_cfs = [{"input_ids": cf_t}]

        with (
            patch(
                f"{MODULE}.prepare_intervenable_inputs",
                return_value=(batched_base, batched_cfs, {}, []),
            ),
            patch(f"{MODULE}.set_interventions_interpolation"),
        ):
            pipeline.intervenable_generate.return_value = {
                "sequences": torch.tensor([[1, 2]])
            }

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            batched_interpolation_intervention(
                pipeline,
                MagicMock(),
                examples=_make_test_dataset(1),
                interchange_target=MagicMock(),
                fn=fn,
                params={},
            )

        # CPU move applied to both batched_base and every counterfactual batch.
        assert base_t.cpu.called
        assert cf_t.cpu.called

    def test_returns_pipeline_output_dict_verbatim(self) -> None:
        """The dict from ``intervenable_generate`` is returned unwrapped
        (the public contract is "dict with sequences and optionally scores").
        """
        pipeline = MagicMock()
        expected = {
            "sequences": torch.tensor([[1, 2, 3]]),
            "scores": [torch.tensor([[0.1, 0.2, 0.3]])],
        }
        pipeline.intervenable_generate.return_value = expected

        with (
            patch(
                f"{MODULE}.prepare_intervenable_inputs",
                return_value=({}, [{}], {}, []),
            ),
            patch(f"{MODULE}.set_interventions_interpolation"),
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            out = batched_interpolation_intervention(
                pipeline,
                MagicMock(),
                examples=_make_test_dataset(1),
                interchange_target=MagicMock(),
                fn=fn,
                params={},
            )

        assert out is expected

    def test_cleans_up_intervenable_model(self) -> None:
        """Batched form should also clean up — pin the *desired* behaviour."""
        pipeline = MagicMock()
        pipeline.intervenable_generate.return_value = {
            "sequences": torch.tensor([[1, 2]])
        }
        intervenable_model = MagicMock()

        with (
            patch(
                f"{MODULE}.prepare_intervenable_inputs",
                return_value=({}, [{}], {}, []),
            ),
            patch(f"{MODULE}.set_interventions_interpolation"),
            patch(f"{MODULE}.delete_intervenable_model") as mock_delete,
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            batched_interpolation_intervention(
                pipeline,
                intervenable_model,
                examples=_make_test_dataset(1),
                interchange_target=MagicMock(),
                fn=fn,
                params={},
            )

            mock_delete.assert_called_once_with(intervenable_model)


# --------------------------------------------------------------------------- #
#  run_interpolation_interventions                                            #
# --------------------------------------------------------------------------- #
class TestRunInterpolationInterventionsUnit:
    """``run_interpolation_interventions`` is the public batched entrypoint.

    Constructs an ``IntervenableModel`` once, iterates the dataset in
    ``batch_size`` chunks, and aggregates per-batch dicts into the final
    ``{key: [batch_outputs...]}`` shape. Owns the IntervenableModel lifecycle:
    cleanup is unconditional.
    """

    pytestmark = pytest.mark.unit

    def test_constructs_intervenable_model_with_interpolation_type(self) -> None:
        """``prepare_intervenable_model`` is called once with
        ``intervention_type="interpolation"``.
        """
        pipeline = MagicMock()
        target = MagicMock()
        mock_iv = MagicMock()
        dataset = _make_test_dataset(2)

        with (
            patch(
                f"{MODULE}.prepare_intervenable_model", return_value=mock_iv
            ) as mock_prepare,
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                return_value={"sequences": torch.tensor([[1, 2]])},
            ),
            patch(f"{MODULE}.delete_intervenable_model"),
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            run_interpolation_interventions(
                pipeline,
                dataset,
                target,
                fn=fn,
                params={},
                batch_size=2,
                output_scores=False,
            )

            mock_prepare.assert_called_once_with(
                pipeline, target, intervention_type="interpolation"
            )

    def test_cleans_up_intervenable_model(self) -> None:
        """``delete_intervenable_model`` is called after the batch loop."""
        mock_iv = MagicMock()

        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                return_value={"sequences": torch.tensor([[1, 2]])},
            ),
            patch(f"{MODULE}.delete_intervenable_model") as mock_delete,
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                fn=fn,
                params={},
                batch_size=2,
                output_scores=False,
            )

            mock_delete.assert_called_once_with(mock_iv)

    def test_batches_dataset_by_batch_size(self) -> None:
        """A dataset of 5 examples with batch_size=2 produces 3 batched calls
        (sizes 2, 2, 1).
        """
        mock_iv = MagicMock()
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
            ) as mock_batched,
            patch(f"{MODULE}.delete_intervenable_model"),
        ):
            mock_batched.side_effect = [
                {"sequences": torch.tensor([[1, 2]])},  # batch of 2
                {"sequences": torch.tensor([[3, 4]])},  # batch of 2
                {"sequences": torch.tensor([[5]])},  # batch of 1
            ]

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            results = run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(5),
                MagicMock(),
                fn=fn,
                params={},
                batch_size=2,
                output_scores=False,
            )

            assert mock_batched.call_count == 3
            assert "sequences" in results
            assert len(results["sequences"]) == 3

    def test_forwards_fn_and_params_to_batched(self) -> None:
        """``fn`` and ``params`` are plumbed through verbatim to each batched call."""
        mock_iv = MagicMock()
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                return_value={"sequences": torch.tensor([[1, 2]])},
            ) as mock_batched,
            patch(f"{MODULE}.delete_intervenable_model"),
        ):

            def fn(
                *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
            ) -> torch.Tensor:
                return f_base

            run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                fn=fn,
                params={"alpha": 0.7},
                batch_size=4,
                output_scores=False,
            )

            kwargs = mock_batched.call_args.kwargs
            assert kwargs["fn"] is fn
            assert kwargs["params"] == {"alpha": 0.7}

    def test_output_scores_false_omits_scores_key(self) -> None:
        """With ``output_scores=False``, the aggregated dict has no ``scores`` key."""
        mock_iv = MagicMock()
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                return_value={"sequences": torch.tensor([[1, 2]])},
            ),
            patch(f"{MODULE}.delete_intervenable_model"),
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            results = run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                fn=fn,
                params={},
                output_scores=False,
            )
            assert "scores" not in results
            assert "sequences" in results

    def test_output_scores_int_invokes_top_k_conversion(self) -> None:
        """An ``int`` ``output_scores`` value triggers ``convert_to_top_k``."""
        mock_iv = MagicMock()
        batched_dicts = [
            {
                "sequences": torch.tensor([[1, 2]]),
                "scores": [torch.tensor([[0.1, 0.2]])],
            },
        ]
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                side_effect=batched_dicts,
            ),
            patch(f"{MODULE}.delete_intervenable_model"),
            patch(
                f"{MODULE}.convert_to_top_k",
                return_value=batched_dicts,
            ) as mock_convert,
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            results = run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                fn=fn,
                params={},
                output_scores=5,
            )

            mock_convert.assert_called_once()
            # k argument forwarded as the integer output_scores value
            call_kwargs = mock_convert.call_args
            assert (
                call_kwargs.kwargs.get(
                    "k", call_kwargs.args[-1] if call_kwargs.args else None
                )
                == 5
            )
            assert "scores" in results

    def test_output_scores_true_skips_top_k_conversion(self) -> None:
        """``output_scores=True`` keeps full-vocab scores (no top-k pass)."""
        mock_iv = MagicMock()
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                return_value={
                    "sequences": torch.tensor([[1, 2]]),
                    "scores": [torch.tensor([[0.1, 0.2]])],
                },
            ),
            patch(f"{MODULE}.delete_intervenable_model"),
            patch(f"{MODULE}.convert_to_top_k") as mock_convert,
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                fn=fn,
                params={},
                output_scores=True,
            )

            mock_convert.assert_not_called()

    def test_final_output_is_moved_to_cpu(self) -> None:
        """``move_outputs_to_cpu`` is invoked on the aggregated batch list."""
        mock_iv = MagicMock()
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                return_value={"sequences": torch.tensor([[1, 2]])},
            ),
            patch(f"{MODULE}.delete_intervenable_model"),
            patch(
                f"{MODULE}.move_outputs_to_cpu",
                side_effect=lambda outs: outs,
            ) as mock_move,
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                fn=fn,
                params={},
                output_scores=False,
            )

            mock_move.assert_called_once()

    def test_aggregates_per_batch_dicts_into_lists(self) -> None:
        """Two batches → ``results["sequences"]`` has two entries."""
        mock_iv = MagicMock()
        with (
            patch(f"{MODULE}.prepare_intervenable_model", return_value=mock_iv),
            patch(
                f"{MODULE}.batched_interpolation_intervention",
                side_effect=[
                    {"sequences": torch.tensor([[1]])},
                    {"sequences": torch.tensor([[2]])},
                ],
            ),
            patch(f"{MODULE}.delete_intervenable_model"),
        ):

            def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
                return f_base

            results = run_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(2),
                MagicMock(),
                fn=fn,
                params={},
                batch_size=1,
                output_scores=False,
            )

            assert isinstance(results["sequences"], list)
            assert len(results["sequences"]) == 2


# --------------------------------------------------------------------------- #
#  sweep_interpolation_interventions                                          #
# --------------------------------------------------------------------------- #
class TestSweepInterpolationInterventionsUnit:
    """``sweep_interpolation_interventions`` runs ``run_*`` once per named config.

    For each ``(name, (featurizer, fn, params))`` entry, sets the
    featurizer on the interchange target *exactly once* and then dispatches
    a full ``run_interpolation_interventions`` call. Returns a dict keyed by
    config name. Per the docstring contract: the target retains the **last**
    featurizer after the sweep completes (no restore).
    """

    pytestmark = pytest.mark.unit

    def test_iterates_configs_in_insertion_order(self) -> None:
        """Each config in ``configs`` is run, and the result dict preserves keys."""
        target = MagicMock()
        feat_a = MagicMock(name="featurizer_a")
        feat_b = MagicMock(name="featurizer_b")

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        configs = {
            "a": (feat_a, fn, {"alpha": 0.0}),
            "b": (feat_b, fn, {"alpha": 1.0}),
        }

        # Sentinel payload distinguishes per-config calls.
        with patch(
            f"{MODULE}.run_interpolation_interventions",
            side_effect=[
                {"sequences": ["A"]},
                {"sequences": ["B"]},
            ],
        ) as mock_run:
            results = sweep_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                target,
                configs=configs,
            )

            assert list(results.keys()) == ["a", "b"]
            assert results["a"] == {"sequences": ["A"]}
            assert results["b"] == {"sequences": ["B"]}
            assert mock_run.call_count == 2

    def test_calls_set_featurizer_once_per_config(self) -> None:
        """The interchange target's ``set_featurizer`` is called once per config,
        in order, with the supplied featurizer.
        """
        target = MagicMock()
        feat_a = MagicMock(name="featurizer_a")
        feat_b = MagicMock(name="featurizer_b")

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        configs = {"a": (feat_a, fn, {}), "b": (feat_b, fn, {})}

        with patch(
            f"{MODULE}.run_interpolation_interventions",
            return_value={"sequences": []},
        ):
            sweep_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                target,
                configs=configs,
            )

        # Two calls, in insertion order.
        assert target.set_featurizer.call_count == 2
        observed_featurizers = [c.args[0] for c in target.set_featurizer.call_args_list]
        assert observed_featurizers == [feat_a, feat_b]

    def test_forwards_fn_params_dataset_to_run_for_each_config(self) -> None:
        """Each per-config dispatch receives the matching fn + params."""
        target = MagicMock()
        feat = MagicMock()
        dataset = _make_test_dataset(2)

        def fn_a(
            *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return f_base

        def fn_b(
            *, f_base: torch.Tensor, f_src: torch.Tensor, beta: float
        ) -> torch.Tensor:
            return f_src

        configs = {
            "a": (feat, fn_a, {"alpha": 0.25}),
            "b": (feat, fn_b, {"beta": 0.75}),
        }

        with patch(
            f"{MODULE}.run_interpolation_interventions",
            return_value={"sequences": []},
        ) as mock_run:
            sweep_interpolation_interventions(
                MagicMock(),
                dataset,
                target,
                configs=configs,
                batch_size=8,
                output_scores=False,
            )

        assert mock_run.call_count == 2
        first_call = mock_run.call_args_list[0]
        second_call = mock_run.call_args_list[1]
        # run_interpolation_interventions takes positional args:
        # (pipeline, counterfactual_dataset, interchange_target, fn, params, batch_size, output_scores)
        assert first_call.args[3] is fn_a
        assert first_call.args[4] == {"alpha": 0.25}
        assert second_call.args[3] is fn_b
        assert second_call.args[4] == {"beta": 0.75}
        assert first_call.args[5] == 8
        assert first_call.args[6] is False

    def test_target_retains_last_featurizer_after_sweep(self) -> None:
        """Per the docstring contract: the target is **mutated** in place
        and ends with the last featurizer in the config dict — no restore.
        """
        target = MagicMock()
        feat_a = MagicMock(name="featurizer_a")
        feat_last = MagicMock(name="featurizer_last")

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        configs = {"a": (feat_a, fn, {}), "last": (feat_last, fn, {})}

        with patch(
            f"{MODULE}.run_interpolation_interventions",
            return_value={"sequences": []},
        ):
            sweep_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                target,
                configs=configs,
            )

        # Last set_featurizer call uses feat_last — no implicit restore step.
        last_call = target.set_featurizer.call_args_list[-1]
        assert last_call.args[0] is feat_last

    def test_empty_configs_returns_empty_dict(self) -> None:
        """``configs={}`` → no ``run_*`` calls, empty result dict."""
        with patch(
            f"{MODULE}.run_interpolation_interventions",
        ) as mock_run:
            results = sweep_interpolation_interventions(
                MagicMock(),
                _make_test_dataset(1),
                MagicMock(),
                configs={},
            )
        assert results == {}
        mock_run.assert_not_called()


# --------------------------------------------------------------------------- #
#  Property tier — end-to-end against tiny_pipeline                           #
# --------------------------------------------------------------------------- #
class TestInterpolationProperty:
    """End-to-end invariants on the real (tiny) :class:`IntervenableModel`.

    These tests drive ``run_interpolation_interventions`` /
    ``sweep_interpolation_interventions`` end-to-end against
    ``tests/neural/conftest.py::tiny_pipeline``, exercising the same
    ``prepare_intervenable_model`` → ``prepare_intervenable_inputs`` →
    ``intervenable_generate`` → ``convert_to_top_k`` → ``move_outputs_to_cpu``
    path that production callers (``methods/steer/collect``,
    ``methods/pullback/optimization``) hit. They mirror the canonical pattern
    established by ``tests/neural/activations/test_collect.py``.

    None of these tests mock internal numerical helpers — only real pyvene
    primitives run.
    """

    pytestmark = pytest.mark.property

    def test_identity_fn_collapses_to_base_forward(self, tiny_pipeline) -> None:
        """``fn(f_base, f_src) -> f_base`` reproduces the unintervened base.

        The interpolation intervention with the identity function (alpha=0
        in the linear case) must produce sequences matching the pipeline's
        base generation on the same inputs. We pin **equality** of the
        generated sequences against a base-only generation through the same
        pipeline — no model numerics are asserted, only the contract that
        identity collapses to base.
        """
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(2)

        def identity_fn(*, f_base, f_src):
            return f_base

        result = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target,
            fn=identity_fn,
            params={},
            batch_size=2,
            output_scores=False,
        )

        # Generate base sequences without any intervention through the same pipeline.
        base_inputs = tiny_pipeline.load([ex["input"] for ex in dataset])
        base_inputs = {
            k: v.to(tiny_pipeline.model.device) for k, v in base_inputs.items()
        }
        with torch.no_grad():
            base_out = tiny_pipeline.model.generate(
                **base_inputs,
                max_new_tokens=tiny_pipeline.max_new_tokens,
                pad_token_id=tiny_pipeline.tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
        base_seqs = (
            base_out.sequences[:, -tiny_pipeline.max_new_tokens :].detach().cpu()
        )

        interv_seqs = torch.cat(result["sequences"], dim=0)
        assert torch.equal(interv_seqs, base_seqs)

    def test_full_interchange_fn_matches_run_interchange(self, tiny_pipeline) -> None:
        """``fn(f_base, f_src) -> f_src`` reproduces a full interchange.

        Compared against :func:`run_interchange_interventions` on the same
        target — both routes patch the source activation in unchanged
        (identity featurizer), so per-example sequences must match.
        """
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(2)

        def interchange_fn(*, f_base, f_src):
            return f_src

        interp_result = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target,
            fn=interchange_fn,
            params={},
            batch_size=2,
            output_scores=False,
        )

        # Fresh target for the interchange runner — the interpolation runner
        # may have mutated featurizer state on the units (it doesn't, but
        # rebuilding is the safe choice and matches caller patterns).
        unit2 = _make_residual_unit(tiny_pipeline, layer=0)
        target2 = InterchangeTarget([[unit2]])
        interchange_result = run_interchange_interventions(
            tiny_pipeline,
            dataset,
            target2,
            batch_size=2,
            output_scores=False,
        )

        interp_seqs = torch.cat(interp_result["sequences"], dim=0)
        interchange_seqs = torch.cat(interchange_result["sequences"], dim=0)
        assert torch.equal(interp_seqs, interchange_seqs)

    def test_batch_invariance_in_aggregated_payload(self, tiny_pipeline) -> None:
        """Running with ``batch_size=1`` vs ``batch_size=N`` produces identical
        concatenated sequences end-to-end on the tiny pipeline.

        Determinism comes from ``do_sample=False`` in
        :meth:`LMPipeline.intervenable_generate`.
        """
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target_small = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(4)

        def identity_fn(*, f_base, f_src):
            return f_base

        out_small = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target_small,
            fn=identity_fn,
            params={},
            batch_size=1,
            output_scores=False,
        )

        # Rebuild target+unit to avoid any in-place pyvene state carrying over.
        unit_big = _make_residual_unit(tiny_pipeline, layer=0)
        target_big = InterchangeTarget([[unit_big]])
        out_large = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target_big,
            fn=identity_fn,
            params={},
            batch_size=4,
            output_scores=False,
        )

        cat_small = torch.cat(out_small["sequences"], dim=0)
        cat_large = torch.cat(out_large["sequences"], dim=0)
        assert torch.equal(cat_small, cat_large)

    def test_cpu_residency_of_aggregated_outputs(self, tiny_pipeline) -> None:
        """Every tensor reachable through ``run_*`` output lands on CPU.

        Pins the ``move_outputs_to_cpu`` contract that downstream
        path-steering / pullback artifacts depend on.
        """
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(2)

        def fn(*, f_base, f_src):
            return f_base

        result = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target,
            fn=fn,
            params={},
            batch_size=2,
            output_scores=True,
        )

        for seq in result["sequences"]:
            assert seq.device.type == "cpu"
        for score_batch in result.get("scores", []):
            for s in score_batch:
                assert s.device.type == "cpu"

    def test_sweep_single_config_equals_run(self, tiny_pipeline) -> None:
        """``sweep_*({"a": (feat, fn, params)})["a"]`` matches a plain
        ``set_featurizer + run_*`` end-to-end on the tiny pipeline.

        Pins the sweep-equivalence contract: the wrapper does nothing more
        than apply the featurizer and delegate. Compared on per-example
        ``sequences``.
        """
        feat = Featurizer(id="null")

        def fn(*, f_base, f_src):
            return f_base

        dataset = _make_cf_dataset(2)

        # Sweep path: build target with the unit, sweep wraps set_featurizer.
        unit_sweep = _make_residual_unit(tiny_pipeline, layer=0)
        target_sweep = InterchangeTarget([[unit_sweep]])
        sweep_result = sweep_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target_sweep,
            configs={"a": (feat, fn, {})},
            batch_size=2,
            output_scores=False,
        )

        # Manual path: set the featurizer ourselves, then call run_* directly.
        unit_manual = _make_residual_unit(tiny_pipeline, layer=0)
        target_manual = InterchangeTarget([[unit_manual]])
        target_manual.set_featurizer(feat)
        run_result = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target_manual,
            fn=fn,
            params={},
            batch_size=2,
            output_scores=False,
        )

        sweep_seqs = torch.cat(sweep_result["a"]["sequences"], dim=0)
        run_seqs = torch.cat(run_result["sequences"], dim=0)
        assert torch.equal(sweep_seqs, run_seqs)

    def test_top_k_clamp_does_not_raise_for_large_k(self, tiny_pipeline) -> None:
        """``output_scores=int`` with ``k > vocab_size`` must not raise.

        Defers numerical correctness of the clamp to ``convert_to_top_k``'s
        own tests (``tests/neural/test_data_utils.py``); this property test
        only pins that the path-steering caller can pass an arbitrarily
        large k without surprises.
        """
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(1)

        def fn(*, f_base, f_src):
            return f_base

        vocab_size = tiny_pipeline.model.config.vocab_size
        result = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target,
            fn=fn,
            params={},
            batch_size=1,
            output_scores=vocab_size + 1_000,
        )

        # Scores still emitted; convert_to_top_k clamped k internally.
        assert "scores" in result
        assert len(result["scores"]) == 1

    def test_run_aggregates_one_entry_per_batch(self, tiny_pipeline) -> None:
        """``results["sequences"]`` has one entry per batch processed.

        End-to-end shape contract: 4 examples / batch_size=2 → 2 batches →
        2 entries in the aggregated list. Downstream collectors rely on
        ``len(results["sequences"])`` matching the batch count.
        """
        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(4)

        def fn(*, f_base, f_src):
            return f_base

        result = run_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target,
            fn=fn,
            params={},
            batch_size=2,
            output_scores=False,
        )

        assert len(result["sequences"]) == 2

    def test_sweep_in_insertion_order_preserves_keys(self, tiny_pipeline) -> None:
        """``sweep_*`` returns a dict whose keys preserve the insertion order
        of ``configs`` on the real pipeline.

        Caller iteration patterns (e.g. path-steering grid plots) rely on
        this contract — they index sweep results positionally and expect
        them to align with the config dict's key order.
        """

        def fn(*, f_base, f_src):
            return f_base

        unit = _make_residual_unit(tiny_pipeline, layer=0)
        target = InterchangeTarget([[unit]])
        dataset = _make_cf_dataset(1)

        configs = {
            "first": (Featurizer(id="null"), fn, {}),
            "second": (Featurizer(id="null"), fn, {}),
        }
        sweep_result = sweep_interpolation_interventions(
            tiny_pipeline,
            dataset,
            target,
            configs=configs,
            batch_size=1,
            output_scores=False,
        )
        assert list(sweep_result.keys()) == ["first", "second"]
        for key in ("first", "second"):
            assert "sequences" in sweep_result[key]


# --------------------------------------------------------------------------- #
#  Property-tier mock-free wiring assertion (kept lightweight)                #
# --------------------------------------------------------------------------- #
class TestSweepWiringProperty:
    """Sweep-equivalence at the dispatch level.

    These two assertions hover at the wiring layer of
    :func:`sweep_interpolation_interventions` (the only piece whose
    end-to-end equivalence is more naturally pinned through the per-call
    contract than through expensive tiny-real round-trips on every config).
    Kept as a *property* class because they assert an equivalence between
    two public API paths rather than testing a single function's wiring.
    """

    pytestmark = pytest.mark.property

    def test_sweep_dispatches_each_config_through_run(self) -> None:
        """``sweep_*`` dispatches one ``run_*`` call per config with the
        correct fn/params pair — equivalent to a hand-written loop.
        """
        target = MagicMock()
        feat = MagicMock()
        dataset = _make_test_dataset(2)

        def fn(*, f_base: torch.Tensor, f_src: torch.Tensor) -> torch.Tensor:
            return f_base

        payload = {"sequences": ["X"]}
        with patch(
            f"{MODULE}.run_interpolation_interventions", return_value=payload
        ) as mock_run:
            results = sweep_interpolation_interventions(
                MagicMock(),
                dataset,
                target,
                configs={"a": (feat, fn, {"alpha": 0.5})},
                batch_size=8,
                output_scores=False,
            )

        assert results == {"a": payload}
        target.set_featurizer.assert_called_once_with(feat)
        mock_run.assert_called_once_with(
            ANY, dataset, target, fn, {"alpha": 0.5}, 8, False
        )
