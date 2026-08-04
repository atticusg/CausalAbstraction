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


def _lerp(f_base: torch.Tensor, f_src: torch.Tensor, alpha: float) -> torch.Tensor:
    """The canonical blend: ``(1-alpha)*f_base + alpha*f_src``."""
    return (1 - alpha) * f_base + alpha * f_src


# --------------------------------------------------------------------------- #
#  set_interventions_interpolation                                            #
# --------------------------------------------------------------------------- #
class TestSetInterventionsInterpolationUnit:
    """``set_interventions_interpolation`` pushes ``fn`` + ``params`` onto every
    intervention exposing ``set_interpolation``.

    This is the single entry point by which the runner gets a user's
    interpolation function onto the intervention objects. If it silently drops
    the kwargs or skips an intervention, the model runs with stale or no
    interpolation and produces silently-wrong activations.

    It takes a plain sequence of interventions; the pyvene-era tuple-unwrapping
    branch is gone, because there is no intervention dict whose values might be
    ``(module, ...)`` tuples any more.
    """

    pytestmark = pytest.mark.unit

    def test_pushes_fn_and_params_onto_every_intervention(self) -> None:
        interventions = [_FakeIntervention(), _FakeIntervention()]

        def linear(
            *, f_base: torch.Tensor, f_src: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return (1 - alpha) * f_base + alpha * f_src

        set_interventions_interpolation(interventions, linear, alpha=0.3)

        for inter in interventions:
            assert inter.fn is linear
            assert inter.params == {"alpha": 0.3}
            assert inter.set_calls == 1

    def test_skips_intervention_without_set_interpolation(self) -> None:
        """A mixed run (interpolation at one site, another mode elsewhere) must
        not crash on the sites that take no interpolation function."""
        inter = _FakeIntervention()
        set_interventions_interpolation([inter, _BarePyveneStub()], _lerp, alpha=0.1)
        assert inter.set_calls == 1

    def test_calling_twice_replaces_rather_than_stacks(self) -> None:
        inter = _FakeIntervention()
        set_interventions_interpolation([inter], _lerp, alpha=0.1)
        set_interventions_interpolation([inter], _lerp, alpha=0.9)
        assert inter.params == {"alpha": 0.9}
        assert inter.set_calls == 2


class TestBatchedInterpolationInterventionUnit:
    """One batch: read the sources, blend, generate.

    These ran against a mocked pyvene ``IntervenableModel`` and asserted its
    lifecycle (constructed with ``intervention_type="interpolation"``, deleted
    afterwards). The nnsight engine has no such object — an intervention lives
    only for a trace — so the assertions are re-aimed at the behaviour that
    actually matters: the blend the caller asked for is the blend applied, and
    the returned dict has the documented shape.

    ``docs/PATH_PATCHING.md``-style numerical equivalence lives in
    ``test_interpolation_hook_oracle.py``; this class covers the wiring.
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _run(pipeline, fn, params, *, output_scores=True):
        target = InterchangeTarget([[_make_residual_unit(pipeline, layer=1)]])
        return batched_interpolation_intervention(
            pipeline,
            _make_cf_dataset(2),
            target,
            fn=fn,
            params=params,
            output_scores=output_scores,
        )

    def test_alpha_zero_leaves_the_base_untouched(self, tiny_pipeline) -> None:
        """``(1-α)·f_base + α·f_src`` at α=0 is the identity, so the logits must
        equal an un-intervened run's."""
        blended = self._run(tiny_pipeline, _lerp, {"alpha": 0.0})
        target = InterchangeTarget([[_make_residual_unit(tiny_pipeline, layer=1)]])
        untouched = run_interchange_interventions(
            tiny_pipeline,
            [
                {"input": ex["input"], "counterfactual_inputs": [ex["input"]]}
                for ex in _make_cf_dataset(2)
            ],
            target,
            batch_size=2,
            output_scores=True,
        )
        torch.testing.assert_close(
            blended["scores"][0], untouched["scores"][0][0], atol=1e-5, rtol=1e-4
        )

    def test_alpha_one_equals_interchange(self, tiny_pipeline) -> None:
        """At α=1 the blend keeps only the source — i.e. a plain interchange."""
        blended = self._run(tiny_pipeline, _lerp, {"alpha": 1.0})
        target = InterchangeTarget([[_make_residual_unit(tiny_pipeline, layer=1)]])
        swapped = run_interchange_interventions(
            tiny_pipeline,
            _make_cf_dataset(2),
            target,
            batch_size=2,
            output_scores=True,
        )
        torch.testing.assert_close(
            blended["scores"][0], swapped["scores"][0][0], atol=1e-5, rtol=1e-4
        )

    def test_params_reach_the_interpolation_function(self, tiny_pipeline) -> None:
        """A different ``alpha`` must produce a different blend — otherwise the
        parameters are being dropped somewhere between caller and intervention."""
        quarter = self._run(tiny_pipeline, _lerp, {"alpha": 0.25})
        three_quarters = self._run(tiny_pipeline, _lerp, {"alpha": 0.75})
        assert not torch.allclose(
            quarter["scores"][0], three_quarters["scores"][0], atol=1e-6
        )

    def test_arbitrary_callable_is_used(self, tiny_pipeline) -> None:
        """Any ``fn(f_base, f_src, **params)`` works, not just a linear blend."""

        def elementwise_max(f_base, f_src):
            return torch.maximum(f_base, f_src)

        out = self._run(tiny_pipeline, elementwise_max, {})
        lerped = self._run(tiny_pipeline, _lerp, {"alpha": 0.5})
        assert not torch.allclose(out["scores"][0], lerped["scores"][0], atol=1e-6)

    def test_returns_documented_keys(self, tiny_pipeline) -> None:
        out = self._run(tiny_pipeline, _lerp, {"alpha": 0.5})
        assert set(out) == {"sequences", "scores", "string"}
        assert out["sequences"].shape[0] == 2

    def test_output_scores_false_omits_scores(self, tiny_pipeline) -> None:
        out = self._run(tiny_pipeline, _lerp, {"alpha": 0.5}, output_scores=False)
        assert "scores" not in out


class TestRunInterpolationInterventionsUnit:
    """The dataset loop: batch, blend, aggregate.

    Like the class above, re-aimed from a mocked ``IntervenableModel``'s
    lifecycle onto the observable contract.
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _run(pipeline, n, batch_size, *, output_scores=True):
        target = InterchangeTarget([[_make_residual_unit(pipeline, layer=1)]])
        return run_interpolation_interventions(
            pipeline,
            _make_cf_dataset(n),
            target,
            fn=_lerp,
            params={"alpha": 0.5},
            batch_size=batch_size,
            output_scores=output_scores,
        )

    def test_aggregates_one_entry_per_batch(self, tiny_pipeline) -> None:
        """5 examples at batch_size=2 -> 3 batches, and the per-batch entries
        together cover every example."""
        out = self._run(tiny_pipeline, 5, 2)
        assert len(out["sequences"]) == 3
        assert sum(s.shape[0] for s in out["sequences"]) == 5

    def test_batching_does_not_change_results(self, tiny_pipeline) -> None:
        """Batch size is a memory knob, not a semantic one: the same dataset
        scored one-at-a-time and all-at-once must agree."""
        one_at_a_time = self._run(tiny_pipeline, 4, 1)
        all_at_once = self._run(tiny_pipeline, 4, 4)
        single = torch.cat([s[0] for s in one_at_a_time["scores"]])
        batched = all_at_once["scores"][0][0]
        torch.testing.assert_close(single, batched, atol=1e-5, rtol=1e-4)

    def test_output_scores_false_omits_scores_key(self, tiny_pipeline) -> None:
        out = self._run(tiny_pipeline, 2, 2, output_scores=False)
        assert "scores" not in out
        assert "sequences" in out

    def test_output_scores_int_returns_top_k(self, tiny_pipeline) -> None:
        out = self._run(tiny_pipeline, 2, 2, output_scores=4)
        assert "scores" in out
        top_k = out["scores"][0][0]
        assert top_k["top_k_logits"].shape[-1] == 4

    def test_results_are_on_cpu(self, tiny_pipeline) -> None:
        out = self._run(tiny_pipeline, 2, 2)
        for sequences in out["sequences"]:
            assert sequences.device.type == "cpu"
        for batch_scores in out["scores"]:
            for step in batch_scores:
                assert step.device.type == "cpu"


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
