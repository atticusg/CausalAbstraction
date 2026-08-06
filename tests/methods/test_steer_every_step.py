"""Steering that stays on for the whole generation, not just the prompt.

By default an intervention lands on each site's first occurrence — the prompt
prefill — so it changes how the model *reads* the input but not what it does
while writing. ``every_step=True`` re-applies it to each token the model
generates, which is what "keep the steering on" and "run the whole generation
with this head ablated" mean.

Two details make this more than wrapping the body in a loop, and both are pinned
below. A decode step's activation holds only the token just produced, so a
position resolved against the prompt is out of range there and every plan has to
address the single position that exists. And the loop must be bounded: an
unbounded ``tracer.all()`` would discard ``tracer.result``, so the generated ids
would never come back.

Only modes whose source is position-independent are offered this — a steering
direction or a fixed replacement vector means the same thing at any token. An
interchange source was gathered at particular prompt positions and has no
reading at a token that did not exist yet.
"""

from __future__ import annotations

import pytest
import torch

from causalab.methods.steer.steer import run_steering_interventions
from causalab.neural.LM_units import ResidualStream
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget

from tests.neural.activations.hook_oracle import make_trace

pytestmark = pytest.mark.property

_PROMPT = "the quick brown fox jumps"
_STEPS = 6


def _target(pipeline: LMPipeline) -> tuple[InterchangeTarget, str, int]:
    hidden = pipeline.model.config.hidden_size
    tp = TokenPosition(lambda _x: [1], pipeline, id="p1")
    unit = ResidualStream(
        layer=0, token_indices=tp, target_output=True, shape=(hidden,)
    )
    return InterchangeTarget([[unit]]), unit.id, hidden


def _run(pipeline: LMPipeline, magnitude: float, *, every_step: bool) -> torch.Tensor:
    target, unit_id, hidden = _target(pipeline)
    vectors = {unit_id: torch.full((hidden,), magnitude)}
    out = run_steering_interventions(
        pipeline,
        [{"input": make_trace(_PROMPT), "counterfactual_inputs": []}],
        target,
        vectors,
        batch_size=1,
        every_step=every_step,
    )
    return out["sequences"][0][0]


class TestSteeringDuringGeneration:
    def test_prefill_token_is_unchanged_by_every_step(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """The first generated token is decided by the prefill, which both modes
        treat identically — so it must match exactly.

        This is what separates "the steering now also applies while generating"
        from "the steering changed". If this token moved, the per-step loop would
        have altered the prompt pass as well.
        """
        mock_tiny_lm.max_new_tokens = _STEPS
        prefill_only = _run(mock_tiny_lm, 8.0, every_step=False)
        every_step = _run(mock_tiny_lm, 8.0, every_step=True)
        assert int(prefill_only[-_STEPS]) == int(every_step[-_STEPS])

    def test_later_tokens_diverge(self, mock_tiny_lm: LMPipeline) -> None:
        """Every token after the first must be free to differ — otherwise the
        per-step application is not reaching the decode steps at all."""
        mock_tiny_lm.max_new_tokens = _STEPS
        prefill_only = _run(mock_tiny_lm, 8.0, every_step=False)
        every_step = _run(mock_tiny_lm, 8.0, every_step=True)
        assert not torch.equal(prefill_only[-_STEPS + 1 :], every_step[-_STEPS + 1 :])

    def test_zero_vector_is_a_noop_at_every_step(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A zero steering vector added at every step must change nothing.

        The non-vacuity guard for the two tests above: they show the per-step
        loop *can* alter generation, and this shows it alters it only through the
        vector it was given, rather than by disturbing the decode pass — a
        wrong position or a double-applied intervention would show up here.
        """
        mock_tiny_lm.max_new_tokens = _STEPS
        prefill_only = _run(mock_tiny_lm, 0.0, every_step=False)
        every_step = _run(mock_tiny_lm, 0.0, every_step=True)
        torch.testing.assert_close(prefill_only, every_step, atol=0, rtol=0)

    def test_single_step_generation_is_unaffected(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """With one generated token there are no decode steps to re-apply to, so
        the two modes must agree — the bounded loop simply does not run."""
        mock_tiny_lm.max_new_tokens = 1
        prefill_only = _run(mock_tiny_lm, 8.0, every_step=False)
        every_step = _run(mock_tiny_lm, 8.0, every_step=True)
        torch.testing.assert_close(prefill_only, every_step, atol=0, rtol=0)


class TestEveryStepRejectsPositionBoundSources:
    """Carrying an intervention into generation is only meaningful for some modes.

    A steering direction or a fixed replacement vector is a point in feature
    space: it means the same thing at any token, so re-applying it to one the
    model just wrote is well defined. An interchange source is a *reading* taken
    from another run at particular prompt positions, and a token that did not
    exist when it was gathered has no counterpart among them.

    Left unguarded this failed in two ways, both bad. A multi-position span died
    inside the scatter with a bare shape mismatch naming no cause; a
    single-position one succeeded silently and applied a prompt activation to a
    generated token.
    """

    def _plans(self, pipeline, mode: str, source):
        from causalab.neural.LM_units import ResidualStream
        from causalab.neural.activations.engine import build_plans
        from causalab.neural.token_positions import TokenPosition

        hidden = pipeline.model.config.hidden_size
        unit = ResidualStream(
            layer=0,
            token_indices=TokenPosition(lambda _x: [1], pipeline, id="p1"),
            target_output=True,
            shape=(hidden,),
        )
        return build_plans([unit], [[[1]]], mode, sources=[source])

    def test_interchange_is_refused(self, mock_tiny_lm: LMPipeline) -> None:
        from causalab.neural.activations.engine import generate_with_interventions

        hidden = mock_tiny_lm.model.config.hidden_size
        plans = self._plans(mock_tiny_lm, "interchange", torch.zeros(1, hidden))
        encoding = mock_tiny_lm.load([make_trace(_PROMPT)])
        with pytest.raises(ValueError, match="read at prompt positions"):
            generate_with_interventions(mock_tiny_lm, encoding, plans, every_step=True)

    def test_the_message_names_the_mode_and_unit(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        from causalab.neural.activations.engine import generate_with_interventions

        hidden = mock_tiny_lm.model.config.hidden_size
        plans = self._plans(mock_tiny_lm, "interchange", torch.zeros(1, hidden))
        encoding = mock_tiny_lm.load([make_trace(_PROMPT)])
        with pytest.raises(ValueError, match="InterchangeIntervention"):
            generate_with_interventions(mock_tiny_lm, encoding, plans, every_step=True)

    def test_interchange_still_runs_on_the_prompt(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """The guard is about carrying the source forward, not about interchange:
        the default path must be untouched."""
        from causalab.neural.activations.engine import generate_with_interventions

        hidden = mock_tiny_lm.model.config.hidden_size
        plans = self._plans(mock_tiny_lm, "interchange", torch.zeros(1, hidden))
        encoding = mock_tiny_lm.load([make_trace(_PROMPT)])
        out = generate_with_interventions(mock_tiny_lm, encoding, plans)
        assert out.sequences.shape[0] == 1

    @pytest.mark.parametrize("mode", ["add", "replace"])
    def test_positionless_modes_are_allowed(
        self, mock_tiny_lm: LMPipeline, mode: str
    ) -> None:
        """Steering (``add``) and replace carry over, and must not be caught
        by the guard — otherwise it would block the feature it was written for."""
        from causalab.neural.activations.engine import generate_with_interventions

        hidden = mock_tiny_lm.model.config.hidden_size
        plans = self._plans(mock_tiny_lm, mode, torch.zeros(1, 1, hidden))
        encoding = mock_tiny_lm.load([make_trace(_PROMPT)])
        out = generate_with_interventions(
            mock_tiny_lm, encoding, plans, every_step=True
        )
        assert out.sequences.shape[0] == 1
