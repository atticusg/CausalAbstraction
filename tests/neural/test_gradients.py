"""Gradients must flow through an intervention, and be the *right* gradients.

DAS trains a rotation applied inside the model's forward. Under the nnsight
backbone the rotation is applied by ordinary torch ops in the trace body, so
whether autograd reaches it is a property of the interleaver, not of causalab —
and a wrong answer here is silent: training would run, report scores, and have
learned nothing.

``tests/methods/test_train_intervention.py`` covers the loop (parameters move,
the base stays frozen). This file covers the layer beneath it, against a ground
truth built with raw forward hooks: the same intervention applied by hand, with
the same backward, compared element-wise. Both the forward *and* the gradient
are compared, so a passing gradient cannot be explained by both paths being
wrong in the same way.

Ported from the migration's phase-0 spike, which gated the whole port.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.LM_units import ResidualStream
from causalab.neural.activations.engine import (
    build_plans,
    forward_with_interventions,
)
from causalab.neural.token_positions import TokenPosition
from causalab.neural.units import InterchangeTarget

pytestmark = pytest.mark.unit

LAYER = 1
RANK = 3
BASE = "the capital of france is"
SOURCE = "the capital of italy is"
TARGET_TOKEN = torch.tensor([1234])


def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _rotation(hidden: int, seed: int = 0) -> SubspaceFeaturizer:
    torch.manual_seed(seed)
    return SubspaceFeaturizer(shape=(hidden, RANK), trainable=True)


def _rotation_grad(featurizer: SubspaceFeaturizer) -> torch.Tensor:
    parameter = featurizer.featurizer.rotate.parametrizations.weight.original
    assert parameter.grad is not None, "no gradient reached the rotation"
    return parameter.grad


def _last_token(pipeline) -> TokenPosition:
    def index(trace):
        return [len(pipeline.tokenizer(trace["raw_input"])["input_ids"]) - 1]

    return TokenPosition(index, pipeline, id="last_token")


def _oracle(pipeline, featurizer: SubspaceFeaturizer) -> torch.Tensor:
    """Interchange the last-token residual at LAYER, by hand, and backprop.

    Raw forward hooks only — no engine — so this is an independent statement of
    what the forward and the gradient should be.
    """
    model = pipeline.model
    base = pipeline.load([_trace(BASE)])
    source = pipeline.load([_trace(SOURCE)])

    captured: dict[str, torch.Tensor] = {}

    def grab(_m, _inp, out):
        captured["src"] = (out[0] if isinstance(out, tuple) else out)[:, -1, :]

    handle = model.model.layers[LAYER].register_forward_hook(grab)
    with torch.no_grad():
        model(input_ids=source["input_ids"], attention_mask=source["attention_mask"])
    handle.remove()
    src = captured["src"]

    def patch(_m, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        f_base, error = featurizer.featurize(hidden[:, -1, :])
        f_src, _ = featurizer.featurize(src)
        patched = featurizer.inverse_featurize(f_src, error)
        merged = torch.cat([hidden[:, :-1, :], patched.unsqueeze(1)], dim=1)
        return (merged,) + tuple(out[1:]) if isinstance(out, tuple) else merged

    handle = model.model.layers[LAYER].register_forward_hook(patch)
    logits = model(
        input_ids=base["input_ids"], attention_mask=base["attention_mask"]
    ).logits[:, -1, :]
    handle.remove()
    return logits


def _via_engine(pipeline, featurizer: SubspaceFeaturizer) -> torch.Tensor:
    unit = ResidualStream(
        layer=LAYER,
        token_indices=_last_token(pipeline),
        target_output=True,
        shape=(pipeline.model.config.hidden_size,),
        featurizer=featurizer,
    )
    from causalab.neural.activations.interchange_mode import (
        collect_group_sources,
        prepare_interchange_batch,
    )

    batch = prepare_interchange_batch(
        pipeline,
        [{"input": _trace(BASE), "counterfactual_inputs": [_trace(SOURCE)]}],
        InterchangeTarget([[unit]]),
    )
    sources = collect_group_sources(pipeline, batch)
    plans = build_plans(
        batch.units, batch.base_positions, "interchange", sources=sources
    )
    return forward_with_interventions(pipeline, batch.base_encoding, plans).logits[
        :, -1, :
    ]


class TestGradientsThroughInterventions:
    def test_forward_and_gradient_match_the_hook_oracle(self, tiny_pipeline) -> None:
        """The decisive check: same logits *and* same rotation gradient."""
        hidden = tiny_pipeline.model.config.hidden_size
        tiny_pipeline.model.requires_grad_(False)
        tiny_pipeline.model.eval()

        oracle_featurizer = _rotation(hidden)
        oracle_logits = _oracle(tiny_pipeline, oracle_featurizer)
        nn.functional.cross_entropy(oracle_logits, TARGET_TOKEN).backward()

        engine_featurizer = _rotation(hidden)
        engine_logits = _via_engine(tiny_pipeline, engine_featurizer)
        nn.functional.cross_entropy(engine_logits, TARGET_TOKEN).backward()

        torch.testing.assert_close(
            engine_logits.detach(), oracle_logits.detach(), atol=1e-5, rtol=1e-4
        )
        oracle_grad = _rotation_grad(oracle_featurizer)
        engine_grad = _rotation_grad(engine_featurizer)
        # A vacuous test would pass with both gradients at zero.
        assert float(oracle_grad.norm()) > 1e-8
        torch.testing.assert_close(engine_grad, oracle_grad, atol=1e-6, rtol=1e-4)

    def test_saved_logits_still_carry_a_graph(self, tiny_pipeline) -> None:
        """A trace's output must remain differentiable after the trace exits —
        otherwise the loss would be computed on a detached tensor and every
        backward would be a no-op."""
        hidden = tiny_pipeline.model.config.hidden_size
        tiny_pipeline.model.requires_grad_(False)
        logits = _via_engine(tiny_pipeline, _rotation(hidden))
        assert logits.requires_grad
        assert logits.grad_fn is not None

    def test_no_gradient_leaks_into_the_frozen_model(self, tiny_pipeline) -> None:
        hidden = tiny_pipeline.model.config.hidden_size
        tiny_pipeline.model.requires_grad_(False)
        tiny_pipeline.model.zero_grad(set_to_none=True)

        logits = _via_engine(tiny_pipeline, _rotation(hidden))
        nn.functional.cross_entropy(logits, TARGET_TOKEN).backward()

        leaked = [
            name
            for name, parameter in tiny_pipeline.model.named_parameters()
            if parameter.grad is not None and parameter.grad.any()
        ]
        assert not leaked, f"gradient leaked into the base model: {leaked[:3]}"


class TestPerHeadGradients:
    """A per-head write goes through a reshape and a scatter. Both must stay in
    the autograd graph, or per-head DAS would train nothing and say so only by
    producing a flat score."""

    def test_gradient_reaches_a_per_head_rotation(self, tiny_pipeline) -> None:
        from causalab.neural.LM_units import AttentionHead
        from causalab.neural.activations.interchange_mode import (
            collect_group_sources,
            prepare_interchange_batch,
        )
        from causalab.neural.components import head_layout

        tiny_pipeline.model.requires_grad_(False)
        tiny_pipeline.model.eval()
        _n_q, _n_kv, head_dim = head_layout(tiny_pipeline.model.config)
        featurizer = _rotation(head_dim)

        unit = AttentionHead(
            layer=0,
            head=1,
            token_indices=_last_token(tiny_pipeline),
            shape=(head_dim,),
            featurizer=featurizer,
        )
        batch = prepare_interchange_batch(
            tiny_pipeline,
            [{"input": _trace(BASE), "counterfactual_inputs": [_trace(SOURCE)]}],
            InterchangeTarget([[unit]]),
        )
        sources = collect_group_sources(tiny_pipeline, batch)
        plans = build_plans(
            batch.units, batch.base_positions, "interchange", sources=sources
        )
        logits = forward_with_interventions(
            tiny_pipeline, batch.base_encoding, plans
        ).logits[:, -1, :]

        nn.functional.cross_entropy(logits, TARGET_TOKEN).backward()
        grad = _rotation_grad(featurizer)
        assert torch.isfinite(grad).all()
        assert float(grad.norm()) > 0
