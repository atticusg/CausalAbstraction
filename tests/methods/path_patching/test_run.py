"""Tests for the path-patching runner (``methods.path_patching.run``).

The tiny-random model has no IOI behaviour, so the integration checks pin
*wiring*: every receiver regime runs its Plan lowering end to end through the
public functions and scores one finite value per cell. Property tier
(path-equivalence / shape contracts); ``TestLeftPadGenerateParity`` is the
numerical_unit left-pad position_ids gate.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.metric import InterchangeMetric
from causalab.methods.path_patching.run import (
    run_path_patching,
    run_path_patching_scan,
)
from causalab.methods.path_patching.targets import (
    ReceiverSpec,
    build_restorer_sites,
)
from causalab.neural.activations.site_grids import build_attention_head_sites
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_last_token_index


def _trace(text: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def _dataset() -> list[dict[str, Any]]:
    return [
        {
            "input": _trace("hello world"),
            "counterfactual_inputs": [_trace("blue green")],
        },
        {"input": _trace("red sky"), "counterfactual_inputs": [_trace("cat dog")]},
    ]


def _last_token(pipeline: LMPipeline) -> TokenPosition:
    return TokenPosition(
        lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
    )


def _first_logit_metric() -> InterchangeMetric:
    """Trivial scores-based metric: the patched first-position logit at index 0.

    Exercises the full-vocab ``output_scores=True`` scoring path without needing
    task behaviour — enough to confirm the scan produces one finite value per cell.
    """

    def fn(
        intervention_output: dict[str, Any],
        _expected: dict[str, Any],
        _original: dict[str, Any],
    ) -> float:
        idx = intervention_output["example_idx"]
        return float(intervention_output["scores"][0][idx][0])

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=False,
        needs_scores=True,
    )


class TestOutputScan:
    pytestmark = pytest.mark.property

    def test_run_path_patching_returns_scores(self, mock_tiny_lm: LMPipeline) -> None:
        sender = build_attention_head_sites(
            mock_tiny_lm, [0], [0], _last_token(mock_tiny_lm)
        )[(0, 0)][0][0]
        out = run_path_patching(mock_tiny_lm, _dataset(), sender, batch_size=2)
        assert out.scores

    @pytest.mark.parametrize("restore", [("attention", "mlp"), ("attention",)])
    def test_scan_one_finite_score_per_cell(
        self, mock_tiny_lm: LMPipeline, restore: tuple[str, ...]
    ) -> None:
        layers, heads = [0, 1], [0, 1]
        targets = build_attention_head_sites(
            mock_tiny_lm, layers, heads, _last_token(mock_tiny_lm)
        )
        senders = {key: t[0][0] for key, t in targets.items()}

        scores = run_path_patching_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            metric=_first_logit_metric(),
            restore=restore,
            batch_size=2,
        )

        assert set(scores.keys()) == set(senders.keys())
        assert all(torch.isfinite(torch.tensor(v)) for v in scores.values())


class TestTwoPassReceivers:
    """Two-pass (PASS-1 collect under interchange, PASS-2 inject) for internal
    receivers. The tiny-random model has no real behaviour, so these pin *wiring*:
    each internal receiver runs the mixed collect+interchange forward and the
    source-representation injection, and the scan returns one finite score per
    sender cell."""

    pytestmark = pytest.mark.property

    def _receiver(self, kind: str, pos: TokenPosition) -> ReceiverSpec:
        if kind == "head_value_input":
            return ReceiverSpec(kind=kind, layer=1, head=0, token_position=pos)
        if kind == "residual":
            return ReceiverSpec(kind=kind, layer=1, token_position=pos)
        return ReceiverSpec(kind=kind, layer=1, token_position=pos)

    @pytest.mark.parametrize("kind", ["head_value_input", "mlp_input", "residual"])
    def test_internal_receiver_scan_finite(
        self, mock_tiny_lm: LMPipeline, kind: str
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        targets = build_attention_head_sites(mock_tiny_lm, [0], [0, 1], pos)
        senders = {key: t[0][0] for key, t in targets.items()}

        scores = run_path_patching_scan(
            mock_tiny_lm,
            _dataset(),
            senders,
            metric=_first_logit_metric(),
            receiver=self._receiver(kind, pos),
            restore=("attention", "mlp"),
            batch_size=2,
        )

        assert set(scores.keys()) == set(senders.keys())
        assert all(torch.isfinite(torch.tensor(v)) for v in scores.values())

    def test_gqa_head_value_scan_finite(self, gqa_tiny_lm: LMPipeline) -> None:
        """End-to-end head_value_input on a grouped-query-attention model: the
        query head maps to its KV group, and the two-pass mixed forward + injection
        must run on the GQA value-head shapes (split by num_key_value_heads)."""
        cfg = gqa_tiny_lm.model.config
        assert cfg.num_key_value_heads < cfg.num_attention_heads
        pos = _last_token(gqa_tiny_lm)
        targets = build_attention_head_sites(gqa_tiny_lm, [0], [0, 1], pos)
        senders = {key: t[0][0] for key, t in targets.items()}
        scores = run_path_patching_scan(
            gqa_tiny_lm,
            _dataset(),
            senders,
            metric=_first_logit_metric(),
            # A high query head to exercise the query→KV-group mapping (head 3 -> group 1).
            receiver=ReceiverSpec(
                kind="head_value_input",
                layer=1,
                head=cfg.num_attention_heads - 1,
                token_position=pos,
            ),
            restore=("attention", "mlp"),
            batch_size=2,
        )
        assert set(scores.keys()) == set(senders.keys())
        assert all(torch.isfinite(torch.tensor(v)) for v in scores.values())

    def test_decoupled_head_dim_value_scan_finite(
        self, gqa_decoupled_head_dim_lm: LMPipeline
    ) -> None:
        """A head_value_input scan on a model whose head_dim is decoupled from
        hidden // n_head runs to finite scores: HeadSite slices value vectors at
        the true config.head_dim, lifting pyvene 0.1.8's refusal (which inferred
        the width from the head count and could not realize this receiver)."""
        cfg = gqa_decoupled_head_dim_lm.model.config
        assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
        pos = _last_token(gqa_decoupled_head_dim_lm)
        targets = build_attention_head_sites(gqa_decoupled_head_dim_lm, [0], [0], pos)
        senders = {key: t[0][0] for key, t in targets.items()}
        scores = run_path_patching_scan(
            gqa_decoupled_head_dim_lm,
            _dataset(),
            senders,
            metric=_first_logit_metric(),
            receiver=ReceiverSpec(
                kind="head_value_input", layer=1, head=0, token_position=pos
            ),
            restore=("attention", "mlp"),
            batch_size=2,
        )
        assert set(scores.keys()) == set(senders.keys())
        assert all(torch.isfinite(torch.tensor(v)) for v in scores.values())

    def test_run_path_patching_internal_returns_scores(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = build_attention_head_sites(mock_tiny_lm, [0], [0], pos)[(0, 0)][0][0]
        out = run_path_patching(
            mock_tiny_lm,
            _dataset(),
            sender,
            ReceiverSpec(kind="mlp_input", layer=1, token_position=pos),
            batch_size=2,
        )
        assert out.scores

    def test_empty_restorer_two_pass_through_runner(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        """A top-layer sender under attention-only restore has no restorers between
        it and the receiver, so the two-pass target collapses to a single
        interchange group. The runner still feeds the [source, base] dataset; the
        extra base column is silently dropped (n_groups == 1) and PASS 1 runs with
        sources=[source, None]. Pin that this empty-restorer path works end to end."""
        n_layers = mock_tiny_lm.model.config.num_hidden_layers
        pos = _last_token(mock_tiny_lm)
        sender = build_attention_head_sites(mock_tiny_lm, [n_layers - 1], [0], pos)[
            (n_layers - 1, 0)
        ][0][0]
        receiver = ReceiverSpec(kind="residual", layer=n_layers - 1, token_position=pos)
        # Precondition: this configuration really does produce an empty restorer set.
        assert (
            build_restorer_sites(mock_tiny_lm, sender, receiver, restore=("attention",))
            == []
        )
        out = run_path_patching(
            mock_tiny_lm,
            _dataset(),
            sender,
            receiver,
            restore=("attention",),
            batch_size=2,
        )
        assert out.scores


class TestDownstreamSenderGuard:
    """A sender at/downstream of an internal receiver has no forward path, so the
    methods runner raises rather than spend a (structurally-zero) forward pass."""

    pytestmark = pytest.mark.property

    def test_run_path_patching_raises_on_downstream(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = build_attention_head_sites(mock_tiny_lm, [1], [0], pos)[(1, 0)][0][0]
        with pytest.raises(ValueError, match="not upstream"):
            run_path_patching(
                mock_tiny_lm,
                _dataset(),
                sender,
                ReceiverSpec(kind="mlp_input", layer=0, token_position=pos),
                batch_size=2,
            )

    def test_scan_raises_on_downstream(self, mock_tiny_lm: LMPipeline) -> None:
        pos = _last_token(mock_tiny_lm)
        senders = {
            key: t[0][0]
            for key, t in build_attention_head_sites(
                mock_tiny_lm, [1], [0], pos
            ).items()
        }
        with pytest.raises(ValueError, match="not upstream"):
            run_path_patching_scan(
                mock_tiny_lm,
                _dataset(),
                senders,
                metric=_first_logit_metric(),
                receiver=ReceiverSpec(kind="mlp_input", layer=0, token_position=pos),
                batch_size=2,
            )


class TestTwoPassPropagation:
    """The PASS-1 semantics: the injected v* is the receiver's activation UNDER
    the sender patch (otherwise the two-pass would be a no-op). With the sender
    at layer 0 and an mlp_input receiver at layer 1 (nothing frozen between
    them), a real source moves the final logits away from clean, while a *null*
    interchange (source == base) reproduces the clean run exactly."""

    pytestmark = pytest.mark.property

    def test_source_sender_changes_output_null_does_not(
        self, mock_tiny_lm: LMPipeline
    ) -> None:
        pos = _last_token(mock_tiny_lm)
        sender = build_attention_head_sites(mock_tiny_lm, [0], [0], pos)[(0, 0)][0][0]
        receiver = ReceiverSpec(kind="mlp_input", layer=1, token_position=pos)
        ds = _dataset()
        null_ds = [
            {"input": ex["input"], "counterfactual_inputs": [ex["input"]]} for ex in ds
        ]
        clean = mock_tiny_lm.generate([ex["input"] for ex in ds]).scores[0]
        real = run_path_patching(
            mock_tiny_lm, ds, sender, receiver, restore=("attention",), batch_size=2
        ).scores[0]
        null = run_path_patching(
            mock_tiny_lm,
            null_ds,
            sender,
            receiver,
            restore=("attention",),
            batch_size=2,
        ).scores[0]
        assert not torch.allclose(real, clean, atol=1e-4)
        torch.testing.assert_close(null, clean, atol=1e-4, rtol=1e-3)


class TestReceiverSpanGuard:
    """The two-pass receiver index is spliced into the pyvene locations *outside*
    ``prepare_intervenable_inputs``, so it would bypass the ragged-span guard the
    interchange groups get. A uniform multi-token receiver span is fine, but a
    *ragged* one (width varying across the batch) must raise the actionable
    ``ValueError`` rather than reach pyvene's gather as the cryptic ``expected
    sequence of length N`` error (latent today: every receiver reads at the last
    token, a uniform single-token span)."""

    pytestmark = pytest.mark.property

    def test_ragged_receiver_position_raises(self, mock_tiny_lm: LMPipeline) -> None:
        # Empty-restorer config so the receiver is the *only* unit with a ragged
        # span: a top-layer sender under attention-only restore has no restorers
        # between it and the receiver, so the interchange group is just the
        # (single-token, uniform) sender — the interchange-group guard cannot fire,
        # isolating the receiver path.
        n_layers = mock_tiny_lm.model.config.num_hidden_layers

        # Width follows the prompt's word count, so the span is ragged *across* the
        # batch (the 4-word base selects 2 tokens, the 2-word base selects 1) while
        # each example's own positions are well-formed.
        def ragged_idx(x):
            return [0, 1] if len(x["raw_input"].split()) >= 3 else [0]

        ragged = TokenPosition(ragged_idx, mock_tiny_lm, id="ragged")
        dataset = [
            {
                "input": _trace("the quick brown fox"),
                "counterfactual_inputs": [_trace("a slow lazy dog")],
            },
            {
                "input": _trace("hi there"),
                "counterfactual_inputs": [_trace("yo there")],
            },
        ]
        sender = build_attention_head_sites(
            mock_tiny_lm, [n_layers - 1], [0], _last_token(mock_tiny_lm)
        )[(n_layers - 1, 0)][0][0]
        receiver = ReceiverSpec(
            kind="residual", layer=n_layers - 1, token_position=ragged
        )
        # Precondition: this really is the empty-restorer two-pass path.
        assert (
            build_restorer_sites(mock_tiny_lm, sender, receiver, restore=("attention",))
            == []
        )
        with pytest.raises(
            ValueError, match="path-patching receiver selects a variable number"
        ):
            run_path_patching(
                mock_tiny_lm,
                dataset,
                sender,
                receiver,
                restore=("attention",),
                batch_size=2,
            )


def _mixed_length_dataset() -> list[dict[str, Any]]:
    """Counterfactual examples whose prompts differ in token length, so batching
    them left-pads the short rows. Each base/counterfactual pair is read at the
    last token (a single, uniform-width position), so the length differences do
    not trip the ragged-span guard — they only exercise left padding."""
    return [
        {
            "input": _trace("the quick brown fox jumps over the lazy dog today"),
            "counterfactual_inputs": [
                _trace("a slow green turtle crawls under the old stone bridge now")
            ],
        },
        {"input": _trace("hi there"), "counterfactual_inputs": [_trace("yo friend")]},
        {
            "input": _trace("the cat sat quietly on the warm windowsill"),
            "counterfactual_inputs": [_trace("a dog ran loudly through the cold yard")],
        },
        {"input": _trace("red sky"), "counterfactual_inputs": [_trace("blue sea")]},
    ]


class TestLeftPadGenerateParity:
    """End-to-end guard for the left-pad ``position_ids`` fix on the *generate*
    paths (the output single-pass and PASS 2 of the two-pass).

    A path-patching score over one left-padded mixed-length batch must equal the
    score computed one example per batch (``batch_size=1`` → no padding, the
    reference). On an **absolute-position** model (``tiny_gpt2_lm``) this fails
    without ``position_ids`` on the generate forwards — the padded prompt is
    numbered from the pad tokens and every activation in it is mis-encoded (the
    band bug). On a **RoPE** model (``mock_tiny_lm``) it holds either way, since
    relative positions are invariant to a uniform left-pad shift. Parametrizing
    both pins "GPT-2 is fixed" and "Llama is unchanged" together.
    """

    pytestmark = pytest.mark.numerical_unit

    @staticmethod
    def _assert_left_pads(pipeline: LMPipeline, dataset: list[dict[str, Any]]) -> None:
        mask = pipeline.load([ex["input"] for ex in dataset])["attention_mask"]
        assert (mask == 0).any(), "batch did not introduce left padding"

    @pytest.mark.parametrize("pipeline_fixture", ["mock_tiny_lm", "tiny_gpt2_lm"])
    def test_output_receiver_bs_parity(
        self, pipeline_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        pipeline = request.getfixturevalue(pipeline_fixture)
        dataset = _mixed_length_dataset()
        self._assert_left_pads(pipeline, dataset)
        pos = _last_token(pipeline)
        targets = build_attention_head_sites(pipeline, [0, 1], [0, 1], pos)
        senders = {key: t[0][0] for key, t in targets.items()}

        def scan(batch_size: int) -> dict[Any, float]:
            return run_path_patching_scan(
                pipeline,
                dataset,
                senders,
                metric=_first_logit_metric(),
                restore=("attention",),
                batch_size=batch_size,
            )

        solo, batched = scan(1), scan(len(dataset))
        for key in solo:
            assert abs(solo[key] - batched[key]) < 1e-4, (
                f"[{pipeline_fixture}] sender {key}: left-padded batch diverged from "
                f"bs=1 ({batched[key]} vs {solo[key]}) — position_ids on the generate "
                f"forward is what makes these equal on an absolute-position model"
            )

    @pytest.mark.parametrize("pipeline_fixture", ["mock_tiny_lm", "tiny_gpt2_lm"])
    def test_two_pass_receiver_bs_parity(
        self, pipeline_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Covers PASS 2 (inject v* via intervenable_generate) and PASS 1's
        source-collection forward on a left-padded mixed-length batch."""
        pipeline = request.getfixturevalue(pipeline_fixture)
        dataset = _mixed_length_dataset()
        self._assert_left_pads(pipeline, dataset)
        pos = _last_token(pipeline)
        targets = build_attention_head_sites(pipeline, [0], [0, 1], pos)
        senders = {key: t[0][0] for key, t in targets.items()}
        receiver = ReceiverSpec(kind="residual", layer=1, token_position=pos)

        def scan(batch_size: int) -> dict[Any, float]:
            return run_path_patching_scan(
                pipeline,
                dataset,
                senders,
                metric=_first_logit_metric(),
                receiver=receiver,
                restore=("attention",),
                batch_size=batch_size,
            )

        solo, batched = scan(1), scan(len(dataset))
        for key in solo:
            assert abs(solo[key] - batched[key]) < 1e-4, (
                f"[{pipeline_fixture}] sender {key}: two-pass left-padded batch "
                f"diverged from bs=1 ({batched[key]} vs {solo[key]})"
            )
