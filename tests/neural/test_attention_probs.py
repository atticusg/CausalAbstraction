"""Tests for :mod:`causalab.neural.attention_probs` — CAP4 attention-probability
editing (#457).

Tiers (per docs/TESTS.md; ``causalab/neural`` owes ``unit`` + ``property``, and
a ``golden`` GPU pin is the established pattern for the coherent backbone —
mirroring ``tests/neural/test_modes.py``):

* ``unit`` — the declarative contracts, no model: site validation, mode
  constructors return the expected :class:`Edit` shape, empty selections and
  non-pattern/featurized sites rejected at construction, forward-rank slot.
* ``property`` — on the tiny-random Llama (CPU) loaded through the real gate
  (``LMPipeline(..., enable_attention_probs=True)`` → eager kernel + nnterp's
  ``check_source()`` at load): the pattern is a simplex; knockout changes the
  logits and the read-back pattern equals the same transform computed offline
  from the clean pattern; ``redistribute=False`` removes exactly the knocked
  mass; the standalone ``renormalize`` restores the simplex; untouched heads
  stay bit-identical; a two-layer knockout **chain** lowers through
  :func:`~causalab.neural.plan.run_plan`.
* ``golden`` — the same knockout chain on the real Qwen3-4B backbone (GPU),
  one model load; left for the nightly runner, not run interactively.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.attention_probs import (
    AttentionProbabilitiesSite,
    knockout,
    renormalize,
)
from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.plan import CollectOp, EditOp, Plan, run_plan
from causalab.neural.site import INTRA_BLOCK_RANK, Site

from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME, tiny_random_model

_TEXT = "the quick brown fox jumps"


# --------------------------------------------------------------------------- #
#  unit — declarative contracts, no model                                      #
# --------------------------------------------------------------------------- #
class TestAttentionProbsUnit:
    pytestmark = pytest.mark.unit

    def test_site_rejects_negative_layer(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AttentionProbabilitiesSite(-1)

    def test_forward_rank_slots_inside_the_attention_block(self) -> None:
        site = AttentionProbabilitiesSite(3)
        assert site.layer == 3
        assert INTRA_BLOCK_RANK["value"] < site.forward_rank
        assert site.forward_rank < INTRA_BLOCK_RANK["attention_value"]

    def test_knockout_returns_a_writing_edit_over_the_site(self) -> None:
        site = AttentionProbabilitiesSite(1)
        edit = knockout(site, key_positions=0)
        assert isinstance(edit, Edit)
        assert edit.site.site is site
        assert edit.g is not None
        assert edit.read_sources == ()
        assert edit.positions is None

    def test_renormalize_returns_a_writing_edit_over_the_site(self) -> None:
        site = AttentionProbabilitiesSite(1)
        edit = renormalize(site, heads=[0])
        assert isinstance(edit, Edit)
        assert edit.site.site is site
        assert edit.g is not None

    def test_constructors_accept_a_prewrapped_featurized_site(self) -> None:
        fsite = FeaturizedSite(AttentionProbabilitiesSite(0))
        assert knockout(fsite, key_positions=[0]).site is fsite
        assert renormalize(fsite).site is fsite

    @pytest.mark.parametrize("kwarg", ["heads", "query_positions", "key_positions"])
    def test_empty_selection_rejected(self, kwarg: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            knockout(AttentionProbabilitiesSite(0), **{kwarg: []})

    def test_renormalize_empty_selection_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            renormalize(AttentionProbabilitiesSite(0), heads=[])

    def test_non_pattern_site_rejected(self) -> None:
        with pytest.raises(ValueError, match="AttentionProbabilitiesSite"):
            knockout(FeaturizedSite(Site("block_output", 0)), key_positions=[0])

    def test_feature_ids_rejected(self) -> None:
        fsite = FeaturizedSite(AttentionProbabilitiesSite(0), feature_ids=(0, 1))
        with pytest.raises(ValueError, match="feature_ids"):
            knockout(fsite, key_positions=[0])

    def test_bool_selection_rejected(self) -> None:
        # bool subclasses int — heads=True would otherwise coerce to head [1].
        with pytest.raises(ValueError, match="bool"):
            knockout(AttentionProbabilitiesSite(0), heads=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="bool"):
            renormalize(AttentionProbabilitiesSite(0), heads=False)  # type: ignore[arg-type]

    def test_per_row_positions_refused_before_any_model_access(self) -> None:
        # _flat_index normalizes positions before the accessor is touched, so
        # the refusal needs no model / trace at all.
        site = AttentionProbabilitiesSite(0)
        with pytest.raises(NotImplementedError, match="flat row"):
            site.read(None, [[0], [1, 2]])  # type: ignore[arg-type]

    def test_absent_accessor_treated_as_unavailable(self) -> None:
        # A model without the attribute at all (not a StandardizedTransformer)
        # gets the same load-flag remedy, not a raw AttributeError downstream.
        from types import SimpleNamespace

        with pytest.raises(ValueError, match="enable_attention_probs=True"):
            AttentionProbabilitiesSite(0)._check_available(SimpleNamespace())  # type: ignore[arg-type]

    def test_enable_attention_probs_with_check_renaming_false_fails_at_load(
        self,
    ) -> None:
        # nnterp silently disables the accessor when its load-time checks are
        # skipped; the pipeline names the real cause at load instead of letting
        # a misleading "load it with enable_attention_probs=True" surface at
        # first read. Raised before any HF access, so this stays unit-tier.
        with pytest.raises(ValueError, match="check_renaming=False"):
            LMPipeline(
                TINY_RANDOM_MODEL_NAME,
                enable_attention_probs=True,
                check_renaming=False,
            )


# --------------------------------------------------------------------------- #
#  property — the semantics hold on the tiny-random backbone (CPU)             #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def attn_pipeline() -> LMPipeline:
    """Tiny-random Llama loaded through the real gate: the pipeline flag makes
    nnterp force eager attention and run ``check_source()`` at load."""
    return LMPipeline(
        TINY_RANDOM_MODEL_NAME,
        max_new_tokens=1,
        padding_side="left",
        enable_attention_probs=True,
        device="cpu",
    )


@pytest.fixture(scope="module")
def attn_inputs(attn_pipeline: LMPipeline) -> dict:
    return attn_pipeline.load([{"raw_input": _TEXT}])


def _clean_logits(st, inputs) -> torch.Tensor:
    with st.trace(inputs):
        logits = st.logits[:, -1, :].cpu().save()
    return logits


def _edited(st, edit: Edit, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``edit`` in one trace; return (last-token logits, read-back pattern)."""
    site = edit.site.site
    with st.trace(inputs):
        edit.apply(st)
        pattern = site.read(st).cpu().save()
        logits = st.logits[:, -1, :].cpu().save()
    return logits, pattern


class TestAttentionProbsProperty:
    pytestmark = pytest.mark.property

    def test_load_gating_forces_eager_and_enables_the_accessor(
        self, attn_pipeline: LMPipeline
    ) -> None:
        assert attn_pipeline.model.config._attn_implementation == "eager"
        assert attn_pipeline.model.attn_probs_available
        # The module's fail-fast reads accessor.enabled; the public readiness
        # bit is attn_probs_available — assert the two nnterp attributes agree.
        assert (
            attn_pipeline.model.attention_probabilities.enabled
            == attn_pipeline.model.attn_probs_available
        )

    def test_preloaded_non_eager_module_fails_fast(self) -> None:
        # The pre-loaded wrap path never rewrites the caller's config, and the
        # cached tiny module resolves to sdpa — the pipeline must refuse with
        # the remedy rather than let nnterp's check_source die mid-trace.
        with pytest.raises(ValueError, match="eager"):
            LMPipeline(tiny_random_model(), enable_attention_probs=True)

    def test_pattern_is_a_simplex(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        pattern = AttentionProbabilitiesSite(1).collect(st, attn_inputs)
        batch, seq = attn_inputs["input_ids"].shape
        assert pattern.shape == (batch, st.num_heads, seq, seq)
        sums = pattern.sum(-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)

    def test_query_positions_slice_the_query_axis(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        site = AttentionProbabilitiesSite(1)
        full = site.collect(st, attn_inputs)
        last = site.collect(st, attn_inputs, positions=[-1])
        assert last.shape == (full.shape[0], full.shape[1], 1, full.shape[3])
        torch.testing.assert_close(last, full[:, :, -1:, :])

    def test_knockout_changes_logits_and_matches_offline_transform(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        site = AttentionProbabilitiesSite(1)
        clean_pattern = site.collect(st, attn_inputs)
        clean = _clean_logits(st, attn_inputs)

        edited, pattern = _edited(st, knockout(site, key_positions=0), attn_inputs)
        assert not torch.allclose(edited, clean, atol=1e-6)

        # The read-back pattern is exactly the declared transform of the clean
        # pattern: zero the key-0 column, renormalize every (fully selected) row.
        expected = clean_pattern.clone()
        expected[..., 0] = 0.0
        expected = expected / expected.sum(-1, keepdim=True).clamp_min(1e-12)
        torch.testing.assert_close(pattern, expected, atol=1e-6, rtol=0)

        # Query row 0 attends only to key 0 (causal), so its whole support was
        # knocked out: it stays zero. Every other row renormalizes to 1.
        sums = pattern.sum(-1)
        assert torch.all(sums[:, :, 0].abs() < 1e-6)
        torch.testing.assert_close(
            sums[:, :, 1:], torch.ones_like(sums[:, :, 1:]), atol=1e-5, rtol=0
        )

    def test_knockout_without_redistribute_removes_exactly_the_knocked_mass(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        site = AttentionProbabilitiesSite(1)
        clean_pattern = site.collect(st, attn_inputs)

        _, pattern = _edited(
            st, knockout(site, key_positions=0, redistribute=False), attn_inputs
        )
        torch.testing.assert_close(
            pattern.sum(-1),
            clean_pattern.sum(-1) - clean_pattern[..., 0],
            atol=1e-5,
            rtol=0,
        )

    def test_standalone_renormalize_restores_the_simplex(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        site = AttentionProbabilitiesSite(1)
        with st.trace(attn_inputs):
            knockout(site, key_positions=0, redistribute=False).apply(st)
            renormalize(site).apply(st)
            pattern = site.read(st).cpu().save()
        sums = pattern.sum(-1)
        assert torch.all(sums[:, :, 0].abs() < 1e-6)  # fully-knocked row stays zero
        torch.testing.assert_close(
            sums[:, :, 1:], torch.ones_like(sums[:, :, 1:]), atol=1e-5, rtol=0
        )

    def test_scoped_knockout_leaves_unselected_heads_and_rows_untouched(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        site = AttentionProbabilitiesSite(1)
        clean_pattern = site.collect(st, attn_inputs)

        _, pattern = _edited(
            st,
            knockout(site, heads=0, query_positions=[-1], key_positions=[0]),
            attn_inputs,
        )
        # Unselected heads and unselected query rows: bit-identical.
        torch.testing.assert_close(pattern[:, 1:], clean_pattern[:, 1:], atol=0, rtol=0)
        torch.testing.assert_close(
            pattern[:, 0, :-1], clean_pattern[:, 0, :-1], atol=0, rtol=0
        )
        # The selected edge is zero and its row renormalized.
        assert torch.all(pattern[:, 0, -1, 0] == 0)
        row_sum = pattern[:, 0, -1, :].sum(-1)
        torch.testing.assert_close(row_sum, torch.ones_like(row_sum), atol=1e-5, rtol=0)

    def test_knockout_chain_lowers_through_run_plan(
        self, attn_pipeline: LMPipeline, attn_inputs: dict
    ) -> None:
        st = attn_pipeline.model
        clean = _clean_logits(st, attn_inputs)
        single, _ = _edited(
            st, knockout(AttentionProbabilitiesSite(0), key_positions=0), attn_inputs
        )

        plan = Plan(
            inputs={"base": attn_inputs},
            ops=(
                EditOp(
                    "base", knockout(AttentionProbabilitiesSite(0), key_positions=0)
                ),
                EditOp(
                    "base", knockout(AttentionProbabilitiesSite(1), key_positions=0)
                ),
                CollectOp(
                    "base",
                    FeaturizedSite(AttentionProbabilitiesSite(1)),
                    key="probs_l1",
                ),
            ),
            save_logits=("base",),
        )
        result = run_plan(st, plan)
        chained = result.logits["base"][:, -1, :]

        assert not torch.allclose(chained, clean, atol=1e-6)
        assert not torch.allclose(chained, single, atol=1e-6)
        # The collect fires at the same site as the layer-1 edit, declared
        # after it — it sees the knocked-out, renormalized pattern.
        sums = result.collects["probs_l1"].sum(-1)
        assert torch.all(sums[:, :, 0].abs() < 1e-6)
        torch.testing.assert_close(
            sums[:, :, 1:], torch.ones_like(sums[:, :, 1:]), atol=1e-5, rtol=0
        )

    def test_accessor_disabled_without_the_flag(self) -> None:
        # Without enable_attention_probs nnterp disables the accessor; the
        # site fails fast with the load-flag remedy (before any trace work).
        pipe = LMPipeline(
            TINY_RANDOM_MODEL_NAME,
            max_new_tokens=1,
            padding_side="left",
            device="cpu",
        )
        with pytest.raises(ValueError, match="enable_attention_probs=True"):
            AttentionProbabilitiesSite(0)._check_available(pipe.model)


# --------------------------------------------------------------------------- #
#  golden — the knockout chain holds on the real Qwen3-4B backbone (GPU)       #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """The CAP4 knockout chain on the coherent GPU backbone, one model load:
    the pipeline gate (eager + nnterp ``check_source()``) resolves on a real
    GQA architecture, a two-layer BOS-column knockout chain lowers through
    ``run_plan``, changes the logits, and the read-back patterns stay a
    (renormalized) simplex in bf16."""

    pytestmark = pytest.mark.golden

    def test_knockout_chain_on_coherent_model(self) -> None:
        pipeline = LMPipeline(
            "Qwen/Qwen3-4B-Instruct-2507",
            max_new_tokens=1,
            padding_side="left",
            enable_attention_probs=True,
        )
        st = pipeline.model
        assert st.config._attn_implementation == "eager"
        assert st.attn_probs_available  # check_source() passed at load

        inputs = pipeline.load([{"raw_input": "The capital of France is"}])
        mid = pipeline.get_num_layers() // 2
        layers = (mid, mid + 1)

        clean = _clean_logits(st, inputs)

        plan = Plan(
            inputs={"base": inputs},
            ops=tuple(
                EditOp("base", knockout(AttentionProbabilitiesSite(L), key_positions=0))
                for L in layers
            )
            + (
                CollectOp(
                    "base",
                    FeaturizedSite(AttentionProbabilitiesSite(layers[-1])),
                    key="probs",
                ),
            ),
            save_logits=("base",),
        )
        result = run_plan(st, plan)
        chained = result.logits["base"][:, -1, :]

        # Knocking out all attention to the first token across two layers is a
        # real intervention: the next-token logits move.
        assert not torch.allclose(chained.float(), clean.float(), atol=1e-4)

        # The read-back pattern under the chain: key-0 column zeroed, every
        # surviving row renormalized to 1 (bf16 tolerance), the fully-knocked
        # query row 0 zero.
        pattern = result.collects["probs"].float()
        assert torch.all(pattern[..., 0] == 0)
        sums = pattern.sum(-1)
        assert torch.all(sums[:, :, 0].abs() < 1e-6)
        torch.testing.assert_close(
            sums[:, :, 1:], torch.ones_like(sums[:, :, 1:]), atol=2e-2, rtol=0
        )
