"""Tests for :mod:`causalab.neural.head_view` — the F4 per-head reshape spike (#395)
and the ST4 HeadView adapter (#399: fused-QKV support, the per-head write path, and
the Site-protocol ``HeadSite``).

Tiers (mirroring ``tests/neural/test_validate.py``; ``causalab/neural`` owes ``unit`` +
``property``, and a ``golden`` GPU pin is the established pattern for the real
decoupled-``head_dim`` contract):

* ``unit`` — the contract math (true ``head_dim``, GQA KV-head remap, fused detection
  + the fused-layout boundary, model-aware ``kind_rank`` ordering, ``HeadSite``
  validation), no forward pass.
* ``property`` — on tiny coupled / GQA / **decoupled-``head_dim`` GQA** Llamas *and a
  tiny fused-QKV GPT-2* (CPU), every receiver matches a raw-``register_forward_hook``
  oracle at the true ``head_dim``; KV-space value addressing; the forward-order
  collector (incl. the fused reordering); the per-head write path (both families);
  ``HeadSite`` read/write/collect parity; mixed ``Site`` + ``HeadSite`` single-pass
  collection.
* ``golden`` — the same equivalence on the real Qwen3-4B backbone (decoupled
  ``head_dim=128``), the case pyvene 0.1.8 gets wrong.

The oracle hooks the *same underlying model* the ``StandardizedTransformer`` wraps
(``st._model``), so weights are shared and the comparison is exact — it never imports
the backbone under test, exactly like ``tests/neural/activations/hook_oracle.py``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.neural.head_view import HeadSite, HeadView
from causalab.neural.site import INTRA_BLOCK_RANK, Site, collect_sites
from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama

_TEXT = "the quick brown fox jumps"


# --------------------------------------------------------------------------- #
#  Fixtures — tiny StandardizedTransformers + a handle on the raw HF model     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class HVCase:
    raw: Any  # underlying HF model (for the raw-hook oracle)
    tok: Any
    hv: HeadView

    def inputs(self, text: str = _TEXT) -> dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _llama_case(*, gqa: bool, decoupled: bool) -> HVCase:
    def mutate(cfg: Any) -> None:
        if gqa:
            assert cfg.num_attention_heads % 2 == 0
            cfg.num_key_value_heads = cfg.num_attention_heads // 2
        if decoupled:
            # head_dim != hidden // n_heads — the Qwen3 case pyvene mis-slices.
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads + 2

    model, tok = fresh_tiny_random_llama(mutate_config=mutate)
    st = StandardizedTransformer(model, tokenizer=tok, check_renaming=True)
    return HVCase(raw=model, tok=tok, hv=HeadView(st))


@pytest.fixture(scope="module")
def coupled_case() -> HVCase:
    return _llama_case(gqa=False, decoupled=False)


@pytest.fixture(scope="module")
def gqa_case() -> HVCase:
    return _llama_case(gqa=True, decoupled=False)


@pytest.fixture(scope="module")
def decoupled_case() -> HVCase:
    return _llama_case(gqa=True, decoupled=True)


@pytest.fixture(scope="module")
def fused_case() -> HVCase:
    """A tiny fused-QKV GPT-2 (real tokenizer-matched vocab, so text tokenizes to
    in-range ids)."""
    model, tok = fresh_tiny_random_gpt2()
    st = StandardizedTransformer(model, tokenizer=tok, check_renaming=True)
    return HVCase(raw=model, tok=tok, hv=HeadView(st))


# --------------------------------------------------------------------------- #
#  Oracle — raw forward hooks on the shared underlying model (no backbone)     #
# --------------------------------------------------------------------------- #
def _raw_projections(
    raw: Any, layer: int, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Flat projection tensors via our own hooks: ``q_proj``/``v_proj`` outputs and
    ``o_proj``'s input — the ground truth the head views slice from."""
    grabbed: dict[str, torch.Tensor] = {}
    attn = raw.model.layers[layer].self_attn
    handles = [
        attn.q_proj.register_forward_hook(
            lambda _m, _i, o: grabbed.__setitem__("query", o.detach().clone())
        ),
        attn.v_proj.register_forward_hook(
            lambda _m, _i, o: grabbed.__setitem__("value", o.detach().clone())
        ),
        attn.o_proj.register_forward_pre_hook(
            lambda _m, a: grabbed.__setitem__("attention_value", a[0].detach().clone())
        ),
    ]
    try:
        with torch.no_grad():
            raw(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return grabbed


def _raw_fused_projections(
    raw: Any, layer: int, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Flat ground-truth tensors for a fused-QKV GPT-2 via our own hooks: the
    ``c_attn`` output's ``[q | k | v]`` columns and ``c_proj``'s input (sender)."""
    grabbed: dict[str, torch.Tensor] = {}
    attn = raw.transformer.h[layer].attn
    hidden = raw.config.hidden_size
    handles = [
        attn.c_attn.register_forward_hook(
            lambda _m, _i, o: grabbed.__setitem__("c_attn", o.detach().clone())
        ),
        attn.c_proj.register_forward_pre_hook(
            lambda _m, a: grabbed.__setitem__("attention_value", a[0].detach().clone())
        ),
    ]
    try:
        with torch.no_grad():
            raw(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return {
        "query": grabbed["c_attn"][..., 0:hidden],
        "value": grabbed["c_attn"][..., 2 * hidden : 3 * hidden],
        "attention_value": grabbed["attention_value"],
    }


def _expected_head(flat: torch.Tensor, head: int, head_dim: int) -> torch.Tensor:
    return flat[:, :, head * head_dim : (head + 1) * head_dim]


def _stacked_inputs(case: HVCase, texts: list[str]) -> dict[str, torch.Tensor]:
    """A padding-free batch: tokenize each text and truncate all rows to the
    shortest, so per-row position tests never depend on a pad token."""
    encs = [case.tok(t, return_tensors="pt") for t in texts]
    length = min(int(e["input_ids"].shape[1]) for e in encs)
    return {
        "input_ids": torch.cat([e["input_ids"][:, :length] for e in encs]),
        "attention_mask": torch.cat([e["attention_mask"][:, :length] for e in encs]),
    }


# --------------------------------------------------------------------------- #
#  unit — the contract math                                                    #
# --------------------------------------------------------------------------- #
class TestContractUnit:
    pytestmark = pytest.mark.unit

    def test_head_dim_honours_config(self, decoupled_case, coupled_case) -> None:
        d = decoupled_case.hv
        cfg = decoupled_case.raw.config
        assert d.head_dim == cfg.head_dim
        assert d.head_dim != cfg.hidden_size // cfg.num_attention_heads  # decoupled
        c = coupled_case.hv
        ccfg = coupled_case.raw.config
        assert c.head_dim == ccfg.hidden_size // ccfg.num_attention_heads  # coupled

    def test_gqa_group_and_kv_remap(self, gqa_case, coupled_case) -> None:
        hv = gqa_case.hv
        assert hv.n_kv_heads == hv.n_heads // 2
        assert hv.group_size == 2
        # query heads 0,1 -> KV 0; 2,3 -> KV 1; ...
        for q in range(hv.n_heads):
            assert hv.kv_head_for(q) == q // 2
        # non-grouped: identity
        c = coupled_case.hv
        assert c.group_size == 1
        assert all(c.kv_head_for(q) == q for q in range(c.n_heads))

    def test_head_column_slice(self, decoupled_case) -> None:
        hv = decoupled_case.hv
        d = hv.head_dim
        assert hv.head_column_slice(0) == slice(0, d)
        assert hv.head_column_slice(3) == slice(3 * d, 4 * d)

    def test_is_fused_false_for_llama(self, coupled_case, decoupled_case) -> None:
        assert coupled_case.hv.is_fused is False
        assert decoupled_case.hv.is_fused is False

    def test_fused_model_detected(self, fused_case) -> None:
        assert fused_case.hv.is_fused is True

    def test_kind_rank_separate_matches_intra_block_order(self, gqa_case) -> None:
        """Separate projections fire q → v → o-input — ranks come straight off the
        shared ``INTRA_BLOCK_RANK`` scale."""
        hv = gqa_case.hv
        for kind in ("query", "value", "attention_value"):
            assert hv.kind_rank(kind) == INTRA_BLOCK_RANK[kind]
        with pytest.raises(ValueError, match="unknown head kind"):
            hv.kind_rank("key")

    def test_kind_rank_fused_collapses_query_and_value(self, fused_case) -> None:
        """On fused QKV, query and value both read the single ``c_attn`` output —
        which fires first — so both collapse to the ``query`` rank while the sender
        keeps its later slot (the naive q<v order would deadlock a one-pass read)."""
        hv = fused_case.hv
        assert hv.kind_rank("query") == INTRA_BLOCK_RANK["query"]
        assert hv.kind_rank("value") == INTRA_BLOCK_RANK["query"]
        assert hv.kind_rank("attention_value") == INTRA_BLOCK_RANK["attention_value"]

    def test_unsupported_fused_layout_refused(self, fused_case) -> None:
        """The honest boundary: a fused model outside the GPT-2-style equal-width
        ``[q|k|v]`` layout (e.g. fused GQA, or a decoupled ``head_dim``) raises
        rather than mis-slicing."""
        hv = HeadView(fused_case.hv.model)
        hv.n_kv_heads = hv.n_heads // 2  # a hypothetical fused-GQA layout
        with pytest.raises(NotImplementedError, match="fused-QKV"):
            hv.values(0)
        hv2 = HeadView(fused_case.hv.model)
        hv2.head_dim += 2  # a hypothetical fused decoupled-head_dim layout
        with pytest.raises(NotImplementedError, match="fused-QKV"):
            hv2.queries(0)

    def test_head_bounds_checked(self, gqa_case, fused_case) -> None:
        """Out-of-range heads fail loudly (KV space for value) — before any trace."""
        hv = gqa_case.hv
        with pytest.raises(IndexError, match="KV heads"):
            hv.write(0, "value", hv.n_kv_heads, 0.0)
        with pytest.raises(IndexError, match="query heads"):
            fused_case.hv.write(0, "query", fused_case.hv.n_heads, 0.0)

    def test_head_site_validation(self) -> None:
        with pytest.raises(ValueError, match="unknown head kind"):
            HeadSite("key", 0, 0)
        with pytest.raises(ValueError, match="layer"):
            HeadSite("value", -1, 0)
        with pytest.raises(ValueError, match="head"):
            HeadSite("query", 0, -1)

    def test_head_site_forward_rank(self, gqa_case, fused_case) -> None:
        """``forward_rank`` is the separate-projection default off the shared
        scale; ``forward_rank_on`` resolves per model (fused reordering)."""
        site = HeadSite("value", 0, 0)
        assert site.forward_rank == INTRA_BLOCK_RANK["value"]
        assert site.forward_rank_on(gqa_case.hv.model) == INTRA_BLOCK_RANK["value"]
        assert site.forward_rank_on(fused_case.hv.model) == INTRA_BLOCK_RANK["query"]


class TestTextConfigModelsUnit:
    """#449 finding 1: the contract fields must come from nnterp's instance
    attributes / the *text* config, so ``text_config``-nesting models (Gemma3)
    neither crash (`hidden_size` is not top-level) nor silently pick the
    wrong GQA fallbacks (kv-heads / ``head_dim``)."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _gemma3_config():
        from transformers import Gemma3Config

        return Gemma3Config()

    def test_nnterp_attributes_win_when_set(self) -> None:
        cfg = self._gemma3_config()
        text = cfg.text_config
        model = SimpleNamespace(
            config=cfg,
            num_heads=text.num_attention_heads,
            hidden_size=text.hidden_size,
            num_layers=text.num_hidden_layers,
        )
        hv = HeadView(model)  # type: ignore[arg-type]
        assert hv.n_heads == text.num_attention_heads
        assert hv.hidden_size == text.hidden_size
        assert hv.num_layers == text.num_hidden_layers
        # GQA fields nnterp does not standardize: from the text config, not
        # the (absent) top-level keys — the silently-wrong fallbacks were
        # n_heads (8) and hidden // n_heads (288).
        assert hv.n_kv_heads == text.num_key_value_heads == 4
        assert hv.head_dim == text.head_dim == 256

    def test_none_guarded_fallback_to_text_config(self) -> None:
        # nnterp sets num_heads / hidden_size with raise_error=False, so they
        # can be None — the fallback must read the nested text config.
        model = SimpleNamespace(
            config=self._gemma3_config(),
            num_heads=None,
            hidden_size=None,
            num_layers=None,
        )
        hv = HeadView(model)  # type: ignore[arg-type]
        text = model.config.text_config
        assert (hv.n_heads, hv.hidden_size, hv.num_layers) == (
            text.num_attention_heads,
            text.hidden_size,
            text.num_hidden_layers,
        )
        assert (hv.n_kv_heads, hv.head_dim) == (4, 256)


# --------------------------------------------------------------------------- #
#  property — reads match the raw-hook oracle at the true head_dim             #
# --------------------------------------------------------------------------- #
class TestReadsMatchOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["coupled", "gqa", "decoupled"])
    def case(self, request, coupled_case, gqa_case, decoupled_case) -> HVCase:
        return {"coupled": coupled_case, "gqa": gqa_case, "decoupled": decoupled_case}[
            request.param
        ]

    def test_query_matches_qproj_slice_per_query_head(self, case) -> None:
        hv, inp = case.hv, case.inputs()
        flat = _raw_projections(case.raw, 0, inp)["query"]
        for h in range(hv.n_heads):
            got = hv.collect_query(inp, 0, h)
            assert got.shape[-1] == hv.head_dim
            torch.testing.assert_close(
                got, _expected_head(flat, h, hv.head_dim), atol=1e-5, rtol=1e-4
            )

    def test_value_matches_vproj_slice_in_kv_space(self, case) -> None:
        hv, inp = case.hv, case.inputs()
        flat = _raw_projections(case.raw, 0, inp)["value"]
        collected = [hv.collect_value(inp, 0, kv) for kv in range(hv.n_kv_heads)]
        for kv in range(hv.n_kv_heads):
            assert collected[kv].shape[-1] == hv.head_dim
            torch.testing.assert_close(
                collected[kv],
                _expected_head(flat, kv, hv.head_dim),
                atol=1e-5,
                rtol=1e-4,
            )
        if hv.n_kv_heads > 1:  # distinct KV heads => distinct vectors
            assert not torch.allclose(collected[0], collected[1], atol=1e-4)

    def test_attention_value_sender_matches_oproj_input(self, case) -> None:
        hv, inp = case.hv, case.inputs()
        flat = _raw_projections(case.raw, 0, inp)["attention_value"]
        for h in range(hv.n_heads):
            got = hv.collect_attention_value(inp, 0, h)
            assert got.shape[-1] == hv.head_dim
            torch.testing.assert_close(
                got, _expected_head(flat, h, hv.head_dim), atol=1e-5, rtol=1e-4
            )

    def test_decoupled_width_is_true_head_dim(self, decoupled_case) -> None:
        """The pyvene bug in one line: on a decoupled model every receiver is
        ``config.head_dim``-wide, not ``hidden // n_heads``."""
        hv, inp = decoupled_case.hv, decoupled_case.inputs()
        naive = decoupled_case.raw.config.hidden_size // hv.n_heads
        assert hv.head_dim != naive
        assert hv.collect_value(inp, 0, 0).shape[-1] == hv.head_dim
        assert hv.collect_query(inp, 0, 0).shape[-1] == hv.head_dim

    def test_collect_orders_reads_by_forward_position(self, decoupled_case) -> None:
        """A mixed request list in *non*-forward order reads in one pass (no
        ``MissedProviderError``) and every result still matches the oracle."""
        hv, inp = decoupled_case.hv, decoupled_case.inputs()
        gt0 = _raw_projections(case_raw := decoupled_case.raw, 0, inp)
        gt1 = _raw_projections(case_raw, 1, inp)
        # deliberately out of forward order (value/attn before query, layer 1 before 0)
        requests = [
            HeadSite("attention_value", 1, 0),
            HeadSite("value", 0, 0),
            HeadSite("query", 1, 1),
            HeadSite("query", 0, 0),
        ]
        out = hv.collect(inp, requests)
        d = hv.head_dim
        torch.testing.assert_close(
            out[0], _expected_head(gt1["attention_value"], 0, d), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            out[1], _expected_head(gt0["value"], 0, d), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            out[2], _expected_head(gt1["query"], 1, d), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            out[3], _expected_head(gt0["query"], 0, d), atol=1e-5, rtol=1e-4
        )

    def test_per_head_write_moves_logits_and_matches_hook(self, decoupled_case) -> None:
        """Writing a KV-head value slice (the ST4/ED1 intervention path) reproduces a
        hand-rolled ``v_proj`` hook edit and is non-vacuous."""
        import nnsight

        hv, raw = decoupled_case.hv, decoupled_case.raw
        inp = decoupled_case.inputs()
        sl = hv.head_column_slice(0)  # KV head 0

        with hv.model.trace(inp):
            clean = nnsight.save(hv.model.logits)
        with hv.model.trace(inp):
            v = hv.model.model.layers[0].self_attn.v_proj.output
            v[:, :, sl] = v[:, :, sl] + 10.0
            edited = nnsight.save(hv.model.logits)

        def hooked() -> torch.Tensor:
            def hook(_m, _i, o):
                o = o.clone()
                o[:, :, sl] = o[:, :, sl] + 10.0
                return o

            h = raw.model.layers[0].self_attn.v_proj.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    return raw(
                        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                    ).logits
            finally:
                h.remove()

        manual = hooked()
        assert not torch.allclose(edited, clean, atol=1e-4)  # non-vacuous
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  property — fused QKV (GPT-2): the ✗ gap ST4 closes                          #
# --------------------------------------------------------------------------- #
class TestFusedMatchesOracle:
    """Per-head receivers on a fused ``c_attn`` model match the raw-hook oracle —
    reads, the reordered one-pass collect, and the write path."""

    pytestmark = pytest.mark.property

    def test_reads_match_cattn_column_slices_per_head(self, fused_case) -> None:
        hv, inp = fused_case.hv, fused_case.inputs()
        flat = _raw_fused_projections(fused_case.raw, 0, inp)
        d = hv.head_dim
        for h in range(hv.n_heads):
            for kind, collect in (
                ("query", hv.collect_query),
                ("value", hv.collect_value),
                ("attention_value", hv.collect_attention_value),
            ):
                got = collect(inp, 0, h)
                assert got.shape[-1] == d
                torch.testing.assert_close(
                    got, _expected_head(flat[kind], h, d), atol=1e-5, rtol=1e-4
                )

    def test_collect_reorders_for_fused_execution(self, fused_case) -> None:
        """Query-before-value requests read in one pass on a fused model — the
        naive separate-projection order (q → v) would miss the provider, since
        both kinds tap the single ``c_attn`` output."""
        hv, inp = fused_case.hv, fused_case.inputs()
        gt0 = _raw_fused_projections(fused_case.raw, 0, inp)
        gt1 = _raw_fused_projections(fused_case.raw, 1, inp)
        requests = [
            HeadSite("attention_value", 1, 0),
            HeadSite("query", 0, 0),  # deliberately requested before value
            HeadSite("value", 0, 1),
            HeadSite("query", 1, 1),
        ]
        out = hv.collect(inp, requests)
        d = hv.head_dim
        torch.testing.assert_close(
            out[0], _expected_head(gt1["attention_value"], 0, d), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            out[1], _expected_head(gt0["query"], 0, d), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            out[2], _expected_head(gt0["value"], 1, d), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            out[3], _expected_head(gt1["query"], 1, d), atol=1e-5, rtol=1e-4
        )

    def test_fused_value_write_matches_hand_rolled_hook(self, fused_case) -> None:
        """A per-head value write on the fused model reproduces a hand-rolled
        ``c_attn`` column edit and is non-vacuous — the write half of the ✗ gap."""
        import nnsight

        hv, raw = fused_case.hv, fused_case.raw
        inp = fused_case.inputs()
        hidden, d = hv.hidden_size, hv.head_dim
        cols = slice(2 * hidden + 0 * d, 2 * hidden + 1 * d)  # value head 0

        with hv.model.trace(inp):
            clean = nnsight.save(hv.model.logits)
        with hv.model.trace(inp):
            hv.write(0, "value", 0, torch.zeros(d))
            edited = nnsight.save(hv.model.logits)

        def hooked() -> torch.Tensor:
            def hook(_m, _i, o):
                o = o.clone()
                o[:, :, cols] = 0.0
                return o

            h = raw.transformer.h[0].attn.c_attn.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    return raw(
                        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                    ).logits
            finally:
                h.remove()

        manual = hooked()
        assert not torch.allclose(edited, clean, atol=1e-4)  # non-vacuous
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_fused_positional_query_write(self, fused_case) -> None:
        """A position-sliced query write edits only the requested positions."""
        import nnsight

        hv, raw = fused_case.hv, fused_case.raw
        inp = fused_case.inputs()
        d = hv.head_dim
        positions = [1, 2]
        cols = slice(1 * d, 2 * d)  # query head 1 (query columns start at 0)

        with hv.model.trace(inp):
            hv.write(0, "query", 1, torch.zeros(d), positions=positions)
            edited = nnsight.save(hv.model.logits)

        def hooked() -> torch.Tensor:
            def hook(_m, _i, o):
                o = o.clone()
                o[:, positions, cols] = 0.0
                return o

            h = raw.transformer.h[0].attn.c_attn.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    return raw(
                        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                    ).logits
            finally:
                h.remove()

        torch.testing.assert_close(edited, hooked(), atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  property — HeadSite speaks the Site protocol                                #
# --------------------------------------------------------------------------- #
class TestHeadSiteProtocol:
    """``HeadSite`` mirrors ``Site``: read/write/collect with resolved positions,
    KV-space value addressing, and mixed-site single-pass collection."""

    pytestmark = pytest.mark.property

    def test_collect_matches_head_view_and_is_cpu(self, gqa_case) -> None:
        hv, inp = gqa_case.hv, gqa_case.inputs()
        for site, via_hv in (
            (HeadSite("value", 0, 1), hv.collect_value(inp, 0, 1)),
            (HeadSite("query", 1, 2), hv.collect_query(inp, 1, 2)),
            (HeadSite("attention_value", 0, 3), hv.collect_attention_value(inp, 0, 3)),
        ):
            got = site.collect(hv.model, inp)
            assert got.device.type == "cpu"
            torch.testing.assert_close(got, via_hv.cpu(), atol=1e-5, rtol=1e-4)

    def test_collect_positions_slice(self, gqa_case) -> None:
        hv, inp = gqa_case.hv, gqa_case.inputs()
        site = HeadSite("value", 0, 0)
        full = site.collect(hv.model, inp)
        sliced = site.collect(hv.model, inp, positions=[0, 2])
        assert sliced.shape[1] == 2
        torch.testing.assert_close(sliced, full[:, [0, 2]], atol=1e-6, rtol=1e-6)

    def test_write_matches_hand_rolled_hook(self, gqa_case) -> None:
        """``HeadSite.write`` (KV head 0) reproduces the same hand-rolled
        ``v_proj`` hook edit the F4 spike validated."""
        import nnsight

        hv, raw = gqa_case.hv, gqa_case.raw
        inp = gqa_case.inputs()
        sl = hv.head_column_slice(0)  # KV head 0
        site = HeadSite("value", 0, 0)

        with hv.model.trace(inp):
            clean = nnsight.save(hv.model.logits)
        with hv.model.trace(inp):
            site.write(hv.model, torch.zeros(hv.head_dim))
            edited = nnsight.save(hv.model.logits)

        def hooked() -> torch.Tensor:
            def hook(_m, _i, o):
                o = o.clone()
                o[:, :, sl] = 0.0
                return o

            h = raw.model.layers[0].self_attn.v_proj.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    return raw(
                        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                    ).logits
            finally:
                h.remove()

        manual = hooked()
        assert not torch.allclose(edited, clean, atol=1e-4)  # non-vacuous
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_out_of_range_layer_and_head_fail_loudly(self, gqa_case) -> None:
        hv, inp = gqa_case.hv, gqa_case.inputs()
        with pytest.raises(IndexError, match="layer"):
            HeadSite("value", hv.num_layers, 0).collect(hv.model, inp)
        with pytest.raises(IndexError, match="KV heads"):
            HeadSite("value", 0, hv.n_kv_heads).collect(hv.model, inp)

    def test_per_row_read_gathers_each_rows_indices(self, gqa_case) -> None:
        """Per-row ``(batch, k)`` positions (the ST2 bridge output) gather each
        row's own indices — the PL5 path-patching case where a receiver's
        position varies per example (e.g. an IOI name position)."""
        hv = gqa_case.hv
        inp = _stacked_inputs(gqa_case, [_TEXT, "a slow lazy old dog sits today"])
        site = HeadSite("value", 0, 1)
        full = site.collect(hv.model, inp)
        got = site.collect(hv.model, inp, positions=[[1], [2]])
        assert got.shape[:2] == (2, 1)
        torch.testing.assert_close(got[0, 0], full[0, 1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(got[1, 0], full[1, 2], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("family", ["llama", "gpt2"])
    def test_per_row_write_matches_hand_rolled_hook(
        self, family, gqa_case, fused_case
    ) -> None:
        """A per-row positional ``HeadSite.write`` scatters each row's own
        indices, matching a hand-rolled hook that edits row 0 at one position
        and row 1 at another — on both projection families (the fused family
        exercises the column offset composed with the per-row key)."""
        import nnsight

        case = {"llama": gqa_case, "gpt2": fused_case}[family]
        hv, raw = case.hv, case.raw
        inp = _stacked_inputs(case, [_TEXT, "a slow lazy old dog sits today"])
        site = HeadSite("value", 0, 0)
        rows = [[1], [2]]  # row 0 edits position 1, row 1 edits position 2

        with hv.model.trace(inp):
            clean = nnsight.save(hv.model.logits)
        with hv.model.trace(inp):
            site.write(hv.model, torch.zeros(hv.head_dim), positions=rows)
            edited = nnsight.save(hv.model.logits)

        if family == "llama":
            module = raw.model.layers[0].self_attn.v_proj
            cols = hv.head_column_slice(0)
        else:
            module = raw.transformer.h[0].attn.c_attn
            offset = 2 * raw.config.hidden_size  # fused [q | k | v] value columns
            cols = slice(offset, offset + hv.head_dim)

        def hook(_m, _i, o):
            o = o.clone()
            for r, (p,) in enumerate(rows):
                o[r, p, cols] = 0.0
            return o

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                manual = raw(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                ).logits
        finally:
            handle.remove()

        assert not torch.allclose(edited, clean, atol=1e-4)  # non-vacuous
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("family", ["llama", "gpt2"])
    def test_mixed_site_and_head_site_single_pass(
        self, family, gqa_case, fused_case
    ) -> None:
        """``collect_sites`` mixes whole-component ``Site``s and per-head
        ``HeadSite``s in one ordered pass on both projection families (the fused
        family exercises ``forward_rank_on``'s reordering); every result matches
        its single-tap collect."""
        case = {"llama": gqa_case, "gpt2": fused_case}[family]
        model, inp = case.hv.model, case.inputs()
        sites = [  # deliberately out of forward order
            Site("mlp_output", 1),
            HeadSite("query", 1, 1),
            Site("block_input", 0),
            HeadSite("value", 0, 0),
            Site("attention_output", 0),
        ]
        got = collect_sites(model, inp, sites)
        for one, batched in zip(sites, got):
            torch.testing.assert_close(
                batched, one.collect(model, inp), atol=1e-5, rtol=1e-4
            )


# --------------------------------------------------------------------------- #
#  golden — the real decoupled-head_dim Qwen3 contract (GPU)                   #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """Per-head receivers on the coherent GPU backbone (Qwen3-4B): the decoupled
    ``head_dim=128`` contract pyvene 0.1.8 gets wrong (returns 80-wide)."""

    pytestmark = pytest.mark.golden

    def test_receivers_match_projection_slices_at_true_head_dim(self) -> None:
        # dispatch=True materializes real weights (the default lazy load leaves
        # `_model` on meta, so the raw-hook oracle's forward can't run).
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        hv = HeadView(st)
        raw = st._model
        cfg = raw.config
        assert hv.head_dim == 128
        assert hv.head_dim != cfg.hidden_size // cfg.num_attention_heads  # 128 != 80

        tok = st.tokenizer
        enc = tok(_TEXT, return_tensors="pt")
        device = next(raw.parameters()).device
        inp = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        layer = hv.num_layers // 2
        # Qwen3 shares the Llama attention module names, so the oracle applies directly.
        flat = _raw_projections(raw, layer, inp)
        d = hv.head_dim

        q = hv.collect_query(inp, layer, 0).cpu().float()
        v = hv.collect_value(inp, layer, 0).cpu().float()
        s = hv.collect_attention_value(inp, layer, 0).cpu().float()
        torch.testing.assert_close(
            q, _expected_head(flat["query"], 0, d).cpu().float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            v, _expected_head(flat["value"], 0, d).cpu().float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            s,
            _expected_head(flat["attention_value"], 0, d).cpu().float(),
            atol=1e-3,
            rtol=1e-3,
        )
        assert q.shape[-1] == d == 128

        # ST4: the Site-protocol adapter resolves to the same slices (CPU offload).
        v_site = HeadSite("value", layer, 0).collect(st, inp).float()
        torch.testing.assert_close(v_site, v, atol=1e-3, rtol=1e-3)
        q_site = HeadSite("query", layer, 0).collect(st, inp).float()
        torch.testing.assert_close(q_site, q, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
#  property — per-row / ragged positions on HeadSite (PL3, #405)               #
# --------------------------------------------------------------------------- #
class TestHeadSitePerRowPositions:
    """``HeadSite`` shares ``Site``'s position normalization: equal-width
    per-row rows gather/scatter each batch row's own indices, ragged rows ride
    the flat advanced index — pinned against per-row slices of the full head
    read and a hand-rolled projection-hook edit."""

    pytestmark = pytest.mark.property

    def _batch(self, case) -> dict[str, torch.Tensor]:
        enc = case.tok([_TEXT, _TEXT], return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def test_per_row_read_gathers_each_rows_indices(self, gqa_case) -> None:
        inp = self._batch(gqa_case)
        site = HeadSite("value", 0, 0)
        full = site.collect(gqa_case.hv.model, inp)
        rows = [[1], [2]]
        got = site.collect(gqa_case.hv.model, inp, positions=rows)
        assert got.shape == (2, 1, full.shape[-1])
        for i, row in enumerate(rows):
            torch.testing.assert_close(got[i], full[i, row, :], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(got[0], got[1], atol=1e-5)

    def test_ragged_read_is_flat_and_matches_full(self, gqa_case) -> None:
        inp = self._batch(gqa_case)
        site = HeadSite("attention_value", 0, 1)
        full = site.collect(gqa_case.hv.model, inp)
        rows = [[1], [2, 3]]
        got = site.collect(gqa_case.hv.model, inp, positions=rows)
        assert got.shape == (3, full.shape[-1])
        per_row = torch.split(got, [1, 2])
        for i, row in enumerate(rows):
            torch.testing.assert_close(
                per_row[i], full[i, row, :], atol=1e-6, rtol=1e-6
            )

    def test_per_row_write_matches_hand_rolled_hook(self, gqa_case) -> None:
        import nnsight

        hv, raw = gqa_case.hv, gqa_case.raw
        inp = self._batch(gqa_case)
        sl = hv.head_column_slice(0)  # KV head 0
        site = HeadSite("value", 0, 0)
        rows = [[1], [2]]

        with hv.model.trace(inp):
            clean = nnsight.save(hv.model.logits)
        with hv.model.trace(inp):
            site.write(hv.model, torch.zeros(hv.head_dim), positions=rows)
            edited = nnsight.save(hv.model.logits)

        def hook(_m, _i, o):
            o = o.clone()
            for i, row in enumerate(rows):
                o[i, row, sl] = 0.0
            return o

        h = raw.model.layers[0].self_attn.v_proj.register_forward_hook(hook)
        try:
            with torch.no_grad():
                manual = raw(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
                ).logits
        finally:
            h.remove()

        assert not torch.allclose(edited, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)
