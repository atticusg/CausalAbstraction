"""Tests for :mod:`causalab.neural.site` — the ST1 Site core (#396).

Tiers (mirroring ``tests/neural/test_head_view.py``; ``causalab/neural`` owes
``unit`` + ``property``, and a ``golden`` GPU pin is the F4-established pattern for
the real coherent backbone):

* ``unit`` — the contract (component vocabulary incl. the layer-less
  ``embeddings``, the gapped ``forward_rank`` ordering with the reserved ST4
  head slots, ``mlp_activation`` architecture detection), no forward pass.
* ``property`` — on tiny Llama **and** GPT-2 (CPU): every component read matches a
  raw-``register_forward_hook`` oracle (full and position-sliced); **every**
  component's ``write`` reproduces a hand-rolled hook edit on **both** branches
  (positional slice write and whole-tensor replacement); a wrong-dtype value is
  coerced rather than crashing (the same ``.to`` moves values to a sharded site's
  device); ``collect_sites`` reads an out-of-forward-order request list in one pass.
* ``golden`` — the same read-matches-oracle equivalence on the real Qwen3-4B
  backbone (GPU) for all components in **one** ``collect_sites`` pass, confirming
  the standardized accessors resolve on a coherent model.

The oracle is the **existing** backbone-agnostic one in
``tests/neural/activations/hook_oracle.py`` — it already maps these exact
components (``component_module`` / ``capture_component`` /
``component_edited_logits``) against a model's raw HF module via
``register_forward_hook`` and never imports the intervention backbone, so a Site
read tapping the ``StandardizedTransformer`` is checked against the same ground
truth pyvene is. Those helpers only need a ``.hf_model`` attribute, so we hand
them a ``SimpleNamespace`` over the same underlying module the
``StandardizedTransformer`` wraps — the comparison is exact.

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py`` —
never the session-cached ``tiny_random_model`` singleton, whose leftover pyvene
forward hooks break a later nnsight trace (see the factory docstrings).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.neural.site import (
    COMPONENTS,
    INTRA_BLOCK_RANK,
    RaggedIndex,
    Site,
    _check_write_fits,
    _sequence_index,
    _write_slice_shape,
    collect_sites,
    hf_text_config,
)

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_edited_logits,
    component_module,
)

_TEXT = "the quick brown fox jumps"


def _layer_for(component: str, layer: int) -> int:
    """``layer`` for per-layer components, the pinned ``0`` for ``embeddings``."""
    return 0 if component == "embeddings" else layer


# --------------------------------------------------------------------------- #
#  Fixtures — fresh (uncached) StandardizedTransformers + a raw-model shim     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SiteCase:
    st: StandardizedTransformer  # Site reads/writes tap this
    oracle: Any  # SimpleNamespace(hf_model=raw) — for the hook_oracle helpers
    tok: Any

    def inputs(self, text: str = _TEXT) -> dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def capture(self, component: str, layer: int, inputs: Any) -> torch.Tensor:
        """The raw-hook ground truth for ``(component, layer)`` — full ``(b, seq, d)``."""
        module, kind = component_module(self.oracle, layer, component)
        return capture_component(self.oracle, module, kind, inputs)

    def feature_width(self, component: str) -> int:
        """The site's feature width, from config (no forward pass): the hidden
        size, except ``mlp_activation`` which is intermediate-width. GPT-2-style
        configs are checked for ``n_inner`` first (``None`` → ``4*hidden``, per
        ``GPT2MLP``): the tiny-random-gpt2 config carries a stray
        ``intermediate_size`` field the architecture ignores."""
        cfg = self.oracle.hf_model.config
        if component == "mlp_activation":
            if hasattr(cfg, "n_inner"):  # GPT-2 family
                return int(cfg.n_inner or 4 * cfg.hidden_size)
            return int(cfg.intermediate_size)
        return int(cfg.hidden_size)


def _case(raw: Any, tok: Any) -> SiteCase:
    st = StandardizedTransformer(raw, tokenizer=tok, check_renaming=True)
    # Dispatch materializes real weights so the first trace runs directly instead
    # of nnterp's shape-scan fallback (which, for GPT-2, hits a FakeTensor
    # data-dependent guard) — matching the F3 LMPipeline load path.
    st.dispatch()
    return SiteCase(st=st, oracle=SimpleNamespace(hf_model=raw), tok=tok)


@pytest.fixture(scope="module")
def llama_case() -> SiteCase:
    return _case(*fresh_tiny_random_llama())


@pytest.fixture(scope="module")
def gpt2_case() -> SiteCase:
    return _case(*fresh_tiny_random_gpt2())


# --------------------------------------------------------------------------- #
#  unit — the contract, no forward pass                                        #
# --------------------------------------------------------------------------- #
class TestContractUnit:
    pytestmark = pytest.mark.unit

    def test_rejects_unknown_component(self) -> None:
        with pytest.raises(ValueError, match="unknown component"):
            Site("resid", 0)  # type: ignore[arg-type]

    def test_rejects_negative_layer(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Site("block_output", -1)

    def test_embeddings_is_layer_less(self) -> None:
        with pytest.raises(ValueError, match="layer-less"):
            Site("embeddings", 1)
        # ... and sorts before the first block's input.
        assert Site("embeddings", 0).forward_rank < Site("block_input", 0).forward_rank

    def test_forward_rank_orders_within_a_block(self) -> None:
        # embeddings < block_input < attention_output < mlp_input < mlp_activation
        # < mlp_output < block_output — the intra-block execution order.
        order = [
            "embeddings",
            "block_input",
            "attention_output",
            "mlp_input",
            "mlp_activation",
            "mlp_output",
            "block_output",
        ]
        ranks = [Site(c, 0).forward_rank for c in order]  # type: ignore[arg-type]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(order)  # strictly increasing (no ties)

    def test_reserved_head_ranks_slot_inside_attention(self) -> None:
        # The ST4 per-head taps (q → k → v → o-input; head_view._KIND_RANK derives
        # from these) must sort strictly between block_input and attention_output —
        # the gapped-rank contract that lets ST4 fold into collect_sites without
        # renumbering the published forward_rank ordering.
        lo = INTRA_BLOCK_RANK["block_input"]
        hi = INTRA_BLOCK_RANK["attention_output"]
        ranks = [
            INTRA_BLOCK_RANK[k] for k in ("query", "key", "value", "attention_value")
        ]
        assert ranks == sorted(ranks)
        assert all(lo < r < hi for r in ranks)

    def test_is_mlp_activation_flag(self) -> None:
        assert Site("mlp_activation", 3).is_mlp_activation is True
        assert Site("block_output", 3).is_mlp_activation is False

    def test_all_components_addressable(self) -> None:
        # Every declared component builds a Site and reports a rank.
        for c in COMPONENTS:
            assert isinstance(Site(c, _layer_for(c, 1)).forward_rank, int)

    def test_mlp_activation_kind_llama(self, llama_case: SiteCase) -> None:
        # SwiGLU Llama exposes the intermediate activation as act_fn's output.
        assert Site("mlp_activation", 0).mlp_activation_kind(llama_case.st) == (
            "act_fn",
            "output",
        )

    def test_mlp_activation_kind_gpt2(self, gpt2_case: SiteCase) -> None:
        # GPT-2 exposes it as c_proj's input.
        assert Site("mlp_activation", 0).mlp_activation_kind(gpt2_case.st) == (
            "c_proj",
            "input",
        )


class TestHfTextConfigUnit:
    """`hf_text_config` — nnterp's ``text_config`` nesting rule over the
    standardized/underlying config resolution (#449 finding 1)."""

    pytestmark = pytest.mark.unit

    def test_plain_config_passes_through(self) -> None:
        from transformers import LlamaConfig

        cfg = LlamaConfig()
        assert hf_text_config(SimpleNamespace(config=cfg)) is cfg

    def test_nested_text_config_is_unwrapped(self) -> None:
        gemma3 = pytest.importorskip("transformers").Gemma3Config()
        resolved = hf_text_config(SimpleNamespace(config=gemma3))
        assert resolved is gemma3.text_config
        # The fields the raw top-level read gets silently wrong on Gemma3.
        assert resolved.num_key_value_heads == 4
        assert resolved.head_dim == 256

    def test_falls_back_to_underlying_model_config(self) -> None:
        from transformers import LlamaConfig

        cfg = LlamaConfig()
        wrapper = SimpleNamespace(config=None, _model=SimpleNamespace(config=cfg))
        assert hf_text_config(wrapper) is cfg

    def test_no_config_raises(self) -> None:
        with pytest.raises(ValueError, match="no HF config"):
            hf_text_config(SimpleNamespace(config=None, _model=None))


# --------------------------------------------------------------------------- #
#  property — reads/writes match the raw-hook oracle on Llama and GPT-2        #
# --------------------------------------------------------------------------- #
class TestReadsMatchOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: SiteCase, gpt2_case: SiteCase) -> SiteCase:
        """A tiny-random case across the two families whose MLPs expose
        ``mlp_activation`` differently (Llama ``act_fn`` output / GPT-2 ``c_proj``
        input) — the meaningful architecture axis for the whole-sublayer components."""
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    @pytest.mark.parametrize("positions", [None, [1]], ids=["all", "pos1"])
    @pytest.mark.parametrize("component", COMPONENTS)
    def test_read_matches_capture(
        self, case: SiteCase, component: str, positions: list[int] | None
    ) -> None:
        inputs = case.inputs()
        got = Site(component, 0).collect(case.st, inputs, positions=positions)  # type: ignore[arg-type]
        expected = case.capture(component, 0, inputs)
        if positions is not None:
            expected = expected[:, positions, :]
        assert got.shape == expected.shape
        assert got.device.type == "cpu"  # collect offloads (package convention)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    def test_read_second_layer(self, case: SiteCase) -> None:
        # Layer index is honoured (not silently pinned to 0).
        inputs = case.inputs()
        got = Site("block_output", 1).collect(case.st, inputs)
        expected = case.capture("block_output", 1, inputs)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    def _assert_write_matches_hook(
        self, case: SiteCase, component: str, whole_tensor: bool
    ) -> None:
        """``Site.write`` reproduces a hand-rolled forward-hook edit at the same
        component (the ED1 intervention path), on the branch under test:
        positional slice write at the last token, or whole-tensor replacement
        (``positions=None`` — the branch that must stay tuple-safe on accessors
        whose tuple-ness was never pre-detected). Editing (at least) the last
        position guarantees the last-token logits move, so the ground-truth hook
        is non-vacuous on every architecture; the equivalence assertion is then
        what catches a write that fails to propagate.

        The perturbation is a **non-uniform** ramp across the feature dim, not a
        uniform additive constant: a pre-LayerNorm model (GPT-2) subtracts the
        per-position mean, so a uniform shift is cancelled and the edit reads as
        vacuous. A ramp shifts the variance/shape and survives both LayerNorm and
        RMSNorm.
        """
        model = case.st
        layer = _layer_for(component, 0)
        site = Site(component, layer)  # type: ignore[arg-type]
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        delta = torch.linspace(-10.0, 10.0, case.feature_width(component))

        with model.trace(inputs):
            clean = model.logits[:, -1, :].cpu().save()
        with model.trace(inputs):
            if whole_tensor:
                site.write(model, site.read(model) + delta, positions=None)
            else:
                site.write(model, site.read(model, [last]) + delta, positions=[last])
            edited = model.logits[:, -1, :].cpu().save()

        module, kind = component_module(case.oracle, layer, component)

        def edit(h: torch.Tensor) -> None:
            if whole_tensor:
                h += delta
            else:
                h[:, last, :] = h[:, last, :] + delta

        manual = component_edited_logits(case.oracle, inputs, module, kind, edit)

        # The edit is non-vacuous (the oracle itself moves the logits) ...
        assert not torch.allclose(manual, clean, atol=1e-4)
        # ... and Site.write reproduces the oracle exactly.
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("whole_tensor", [False, True], ids=["slice", "whole"])
    @pytest.mark.parametrize("component", COMPONENTS)
    def test_write_matches_hook(
        self, case: SiteCase, component: str, whole_tensor: bool
    ) -> None:
        """Every component's write path, on both branches — the full ST1
        intervention surface is oracle-pinned, not just the residual stream."""
        self._assert_write_matches_hook(case, component, whole_tensor)

    def test_write_coerces_value_dtype(self, case: SiteCase) -> None:
        """A wrong-dtype value is moved to the site's dtype instead of crashing
        (the positional path lowers to ``index_put_``, which requires an exact
        dtype/device match — the same ``.to`` moves values to a sharded site's
        device under ``hf_device_map``). float32 → float64 → float32 round-trips
        exactly, so the result must equal the plain-dtype oracle edit."""
        model = case.st
        site = Site("block_output", 0)
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        delta = torch.linspace(-10.0, 10.0, case.feature_width("block_output"))

        with model.trace(inputs):
            value = (site.read(model, [last]) + delta).to(torch.float64)
            site.write(model, value, positions=[last])
            edited = model.logits[:, -1, :].cpu().save()

        module, kind = component_module(case.oracle, 0, "block_output")

        def edit(h: torch.Tensor) -> None:
            h[:, last, :] = h[:, last, :] + delta

        manual = component_edited_logits(case.oracle, inputs, module, kind, edit)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_collect_sites_orders_reads_by_forward_position(
        self, case: SiteCase
    ) -> None:
        """A mixed request list in *non*-forward order reads in one pass (no
        ``MissedProviderError``) and every result still matches its own capture."""
        inputs = case.inputs()
        # deliberately out of forward order: layer 1 before 0, mlp_output before
        # attention_output, block_output (last) before block_input (first), and
        # the embeddings (earliest tap of all) requested dead last.
        sites = [
            Site("block_output", 1),
            Site("attention_output", 0),
            Site("mlp_output", 1),
            Site("block_input", 0),
            Site("embeddings", 0),
        ]
        got = collect_sites(case.st, inputs, sites)
        assert len(got) == len(sites)  # aligned with `sites`, not read order
        for site, tensor in zip(sites, got):
            expected = case.capture(site.component, site.layer, inputs)
            torch.testing.assert_close(tensor, expected, atol=1e-5, rtol=1e-4)

    def test_collect_sites_rejects_mismatched_positions(self, case: SiteCase) -> None:
        inputs = case.inputs()
        with pytest.raises(ValueError, match="positions has length"):
            collect_sites(
                case.st,
                inputs,
                [Site("block_output", 0), Site("block_input", 0)],
                positions=[[1]],  # 1 entry for 2 sites
            )


# --------------------------------------------------------------------------- #
#  unit — per-row position normalization (the ST2 consumption contract)        #
# --------------------------------------------------------------------------- #
class TestPositionNormalizationUnit:
    """``_sequence_index`` — how ``read``/``write`` tell one flat row (uniform
    across the batch) from per-row padded-frame indices (what the ST2 bridge
    ``causalab.neural.positions.resolve_positions`` produces)."""

    pytestmark = pytest.mark.unit

    def test_none_passes_through(self) -> None:
        assert _sequence_index(None) is None

    def test_flat_row_stays_uniform(self) -> None:
        assert _sequence_index([1, 3]) == [1, 3]

    def test_1d_tensor_stays_uniform(self) -> None:
        idx = torch.tensor([1, 3])
        assert _sequence_index(idx) is idx

    def test_nested_rows_become_a_per_row_long_tensor(self) -> None:
        got = _sequence_index([[1, 2], [3, 4]])
        assert isinstance(got, torch.Tensor)
        assert got.dtype == torch.long
        assert got.tolist() == [[1, 2], [3, 4]]

    def test_2d_tensor_is_per_row(self) -> None:
        got = _sequence_index(torch.tensor([[1], [2]]))
        assert got.dtype == torch.long and got.shape == (2, 1)

    def test_ragged_rows_become_a_flat_advanced_index(self) -> None:
        # Ragged per-row spans batch as one flat gather/scatter (PL3, #405):
        # row_ids pair each selected position with its batch row, widths let
        # consumers re-nest the flat (total, hidden) view per example.
        got = _sequence_index([[1], [2, 3]])
        assert isinstance(got, RaggedIndex)
        assert got.row_ids.tolist() == [0, 1, 1]
        assert got.col_ids.tolist() == [1, 2, 3]
        assert got.widths == (1, 2)

    def test_3d_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="positions tensor"):
            _sequence_index(torch.zeros(1, 1, 1, dtype=torch.long))


# --------------------------------------------------------------------------- #
#  unit — write width guard (the scan-preflight width-mismatch catch, #458)    #
# --------------------------------------------------------------------------- #
class TestWriteWidthGuardUnit:
    """``_write_slice_shape`` + ``_check_write_fits`` — the explicit setitem
    broadcast check ``Site.write``/``HeadView.write`` run on tensor values.
    The real backend raises on these mismatches anyway (opaquely); under
    ``model.scan()`` fake tensors do NOT value-check advanced-indexing writes,
    so this guard is what makes the width-mismatch class visible to the CAP5
    preflight — with one legible error in both modes."""

    pytestmark = pytest.mark.unit

    _PROXY = (2, 9, 16)  # (batch, seq, hidden)

    def test_slice_shape_forms(self) -> None:
        assert _write_slice_shape(self._PROXY, None) == (2, 9, 16)
        assert _write_slice_shape(self._PROXY, [3, 5]) == (2, 2, 16)
        assert _write_slice_shape(self._PROXY, torch.tensor([3])) == (2, 1, 16)
        per_row = _sequence_index([[1, 2], [3, 4]])
        assert _write_slice_shape(self._PROXY, per_row) == (2, 2, 16)
        ragged = _sequence_index([[1], [2, 3]])
        assert _write_slice_shape(self._PROXY, ragged) == (3, 16)

    def test_broadcastable_values_fit(self) -> None:
        slice_shape = (2, 1, 16)
        for shape in [(2, 1, 16), (1, 16), (16,), (1, 1, 16)]:
            _check_write_fits(torch.zeros(shape), slice_shape, "Site")  # no raise

    def test_width_mismatch_refused_legibly(self) -> None:
        with pytest.raises(ValueError, match="Widths must pair up"):
            _check_write_fits(torch.zeros(2, 2, 16), (2, 1, 16), "Site(test)")

    def test_extra_leading_dims_refused(self) -> None:
        # setitem broadcasting is one-way: a value with MORE dims than the
        # slice never fits, even with leading 1s.
        with pytest.raises(ValueError, match="does not broadcast"):
            _check_write_fits(torch.zeros(1, 2, 1, 16), (2, 1, 16), "Site(test)")

    def test_ragged_total_mismatch_refused(self) -> None:
        with pytest.raises(ValueError, match="does not broadcast"):
            _check_write_fits(torch.zeros(4, 16), (3, 16), "Site(test)")


class TestWriteWidthGuardProperty:
    """The guard fires inside a real trace with the legible message (instead
    of the backend's opaque index error) — and a broadcastable vector write
    still passes (the oracle-pinned write tests cover its correctness)."""

    pytestmark = pytest.mark.property

    def test_real_trace_width_mismatch_raises_legibly(
        self, llama_case: SiteCase
    ) -> None:
        model = llama_case.st
        inputs = llama_case.inputs()
        width = llama_case.feature_width("block_output")
        with pytest.raises(ValueError, match="Widths must pair up"):
            with model.trace(inputs):
                Site("block_output", 0).write(
                    model, torch.zeros(1, 2, width), positions=[-1]
                )

    def test_real_trace_broadcast_vector_write_passes(
        self, llama_case: SiteCase
    ) -> None:
        model = llama_case.st
        inputs = llama_case.inputs()
        width = llama_case.feature_width("block_output")
        with model.trace(inputs):
            Site("block_output", 0).write(model, torch.zeros(width), positions=[-1])
            logits = model.logits.cpu().save()
        assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
#  property — per-row reads/writes match the raw-hook oracle                   #
# --------------------------------------------------------------------------- #
class TestPerRowPositionsProperty:
    """Per-row positions gather/scatter each batch row's *own* indices — pinned
    against the raw-hook oracle on a batch whose rows deliberately use different
    positions, so a broadcast bug (every row reading row 0's indices) fails."""

    pytestmark = pytest.mark.property

    _ROWS = [[1], [2]]

    def _batch_inputs(self, case: SiteCase) -> dict[str, torch.Tensor]:
        # The same text twice: equal-length rows, no padding needed — per-row
        # indexing itself is what's under test (the padding-frame story lives
        # with the bridge in tests/neural/test_positions.py).
        enc = case.tok([_TEXT, _TEXT], return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def test_per_row_read_gathers_each_rows_indices(self, llama_case: SiteCase) -> None:
        inputs = self._batch_inputs(llama_case)
        got = Site("block_output", 0).collect(
            llama_case.st, inputs, positions=self._ROWS
        )
        full = llama_case.capture("block_output", 0, inputs)
        assert got.shape == (2, 1, full.shape[-1])
        for i, row in enumerate(self._ROWS):
            torch.testing.assert_close(got[i], full[i, row, :], atol=1e-5, rtol=1e-4)
        # Identical inputs, different rows: the two results must differ, so a
        # broadcast bug cannot pass the row-wise equality above vacuously.
        assert not torch.allclose(got[0], got[1], atol=1e-4)

    def test_per_row_write_matches_per_row_hook_edit(
        self, llama_case: SiteCase
    ) -> None:
        model = llama_case.st
        site = Site("block_output", 0)
        inputs = self._batch_inputs(llama_case)
        delta = torch.linspace(-10.0, 10.0, llama_case.feature_width("block_output"))

        with model.trace(inputs):
            clean = model.logits[:, -1, :].cpu().save()
        with model.trace(inputs):
            site.write(
                model, site.read(model, self._ROWS) + delta, positions=self._ROWS
            )
            edited = model.logits[:, -1, :].cpu().save()

        module, kind = component_module(llama_case.oracle, 0, "block_output")

        def edit(h: torch.Tensor) -> None:
            for i, row in enumerate(self._ROWS):
                h[i, row, :] = h[i, row, :] + delta

        manual = component_edited_logits(llama_case.oracle, inputs, module, kind, edit)
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_per_row_with_identical_rows_equals_uniform_slice(
        self, llama_case: SiteCase
    ) -> None:
        inputs = self._batch_inputs(llama_case)
        site = Site("block_output", 0)
        per_row = site.collect(llama_case.st, inputs, positions=[[1], [1]])
        uniform = site.collect(llama_case.st, inputs, positions=[1])
        torch.testing.assert_close(per_row, uniform, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  golden — the standardized accessors resolve on the real Qwen3-4B backbone   #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """Every component reads correctly on the coherent GPU backbone (Qwen3-4B),
    matched against the same raw-hook oracle — all in **one** ``collect_sites``
    forward pass (the oracle side necessarily runs one capture per component)."""

    pytestmark = pytest.mark.golden

    def test_reads_match_capture_on_coherent_model(self) -> None:
        # dispatch=True materializes real weights so the raw-hook oracle can run.
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        oracle = SimpleNamespace(hf_model=raw)
        tok = st.tokenizer
        enc = tok(_TEXT, return_tensors="pt")
        device = next(raw.parameters()).device
        inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        layer = int(st.num_layers) // 2
        sites = [Site(c, _layer_for(c, layer)) for c in COMPONENTS]
        got_list = collect_sites(st, inputs, sites)
        for site, got in zip(sites, got_list):
            assert got.device.type == "cpu"  # offloaded as collected
            module, kind = component_module(oracle, site.layer, site.component)
            expected = capture_component(oracle, module, kind, inputs)
            assert got.shape == expected.shape
            torch.testing.assert_close(
                got.float(), expected.cpu().float(), atol=1e-3, rtol=1e-3
            )


# --------------------------------------------------------------------------- #
#  property — ragged per-row reads/writes match the raw-hook oracle (PL3)      #
# --------------------------------------------------------------------------- #
class TestRaggedPositionsProperty:
    """Ragged per-row positions batch as ONE flat gather/scatter — pinned
    against the raw-hook oracle on rows of deliberately different widths, so
    both a broadcast bug and a row/column pairing bug fail loudly."""

    pytestmark = pytest.mark.property

    _ROWS = [[1], [2, 3]]  # widths 1 and 2 — genuinely ragged

    def _batch_inputs(self, case: SiteCase) -> dict[str, torch.Tensor]:
        enc = case.tok([_TEXT, _TEXT], return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def test_ragged_read_gathers_flat_positions(self, llama_case: SiteCase) -> None:
        inputs = self._batch_inputs(llama_case)
        got = Site("block_output", 0).collect(
            llama_case.st, inputs, positions=self._ROWS
        )
        full = llama_case.capture("block_output", 0, inputs)
        # Flat (total_positions, hidden) view, re-nestable by widths.
        assert got.shape == (3, full.shape[-1])
        per_row = torch.split(got, [len(r) for r in self._ROWS])
        for i, row in enumerate(self._ROWS):
            torch.testing.assert_close(
                per_row[i], full[i, row, :], atol=1e-5, rtol=1e-4
            )

    def test_ragged_write_matches_per_row_hook_edit(self, llama_case: SiteCase) -> None:
        model = llama_case.st
        site = Site("block_output", 0)
        inputs = self._batch_inputs(llama_case)
        delta = torch.linspace(-10.0, 10.0, llama_case.feature_width("block_output"))

        with model.trace(inputs):
            clean = model.logits[:, -1, :].cpu().save()
        with model.trace(inputs):
            site.write(
                model, site.read(model, self._ROWS) + delta, positions=self._ROWS
            )
            edited = model.logits[:, -1, :].cpu().save()

        module, kind = component_module(llama_case.oracle, 0, "block_output")

        def edit(h: torch.Tensor) -> None:
            for i, row in enumerate(self._ROWS):
                h[i, row, :] = h[i, row, :] + delta

        manual = component_edited_logits(llama_case.oracle, inputs, module, kind, edit)
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)
