"""Tests for :mod:`causalab.neural.edit` — the ED1 wrap (#400).

Tiers (mirroring ``tests/neural/test_featurized_site.py``; ``causalab/neural``
owes ``unit`` + ``property``, and a ``golden`` GPU pin is the established
pattern for the real coherent backbone):

* ``unit`` — the contract (``read_sources`` requires ``g``; a read-only
  ``Edit`` refuses ``.apply()``), no forward pass.
* ``property`` — on tiny Llama **and** GPT-2 (CPU), against the same
  raw-``register_forward_hook`` oracle ST1/ST3 are pinned to: the collect
  shape matches ``FeaturizedSite.collect``; the replace shape (``g`` ignoring
  ``f``) matches the oracle; the general-RMW shape via one and two
  ``read_sources`` (the cross-site transplant and an interpolate-style blend)
  matches the oracle; a non-``FeaturizedSite`` (constant) ``read_sources``
  entry is coerced to the site's device/dtype before ``g`` runs; a
  forward-order violation raises.
* ``golden`` — the cross-site RMW shape on the real Qwen3-4B backbone (GPU)
  against the same oracle, one model load.

Models come from the fresh (uncached) factories in ``tests/_helpers/tiny.py``
— never the session-cached ``tiny_random_model`` singleton, whose leftover
pyvene forward hooks break a later nnsight trace (see the factory docstrings).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from nnterp import StandardizedTransformer

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import Site

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.activations.hook_oracle import (
    capture_component,
    component_edited_logits,
    component_module,
)

_TEXT = "the quick brown fox jumps"


def _subspace(width: int, k: int, seed: int = 0) -> SubspaceFeaturizer:
    """A deterministic frozen rotation ``width → k`` (seeded orthogonal init)."""
    torch.manual_seed(seed)
    return SubspaceFeaturizer(shape=(width, k), trainable=False)


# --------------------------------------------------------------------------- #
#  Fixtures — fresh (uncached) StandardizedTransformers + a raw-model shim     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SiteCase:
    st: StandardizedTransformer  # Edit reads/writes tap this
    oracle: Any  # SimpleNamespace(hf_model=raw) — for the hook_oracle helpers
    tok: Any

    def inputs(self, text: str = _TEXT) -> dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def capture(self, component: str, layer: int, inputs: Any) -> torch.Tensor:
        """The raw-hook ground truth for ``(component, layer)`` — full ``(b, seq, d)``."""
        module, kind = component_module(self.oracle, layer, component)
        return capture_component(self.oracle, module, kind, inputs)

    def clean_logits(self, inputs: Any) -> torch.Tensor:
        with self.st.trace(inputs):
            clean = self.st.logits[:, -1, :].cpu().save()
        return clean

    def hidden(self) -> int:
        return int(self.oracle.hf_model.config.hidden_size)

    def oracle_edit_logits(
        self,
        site: Site,
        feat,
        g,
        positions: list[int] | None,
        inputs: Any,
    ) -> torch.Tensor:
        """Ground truth: a hand-rolled forward hook applying featurize → ``g`` →
        inverse (with the base error) on the raw activation, offline."""
        module, kind = component_module(self.oracle, site.layer, site.component)

        def edit(h: torch.Tensor) -> None:
            sel = slice(None) if positions is None else positions
            f, err = feat.featurize(h[:, sel])
            h[:, sel] = feat.inverse_featurize(g(f), err).to(h.dtype)

        return component_edited_logits(self.oracle, inputs, module, kind, edit)


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

    def test_read_sources_without_g_rejected(self) -> None:
        fsite = FeaturizedSite(Site("block_output", 0))
        src = ReadSource(FeaturizedSite(Site("block_output", 0)))
        with pytest.raises(ValueError, match="read_sources requires g"):
            Edit(fsite, read_sources=(src,))

    def test_collect_only_edit_defaults(self) -> None:
        edit = Edit(FeaturizedSite(Site("block_output", 0)))
        assert edit.g is None
        assert edit.read_sources == ()

    def test_apply_on_read_only_edit_rejected(self) -> None:
        edit = Edit(FeaturizedSite(Site("block_output", 0)))
        with pytest.raises(ValueError, match="read-only Edit"):
            edit.apply(model=None)  # never reaches model — rejected before use

    def test_read_source_is_site_flag(self) -> None:
        assert ReadSource(FeaturizedSite(Site("block_output", 0))).is_site
        assert not ReadSource(torch.zeros(3)).is_site

    def test_cross_input_read_source_must_be_site_backed(self) -> None:
        """`input=` marks a read under another plan input — meaningless for a
        constant, which carries no notion of the input it is read under."""
        with pytest.raises(ValueError, match="site-backed"):
            ReadSource(torch.zeros(3), input="source")

    def test_apply_refuses_cross_input_read_sources(self) -> None:
        """A single Edit.apply() runs over one input; staging a cross-input
        read (and the barrier that moves it) is the plan compiler's job."""
        edit = Edit(
            FeaturizedSite(Site("block_output", 0)),
            g=lambda f, f_src: f_src,
            read_sources=(
                ReadSource(FeaturizedSite(Site("block_output", 0)), input="source"),
            ),
        )
        with pytest.raises(ValueError, match="Plan"):
            edit.apply(model=None)  # never reaches model — rejected before use


# --------------------------------------------------------------------------- #
#  property — Edit matches the raw-hook oracle                                 #
# --------------------------------------------------------------------------- #
class TestEditMatchesOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: SiteCase, gpt2_case: SiteCase) -> SiteCase:
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    # ------------------------------ collect ---------------------------------- #
    def test_collect_matches_featurized_site_collect(self, case: SiteCase) -> None:
        inputs = case.inputs()
        fsite = FeaturizedSite(Site("block_output", 0), _subspace(case.hidden(), 3))
        edit = Edit(fsite)
        got = edit.collect(case.st, inputs)
        expected = fsite.collect(case.st, inputs)
        torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)

    # ------------------------------ replace ---------------------------------- #
    def test_replace_shape_matches_oracle(self, case: SiteCase) -> None:
        """``g`` ignoring ``f`` entirely — the write-only/"replace" shape."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        value = torch.linspace(-30.0, 30.0, 3)

        clean = case.clean_logits(inputs)
        edit = Edit(FeaturizedSite(site, feat), g=lambda f: value, positions=[last])
        with case.st.trace(inputs):
            edit.apply(case.st)
            edited = case.st.logits[:, -1, :].cpu().save()

        manual = case.oracle_edit_logits(site, feat, lambda f: value, [last], inputs)
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ general RMW ------------------------------ #
    def test_single_read_source_matches_cross_site_transplant(
        self, case: SiteCase
    ) -> None:
        """One ``ReadSource`` — the ED1 interchange shape: read an earlier
        site's features, write them at a later one, in a single trace."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src = FeaturizedSite(Site("block_output", 0), feat)
        dst_site = Site("block_output", 1)

        edit = Edit(
            FeaturizedSite(dst_site, feat),
            g=lambda f, f_src: f_src,
            read_sources=(ReadSource(src, positions=[last]),),
            positions=[last],
        )
        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            edit.apply(case.st)
            edited = case.st.logits[:, -1, :].cpu().save()

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        manual = case.oracle_edit_logits(
            dst_site, feat, lambda f: f_src, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_two_read_sources_blend_matches_oracle(self, case: SiteCase) -> None:
        """Two ``read_sources`` — proves ``g(f, *aux)``'s arity beyond one aux
        (an interpolate-style blend of two earlier sites, ignoring the
        destination's own base features)."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src0 = FeaturizedSite(Site("block_output", 0), feat)
        src1 = FeaturizedSite(Site("mlp_output", 0), feat)
        dst_site = Site("block_output", 1)

        def blend(f: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
            return 0.5 * f0 + 0.5 * f1

        edit = Edit(
            FeaturizedSite(dst_site, feat),
            g=blend,
            read_sources=(
                ReadSource(src0, positions=[last]),
                ReadSource(src1, positions=[last]),
            ),
            positions=[last],
        )
        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            edit.apply(case.st)
            edited = case.st.logits[:, -1, :].cpu().save()

        f0, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        f1, _ = feat.featurize(case.capture("mlp_output", 0, inputs)[:, [last]])
        manual = case.oracle_edit_logits(
            dst_site, feat, lambda f: 0.5 * f0 + 0.5 * f1, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
    def test_constant_read_source_coerced_to_site_dtype(
        self, case: SiteCase, value_dtype: torch.dtype
    ) -> None:
        """A non-``FeaturizedSite`` ``ReadSource`` (a plain constant, e.g. a
        steering vector) is coerced to the base features' device/dtype before
        ``g`` runs, the same as a site-backed aux — resolving and placing
        every aux input is this layer's job, not each mode's."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        value = torch.zeros(3, dtype=value_dtype)

        edit = Edit(
            FeaturizedSite(site, feat),
            g=lambda f, v: v,
            read_sources=(ReadSource(value),),
            positions=[last],
        )
        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            edit.apply(case.st)
            edited = case.st.logits[:, -1, :].cpu().save()

        manual = case.oracle_edit_logits(
            site, feat, lambda f: torch.zeros_like(f), [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ forward-order boundary ------------------- #
    def test_read_source_after_destination_rejected(self, case: SiteCase) -> None:
        """A ``ReadSource`` firing *after* the destination site in forward
        order can't be resolved inside one trace — the honest boundary PL1
        (cross-pass plans) exists to lift."""
        inputs = case.inputs()
        feat = _subspace(case.hidden(), 3)
        later_src = FeaturizedSite(Site("block_output", 1), feat)
        earlier_dst = Site("block_output", 0)

        edit = Edit(
            FeaturizedSite(earlier_dst, feat),
            g=lambda f, f_src: f_src,
            read_sources=(ReadSource(later_src),),
        )
        with pytest.raises(ValueError, match="forward order"):
            with case.st.trace(inputs):
                edit.apply(case.st)


# --------------------------------------------------------------------------- #
#  golden — the wrap holds on the real Qwen3-4B backbone                       #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """The cross-site RMW shape on the coherent GPU backbone (Qwen3-4B),
    matched against the same offline-featurized raw-hook oracle."""

    pytestmark = pytest.mark.golden

    def test_cross_site_edit_matches_oracle_on_coherent_model(self) -> None:
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        oracle = SimpleNamespace(hf_model=raw)
        case = SiteCase(st=st, oracle=oracle, tok=st.tokenizer)
        enc = st.tokenizer(_TEXT, return_tensors="pt")
        device = next(raw.parameters()).device
        inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        last = int(inputs["input_ids"].shape[1] - 1)
        layer = int(st.num_layers) // 2
        feat = _subspace(int(raw.config.hidden_size), 8)
        src = FeaturizedSite(Site("block_output", layer), feat)
        dst_site = Site("block_output", layer + 1)

        edit = Edit(
            FeaturizedSite(dst_site, feat),
            g=lambda f, f_src: f_src,
            read_sources=(ReadSource(src, positions=[last]),),
            positions=[last],
        )
        clean = case.clean_logits(inputs)
        with st.trace(inputs):
            edit.apply(st)
            edited = st.logits[:, -1, :].cpu().save()

        f_src, _ = feat.featurize(
            case.capture("block_output", layer, inputs)[:, [last]]
        )
        manual = case.oracle_edit_logits(
            dst_site, feat, lambda f: f_src, [last], inputs
        )
        assert not torch.allclose(manual, clean.float(), atol=1e-2)
        torch.testing.assert_close(
            edited.float(), manual.cpu().float(), atol=1e-3, rtol=1e-3
        )
