"""Tests for :mod:`causalab.neural.modes` — the ED2 mode constructors (#401).

Tiers (mirroring ``tests/neural/test_edit.py``; ``causalab/neural`` owes
``unit`` + ``property``, and a ``golden`` GPU pin is the established pattern
for the real coherent backbone):

* ``unit`` — the declarative contracts (each constructor returns the expected
  :class:`Edit` shape; width/count mismatches rejected at construction) and
  the two stateful helpers on plain tensors: :class:`SeededNoise` (same seed →
  same stream; the stream *advances* across calls; ``reset()`` restarts it)
  and :class:`MaskGate` (soft gate in training needs a temperature, hard 0/1
  gate in eval, L1 sparsity loss).
* ``property`` — on tiny Llama **and** GPT-2 (CPU), against the same
  raw-``register_forward_hook`` oracle ST1/ST3/ED1 are pinned to: pyvene-parity
  semantics for replace / steer / interchange (full **and** ``feature_ids``
  subspace scatter) / interpolate (the ``fn(f_base=..., f_src=...)`` keyword
  contract) / seeded noise (exact draw reproduced offline from the same seed)
  / mask (soft and hard gate).
* ``golden`` — steer against the oracle + seeded-noise determinism (a CUDA
  generator) on the real Qwen3-4B backbone (GPU), one model load; left for the
  nightly runner, not run interactively.

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
from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import (
    MaskGate,
    SeededNoise,
    collect,
    interchange,
    interpolate,
    mask,
    noise,
    replace,
    steer,
)
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
    st: StandardizedTransformer  # mode edits tap this
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

    def edited_logits(self, edit: Edit, inputs: Any) -> torch.Tensor:
        with self.st.trace(inputs):
            edit.apply(self.st)
            edited = self.st.logits[:, -1, :].cpu().save()
        return edited

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
#  unit — declarative contracts + the stateful helpers, no forward pass        #
# --------------------------------------------------------------------------- #
class TestConstructorContractsUnit:
    pytestmark = pytest.mark.unit

    def test_collect_is_read_only(self) -> None:
        edit = collect(Site("block_output", 0))
        assert edit.g is None and edit.read_sources == ()
        assert isinstance(edit.site, FeaturizedSite)  # bare Site wrapped (identity)

    def test_source_modes_wire_one_read_source(self) -> None:
        src_site = Site("block_output", 0)
        dst = Site("block_output", 1)
        for edit in (
            interchange(dst, src_site),
            interpolate(dst, src_site, lambda f_base, f_src: f_base),
            mask(dst, src_site, MaskGate(tie=True)),
        ):
            (rs,) = edit.read_sources
            assert rs.is_site and isinstance(rs.value, FeaturizedSite)

    def test_constant_modes_wire_one_constant(self) -> None:
        site = Site("block_output", 0)
        for edit in (replace(site, torch.zeros(4)), steer(site, torch.zeros(4))):
            (rs,) = edit.read_sources
            assert not rs.is_site

    def test_noise_g_is_the_seeded_stream(self) -> None:
        state = SeededNoise(7)
        edit = noise(Site("block_output", 0), 3.0, seed=state)
        assert edit.g is state  # caller-held stream: reset() reaches the edit
        assert isinstance(noise(Site("block_output", 0), 3.0, seed=7).g, SeededNoise)

    def test_feature_width_mismatch_rejected(self) -> None:
        fsite = FeaturizedSite(Site("block_output", 0), _subspace(16, 3))
        with pytest.raises(ValueError, match="feature width"):
            steer(fsite, torch.zeros(5))
        with pytest.raises(ValueError, match="feature width"):
            replace(fsite, torch.zeros(5))
        # scalars and width-1 tensors broadcast — accepted
        steer(fsite, torch.zeros(1))
        noise(fsite, torch.tensor(3.0))

    def test_feature_ids_gather_sets_the_width(self) -> None:
        fsite = FeaturizedSite(
            Site("block_output", 0), _subspace(16, 4), feature_ids=(0, 2)
        )
        replace(fsite, torch.zeros(2))  # matches the gather, not n_features
        with pytest.raises(ValueError, match="feature width"):
            replace(fsite, torch.zeros(4))

    def test_tensor_source_with_source_positions_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_positions"):
            interchange(Site("block_output", 1), torch.zeros(4), source_positions=[0])

    def test_replace_g_preserves_rank_and_applies_scale(self) -> None:
        """#449 finding 2: ``replace``'s feature transform expands the value
        to the base features' shape (a bare broadcast vector would collapse
        the rank and break the error-term rebuild of lossy split featurizers)
        and multiplies by ``scale`` (the ``EditSpec`` replace contract)."""
        value = torch.linspace(1.0, 4.0, 4)
        edit = replace(Site("block_output", 0), value, scale=2.0)
        f = torch.zeros(2, 3, 4)
        out = edit.g(f, value)
        assert out.shape == f.shape  # rank preserved, not collapsed to (4,)
        torch.testing.assert_close(out, (2.0 * value).expand_as(f))

    def test_mask_gate_count_mismatch_rejected(self) -> None:
        fsite = FeaturizedSite(Site("block_output", 1), _subspace(16, 3))
        with pytest.raises(ValueError, match="gate"):
            mask(fsite, Site("block_output", 0), MaskGate(5))
        mask(fsite, Site("block_output", 0), MaskGate(3))
        mask(fsite, Site("block_output", 0), MaskGate(tie=True))  # scalar broadcasts


class TestSeededNoiseUnit:
    pytestmark = pytest.mark.unit

    def test_same_seed_reproduces_stream(self) -> None:
        f = torch.zeros(2, 3)
        a, b = SeededNoise(7), SeededNoise(7)
        torch.testing.assert_close(a(f, 1.0), b(f, 1.0), atol=0.0, rtol=0.0)
        assert not torch.equal(a(f, 1.0), SeededNoise(8)(f, 1.0))

    def test_stream_advances_across_calls(self) -> None:
        f = torch.zeros(2, 3)
        state = SeededNoise(7)
        assert not torch.equal(state(f, 1.0), state(f, 1.0))

    def test_reset_restarts_the_stream(self) -> None:
        f = torch.zeros(2, 3)
        state = SeededNoise(7)
        first = state(f, 1.0)
        state(f, 1.0)
        state.reset()
        torch.testing.assert_close(state(f, 1.0), first, atol=0.0, rtol=0.0)

    def test_scale_and_dtype(self) -> None:
        f = torch.ones(4, dtype=torch.float64)
        out = SeededNoise(0)(f, 0.0)
        torch.testing.assert_close(out, f, atol=0.0, rtol=0.0)  # scale 0 = identity
        assert out.dtype == torch.float64


class TestMaskGateUnit:
    pytestmark = pytest.mark.unit

    def test_training_soft_gate_needs_temperature(self) -> None:
        gate = MaskGate(3).train()
        with pytest.raises(ValueError, match="temperature"):
            gate(torch.zeros(1, 3), torch.ones(1, 3))

    def test_training_blends_by_soft_gate(self) -> None:
        gate = MaskGate(3).train()
        gate.set_temperature(0.5)
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([-8.0, 8.0, 0.0]))
        f_base, f_src = torch.zeros(1, 3), torch.ones(1, 3)
        expected = torch.sigmoid(gate.mask / 0.5) * f_src  # (1-g)*0 + g*1
        torch.testing.assert_close(gate(f_base, f_src), expected)

    def test_eval_gate_is_hard_and_needs_no_temperature(self) -> None:
        gate = MaskGate(3).eval()
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([-2.0, 2.0, 0.0]))  # sigmoid(0)=0.5 → off
        out = gate(torch.zeros(1, 3), torch.ones(1, 3))
        torch.testing.assert_close(out, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_sparsity_loss_is_soft_gate_l1(self) -> None:
        gate = MaskGate(2)
        gate.set_temperature(1.0)
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(gate.sparsity_loss(), torch.tensor(1.0))  # 2×0.5
        gate.temperature = None
        with pytest.raises(ValueError, match="temperature"):
            gate.sparsity_loss()

    def test_per_feature_needs_n_features(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            MaskGate()
        assert MaskGate(tie=True).mask.numel() == 1


# --------------------------------------------------------------------------- #
#  property — every mode matches the raw-hook oracle                           #
# --------------------------------------------------------------------------- #
class TestModesMatchOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: SiteCase, gpt2_case: SiteCase) -> SiteCase:
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    # ------------------------------ collect ---------------------------------- #
    def test_collect_matches_featurized_site_collect(self, case: SiteCase) -> None:
        inputs = case.inputs()
        fsite = FeaturizedSite(Site("block_output", 0), _subspace(case.hidden(), 3))
        got = collect(fsite).collect(case.st, inputs)
        torch.testing.assert_close(
            got, fsite.collect(case.st, inputs), atol=0.0, rtol=0.0
        )

    # ------------------------------ replace ---------------------------------- #
    def test_replace_matches_oracle(self, case: SiteCase) -> None:
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        value = torch.linspace(-30.0, 30.0, 3)

        clean = case.clean_logits(inputs)
        edit = replace(FeaturizedSite(site, feat), value, positions=[last])
        edited = case.edited_logits(edit, inputs)

        manual = case.oracle_edit_logits(site, feat, lambda f: value, [last], inputs)
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_replace_scale_matches_oracle(self, case: SiteCase) -> None:
        """``scale`` writes ``scale·value`` (#449 finding 2 — the legacy
        replace contract upstreamed into the constructor)."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        value = torch.linspace(-30.0, 30.0, 3)
        scale = 2.5

        edit = replace(FeaturizedSite(site, feat), value, scale=scale, positions=[last])
        edited = case.edited_logits(edit, inputs)

        manual = case.oracle_edit_logits(
            site, feat, lambda f: scale * value, [last], inputs
        )
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ steer ------------------------------------ #
    def test_steer_matches_oracle(self, case: SiteCase) -> None:
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        vector = torch.linspace(3.0, 9.0, 3)
        factor = 2.5

        clean = case.clean_logits(inputs)
        edit = steer(
            FeaturizedSite(site, feat), vector, factor=factor, positions=[last]
        )
        edited = case.edited_logits(edit, inputs)

        manual = case.oracle_edit_logits(
            site, feat, lambda f: f + factor * vector, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ interchange ------------------------------ #
    def test_interchange_site_source_matches_transplant(self, case: SiteCase) -> None:
        """The full-space swap — ED1's cross-site transplant, via the
        constructor (single trace, single input; cross-input is PL1)."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src = FeaturizedSite(Site("block_output", 0), feat)
        dst_site = Site("block_output", 1)

        clean = case.clean_logits(inputs)
        edit = interchange(
            FeaturizedSite(dst_site, feat),
            src,
            source_positions=[last],
            positions=[last],
        )
        edited = case.edited_logits(edit, inputs)

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        manual = case.oracle_edit_logits(
            dst_site, feat, lambda f: f_src, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_interchange_feature_ids_swaps_only_selected_columns(
        self, case: SiteCase
    ) -> None:
        """The subspace swap (pyvene ``subspaces`` / ``_do_intervention_by_swap``
        semantics): only the ``feature_ids`` columns come from source — the
        untouched columns and the reconstruction error stay base's."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 4)
        ids = (0, 2)
        src = FeaturizedSite(Site("block_output", 0), feat, feature_ids=ids)
        dst_site = Site("block_output", 1)

        clean = case.clean_logits(inputs)
        edit = interchange(
            FeaturizedSite(dst_site, feat, feature_ids=ids),
            src,
            source_positions=[last],
            positions=[last],
        )
        edited = case.edited_logits(edit, inputs)

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        sel = f_src[..., list(ids)]

        def swap_selected(f: torch.Tensor) -> torch.Tensor:
            out = f.clone()
            out[..., list(ids)] = sel
            return out

        manual = case.oracle_edit_logits(dst_site, feat, swap_selected, [last], inputs)
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ interpolate ------------------------------ #
    def test_interpolate_matches_oracle(self, case: SiteCase) -> None:
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src = FeaturizedSite(Site("block_output", 0), feat)
        dst_site = Site("block_output", 1)

        def linear(f_base: torch.Tensor, f_src: torch.Tensor, alpha: float):
            return (1 - alpha) * f_base + alpha * f_src

        clean = case.clean_logits(inputs)
        edit = interpolate(
            FeaturizedSite(dst_site, feat),
            src,
            linear,
            source_positions=[last],
            positions=[last],
            alpha=0.3,
        )
        edited = case.edited_logits(edit, inputs)

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        manual = case.oracle_edit_logits(
            dst_site, feat, lambda f: 0.7 * f + 0.3 * f_src, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ noise ------------------------------------ #
    def test_noise_draw_matches_oracle_from_same_seed(self, case: SiteCase) -> None:
        """The in-trace draw is reproduced offline from the same seed — the
        determinism contract golden tests rely on (design doc: carry seeds
        through ``torch.Generator`` inside the trace)."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        scale, seed = 3.0, 7

        clean = case.clean_logits(inputs)
        edit = noise(FeaturizedSite(site, feat), scale, seed=seed, positions=[last])
        edited = case.edited_logits(edit, inputs)

        b = inputs["input_ids"].shape[0]
        gen = torch.Generator().manual_seed(seed)
        draw = torch.randn((b, 1, 3), generator=gen)
        manual = case.oracle_edit_logits(
            site, feat, lambda f: f + scale * draw, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_noise_stream_advances_and_resets_across_traces(
        self, case: SiteCase
    ) -> None:
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        fsite = FeaturizedSite(Site("block_output", 0), _subspace(case.hidden(), 3))
        state = SeededNoise(7)
        edit = noise(fsite, 3.0, seed=state, positions=[last])

        first = case.edited_logits(edit, inputs)
        second = case.edited_logits(edit, inputs)  # stream advanced → new draw
        assert not torch.allclose(first, second, atol=1e-4)
        state.reset()
        torch.testing.assert_close(
            case.edited_logits(edit, inputs), first, atol=0.0, rtol=0.0
        )

    # ------------------------------ mask -------------------------------------- #
    def test_mask_soft_gate_matches_oracle(self, case: SiteCase) -> None:
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src = FeaturizedSite(Site("block_output", 0), feat)
        dst_site = Site("block_output", 1)

        gate = MaskGate(3).train()
        gate.set_temperature(0.7)
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([-4.0, 2.0, 0.5]))

        clean = case.clean_logits(inputs)
        edit = mask(
            FeaturizedSite(dst_site, feat),
            src,
            gate,
            source_positions=[last],
            positions=[last],
        )
        edited = case.edited_logits(edit, inputs)

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        g = torch.sigmoid(gate.mask.detach() / 0.7)
        manual = case.oracle_edit_logits(
            dst_site, feat, lambda f: (1 - g) * f + g * f_src, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_mask_eval_gate_is_a_hard_selected_swap(self, case: SiteCase) -> None:
        """Eval mode: features whose gate crosses 0.5 come wholly from source,
        the rest wholly from base — the hard-threshold DBM readout."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src = FeaturizedSite(Site("block_output", 0), feat)
        dst_site = Site("block_output", 1)

        gate = MaskGate(3).eval()
        with torch.no_grad():
            gate.mask.copy_(torch.tensor([-2.0, 2.0, -1.0]))  # only feature 1 on

        clean = case.clean_logits(inputs)
        edit = mask(
            FeaturizedSite(dst_site, feat),
            src,
            gate,
            source_positions=[last],
            positions=[last],
        )
        edited = case.edited_logits(edit, inputs)

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])

        def hard_swap(f: torch.Tensor) -> torch.Tensor:
            out = f.clone()
            out[..., [1]] = f_src[..., [1]]
            return out

        manual = case.oracle_edit_logits(dst_site, feat, hard_swap, [last], inputs)
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  golden — the device-sensitive modes hold on the real Qwen3-4B backbone      #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """Steer against the offline-featurized raw-hook oracle, plus seeded-noise
    reproducibility (a CUDA generator), on the coherent GPU backbone — both on
    one model load, since the load dominates the nightly cost."""

    pytestmark = pytest.mark.golden

    def test_steer_matches_oracle_and_noise_is_seeded_on_coherent_model(self) -> None:
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        case = SiteCase(st=st, oracle=SimpleNamespace(hf_model=raw), tok=st.tokenizer)
        enc = st.tokenizer(_TEXT, return_tensors="pt")
        device = next(raw.parameters()).device
        inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        last = int(inputs["input_ids"].shape[1] - 1)
        layer = int(st.num_layers) // 2
        feat = _subspace(int(raw.config.hidden_size), 8)
        site = Site("block_output", layer)
        vector = torch.linspace(3.0, 10.0, 8)

        clean = case.clean_logits(inputs)

        # steer — oracle-pinned
        edit = steer(FeaturizedSite(site, feat), vector, factor=2.0, positions=[last])
        edited = case.edited_logits(edit, inputs)
        manual = case.oracle_edit_logits(
            site,
            feat,
            # device= too: the oracle hook runs on the CUDA model, and this
            # hand-rolled g gets the on-device features (pre-existing test bug
            # surfaced by the first full golden-tier run on the branch).
            lambda f: f + 2.0 * vector.to(device=f.device, dtype=f.dtype),
            [last],
            inputs,
        )
        assert not torch.allclose(manual, clean.float(), atol=1e-2)
        torch.testing.assert_close(
            edited.float(), manual.cpu().float(), atol=1e-3, rtol=1e-3
        )

        # noise — fresh same-seed streams reproduce on a CUDA generator
        fsite = FeaturizedSite(site, feat)
        first = case.edited_logits(noise(fsite, 3.0, seed=7, positions=[last]), inputs)
        again = case.edited_logits(noise(fsite, 3.0, seed=7, positions=[last]), inputs)
        assert not torch.allclose(first, clean, atol=1e-2)
        torch.testing.assert_close(again, first, atol=0.0, rtol=0.0)
