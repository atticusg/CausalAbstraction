"""Tests for :mod:`causalab.neural.featurized_site` — the ST3 wrap (#398).

Tiers (mirroring ``tests/neural/test_site.py``; ``causalab/neural`` owes
``unit`` + ``property``, and a ``golden`` GPU pin is the established pattern for
the real coherent backbone):

* ``unit`` — the contract (identity default, ``feature_ids`` validation and
  normalization), no forward pass.
* ``property`` — on tiny Llama **and** GPT-2 (CPU), against the same
  raw-``register_forward_hook`` oracle ST1 is pinned to, with the featurizer
  applied *offline* on the oracle side (captured tensor → ``featurize`` /
  hook edit → ``featurize``→modify→``inverse_featurize``): the identity wrap is
  exactly a plain :class:`Site`; a ``SubspaceFeaturizer`` read matches offline
  featurization; an identity-``g`` edit is a logits no-op (the error term makes
  ``inverse(featurize(x)) = x``); feature-space edits/writes reproduce the
  oracle on **both** branches (positional slice and whole tensor);
  ``feature_ids`` gathers on read and scatters on write (untouched columns from
  base); a two-stage :class:`ComposedFeaturizer` threads per-stage errors; and
  the interchange pattern — write one site's :meth:`read` into a later site —
  works inside a single trace.
* ``golden`` — featurized read + edit on the real Qwen3-4B backbone (GPU)
  against the same oracle, one model load.

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

from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.featurizer import ComposedFeaturizer, Featurizer
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
    st: StandardizedTransformer  # FeaturizedSite reads/writes tap this
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

    def test_defaults_to_identity_featurizer(self) -> None:
        fsite = FeaturizedSite(Site("block_output", 0))
        assert fsite.featurizer.is_trivial()
        assert fsite.feature_ids is None

    def test_feature_ids_normalized_to_tuple(self) -> None:
        fsite = FeaturizedSite(Site("block_output", 0), feature_ids=[3, 1])  # type: ignore[arg-type]
        assert fsite.feature_ids == (3, 1)

    def test_rejects_empty_feature_ids(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FeaturizedSite(Site("block_output", 0), feature_ids=())

    def test_rejects_negative_feature_ids(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FeaturizedSite(Site("block_output", 0), feature_ids=(0, -1))

    def test_rejects_duplicate_feature_ids(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            FeaturizedSite(Site("block_output", 0), feature_ids=(2, 2))

    def test_rejects_feature_ids_beyond_n_features(self) -> None:
        feat = Featurizer(n_features=4, id="gated")
        with pytest.raises(ValueError, match="out of range"):
            FeaturizedSite(Site("block_output", 0), feat, feature_ids=(4,))

    def test_unknown_n_features_skips_range_check(self) -> None:
        # An identity featurizer carries no n_features — the bound is unknowable
        # at construction, so large ids must not raise here.
        fsite = FeaturizedSite(Site("block_output", 0), feature_ids=(1000,))
        assert fsite.feature_ids == (1000,)


# --------------------------------------------------------------------------- #
#  property — the wrap matches the raw-hook oracle + offline featurization     #
# --------------------------------------------------------------------------- #
class TestWrapMatchesOracle:
    pytestmark = pytest.mark.property

    @pytest.fixture(params=["llama", "gpt2"])
    def case(self, request, llama_case: SiteCase, gpt2_case: SiteCase) -> SiteCase:
        """A tiny-random case across the two families (post- vs pre-LayerNorm,
        different ``mlp_activation`` taps) — the architecture axis that matters
        for a wrap over every Site component."""
        return {"llama": llama_case, "gpt2": gpt2_case}[request.param]

    # ------------------------------ reads ----------------------------------- #
    def test_identity_collect_is_plain_site_collect(self, case: SiteCase) -> None:
        """The default (identity) wrap is exact: same tensors as Site.collect."""
        inputs = case.inputs()
        site = Site("block_output", 0)
        got = FeaturizedSite(site).collect(case.st, inputs)
        expected = site.collect(case.st, inputs)
        assert got.device.type == "cpu"  # collect offloads (package convention)
        torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("positions", [None, [1]], ids=["all", "pos1"])
    @pytest.mark.parametrize("component", ["block_output", "mlp_activation"])
    def test_subspace_read_matches_offline_featurize(
        self, case: SiteCase, component: str, positions: list[int] | None
    ) -> None:
        """An in-trace featurized read equals featurizing the raw-hook capture
        offline — on the residual stream and on the intermediate-width
        ``mlp_activation`` (the one component whose feature width differs)."""
        inputs = case.inputs()
        site = Site(component, 0)  # type: ignore[arg-type]
        width = int(case.capture(component, 0, inputs).shape[-1])
        feat = _subspace(width, 3)
        got = FeaturizedSite(site, feat).collect(case.st, inputs, positions=positions)
        raw = case.capture(component, 0, inputs)
        if positions is not None:
            raw = raw[:, positions, :]
        expected, _ = feat.featurize(raw)
        assert got.shape == expected.shape
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    def test_feature_ids_read_gathers_columns(self, case: SiteCase) -> None:
        """``feature_ids`` selects feature columns, in the requested order."""
        inputs = case.inputs()
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 4)
        got = FeaturizedSite(site, feat, feature_ids=(2, 0)).collect(case.st, inputs)
        expected, _ = feat.featurize(case.capture("block_output", 0, inputs))
        torch.testing.assert_close(got, expected[..., [2, 0]], atol=1e-5, rtol=1e-4)

    # ------------------------------ writes ---------------------------------- #
    def test_identity_g_edit_is_noop(self, case: SiteCase) -> None:
        """The error term reconstructs exactly: ``inverse(featurize(x)) = x``,
        so an identity-``g`` edit through a lossy subspace leaves the logits
        unchanged (the roundtrip contract every write path rides on)."""
        inputs = case.inputs()
        fsite = FeaturizedSite(Site("block_output", 0), _subspace(case.hidden(), 3))
        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            fsite.edit(case.st, lambda f: f)
            edited = case.st.logits[:, -1, :].cpu().save()
        torch.testing.assert_close(edited, clean, atol=1e-5, rtol=1e-4)

    def _oracle_feature_edit_logits(
        self,
        case: SiteCase,
        site: Site,
        feat: Featurizer,
        g,
        positions: list[int] | None,
        inputs: Any,
    ) -> torch.Tensor:
        """Ground truth: a hand-rolled forward hook applying featurize → ``g`` →
        inverse (with the base error) on the raw activation, offline."""
        module, kind = component_module(case.oracle, site.layer, site.component)

        def edit(h: torch.Tensor) -> None:
            sel = slice(None) if positions is None else positions
            f, err = feat.featurize(h[:, sel])
            h[:, sel] = feat.inverse_featurize(g(f), err).to(h.dtype)

        return component_edited_logits(case.oracle, inputs, module, kind, edit)

    @pytest.mark.parametrize("whole_tensor", [False, True], ids=["slice", "whole"])
    def test_subspace_edit_matches_oracle(
        self, case: SiteCase, whole_tensor: bool
    ) -> None:
        """A feature-space steer (``g = f + delta``) reproduces the oracle hook
        on both branches (positional slice / whole tensor)."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        positions = None if whole_tensor else [last]
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)
        delta = torch.linspace(-40.0, 40.0, 3)

        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            FeaturizedSite(site, feat).edit(case.st, lambda f: f + delta, positions)
            edited = case.st.logits[:, -1, :].cpu().save()

        manual = self._oracle_feature_edit_logits(
            case, site, feat, lambda f: f + delta, positions, inputs
        )
        # The edit is non-vacuous (the oracle itself moves the logits) ...
        assert not torch.allclose(manual, clean, atol=1e-4)
        # ... and the in-trace wrap reproduces the oracle exactly.
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
    def test_write_zero_ablates_features_only(
        self, case: SiteCase, value_dtype: torch.dtype
    ) -> None:
        """A zero-vector write removes the feature-space contribution while the
        error term (orthogonal component) survives — pyvene's
        ``FeatureReplaceIntervention`` ablation semantics. A wrong-dtype value
        is coerced to the featurized dtype rather than crashing."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 3)

        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            FeaturizedSite(site, feat).write(
                case.st, torch.zeros(3, dtype=value_dtype), positions=[last]
            )
            edited = case.st.logits[:, -1, :].cpu().save()

        manual = self._oracle_feature_edit_logits(
            case, site, feat, lambda f: torch.zeros_like(f), [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    def test_feature_ids_write_scatters_into_base(self, case: SiteCase) -> None:
        """Writing under ``feature_ids`` replaces only those columns; the other
        feature columns still come from base (scatter, not whole replacement)."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        site = Site("block_output", 0)
        feat = _subspace(case.hidden(), 4)
        ids = [3, 1]
        value = torch.tensor([55.0, -55.0])

        def scatter(f: torch.Tensor) -> torch.Tensor:
            out = f.clone()
            out[..., ids] = value
            return out

        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            FeaturizedSite(site, feat, feature_ids=tuple(ids)).write(
                case.st, value, positions=[last]
            )
            edited = case.st.logits[:, -1, :].cpu().save()

        manual = self._oracle_feature_edit_logits(
            case, site, feat, scatter, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)

    # ------------------------------ composition ----------------------------- #
    def test_composed_featurizer_reads_and_roundtrips(self, case: SiteCase) -> None:
        """A two-stage chain (``d → 4 → 2``) works through the same wrap: the
        read matches the offline chained featurization, and an identity-``g``
        edit is still a logits no-op — the per-stage error list threads through
        ``ComposedFeaturizer`` unchanged (▣ reused, not rewritten)."""
        inputs = case.inputs()
        stage1 = _subspace(case.hidden(), 4, seed=0)
        stage2 = _subspace(4, 2, seed=1)
        feat = stage1 >> stage2
        assert isinstance(feat, ComposedFeaturizer)
        fsite = FeaturizedSite(Site("block_output", 0), feat)

        got = fsite.collect(case.st, inputs)
        expected, errors = feat.featurize(case.capture("block_output", 0, inputs))
        assert len(errors) == 2  # one error slot per stage
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            fsite.edit(case.st, lambda f: f)
            edited = case.st.logits[:, -1, :].cpu().save()
        torch.testing.assert_close(edited, clean, atol=1e-5, rtol=1e-4)

    # ------------------------------ interchange pattern --------------------- #
    def test_cross_site_feature_transplant_in_one_trace(self, case: SiteCase) -> None:
        """The ED1 interchange shape — ``dst.write(model, src.read(model))`` —
        composes inside a single trace (src earlier in forward order than dst),
        matching an oracle that captures the source features offline and
        hook-writes them at the destination."""
        inputs = case.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        feat = _subspace(case.hidden(), 3)
        src = FeaturizedSite(Site("block_output", 0), feat)
        dst = FeaturizedSite(Site("block_output", 1), feat)

        clean = case.clean_logits(inputs)
        with case.st.trace(inputs):
            dst.write(case.st, src.read(case.st, [last]), positions=[last])
            edited = case.st.logits[:, -1, :].cpu().save()

        f_src, _ = feat.featurize(case.capture("block_output", 0, inputs)[:, [last]])
        manual = self._oracle_feature_edit_logits(
            case, Site("block_output", 1), feat, lambda f: f_src, [last], inputs
        )
        assert not torch.allclose(manual, clean, atol=1e-4)
        torch.testing.assert_close(edited, manual, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
#  golden — the wrap holds on the real Qwen3-4B backbone                       #
# --------------------------------------------------------------------------- #
class TestChatCoherentGolden:
    """Featurized read + feature-space write on the coherent GPU backbone
    (Qwen3-4B), matched against the same offline-featurized raw-hook oracle —
    one model load for both checks. The write value stays where a caller would
    make it (CPU, default dtype) so the wrap's device/dtype coercion is
    exercised against a bf16 CUDA site — the axis the CPU property tier
    structurally can't cover."""

    pytestmark = pytest.mark.golden

    def test_read_and_write_match_oracle_on_coherent_model(self) -> None:
        # dispatch=True materializes real weights so the raw-hook oracle can run.
        st = StandardizedTransformer(
            "Qwen/Qwen3-4B-Instruct-2507", dispatch=True, device_map="auto"
        )
        raw = st._model
        oracle = SimpleNamespace(hf_model=raw)
        enc = st.tokenizer(_TEXT, return_tensors="pt")
        device = next(raw.parameters()).device
        inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        layer = int(st.num_layers) // 2
        site = Site("block_output", layer)
        feat = _subspace(int(raw.config.hidden_size), 8)

        # Featurized read == offline featurization of the raw-hook capture.
        got = FeaturizedSite(site, feat).collect(st, inputs)
        assert got.device.type == "cpu"  # offloaded as collected
        module, kind = component_module(oracle, layer, "block_output")
        captured = capture_component(oracle, module, kind, inputs)
        expected, _ = feat.featurize(captured)
        torch.testing.assert_close(
            got.float(), expected.cpu().float(), atol=1e-3, rtol=1e-3
        )

        # Feature-space replace (a CPU fp32 vector into the bf16 CUDA site) ==
        # the hand-rolled featurize→replace→inverse hook.
        last = int(inputs["input_ids"].shape[1] - 1)
        value = torch.linspace(-40.0, 40.0, 8)
        with st.trace(inputs):
            FeaturizedSite(site, feat).write(st, value, positions=[last])
            edited = st.logits[:, -1, :].cpu().save()

        def edit(h: torch.Tensor) -> None:
            _, err = feat.featurize(h[:, [last]])
            h[:, [last]] = feat.inverse_featurize(value.to(err.device), err).to(h.dtype)

        manual = component_edited_logits(oracle, inputs, module, kind, edit)
        torch.testing.assert_close(
            edited.float(), manual.cpu().float(), atol=1e-3, rtol=1e-3
        )
