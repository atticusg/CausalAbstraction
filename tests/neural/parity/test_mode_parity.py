"""SH1 (#410): the per-mode numerical-parity sweep vs the raw-hook oracle.

Tiers (docs/TESTS.md):

* ``unit`` — every :func:`~tests.neural.parity.cases.enumerate_cases` cell:
  the 7 ED2 modes through the new Site/Edit/Plan stack (``Edit.apply`` /
  ``Edit.collect``, ``run_plan`` single-trace and forced-staged, per-head
  sites) against the backbone-agnostic hook oracle
  (``tests/neural/activations/hook_oracle.py``), on eager-forced tiny-random
  llama / gpt2 / gqa (+ decoupled ``head_dim`` for heads). Each intervention
  also proves non-vacuity (the oracle's edited logits differ from clean).
* ``property`` — determinism/seeding: rebuilding a family from its seeded
  recipe reproduces every mode's output byte-identically; ``SeededNoise``
  fresh-same-seed identity, seed sensitivity, and ``reset()`` replay.

The captured-golden replay (the cross-run anchor) is
``test_captured_goldens.py``; the pins are written by ``update_goldens.py``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import SeededNoise, noise
from causalab.neural.site import Site

from tests.neural.parity.cases import (
    MODES,
    ModeCase,
    ParityCase,
    build_family,
    enumerate_cases,
    realize_new_stack,
    realize_oracle,
)

_ATOL, _RTOL = 1e-5, 1e-4  # the established CPU stack-vs-oracle tolerance


# --------------------------------------------------------------------------- #
#  unit — the sweep                                                            #
# --------------------------------------------------------------------------- #
class TestModeParity:
    pytestmark = pytest.mark.unit

    def test_families_are_eager(self, parity_families: dict[str, ParityCase]) -> None:
        """The harness-wide policy: parity + pins run in eager attention
        (the SDPA flip is SH3 #424, post-cutover)."""
        for name, pc in parity_families.items():
            got = pc.oracle.hf_model.config._attn_implementation
            assert got == "eager", f"{name}: {got!r}"

    @pytest.mark.parametrize("mc", enumerate_cases(), ids=lambda mc: mc.case_id)
    def test_new_stack_matches_hook_oracle(
        self, mc: ModeCase, parity_families: dict[str, ParityCase]
    ) -> None:
        pc = parity_families[mc.family]
        want = realize_oracle(mc, pc)
        got = realize_new_stack(mc, pc)
        assert got.kind == want.kind
        if want.clean is not None:  # every write mode must actually move logits
            assert not torch.allclose(want.value, want.clean, atol=1e-4), (
                f"{mc.case_id}: inert intervention — oracle edit equals clean"
            )
        torch.testing.assert_close(got.value, want.value, atol=_ATOL, rtol=_RTOL)


# --------------------------------------------------------------------------- #
#  property — determinism / seeding                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def llama_rebuild_pair() -> tuple[ParityCase, ParityCase]:
    """Two independent builds of the same seeded recipe — the rebuild IS the
    property under test, so these deliberately bypass the session fixture."""
    return build_family("llama"), build_family("llama")


class TestDeterminism:
    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("mode", MODES)
    def test_rebuild_from_recipe_is_byte_identical(
        self, mode: str, llama_rebuild_pair: tuple[ParityCase, ParityCase]
    ) -> None:
        """Same recipe (seeded weights, eager, seeded featurizer/gate/noise) →
        the canonical cell reproduces with zero tolerance on a fresh instance."""
        a, b = llama_rebuild_pair
        mc = ModeCase(family="llama", mode=mode)
        torch.testing.assert_close(
            realize_new_stack(mc, a).value,
            realize_new_stack(mc, b).value,
            atol=0.0,
            rtol=0.0,
        )

    def test_noise_fresh_same_seed_reproduces_and_other_seed_differs(
        self, llama_rebuild_pair: tuple[ParityCase, ParityCase]
    ) -> None:
        a, _ = llama_rebuild_pair
        mc = ModeCase(family="llama", mode="noise")
        first = realize_new_stack(mc, a).value  # fresh SeededNoise(7) inside
        again = realize_new_stack(mc, a).value
        torch.testing.assert_close(again, first, atol=0.0, rtol=0.0)

        inputs = a.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        fsite = FeaturizedSite(Site("block_output", 1))
        other = a.edited_logits(noise(fsite, 3.0, seed=8, positions=[last]), inputs)
        base = a.edited_logits(noise(fsite, 3.0, seed=7, positions=[last]), inputs)
        assert not torch.allclose(other, base, atol=1e-4)

    def test_noise_reset_replays_a_caller_held_stream(
        self, llama_rebuild_pair: tuple[ParityCase, ParityCase]
    ) -> None:
        a, _ = llama_rebuild_pair
        inputs = a.inputs()
        last = int(inputs["input_ids"].shape[1] - 1)
        state = SeededNoise(7)
        edit = noise(
            FeaturizedSite(Site("block_output", 1)), 3.0, seed=state, positions=[last]
        )
        first = a.edited_logits(edit, inputs)
        advanced = a.edited_logits(edit, inputs)  # stream advanced → new draw
        assert not torch.allclose(advanced, first, atol=1e-4)
        state.reset()
        torch.testing.assert_close(
            a.edited_logits(edit, inputs), first, atol=0.0, rtol=0.0
        )
