"""Replay the captured parity goldens through the NEW stack (#410 / SH1).

``numerical_unit`` tier — these are captured numerical pins on fixed seeds
(docs/TESTS.md's term split), not the GPU ``golden`` runner tier. The pins in
``goldens/<family>.json`` were captured from the **hook oracle** by
``update_goldens.py``; this test rebuilds the models and inputs on the fly
from the same seeded recipes and asserts the **new stack** reproduces every
pinned value within tolerance. That closes the two holes live A/B parity
leaves open: both-sides drift (a torch/transformers bump shifting numerics
silently) and the SH2 cutover (after pyvene deletion these pins remain the
frozen pre-migration reference).
"""

from __future__ import annotations

import pytest
import torch

from tests.neural.parity.cases import ParityCase, realize_new_stack
from tests.neural.parity.pins import (
    GOLDEN_FAMILIES,
    ParityGolden,
    golden_cases,
    golden_path,
    pin_values,
)

pytestmark = pytest.mark.numerical_unit


@pytest.mark.parametrize("family", GOLDEN_FAMILIES)
def test_new_stack_replays_captured_golden(
    family: str, parity_families: dict[str, ParityCase]
) -> None:
    path = golden_path(family)
    if not path.is_file():
        pytest.fail(
            f"no captured golden for {family!r} at {path} — bootstrap it with:\n"
            f"  uv run python tests/neural/parity/update_goldens.py "
            f"--family {family} --i-have-reviewed-the-diff"
        )
    golden = ParityGolden.from_path(path)
    assert golden.attn_implementation == "eager"

    pc = parity_families[family]
    got: dict[str, object] = {}
    for mc in golden_cases(family):
        got.update(pin_values(mc.case_id, realize_new_stack(mc, pc)))

    # Registry drift (a case added/renamed/removed without a repin) is its own
    # failure mode — report it by name, not as a KeyError mid-comparison.
    assert set(got) == set(golden.values), (
        "pinned keys diverge from the registry — re-run update_goldens.py "
        f"(missing from golden: {sorted(set(got) - set(golden.values))[:5]}, "
        f"stale in golden: {sorted(set(golden.values) - set(got))[:5]})"
    )

    mismatches = []
    for key, want in sorted(golden.values.items()):
        have = got[key]
        if isinstance(want, list):  # shapes — exact
            if list(have) != want:
                mismatches.append(f"{key}: shape {have} != pinned {want}")
        else:
            tol = golden.tol_for(key)
            if not abs(float(have) - float(want)) <= tol:
                mismatches.append(f"{key}: {have!r} != pinned {want!r} (tol {tol})")
    assert not mismatches, (
        f"{len(mismatches)} pinned value(s) drifted:\n  " + "\n  ".join(mismatches)
    )


def test_replay_is_deterministic_within_a_session(
    parity_families: dict[str, ParityCase],
) -> None:
    """Two back-to-back realizations of a pinned case are byte-identical on
    CPU — the precondition for the tolerance gate above to be meaningful."""
    pc = parity_families["llama"]
    (mc, *_) = golden_cases("llama")
    a = realize_new_stack(mc, pc).value
    b = realize_new_stack(mc, pc).value
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
