"""Regression guard for the IOI logit-diff readout token (#1).

The path-patching / IOI metrics score a *single* vocab row per answer — the
correct or distractor name's token id, resolved by
:func:`causalab.methods.metric.single_token_id`. At the readout position the IOI
prompt ends ``"... gave a drink to"``, so GPT-2 emits the **leading-space** token
``" Mary"`` (id 5335), a different BPE id than the bare ``"Mary"`` for ~half of
the name vocabulary. Reading the bare id silently scores the wrong row and
contaminates every direct-effect number.

This test pins ``single_token_id`` to the emitted (space-prefixed) id for every
IOI name. With the pre-fix ordering (bare form first) it fails for 53 of the 99
names — exactly the bug it guards.
"""

from __future__ import annotations

import pytest

from causalab.neural.pytorch_hooks.metrics import column_token_id
from causalab.tasks.IOI.causal_models import NAMES

pytestmark = pytest.mark.numerical_unit


def test_single_token_id_returns_emitted_leading_space_id(gpt2_pipeline) -> None:
    tokenizer = gpt2_pipeline.tokenizer
    mismatched: list[tuple[str, int, int]] = []
    for name in NAMES:
        emitted = tokenizer.encode(" " + name.strip(), add_special_tokens=False)
        assert len(emitted) == 1, f"{name!r} is not single-token with a leading space"
        resolved = column_token_id(gpt2_pipeline.tokenizer, name)
        if resolved != emitted[0]:
            mismatched.append((name, resolved, emitted[0]))

    assert not mismatched, (
        "single_token_id must return the emitted (leading-space) id for IOI "
        f"names; {len(mismatched)}/{len(NAMES)} read the wrong vocab row, e.g. "
        f"{mismatched[:5]}"
    )
