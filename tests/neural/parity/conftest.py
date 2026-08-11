"""Session-scoped model families for the parity harness.

One build per family per pytest session, shared by the sweep and the
captured-golden replay (both only *read* through traces/hooks, which detach
cleanly). Determinism tests build their own fresh instances — rebuilding from
the recipe is the property under test there.
"""

from __future__ import annotations

import pytest

from tests.neural.parity.cases import ParityCase, build_family


@pytest.fixture(scope="session")
def parity_families() -> dict[str, ParityCase]:
    """Every family the registry addresses, built lazily-once."""
    return {name: build_family(name) for name in ("llama", "gpt2", "gqa", "decoupled")}
