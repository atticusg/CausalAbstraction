"""Captured-golden pins for the parity harness — extraction and loading.

A *parity golden* pins the numerical output of one canonical
:class:`~tests.neural.parity.cases.ModeCase` (the ``golden``-flagged registry
subset) to disk, captured **from the hook oracle** on the eager-forced
tiny-random families. These are `numerical_unit`-tier pins in docs/TESTS.md's
term split — *not* the GPU ``golden`` runner tier — and they are the
pyvene-independent anchor of the nnterp migration: live oracle equivalence
(``test_mode_parity.py``) can't see drift that moves both sides at once (a
torch/transformers bump changing model numerics), and after the SH2 cutover
deletes pyvene these files remain the frozen pre-migration reference.

One JSON per family at ``tests/neural/parity/goldens/<family>.json``::

    {
      "family": "llama",
      "attn_implementation": "eager",
      "captured_from": "hook_oracle",
      "deterministic": true,
      "context": {"torch": "...", "transformers": "..."},   # never asserted
      "tolerance": {"default": 1e-4},
      "values": {"<case_id>.out.mean": ..., "<case_id>.probe.0": ..., ...}
    }

Regenerate with ``uv run python tests/neural/parity/update_goldens.py``
(review-gated, mirrors ``tests/end_to_end/update_goldens.py``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from tests.end_to_end._helpers.golden import _add_tensor_reductions
from tests.neural.parity.cases import ModeCase, Realization, enumerate_cases

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

#: Families that carry a captured-golden file (the registry's golden subset).
GOLDEN_FAMILIES = ("llama", "gpt2", "gqa")

#: Cross-CPU BLAS variance headroom; per-key overrides via the tolerance map.
DEFAULT_TOLERANCE = {"default": 1e-4}

_N_PROBES = 8


def golden_cases(family: str) -> list[ModeCase]:
    """The pinned subset for one family, in registry order."""
    return [c for c in enumerate_cases() if c.golden and c.family == family]


def golden_path(family: str) -> Path:
    return GOLDENS_DIR / f"{family}.json"


def pin_values(case_id: str, real: Realization) -> dict[str, Any]:
    """Flatten one realization into pinnable values: shape + mean/std/first/last
    reductions, ``_N_PROBES`` seeded probe elements (a mean over a 32k-vocab
    logit row is insensitive to localized regressions), and — for write modes —
    the non-vacuity pin ``clean_delta.max`` (the intervention must move logits
    by the same magnitude at capture and at replay)."""
    values: dict[str, Any] = {}
    t = real.value.detach().float()
    _add_tensor_reductions(values, f"{case_id}.out", t)
    flat = t.flatten()
    gen = torch.Generator().manual_seed(0)
    for j, i in enumerate(
        torch.randperm(flat.numel(), generator=gen)[:_N_PROBES].tolist()
    ):
        values[f"{case_id}.probe.{j}"] = float(flat[i])
    if real.clean is not None:
        delta = (real.value.detach() - real.clean.detach()).abs().max()
        values[f"{case_id}.clean_delta.max"] = float(delta)
    return values


@dataclasses.dataclass
class ParityGolden:
    """One family's captured pins: tolerances + expected values."""

    family: str
    attn_implementation: str
    tolerance: dict[str, float]
    values: dict[str, Any]
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> "ParityGolden":
        with path.open() as f:
            data = json.load(f)
        return cls(
            family=data["family"],
            attn_implementation=data.get("attn_implementation", "eager"),
            tolerance=dict(data.get("tolerance", DEFAULT_TOLERANCE)),
            values=dict(data["values"]),
            path=path,
        )

    def tol_for(self, key: str) -> float:
        return float(self.tolerance.get(key, self.tolerance.get("default", 1e-4)))
