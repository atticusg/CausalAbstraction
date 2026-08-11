"""Recompute the parity harness's captured-golden pins (#410 / SH1).

Captures the ``golden``-flagged registry subset (see
``tests/neural/parity/cases.py``) **from the hook oracle** — never from
pyvene, never from the stack under test — on the eager-forced tiny-random
families, and pins the values to ``tests/neural/parity/goldens/<family>.json``.
Before writing, every case is parity-checked (new stack vs oracle): a pin the
stack already fails can never be committed.

Usage::

    # Show the diff for every family; do not write.
    uv run python tests/neural/parity/update_goldens.py

    # One family only.
    uv run python tests/neural/parity/update_goldens.py --family llama

    # Accept the diff (writes the JSONs).
    uv run python tests/neural/parity/update_goldens.py \\
        --i-have-reviewed-the-diff

Legitimate repins are environment moves (a torch/transformers bump shifting
seeded-init weights or kernel numerics) and deliberate registry changes; the
``context`` block records the capturing environment so a repin's cause is
auditable. Mirrors ``tests/end_to_end/update_goldens.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo root is three levels up: tests/neural/parity/update_goldens.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402

from tests.neural.parity.cases import (  # noqa: E402
    build_family,
    realize_new_stack,
    realize_oracle,
)
from tests.neural.parity.pins import (  # noqa: E402
    DEFAULT_TOLERANCE,
    GOLDEN_FAMILIES,
    ParityGolden,
    golden_cases,
    golden_path,
    pin_values,
)

_PARITY_ATOL, _PARITY_RTOL = 1e-5, 1e-4  # the sweep's stack-vs-oracle gate


def _capture_family(family: str) -> dict[str, Any]:
    """Oracle-side pins for one family, refusing any case where the new stack
    diverges from the oracle right now."""
    pc = build_family(family)
    values: dict[str, Any] = {}
    for mc in golden_cases(family):
        oracle = realize_oracle(mc, pc)
        new = realize_new_stack(mc, pc)
        torch.testing.assert_close(
            new.value,
            oracle.value,
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
            msg=lambda m: (
                f"{mc.case_id}: new stack diverges from the hook oracle — refusing "
                f"to pin a golden the stack already fails.\n{m}"
            ),
        )
        values.update(pin_values(mc.case_id, oracle))
    return values


def _format_value(v: Any) -> str:
    return f"{v!r}"


def _print_diff(family: str, old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Print a per-key diff between ``old`` and ``new``. Return True if changed."""
    keys = sorted(set(old) | set(new))
    rows = [
        (
            key,
            _format_value(old.get(key, "<absent>")),
            _format_value(new.get(key, "<absent>")),
        )
        for key in keys
        if old.get(key, "<absent>") != new.get(key, "<absent>")
    ]
    if not rows:
        print(f"[{family}] no diff (every value matches the existing golden).")
        return False
    print(f"[{family}] diff:")
    width = max(len(r[0]) for r in rows)
    for key, a, b in rows:
        print(f"  {key:<{width}}  -{a}")
        print(f"  {' ':<{width}}  +{b}")
    return True


def _write_golden(
    family: str, values: dict[str, Any], tolerance: dict[str, float]
) -> None:
    payload = {
        "family": family,
        "attn_implementation": "eager",
        "captured_from": "hook_oracle",
        "deterministic": True,
        "context": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "tolerance": dict(tolerance),
        "values": values,
    }
    path = golden_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[{family}] wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=GOLDEN_FAMILIES, help="one family only")
    parser.add_argument(
        "--i-have-reviewed-the-diff",
        action="store_true",
        dest="confirm",
        help="actually write the golden JSONs (default: diff only)",
    )
    args = parser.parse_args(argv)

    families = [args.family] if args.family else list(GOLDEN_FAMILIES)
    any_changed = False
    for family in families:
        path = golden_path(family)
        if path.is_file():
            old = ParityGolden.from_path(path)
            old_values, tolerance = old.values, old.tolerance
        else:
            old_values, tolerance = {}, dict(DEFAULT_TOLERANCE)
        new_values = _capture_family(family)
        changed = _print_diff(family, old_values, new_values)
        any_changed = any_changed or changed
        if changed and args.confirm:
            _write_golden(family, new_values, tolerance)

    if any_changed and not args.confirm:
        print(
            "\nNot written. Review the diff above, then re-run with "
            "--i-have-reviewed-the-diff to accept it."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
