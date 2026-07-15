"""Recompute runner-scope golden values.

A *runner golden* pins the full end-to-end pipeline for a runner config
at a fixed seed, against a small model that reliably solves the task
(see docs/TESTS.md "Runner-golden standard"). One JSON per golden config
at ``tests/end_to_end/goldens/<runner>.json``. Composes runner configs
from ``tests/end_to_end/configs/golden/<name>.yaml``, runs them via
:func:`run_baseline_for_golden` (device follows the model fixture),
writes the captured values. New goldens bootstrapped on GPU are pinned
with ``"deterministic": false`` — BLAS / CUDA kernel drift breaks
byte-equality across runs, so per-key reductions inside ``tolerance``
carry the drift signal in that case.

Task-scope numerical pins (the LM-free symbolic causal-model outputs)
live one layer below and are refreshed by a separate script:
``scripts/update_task_pins.py``. They are *not* goldens — see
docs/TESTS.md for the term split.

Usage::

    # Show the diff for every existing runner golden; do not write.
    uv run python tests/end_to_end/update_goldens.py

    # Show the diff for one baseline; do not write.
    uv run python tests/end_to_end/update_goldens.py \\
        --baseline golden/age

    # Recompute all baselines and overwrite the golden JSONs.
    uv run python tests/end_to_end/update_goldens.py \\
        --i-have-reviewed-the-diff

    # Recompute one baseline and overwrite its JSON.
    uv run python tests/end_to_end/update_goldens.py \\
        --baseline golden/age --i-have-reviewed-the-diff

    # Initialise (or backfill) goldens for every golden runner config
    # that doesn't yet have a JSON.
    uv run python tests/end_to_end/update_goldens.py \\
        --init --i-have-reviewed-the-diff

The diff prints in a structured per-key table — one line per metric,
with ``-`` for the old value and ``+`` for the new. Without
``--i-have-reviewed-the-diff`` the script never writes; the operator
either re-runs with the flag or hand-edits the JSON.

Device and pinned seed come from the golden config / JSON, so the
generation path is the same as the test path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# Repo root is three levels up: tests/end_to_end/update_goldens.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.end_to_end._helpers.golden import (  # noqa: E402
    GOLDENS_DIR,
    Golden,
    enumerate_goldens,
    is_gpu_device,
    load_golden,
    load_golden_runner_config,
    run_baseline_for_golden,
)

# Default JSON shape for newly-initialised goldens. ``--init`` discovers
# baselines from disk; the seed and tolerance defaults below match the
# plan ``Phase 7`` doc.
_DEFAULT_SEED = 0
_DEFAULT_TOLERANCE = {"default": 1e-5}


def _runner_to_filename(runner: str) -> str:
    """Convert a runner config name to its golden JSON filename.

    ``golden/mcqa`` → ``mcqa.json`` — flat, one-per-runner. The filename is
    what the test harness parametrises over, so it must be unique. The
    config-group prefix (``golden/``) is implicit.
    """
    leaf = runner.rsplit("/", 1)[-1]
    return f"{leaf}.json"


def _load_or_synth(runner: str) -> Golden:
    """Return the existing golden for ``runner``, or a fresh stub.

    Fresh stubs default to the ``chat-coherent`` fixture (the sole coherent
    golden model); the actual device/fixture comes from the runner config at
    run time, and GPU bootstrap flips ``deterministic`` to false.
    """
    path = GOLDENS_DIR / _runner_to_filename(runner)
    if path.is_file():
        return load_golden(path)
    return Golden(
        runner=runner,
        seed=_DEFAULT_SEED,
        model="chat-coherent",
        tolerance=dict(_DEFAULT_TOLERANCE),
        values={},
        deterministic=True,
        path=path,
    )


def _format_value(v: Any) -> str:
    """Format a metric value for the diff table."""
    if isinstance(v, float):
        return f"{v!r}"
    return repr(v)


def _print_diff(runner: str, old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Print a per-key diff between ``old`` and ``new``. Return True if changed."""
    keys = sorted(set(old) | set(new))
    changed = False
    rows: list[tuple[str, str, str]] = []
    for key in keys:
        a = old.get(key, "<absent>")
        b = new.get(key, "<absent>")
        if a == b:
            continue
        rows.append((key, _format_value(a), _format_value(b)))
        changed = True

    if not changed:
        print(f"[{runner}] no diff (every value matches the existing golden).")
        return False

    print(f"[{runner}] diff:")
    width = max(len(r[0]) for r in rows)
    for key, a, b in rows:
        print(f"  {key:<{width}}  -{a}")
        print(f"  {' ':<{width}}  +{b}")
    return True


def _write_golden(golden: Golden, new_values: dict[str, Any]) -> None:
    """Overwrite the golden JSON on disk with the new values."""
    payload = {
        "runner": golden.runner,
        "seed": golden.seed,
        "model": golden.model,
        "tolerance": dict(golden.tolerance),
        "deterministic": golden.deterministic,
        "values": new_values,
    }
    golden.path.parent.mkdir(parents=True, exist_ok=True)
    with golden.path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[{golden.runner}] wrote {golden.path}")


def _enumerate_baseline_runners() -> list[str]:
    """List every golden runner config under ``tests/end_to_end/configs/golden/``.

    Returns ``golden/<name>`` for each ``*.yaml`` directly under that
    directory. ``--init`` uses this listing to bootstrap goldens for
    every shipped golden config; without ``--init`` we only refresh
    existing JSONs.
    """
    from tests.end_to_end._helpers.enumeration import enumerate_e2e_configs

    return enumerate_e2e_configs("golden")


def _runners_to_process(args: argparse.Namespace) -> list[str]:
    """Decide which runners to update for this invocation."""
    if args.baseline:
        return [args.baseline]
    if args.init:
        # ``--init`` covers every shipped baseline, even ones without a
        # JSON yet. Without ``--init`` we only refresh existing goldens
        # so a stray YAML doesn't surprise an operator.
        return _enumerate_baseline_runners()
    return [load_golden(p).runner for p in enumerate_goldens()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Runner config name to update (e.g. 'golden/mcqa'). "
            "If omitted, every existing runner golden is refreshed."
        ),
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help=(
            "Initialise goldens for every golden runner config under "
            "tests/end_to_end/configs/golden/ that doesn't yet have one."
        ),
    )
    parser.add_argument(
        "--i-have-reviewed-the-diff",
        dest="confirm",
        action="store_true",
        help=(
            "Required to actually write the new values. Without this "
            "flag, the script prints the diff and exits 0."
        ),
    )
    args = parser.parse_args(argv)

    runners = _runners_to_process(args)
    if not runners:
        print("No goldens to update. Use --init to bootstrap.", file=sys.stderr)
        return 0

    any_changed = False
    for runner in runners:
        golden = _load_or_synth(runner)
        # Keep the recorded ``model`` in sync with the config that actually
        # runs (the config — not this JSON field — picks the model at run
        # time, so a stale field is only misleading). Catches a golden
        # migrated across fixtures, e.g. mcqa small-coherent → chat-coherent.
        config_model = str(load_golden_runner_config(golden.runner).model.id)
        if golden.model != config_model:
            golden = dataclasses.replace(golden, model=config_model)
        with tempfile.TemporaryDirectory() as tmp:
            new_values, device = run_baseline_for_golden(golden, Path(tmp))
        # Phase 2 hardware contract: GPU runs default to
        # ``deterministic: false`` because BLAS / CUDA kernel drift
        # breaks byte-equality across same-seed runs. Per-key
        # reductions in ``tolerance`` carry the drift signal instead.
        if is_gpu_device(device) and golden.deterministic:
            golden = dataclasses.replace(golden, deterministic=False)
            print(
                f"[{runner}] resolved device={device}; pinning "
                f"deterministic=false (per-key tolerances carry drift signal)."
            )
        changed = _print_diff(runner, golden.values, new_values)
        if changed:
            any_changed = True
            if not args.confirm:
                print(
                    f"[{runner}] dry-run only — re-run with "
                    f"--i-have-reviewed-the-diff to overwrite."
                )
        if args.confirm:
            # Always re-write under --confirm so newly-introduced
            # JSON fields (e.g. ``deterministic``) land on disk even
            # for runners whose values haven't moved.
            _write_golden(golden, new_values)

    if not args.confirm and any_changed:
        print(
            "\nDry-run complete. Re-run with --i-have-reviewed-the-diff "
            "to write the new values.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
