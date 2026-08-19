"""Capture (or refresh) the chat-coherent drift pins on the canonical GPU.

    uv run python tests/golden/drift/update_drift_goldens.py \\
        --device cuda --dtype bf16 --i-have-reviewed-the-diff

Runs both drift documents through the real CLI, extracts values with the
same code path the replay test uses (tests/golden/drift/_extract.py),
prints a per-key diff against the current pins, and refuses to write
without the review flag — the idiom inherited from the retired tier's
update_goldens.py and tests/protocol/update_corpus_digests.py. The
baseline-accuracy gate (>= 0.9) must hold at capture; a capture that
fails it is not a pin candidate but a task/model regression to
investigate.

Record which device the pins were captured on: the replay compares
against tolerance, and cross-device bf16 numerics (cuda vs mps) are not
guaranteed to sit inside it — the canonical capture device is cuda.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# repo root (NOT tests/ — putting tests/ itself on sys.path would shadow the
# top-level `tasks` package with tests/tasks and break task resolution)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.golden.drift._extract import (  # noqa: E402
    ACCURACY_GATE,
    PINS,
    extract_values,
    load_pins,
    run_drift_documents,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", default="bf16", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--i-have-reviewed-the-diff", action="store_true")
    args = parser.parse_args()

    dirs = run_drift_documents(Path(tempfile.mkdtemp()), args.device, args.dtype)
    values = extract_values(dirs)

    acc = values["interchange.acc.mean"]
    if acc < ACCURACY_GATE:
        print(f"REFUSED: baseline accuracy {acc:.4f} < gate {ACCURACY_GATE}")
        return 1

    pins = load_pins(PINS)
    old = pins.get("values") or {}
    for key in sorted(set(old) | set(values)):
        before, after = old.get(key), values.get(key)
        if before != after:
            print(f"  - {key}: {before}")
            print(f"  + {key}: {after}")

    if not args.i_have_reviewed_the_diff:
        print("dry run: pass --i-have-reviewed-the-diff to write the pins")
        return 1

    pins["values"] = values
    pins["capture"] = {"device": args.device, "dtype": args.dtype}
    PINS.write_text(json.dumps(pins, indent=2) + "\n")
    print(f"wrote {PINS} ({len(values)} keys, acc {acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
