"""Addition fixture generator (Arithmetic in the Wild, arXiv:2605.01148).

Writes tests/golden/fixtures/data/addition/pairs.json: N seeded addition
prompts "{a}+{b}=" with a, b drawn from the paper's full operand range
{1, ..., 199} (App. G.1 — the VeriFires checklist explicitly forbids a
truncated 1-100 range), plus the pre-modulo sum column the Fourier probes
regress against. Purely templatic — no tokenizer or model involved.

Run: uv run python tests/golden/fixtures/generators/gen_addition.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 0
N = 4000
OUT = Path(__file__).resolve().parents[1] / "data" / "addition" / "pairs.json"


def main() -> None:
    rng = random.Random(SEED)
    seen = set()
    rows = []
    while len(rows) < N:
        a, b = rng.randint(1, 199), rng.randint(1, 199)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append({"input": f"{a}+{b}=", "sum": a + b})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(
        f"wrote {OUT} ({len(rows)} rows, sums {min(r['sum'] for r in rows)}..{max(r['sum'] for r in rows)})"
    )


if __name__ == "__main__":
    main()
