"""Weekdays fixture for the chat-coherent drift tier (Qwen3-4B).

Writes tests/golden/fixtures/data/drift/weekdays.json: 64 seeded rows of
the weekday-successor task in raw completion form, with counterfactual
inputs for interchange. Raw completions because protocol v1 has no chat
template code path (the encoding layer's prefix_lengths field is the seam,
unwired) — the old chat-coherent tier's template + system directive is a
recorded fidelity gap, not reproduced here.

Every answer column value is asserted single-token under the
Qwen/Qwen3-4B-Instruct-2507 tokenizer (space-prefixed day names).

Run: uv run python tests/golden/fixtures/generators/gen_drift_weekdays.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

SEED = 0
N = 64
OUT = Path(__file__).resolve().parents[1] / "data" / "drift" / "weekdays.json"
MODEL_KEY = "Qwen/Qwen3-4B-Instruct-2507"

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
TEMPLATES = [
    "If today is {day}, tomorrow is",
    "Today is {day}, so tomorrow is",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_KEY)
    for day in DAYS:
        ids = tok.encode(" " + day, add_special_tokens=False)
        assert len(ids) == 1, f"' {day}' is not a single token under Qwen3"

    rng = random.Random(SEED)
    rows = []
    for i in range(N):
        base = DAYS[i % 7]
        cf = rng.choice([d for d in DAYS if d != base])
        template = TEMPLATES[(i // 7) % 2]
        rows.append(
            {
                "input": template.format(day=base),
                "counterfactual_inputs": [template.format(day=cf)],
                "answer": " " + DAYS[(DAYS.index(base) + 1) % 7],
                "cf_answer": " " + DAYS[(DAYS.index(cf) + 1) % 7],
                "subject": base,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
