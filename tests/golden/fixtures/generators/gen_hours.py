"""Hours fixture generator (Arithmetic in the Wild, arXiv:2605.01148).

Writes tests/golden/fixtures/data/hours/all.json: all 1,152 prompts of the
paper's hours dataset — 24 concepts (00..23) x 48 spelled-out offsets
(one..forty-eight), the exact template from the paper:

    Q: In 24-hour time, it is now {concept}:00. What time will it be in
    {offset} hours?\\nA: In 24-hour time, it will be

Tokenization rule (recorded here because `match` is exact next-token
equality on a single-token column): the Llama-3.1 tokenizer never merges
a space with digits (" 07" is two tokens) but every zero-padded two-digit
hour is one token ("07" = 2589), so the prompt carries the trailing space
("... it will be ") and the answer column is the bare zero-padded hour,
asserted single-token at generation time. Zero-padded two-digit hours are
used throughout, matching the prompt's own concept rendering; the
VeriFires package accepts any consistent, stated tokenization rule.

Run: uv run python tests/golden/fixtures/generators/gen_hours.py
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parents[1] / "data" / "hours" / "all.json"
MODEL_KEY = "meta-llama/Llama-3.1-8B"

TEMPLATE = (
    "Q: In 24-hour time, it is now {concept}:00. "
    "What time will it be in {offset} hours?\n"
    "A: In 24-hour time, it will be "
)

_ONES = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = {2: "twenty", 3: "thirty", 4: "forty"}


def spell(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_KEY)
    rows = []
    for concept in range(24):
        for offset in range(1, 49):
            target = (concept + offset) % 24
            answer = f"{target:02d}"
            ids = tok.encode(answer, add_special_tokens=False)
            assert len(ids) == 1, f"answer {answer!r} is not a single token"
            rows.append(
                {
                    "input": TEMPLATE.format(
                        concept=f"{concept:02d}", offset=spell(offset)
                    ),
                    "answer": answer,
                    "concept": concept,
                    "offset": offset,
                    "target": target,
                }
            )
    assert len(rows) == 1152
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    distinct = sorted({r["answer"] for r in rows})
    print(f"wrote {OUT} ({len(rows)} rows, {len(distinct)} distinct answer tokens)")
    print("answer tokens:", distinct)


if __name__ == "__main__":
    main()
