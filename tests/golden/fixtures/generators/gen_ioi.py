"""Seeded IOI fixture generator (Wang et al. 2023, arXiv:2211.00593).

Writes tests/golden/fixtures/data/ioi/clean.json: N=512 prompts, half ABBA
half BABA, from the paper's template family with single-token names, places
and objects under the gpt2 tokenizer. Columns: input, io, s (the metric
columns are space-prefixed and asserted single-token at generation time, so
the document's logit_diff(a=io, b=s) resolves without tokenizer surprises).

The paper's 3.56 mean logit difference is a 100,000-example figure; this
fixture pins seed 0 / N=512, and the golden band (+/-1.2, from the
VeriFires ioi-circuit checklist) absorbs the smaller sample.

Run: uv run python tests/golden/fixtures/generators/gen_ioi.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

SEED = 0
N = 512
OUT = Path(__file__).resolve().parents[1] / "data" / "ioi" / "clean.json"

NAMES = [
    "Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Scott",
    "Sarah", "Kevin", "Paul", "Anna", "Peter", "Laura", "Mark", "Emily",
    "Jason", "Karen", "Ryan", "Lisa", "Eric", "Susan", "Adam", "Rachel",
]
PLACES = ["store", "park", "school", "office", "restaurant", "garden", "hospital", "station"]
OBJECTS = ["drink", "ring", "bone", "snack", "book", "ball", "necklace", "basketball"]

TEMPLATES_ABBA = [
    "Then, {io} and {s} went to the {place}. {s} gave a {obj} to",
    "Then, {io} and {s} had a lot of fun at the {place}. {s} gave a {obj} to",
    "When {io} and {s} got a {obj} at the {place}, {s} decided to give it to",
    "Friends {io} and {s} found a {obj} at the {place}. {s} gave it to",
]
TEMPLATES_BABA = [
    "Then, {s} and {io} went to the {place}. {s} gave a {obj} to",
    "Then, {s} and {io} had a lot of fun at the {place}. {s} gave a {obj} to",
    "When {s} and {io} got a {obj} at the {place}, {s} decided to give it to",
    "Friends {s} and {io} found a {obj} at the {place}. {s} gave it to",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    for name in NAMES:
        ids = tok.encode(" " + name)
        assert len(ids) == 1, f"name {name!r} is not single-token under gpt2"

    rng = random.Random(SEED)
    rows = []
    for i in range(N):
        io, s = rng.sample(NAMES, 2)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        family = TEMPLATES_ABBA if i % 2 == 0 else TEMPLATES_BABA
        template = rng.choice(family)
        rows.append(
            {
                "input": template.format(io=io, s=s, place=place, obj=obj),
                "io": " " + io,
                "s": " " + s,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
