"""Mixing-mechanisms fixture generator (arXiv:2510.06182).

Writes tests/golden/fixtures/data/mixing/music.json: seeded TargetRebind
interchange pairs for the paper's Music binding task (m=3: name performs
genre on instrument), n=20 entity groups per context, t_entity=2 (the
genre is the target - balanced lexical/reflexive regime, single-token
answers under the gemma-2-2b-it tokenizer).

Pair construction per row (the paper's three-mechanism disambiguation):
pick distinct group positions i_P (the bucketed query position), i_L, i_R
and an original query group g != all three. The counterfactual context is
a pure permutation of the original: the (name, instrument) pair of group
i_L moves to position i_P (so the counterfactual query's pair sits at
i_P), and the genres of i_P and i_R swap (so the counterfactual target is
the genre living at i_R in the original). After patching the
counterfactual's last-token residual into the original run:

- positional predicts the original genre at position i_P (``pos_answer``),
- lexical predicts the genre bound to the query pair in the original,
  i.e. at i_L (``lex_answer``),
- reflexive predicts the counterfactual target, present in the original
  at i_R (``ref_answer``),
- no-intervention predicts the original query's genre (``orig_answer``).

All 20 genres in a context are distinct, so the four predictions are
distinct. Buckets: i_P = 0 (first), 9/10 (middle), 19 (last), recorded in
``bucket``; N_PER_BUCKET rows each. Raw completions with an "Answer:" cue
(protocol v1 has no chat-template path).

Run: uv run python tests/golden/fixtures/generators/gen_mixing.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

SEED = 0
N_GROUPS = 20
N_PER_BUCKET = 150
OUT = Path(__file__).resolve().parents[1] / "data" / "mixing" / "music.json"
MODEL_KEY = "google/gemma-2-2b-it"

NAMES = [
    "John", "Mary", "Peter", "Anna", "James", "Laura", "David", "Emma",
    "Robert", "Alice", "Thomas", "Sarah", "Henry", "Julia", "Paul", "Nina",
    "George", "Clara", "Martin", "Diana", "Oliver", "Sophie", "Frank", "Helen",
]
GENRES = [
    "rock", "pop", "jazz", "blues", "folk", "metal", "funk", "soul",
    "disco", "techno", "opera", "reggae", "punk", "country", "gospel",
    "swing", "salsa", "trance", "grunge", "ska", "house", "indie",
]
INSTRUMENTS = [
    "guitar", "piano", "drums", "violin", "flute", "trumpet", "cello",
    "harp", "banjo", "saxophone", "accordion", "clarinet", "organ",
    "mandolin", "tuba", "harmonica", "ukulele", "trombone", "oboe",
    "bass", "fiddle", "keyboard",
]

CLAUSE = "{name} performed {genre} music on the {instrument}"
QUERY = "What music did {name} play on the {instrument}? Answer:"


def render(groups: list[tuple[str, str, str]], q_name: str, q_instrument: str) -> str:
    clauses = ", ".join(
        CLAUSE.format(name=n, genre=g, instrument=i) for n, g, i in groups
    )
    return (
        f"At the music festival, {clauses}. "
        + QUERY.format(name=q_name, instrument=q_instrument)
    )


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_KEY)
    for genre in GENRES:
        ids = tok.encode(" " + genre, add_special_tokens=False)
        assert len(ids) == 1, f"' {genre}' is not a single token under gemma-2"

    rng = random.Random(SEED)
    rows = []
    for bucket, i_p_choices in (("first", [0]), ("middle", [9, 10]), ("last", [19])):
        for _ in range(N_PER_BUCKET):
            names = rng.sample(NAMES, N_GROUPS)
            genres = rng.sample(GENRES, N_GROUPS)
            instruments = rng.sample(INSTRUMENTS, N_GROUPS)
            groups = list(zip(names, genres, instruments))

            i_p = rng.choice(i_p_choices)
            others = [i for i in range(N_GROUPS) if i != i_p]
            i_l, i_r, g_q = rng.sample(others, 3)

            cf = [list(g) for g in groups]
            # move i_L's (name, instrument) pair to position i_P
            cf[i_p][0], cf[i_l][0] = cf[i_l][0], cf[i_p][0]
            cf[i_p][2], cf[i_l][2] = cf[i_l][2], cf[i_p][2]
            # the counterfactual target genre at i_P is the original i_R genre
            cf[i_p][1], cf[i_r][1] = cf[i_r][1], cf[i_p][1]

            rows.append(
                {
                    "input": render(groups, names[g_q], instruments[g_q]),
                    "counterfactual_inputs": [
                        render(
                            [tuple(g) for g in cf], names[i_l], instruments[i_l]
                        )
                    ],
                    "pos_answer": " " + genres[i_p],
                    "lex_answer": " " + genres[i_l],
                    "ref_answer": " " + genres[i_r],
                    "orig_answer": " " + genres[g_q],
                    "bucket": bucket,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # one file per bucket: a document's single-batch lm_head forward over all
    # 450 rows would materialize ~57GB of logits; 150-row buckets are H100-sized
    for bucket in ("first", "middle", "last"):
        sub = [r for r in rows if r["bucket"] == bucket]
        out = OUT.parent / f"music_{bucket}.json"
        out.write_text(json.dumps(sub, indent=1) + "\n")
        print(f"wrote {out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
