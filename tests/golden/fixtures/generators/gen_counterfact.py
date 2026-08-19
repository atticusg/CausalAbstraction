"""CounterFact fixture generator (ROME, Meng et al. 2022, arXiv:2202.05262).

Writes tests/golden/fixtures/data/counterfact/facts.json: a seeded sample of
CounterFact facts (via NeelNanda/counterfact-tracing) that gpt2-xl answers
correctly, for the causal-tracing golden documents. Also prints the noise
scale the documents pin: nu = 3 * sigma, where sigma is the standard
deviation of gpt2-xl's token-embedding matrix (the paper's corruption
scale — 3*sigma, not sqrt(3*sigma)).

Selection rules, all load-bearing for the golden band:

- **Loose correctness filter** (the VeriFires prompt's stated trap): keep a
  fact when the object's first token is the argmax next token — never a
  high-confidence filter, which would inflate the clean baseline and move
  the ATE with it.
- **Single-token objects by construction**: the answer column holds the
  object's first token's decoded form and must round-trip to one token
  (space-prefixed first, the repo's column_token_id rule).
- **One subject-width bucket**: the reference backend refuses ragged edits,
  and the noise edit targets the subject-token window, so every kept row's
  subject must span the same number of tokens inside its prompt. The
  largest bucket among correct facts is kept.
- N = 200 facts, seed 0, from the dataset's train split in dataset order
  (shuffled once with the seed before filtering caps the compute).

Run (downloads gpt2-xl ~6.4GB on first use; ~minutes of inference):
    uv run python tests/golden/fixtures/generators/gen_counterfact.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "data" / "counterfact" / "facts.json"
MODEL_KEY = "gpt2-xl"
N_KEEP = 200
SEED = 0
CANDIDATE_CAP = 3000  # facts scored before bucketing; caps one-time compute
BATCH = 16


def object_first_token(tokenizer, target: str) -> tuple[int, str] | None:
    """The object's first token in prompt continuation position (space-
    prefixed), decoded so column_token_id resolves back to the same id."""
    ids = tokenizer.encode(" " + target.strip())
    if not ids:
        return None
    token_id = int(ids[0])
    decoded = tokenizer.decode([token_id])
    candidates = (
        [decoded, decoded.lstrip(" ")] if decoded.startswith(" ") else [" " + decoded, decoded]
    )
    for candidate in candidates:
        if tokenizer.encode(candidate) == [token_id]:
            return token_id, decoded
        break  # column_token_id takes the first single-token candidate
    return None


def subject_width(tokenizer, prompt: str, subject: str) -> int | None:
    """Token width of the subject's character span inside the prompt, via
    offset mapping — the same span the backend's variable resolution taps."""
    start = prompt.find(subject)
    if start < 0:
        return None
    end = start + len(subject)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    covered = [
        i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start
    ]
    if not covered or covered != list(range(covered[0], covered[-1] + 1)):
        return None
    return len(covered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # right padding + per-row last-real-token gather: gpt2 computes position
    # ids as arange, so left padding without explicit position_ids garbles
    # shorter rows (the executor passes position_ids; this script must not
    # rely on that)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_KEY, torch_dtype=torch.float32)
    model.to(torch.device(args.device)).eval().requires_grad_(False)

    sigma = float(model.get_input_embeddings().weight.std())
    print(f"embedding sigma = {sigma:.6f}; noise scale 3*sigma = {3 * sigma:.6f}")

    dataset = load_dataset("NeelNanda/counterfact-tracing", split="train")
    order = list(range(len(dataset)))
    random.Random(SEED).shuffle(order)

    candidates = []
    for idx in order[:CANDIDATE_CAP]:
        row = dataset[idx]
        # dataset strings carry leading spaces ("' Danielle Darrieux'");
        # the fixture stores the subject as it reads inside the prompt
        prompt, subject = row["prompt"].rstrip(), row["subject"].strip()
        resolved = object_first_token(tokenizer, row["target_true"])
        width = subject_width(tokenizer, prompt, subject)
        if resolved is None or width is None or not prompt:
            continue
        token_id, decoded = resolved
        candidates.append(
            {
                "input": prompt,
                "subject": subject,
                "answer": decoded,
                "answer_id": token_id,
                "width": width,
            }
        )

    correct = []
    with torch.no_grad():
        for i in range(0, len(candidates), BATCH):
            chunk = candidates[i : i + BATCH]
            enc = tokenizer(
                [c["input"] for c in chunk], return_tensors="pt", padding=True
            ).to(model.device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(dim=1) - 1  # last real token per row
            argmax = logits[torch.arange(len(chunk)), last, :].argmax(dim=-1)
            for c, a in zip(chunk, argmax):
                if int(a) == c["answer_id"]:
                    correct.append(c)

    by_width: dict[int, list[dict]] = {}
    for c in correct:
        by_width.setdefault(c["width"], []).append(c)
    width, bucket = max(by_width.items(), key=lambda kv: len(kv[1]))
    kept = bucket[:N_KEEP]
    print(
        f"{len(candidates)} candidates, {len(correct)} gpt2-xl-correct "
        f"({len(correct) / len(candidates):.1%}); widths "
        f"{ {w: len(v) for w, v in sorted(by_width.items())} }; "
        f"kept width={width} n={len(kept)}"
    )

    rows = [
        {"input": c["input"], "subject": c["subject"], "answer": c["answer"]}
        for c in kept
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows, subject width {width})")


if __name__ == "__main__":
    main()
