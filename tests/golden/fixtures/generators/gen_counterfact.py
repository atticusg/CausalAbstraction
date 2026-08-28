"""ROME causal-tracing fixture generator (Meng et al. 2022, arXiv:2202.05262).

Writes tests/golden/fixtures/data/counterfact/facts_w{2,3,4,5}.json from the
paper's own tracing dataset — the 1,209 facts of ``known_1000.json``
(https://rome.baulab.info/data/dsets/known_1000.json), the set the paper's
§2 numbers (ATE 18.6, clean 27.0, corrupted 8.47) are measured on. An
earlier draft of this fixture sampled NeelNanda/counterfact-tracing and
measured ATE ≈ 36 — a regime mismatch (shorter templated prompts, a more
confident correct-fact distribution), not a tolerance problem; the golden
keeps the paper's facts instead.

Also prints the noise scale the documents pin: nu = 3 * sigma, where sigma
is the standard deviation of gpt2-xl's token-embedding matrix (0.048227,
so 3*sigma = 0.144681 — the paper's 3*sigma, not sqrt(3*sigma); the
subject-token-restricted sigma is within 6%, measured 0.045470).

Selection rules, all load-bearing for the golden band:

- **The paper's fact set, unfiltered**: known_1000.json already encodes
  "facts GPT-2 XL predicts correctly" by the paper's construction; rows
  are kept as-is (the gpt2-xl first-token argmax rate is printed for
  information, not used as a filter). The only drop is an answer whose
  first token fails the repo's single-token column round-trip.
- **One dataset per subject width, pooled proportionally**: the reference
  engine refuses ragged edits and the noise edit targets the
  subject-token window, so a single document needs equal-width subjects.
  Widths 2-5 are each written as their own dataset, sized proportionally
  to the width distribution, ~200 facts total; the golden test runs one
  document per width and pools the per-fact effects.
- Seed 0 shuffles once before the proportional cut.

Run (gpt2-xl for the info rate; ~a minute once cached):
    uv run python tests/golden/fixtures/generators/gen_counterfact.py --device mps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

import torch

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "counterfact"
MODEL_KEY = "gpt2-xl"
N_KEEP = 200
SEED = 0
BATCH = 16
KNOWN_1000_URL = "https://rome.baulab.info/data/dsets/known_1000.json"
KNOWN_1000_SHA256 = "61daca55318bb5260c1e62e133debef9102c7b278bbc7b19dd2ac655543f333a"


def object_first_token(tokenizer, target: str) -> tuple[int, str] | None:
    """The object's first token in prompt continuation position (space-
    prefixed), decoded so column_token_id resolves back to the same id."""
    ids = tokenizer.encode(" " + target.strip())
    if not ids:
        return None
    token_id = int(ids[0])
    decoded = tokenizer.decode([token_id])
    candidates = (
        [decoded, decoded.lstrip(" ")]
        if decoded.startswith(" ")
        else [" " + decoded, decoded]
    )
    for candidate in candidates:
        if tokenizer.encode(candidate) == [token_id]:
            return token_id, decoded
        break  # column_token_id takes the first single-token candidate
    return None


def subject_width(tokenizer, prompt: str, subject: str) -> int | None:
    """Token width of the subject's character span inside the prompt, via
    offset mapping — the same span the engine's variable resolution taps."""
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = urllib.request.urlopen(KNOWN_1000_URL, timeout=60).read()
    digest = hashlib.sha256(raw).hexdigest()
    print(f"known_1000.json sha256 {digest}")
    if KNOWN_1000_SHA256 and digest != KNOWN_1000_SHA256:
        raise RuntimeError(
            "known_1000.json changed upstream — do not regenerate silently"
        )
    facts = json.loads(raw)
    assert len(facts) == 1209, f"expected the paper's 1209 facts, got {len(facts)}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    candidates = []
    for fact in facts:
        prompt, subject = fact["prompt"].rstrip(), fact["subject"].strip()
        resolved = object_first_token(tokenizer, fact["attribute"])
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
    print(f"{len(candidates)}/{len(facts)} facts pass the single-token round-trip")

    # informational only: the paper's set is kept regardless
    model = AutoModelForCausalLM.from_pretrained(MODEL_KEY, torch_dtype=torch.float32)
    model.to(torch.device(args.device)).eval().requires_grad_(False)
    sigma = float(model.get_input_embeddings().weight.std())
    print(f"embedding sigma = {sigma:.6f}; noise scale 3*sigma = {3 * sigma:.6f}")
    hits = 0
    with torch.no_grad():
        for i in range(0, len(candidates), BATCH):
            chunk = candidates[i : i + BATCH]
            enc = tokenizer(
                [c["input"] for c in chunk], return_tensors="pt", padding=True
            ).to(model.device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            argmax = logits[torch.arange(len(chunk)), last, :].argmax(dim=-1)
            hits += sum(int(a) == c["answer_id"] for c, a in zip(chunk, argmax))
    print(f"gpt2-xl first-token argmax rate: {hits}/{len(candidates)} (informational)")

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    by_width: dict[int, list[dict]] = {}
    for c in candidates:
        by_width.setdefault(c["width"], []).append(c)
    print(f"widths { {w: len(v) for w, v in sorted(by_width.items())} }")

    widths = [2, 3, 4, 5]
    covered = sum(len(by_width.get(w, [])) for w in widths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for w in widths:
        bucket = by_width.get(w, [])
        n = min(len(bucket), round(N_KEEP * len(bucket) / covered))
        rows = [
            {"input": c["input"], "subject": c["subject"], "answer": c["answer"]}
            for c in bucket[:n]
        ]
        out = OUT_DIR / f"facts_w{w}.json"
        out.write_text(json.dumps(rows, indent=1) + "\n")
        total += len(rows)
        print(f"wrote {out} ({len(rows)} rows, subject width {w})")
    print(
        f"total {total} facts; widths 2-5 cover {covered}/{len(candidates)} "
        f"({covered / len(candidates):.0%}) of the paper's usable facts"
    )


if __name__ == "__main__":
    main()
