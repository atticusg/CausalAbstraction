"""Hydra-effect fixture generator (McGrath et al. 2023, arXiv:2307.15771).

Writes tests/golden/fixtures/data/hydra/facts.json: N seeded factual-recall
prompts (CounterFact via NeelNanda/counterfact-tracing — short, templated,
safe to commit, and the distribution the VeriFires hydra-effect task uses)
with, per row:

- ``input``: the prompt, kept only when Llama-3.1-8B answers the fact
  correctly (first-token argmax equals the fact's true object — the
  VeriFires hydra-effect regime, "CounterFact factual-recall prompts
  filtered to correct answers"; unconfident prompts contribute near-zero
  direct effects and only attenuate the Fig-7 regression);
- ``resample_input`` .. ``resample_input_6``: six other rows' prompts,
  seeded derangements, the resample-ablation sources (the paper averages
  ~15 resample patches per prompt; the golden document sweeps these six
  and the test averages per-prompt effects over them);
- ``ml_token``: the model's next token (= the correct object's first
  token, given the filter), round-tripping to one token under the repo's
  column_token_id rule (space-prefixed first).

Run (needs the gated Llama-3.1-8B; ~minutes):
    uv run --with datasets python tests/golden/fixtures/generators/gen_hydra.py --device mps
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "data" / "hydra" / "facts.json"
MODEL_KEY = "meta-llama/Llama-3.1-8B"
N_KEEP = 128
SEED = 0
BATCH = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_KEY, torch_dtype=torch.bfloat16)
    model.to(torch.device(args.device)).eval().requires_grad_(False)

    dataset = load_dataset("NeelNanda/counterfact-tracing", split="train")
    order = list(range(len(dataset)))
    rng = random.Random(SEED)
    rng.shuffle(order)

    candidates = []
    for idx in order:
        record = dataset[idx]
        prompt = record["prompt"].rstrip()
        target = record["target_true"].strip()
        ids = tokenizer.encode(" " + target, add_special_tokens=False)
        if not prompt or not ids:
            continue
        candidates.append({"input": prompt, "answer_id": int(ids[0])})
        if len(candidates) >= N_KEEP * 12:  # ~10% correct rate headroom
            break

    rows = []
    with torch.no_grad():
        for i in range(0, len(candidates), BATCH):
            chunk = candidates[i : i + BATCH]
            enc = tokenizer(
                [c["input"] for c in chunk], return_tensors="pt", padding=True
            ).to(model.device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            argmax = logits[torch.arange(len(chunk)), last, :].argmax(dim=-1)
            for c, token_id in zip(chunk, argmax):
                if int(token_id) != c["answer_id"]:
                    continue  # the correctness filter
                decoded = tokenizer.decode([int(token_id)])
                cands = (
                    [decoded, decoded.lstrip(" ")]
                    if decoded.startswith(" ")
                    else [" " + decoded, decoded]
                )
                for candidate in cands:
                    ids = tokenizer.encode(candidate, add_special_tokens=False)
                    if len(ids) == 1:
                        if ids[0] == int(token_id):
                            rows.append({"input": c["input"], "ml_token": decoded})
                        break
            if len(rows) >= N_KEEP:
                break

    rows = rows[:N_KEEP]
    # seeded derangements for the resample sources: shuffled rotations
    for k in range(1, 7):
        order2 = list(range(len(rows)))
        rng.shuffle(order2)
        column = "resample_input" if k == 1 else f"resample_input_{k}"
        for pos, row_idx in enumerate(order2):
            rows[row_idx][column] = rows[order2[(pos + 1) % len(order2)]]["input"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
