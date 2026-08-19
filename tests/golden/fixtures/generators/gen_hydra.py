"""Hydra-effect fixture generator (McGrath et al. 2023, arXiv:2307.15771).

Writes tests/golden/fixtures/data/hydra/facts.json: N seeded factual-recall
prompts (CounterFact via NeelNanda/counterfact-tracing — short, templated,
safe to commit, and the distribution the VeriFires hydra-effect task uses)
with, per row:

- ``input``: the prompt;
- ``resample_input``: another row's prompt, the resample-ablation source
  (a seeded derangement of the sample, so no row resamples from itself);
- ``ml_token``: the clean model's most-likely next token under
  meta-llama/Llama-3.1-8B, decoded, kept only when it round-trips to one
  token under the repo's column_token_id rule (space-prefixed first). The
  paper's per-prompt metric is the maximum-likelihood token's logit; no
  correctness filter is involved.

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

    prompts = []
    for idx in order:
        prompt = dataset[idx]["prompt"].rstrip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) >= N_KEEP * 2:  # headroom for round-trip rejects
            break

    rows = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i : i + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            argmax = logits[torch.arange(len(chunk)), last, :].argmax(dim=-1)
            for prompt, token_id in zip(chunk, argmax):
                decoded = tokenizer.decode([int(token_id)])
                candidates = (
                    [decoded, decoded.lstrip(" ")]
                    if decoded.startswith(" ")
                    else [" " + decoded, decoded]
                )
                for candidate in candidates:
                    ids = tokenizer.encode(candidate, add_special_tokens=False)
                    if len(ids) == 1:
                        if ids[0] == int(token_id):
                            rows.append({"input": prompt, "ml_token": decoded})
                        break
            if len(rows) >= N_KEEP:
                break

    rows = rows[:N_KEEP]
    # seeded derangement for resample sources: rotate by one after a shuffle
    order2 = list(range(len(rows)))
    rng.shuffle(order2)
    for pos, row_idx in enumerate(order2):
        source_idx = order2[(pos + 1) % len(order2)]
        rows[row_idx]["resample_input"] = rows[source_idx]["input"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
