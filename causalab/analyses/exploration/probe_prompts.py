"""``probe`` mode: greedy-decode a batch of prompts and report each output.

Used by the ``exploration`` analysis (``mode: probe``) to check whether the
model solves a candidate prompt and to confirm that editing an essential token
flips the output. Reads ``cfg.exploration.probe.prompts`` (a JSON list of prompt
strings) and writes a JSON list of ``{"prompt": ..., "output": ...}`` (also
printed line by line).
"""

from __future__ import annotations

import json
import os

from omegaconf import DictConfig

from causalab.methods.generation import greedy_output


def run(pipeline, acfg: DictConfig, out_dir: str) -> list[dict]:
    with open(acfg.prompts) as f:
        prompts = json.load(f)

    results = []
    for prompt in prompts:
        output = greedy_output(pipeline, prompt)
        results.append({"prompt": prompt, "output": output})
        print(f"{prompt!r} -> {output!r}")

    out = acfg.get("out") or os.path.join(out_dir, "outputs.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] {len(results)} prompts -> {out}")
    return results
