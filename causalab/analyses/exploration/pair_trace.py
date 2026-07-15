"""``pair`` mode: single base/counterfactual interchange trace for one pair.

One run == one base/CF pair == one SLURM array task == one GPU. Reads a JSONL
manifest (``cfg.exploration.pair.manifest``, one row per pair) and runs the row
selected by ``cfg.exploration.pair.index``, sweeping every (layer, token)
residual-stream cell via ``run_single_pair_trace``. Saves a self-contained
``single_pair_trace.json`` (no pipeline needed to re-plot) plus a
frequency-colored heatmap, mirroring the artifact shape of
``causalab/analyses/locate/main.py``.

Manifest row schema (one JSON object per line)::

    {"token": "<essential token label>", "input_idx": 0,
     "base": "<base prompt>", "counterfactual": "<edited prompt>",
     "out_dir": "<optional abs dir for this pair's artifacts>"}

If ``out_dir`` is omitted, the trace lands under
``<analysis out_dir>/<token>/input_<input_idx>/``.
"""

from __future__ import annotations

import json
import os

from omegaconf import DictConfig

from causalab.analyses.locate.single_pair_trace import save_single_pair_trace
from causalab.methods.generation import greedy_output
from causalab.neural.token_positions import get_list_of_each_token


def _read_manifest_row(manifest: str, index: int) -> dict:
    with open(manifest) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not 0 <= index < len(rows):
        raise IndexError(
            f"index {index} out of range for manifest with {len(rows)} rows"
        )
    return rows[index]


def run(pipeline, acfg: DictConfig, out_dir: str) -> str:
    index = int(acfg.index)
    row = _read_manifest_row(acfg.manifest, index)
    base = row["base"]
    cf = row["counterfactual"]

    safe_token = str(row.get("token", "pair")).replace("/", "_").replace(" ", "_")
    pair_out = row.get("out_dir") or os.path.join(
        out_dir, safe_token, f"input_{row.get('input_idx', index)}"
    )
    os.makedirs(pair_out, exist_ok=True)

    # Record the clean (un-intervened) outputs so the report can confirm the
    # edit actually flipped the prediction — the precondition for the pair.
    base_output = greedy_output(pipeline, base)
    cf_output = greedy_output(pipeline, cf)

    # The locate analysis owns the trace -> artifact layout; this is a thin
    # caller. ``extra_fields`` carry the manifest context + clean outputs into
    # the self-contained JSON.
    token_positions = get_list_of_each_token(base, pipeline)
    save_single_pair_trace(
        pipeline=pipeline,
        prompt=base,
        counterfactual_prompt=cf,
        token_positions=token_positions,
        layers=None,  # [-1] + all layers
        out_dir=pair_out,
        figure_format=acfg.figure_format,
        title=(
            f"Interchange trace: token={row.get('token')!r} "
            f"input={row.get('input_idx')}"
        ),
        extra_fields={
            "token": row.get("token"),
            "input_idx": row.get("input_idx"),
            "base_output": base_output,
            "counterfactual_output": cf_output,
        },
    )
    trace_path = os.path.join(pair_out, "single_pair_trace.json")
    print(
        f"[pair] index={index} -> {trace_path} (base={base_output!r} cf={cf_output!r})"
    )
    return trace_path
