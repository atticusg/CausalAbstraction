# exploration

exploration answers: *what does the model do at the tokens that matter on this task?* It runs one of five exploratory **modes** over raw, hand-authored inputs (a task-less analysis — like `characterize_subspace`, its data is JSON/manifests, not a runner-generated task dataset), reusing shipped primitives to gather the evidence the `explore-behavior` skill synthesizes into a hypothesis. The model comes from `cfg.model`; everything else from `cfg.exploration.<mode>`.

## Overview

The `mode` field selects one op; each reads its own `cfg.exploration.<mode>` block and writes under `${experiment_root}/exploration/<mode>/`:

```
mode: probe       prompts.json ──► greedy_output           ──► outputs.json
mode: logit_lens  inputs.json  ──► run_logit_lens_on_prompts ──► input_NN/ heatmaps + top-k JSON
mode: pair        pairs.jsonl  ──► save_single_pair_trace   ──► <token>/input_N/single_pair_trace.json + heatmap
mode: pca         inputs + essential_tokens ──► collect_features + compute_svd ──► <token>/L<layer>.safetensors
mode: knockout    inputs + span ──► run_ablation_scan_multi / run_ablation_combo_multi ──► <zero|mean>/results.json
```

`pair` and `pca` fan out one unit per task — one pair (`exploration.pair.index`) or one token (`exploration.pca.tokens`) per SLURM-array task. The five ops live in sibling modules, each exposing `run(pipeline, acfg, out_dir)`; `main(cfg)` builds the pipeline from `cfg.model` and dispatches.

## Configuration

**Root config** (`causalab/configs/config.yaml`)
- `experiment_root` — output root; artifacts land under `${experiment_root}/exploration/${mode}/`.
- `model.name` / `model.device` / `model.dtype` — the model to probe (built via a plain `LMPipeline`; no task).

**Module config** (`causalab/configs/analysis/exploration.yaml`)
```yaml
exploration:
  _name_: exploration
  _subdir: ${.mode}                  # output subdir is the mode
  _output_dir: ${experiment_root}/exploration/${._subdir}
  mode: ???                          # probe | logit_lens | pair | pca | knockout
  probe:
    prompts: ???                     # JSON list[str] of candidate prompts
    out: null                        # output JSON; default <out_dir>/outputs.json
    max_new_tokens: 3
  logit_lens:
    inputs: ???                      # JSON list[str] of prompts
    top_k: 10                        # per-cell top-k predicted tokens recorded
    batch_size: 16
    figure_format: png
  pair:
    manifest: ???                    # path to pairs.jsonl (one pair per row)
    index: 0                         # manifest row to run (SLURM-array fan unit)
    figure_format: png
    max_new_tokens: 3
  pca:
    inputs: ???                      # JSON list[str] or [{input, positions}]
    essential_tokens: ???            # JSON list of {label, index?/text?/occurrence?}
    labels: null                     # optional per-input color-scheme rows
    n_components: 10
    layers: null                     # comma-separated indices; null = all layers
    tokens: null                     # comma-separated token indices; null = all (fan unit)
    batch_size: 16
  knockout:
    inputs: ???                      # JSON list[str] or [{input, positions}] of prompts
    span: all                        # all | last | essential (tokens ablated above)
    essential_tokens: null           # JSON list of {label, index?/text?}; required when span=essential
    components: [attention_head, mlp] # families to knock out
    ablation_modes: [zero, mean]     # zero = drop contribution; mean = corpus-mean replace
    mlp_widths: [1, 3, 5, 10]        # contiguous MLP layer-band widths to sweep
    layers: null                     # list[int] or null = all layers
    heads: null                      # list[int] or null = all heads (attention_head only)
    batch_size: 16
    max_new_tokens: 3
```
Only the selected `mode`'s block is read; the other blocks' `???` fields are never accessed.

## Outputs

### Interpretation
- **`probe` → `outputs.json`** — `[{prompt, output}]`. Confirms the model solves a prompt (probe) and that an essential-token edit flips the output (the precondition for a `pair`).
- **`logit_lens` → `input_NN/`** — per-input (layer × position) top-1 heatmap + per-cell top-k JSON. Read *which token is predicted* at each position/depth — where the answer first appears.
- **`pair` → `<token>/input_N/single_pair_trace.json` + heatmap** — per-cell patched output for one base/CF pair. *Most cells don't change the output; the ones that do localize where/when that token's information is read.*
- **`pca` → `<token>/L<layer>.safetensors`** — top-`n_components` projections (+ explained variance). Read whether the residual subspace at a critical token organizes by a meaningful label (value/answer/correctness) and the depth at which it emerges.
- **`knockout` → `<zero|mean>/results.json`** — a top-level `metrics` descriptor and, per family, a `metrics` map carrying one grid per metric. Attention heads get both `metrics.<metric>.drop_grid` (`"layer|head"` → drop) and `metrics.<metric>.widths` (whole-sublayer bands: width → `{band_start_layer: drop}`); MLP gets `metrics.<metric>.widths`. With `complement=true` each family also carries `metrics.<metric>.complement_widths` (sufficiency: keep the band, ablate the rest). Two metrics are scored from one set of generations: `match_drop` (fraction of outputs that flipped) and `logit_diff` (drop in the base-predicted token's logit). Both graded against the un-ablated output; read *which heads / sublayers / MLP bands the behavior routes through* (necessity) and *which bands alone sustain it* (sufficiency).

### Saved artifacts

| File | Shape / Format | Mode | Used by |
|---|---|---|---|
| `outputs.json` | `[{prompt, output}]` | probe | the report's behavioral section |
| `input_NN/…` | heatmap + top-k JSON | logit_lens | the report / web app |
| `<token>/input_N/single_pair_trace.json` | self-contained trace | pair | the report / web app |
| `<token>/L<layer>.safetensors` | `projections [n_inputs, n_components]` + `rotation`, `mean` | pca | the report / web-app PCA explorer |
| `metadata.json`, `inputs.json`, `labels.json` | run config + per-input refs | pca | the web app |
| `<zero\|mean>/results.json` | per-metric head `drop_grid` + MLP `widths` drops (`match_drop`, `logit_diff`) | knockout | the report / web-app knockout tabs |

The `pca` mode's shared top-level files (`metadata.json` / `inputs.json` / `labels.json`) are written only by the first array task (or the single-job run), since the array writers are not atomic.
