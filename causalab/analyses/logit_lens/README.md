# Logit Lens

Reads a transformer's per-layer next-token predictions by projecting each
residual-stream layer through the model's **final layer norm** + **unembedding
head** (`lm_head`) — the logit lens (nostalgebraist, 2020).

## What it produces

Under `${experiment_root}/logit_lens/`:

- `metadata.json` — analysis configuration.
- `top_k/{layer}__{pos_id}.{safetensors,json}` — top-k predicted token ids,
  probabilities, and decoded strings at the **last token**, per layer, across
  the test set.
- `target_track/` — when the task exposes answer tokens, the full-vocab-softmax
  probability mass on those tokens per layer (`answer_mass_mean.json` plus
  per-cell tensors): the "at which depth does the answer emerge?" curve.
- `logit_lens_heatmap.{pdf,png}` — a (layer × token-position) grid of the top-1
  token for one representative prompt (layer -1 = embeddings at the bottom).

## Config

`causalab/configs/analysis/logit_lens.yaml` (`# @package logit_lens`):

| key | meaning |
|-----|---------|
| `layers` | residual-stream layers to read (`null` = all) |
| `top_k` | tokens retained per (sample, layer, position) |
| `batch_size` | forward-pass batch size |
| `apply_final_norm` | apply the model's final norm before unembedding (faithful lens) |
| `visualization.heatmap` | render the single-example heatmap |
| `visualization.figure_format` | `pdf` or `png` |

Requires `task.target_variable` (selects the answer tokens for the answer-mass
track).

## Caveats

The raw logit lens is **systematically biased on non-GPT-2 models** because
intermediate layers can encode information in a rotated/shifted basis relative
to the unembedding.
