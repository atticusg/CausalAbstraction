# Causal Sufficiency

Causal sufficiency answers: *which internal site carries the information the model
needs to produce its answer — i.e. restoring which site's clean activation
recovers the behavior after the input is corrupted?* This is ROME-style causal
tracing: it corrupts the residual stream where the information enters (the
embedding span, via `zero`, `mean`, or seeded `3σ` `noise`), establishing a
broken-behavior floor, then **restores one clean site at a time** — optionally a
centered `window` of consecutive layers (ROME's severed traces) — over a grid of
attention heads / attention-sublayer outputs / MLPs / residual positions, and
reports `recovery = restored_metric − corrupted_floor` (optionally normalized to
the `clean_ceiling − floor` band) as a (layer × head) or (layer × position)
heatmap.

Corruption and restoration are applied together in a single forward pass. Tracing
is behavioral (one score per cell, not per causal variable). This is a terminal
diagnostic — it reads only the task and the model, and no downstream analysis
consumes its outputs. Run `baseline` first to confirm the model actually solves
the task, otherwise the clean ceiling is meaningless.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — `experiment_root`, `seed`.
Dataset-construction knobs (`n_train`, `n_test`, `enumerate_all`,
`resample_variable`) are read from the task config (`cfg.task.*`).

**Module config** (`causalab/configs/analysis/causal_sufficiency.yaml`):

```yaml
# @package causal_sufficiency
_name_: causal_sufficiency
_subdir: ${.restore.component_type}_${.corruption.kind}   # span token appended by the analysis
_output_dir: ${experiment_root}/causal_sufficiency/${._subdir}

# --- corruption: the broken-behavior floor ---
corruption:
  kind: noise          # zero | mean | noise
  layer: -1            # residual-stream layer to corrupt; -1 = embeddings / block-0 input (ROME)
  span: all            # "all" (every token) or a list of task token-position names
  noise_scale: 3.0     # kind=noise only; MULTIPLE of the subject-embedding sigma (ROME's nu = 3*sigma)
  noise_seed: 0        # reproducible seeded Gaussian draw

# --- restore grid: the site swept one cell at a time ---
restore:
  component_type: residual   # attention_head | attention_output | mlp | residual
  layers: null               # list[int] or null = all layers (the cell CENTER layers)
  heads: null                # list[int] or null = all heads (attention_head only)
  span: [last_token]         # restored sites; each named position must resolve to a SINGLE token
  window: 1                  # restore a centered band of consecutive layers per cell (ROME uses ~10)

# --- recovery metric ---
metric:
  kind: prob                 # prob (softmax P(answer), ROME) | logit_diff | logit
  answer_variable: null      # prob/logit: the ex["input"][answer] token to read
  correct_variable: null     # logit_diff: logit[correct] − logit[distractor]
  distractor_variable: null  #   (e.g. IO / name_C)

normalize: recovery          # recovery = restored − floor; fraction = (restored − floor) / (ceiling − floor)
batch_size: 16
top_k: 20

visualization:
  figure_format: ${figure_format}   # png or pdf — invariant 6
```

**Corruption sub-group** — sets the entry site and how it is broken. `zero`/`mean`
replace the entry span with one vector; `noise` adds independent per-token
Gaussian scaled to `noise_scale × σ` of the subject embeddings, so it spans a
multi-token subject. `layer: -1` corrupts the embeddings, the ROME default.

**Restore sub-group** — sets the swept grid. `component_type` selects the grid
axes: `attention_head` gives a (layer × head) grid keyed `(layer, head)`; the
other three give a (layer × span) grid keyed `(layer, span.id)`. `window > 1`
restores a centered band of consecutive layers jointly at each cell while keeping
the heatmap axes unchanged.

**Metric sub-group** — how each cell's raw patched score is measured; recovery
relative to the corrupted floor is computed by the analysis, not the metric.
`prob` (ROME's default) reads P(answer); `logit_diff` needs a
correct/distractor variable pair.

The output dir folds in a sanitized restore `span` token (e.g.
`.../causal_sufficiency/residual_noise_last_token`) so runs differing only in the
restore span don't overwrite each other.

---

## Outputs

All files are written under
`{experiment_root}/causal_sufficiency/{component_type}_{corruption_kind}_{restore_span}/`.

### Interpretation

- **`results.json`** — The direct answer: `clean_ceiling`, `corrupted_floor`, the
  per-cell `recovery_grid` (`restored_metric − corrupted_floor`, keyed `"L|H"` for
  heads or `"L|span"` otherwise), and the `top_k_cells` ranked by recovery. A large
  positive recovery marks a **mediating** site — restoring its clean activation
  rebuilds the behavior the corruption destroyed; recovery near zero means the site
  carries none of the information the answer depends on. Under
  `normalize: fraction`, values sit in ~`[0, 1]` as the fraction of the
  clean-corrupted band recovered. Sanity-check the two scalars first: if the
  `corrupted_floor` is not well below the `clean_ceiling`, the corruption failed to
  break behavior and no cell can meaningfully "recover" anything.

- **`heatmap.{png,pdf}`** — The `recovery_grid` as a (layer × head) or
  (layer × position) heatmap; the colorbar is labelled **"Recovery"**. Look for hot
  cells/bands localizing where the answer's information is restored. A diverging map
  centered at 0 is used when any cell is negative (a site that pushed past the clean
  ceiling or below the floor); otherwise a sequential `[0, max]` (or `[0, 1]` under
  `fraction`) scale. Extension is set by `analysis.visualization.figure_format`.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `results.json` | `{corruption_kind, component_type, normalize, clean_ceiling, corrupted_floor, recovery_grid {"L|H": float}, top_k_cells}` | human reference |
| `heatmap.pdf` / `.png` | (layer × head) or (layer × position) recovery heatmap | human reference |
| `metadata.json` | run config snapshot (corruption/restore/metric settings, layers/heads, task/model, seed) | provenance |

Grid keys are `(center_layer, head)` for `attention_head` and `(center_layer,
span.id)` for the other component types (a single restored-span column). With
`window > 1` the key is still the *center* layer, but the cell restores the whole
windowed band.
