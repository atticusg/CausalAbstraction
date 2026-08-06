# Ablation

Ablation answers: *how much does the model's task accuracy drop when a given
component is removed?* It zero- or mean-ablates each attention head or MLP across
a configured token span, generates, grades the output against the task's
`raw_output`, and reports the **behavioral accuracy drop**
(`drop = base_accuracy − ablated_accuracy`) as a (layer × head) or
(layer × position) heatmap. Explicit unit **combos** can be ablated jointly.

Ablation is a `replace` intervention (zeros, or the corpus-mean
activation) — the same units/featurizer machinery as interchange, so
subspace/feature-index ablation can layer on later. It does not perform greedy
circuit search.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — `experiment_root`, `seed`.

**Module config** (`causalab/configs/analysis/ablation.yaml`):

```yaml
# @package ablation
_name_: ablation
_subdir: ${.component_type}_${.mode}   # span token appended by the analysis
_output_dir: ${experiment_root}/ablation/${._subdir}

component_type: attention_head   # attention_head | mlp
mode: zero                       # zero | mean
span: all                        # "all" (every token) or a list of task
                                 # token-position names, e.g. [last_token]
layers: null                     # list[int] or null = all layers
heads: null                      # list[int] or null = all heads (attention_head only)
batch_size: 16
top_k: 20
combos: null                     # explicit joint-ablation sets:
                                 #   attention_head: [[[L,H],[L,H]], ...]
                                 #   mlp:            [[L,L], ...]
complement_keep: null            # list[int] of layers to KEEP; ablate every other
                                 # grid layer jointly (sufficiency). null = skip.

visualization:
  figure_format: ${figure_format}   # png or pdf — invariant 6
```

- **`component_type`** — `attention_head` scans a (layer × head) grid;
  `mlp` scans a (layer × position) grid. Head ablation is GQA-correct
  (`head_attention_value_output` is post-KV-repeat).
- **`mode`** — `zero` drops the feature contribution (the featurizer's orthogonal
  error term is preserved); `mean` replaces with the corpus-mean activation
  collected over the **train** split.
- **`span`** — `all` ablates every token position; a list of names ablates the
  union of those named task positions (resolved via `task.create_token_positions`).
  Examples of different length simply select different numbers of positions; the
  batch does not have to be regrouped.
- **`combos`** — joint ablations scored as a single drop, in addition to the grid.
- **`complement_keep`** — list of layers to *keep*; every other grid layer is
  ablated jointly (a sufficiency check). `null` skips it.

The output dir folds in a sanitized `span` token (e.g.
`.../ablation/attention_head_zero_all`), so runs differing only in `span` don't
overwrite each other.

**Task config** must set `target_variable` (or `target_variables[0]`); it only
selects which task to grade `raw_output` against.

---

## Outputs

All files are written under
`{experiment_root}/ablation/{component_type}_{mode}_{span}/`.

### Interpretation

- **`results.json`** — The direct answer: `base_accuracy` plus the per-cell
  `drop_grid` (`base_accuracy − ablated_accuracy`, keyed `"L|H"` for heads or
  `"L|pos"` for MLPs) and the `top_k_cells` ranked by drop. A large positive drop
  marks a component the task behavior depends on; a drop near zero means ablating
  it doesn't move accuracy. A **negative** drop means ablation *improved*
  accuracy (the component was hurting on this task). `combos` holds any
  joint-ablation drops; `complement` is non-null only when `complement_keep` is
  set — it ablates every grid layer *except* the kept layers jointly (a
  sufficiency check) and reports `{kept_layers, ablated_layers,
  ablated_accuracy, drop}`.

- **`heatmap.{pdf,png}`** — The `drop_grid` as a (layer × head) or
  (layer × position) heatmap; the colorbar is labelled **"Accuracy drop"**. Look
  for hot cells/bands localizing the behavior to specific layers or heads. When
  any cell is negative the plot switches to a diverging colormap centered at 0
  (symmetric bounds) so improvements are visually distinct from true zero drops;
  otherwise it uses the `[0, 1]` sequential scale. Extension is set by
  `analysis.visualization.figure_format`.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `results.json` | `{base_accuracy, drop_grid {"L|H": float}, top_k_cells, combos, complement}` | human reference |
| `heatmap.pdf` / `.png` | (layer × component) accuracy-drop heatmap | human reference |
| `metadata.json` | run config snapshot (component_type, mode, span, layers/heads, complement_keep, task/model, seed) | provenance |
