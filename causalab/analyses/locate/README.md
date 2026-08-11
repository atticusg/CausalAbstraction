# Locate

Locate answers: *for each target causal variable, which **(layer, token_position)** cell in the residual stream most strongly encodes it?* It scans a (layer × token_position) grid via either interchange interventions or DBM binary masking, reports the best cell per variable, and saves a heatmap over the grid.

Run `baseline` first — locate can reuse `accuracy.json` and `per_class_output_dists.safetensors` from the baseline output directory.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — `experiment_root`, `batch_size`.

**Module config** (`causalab/configs/analysis/locate.yaml`):

```yaml
analysis:
  _name_: locate
  _subdir: ${analysis.method}
  _output_dir: ${experiment_root}/locate/${analysis._subdir}

  method: interchange         # "interchange" | "dbm_binary"
  mode: pairwise              # interchange only: "pairwise" (default) | "centroid"
  batch_size: ${batch_size}
  layers: null                # null = all hidden layers
  token_positions: null       # list of position names from the task; null = all
  seed: 42
  n_train: ${task.n_train}
  n_test: ${task.n_test}
  n_steer: 50                 # centroid mode: steer examples per class

  dbm:                        # used when method: dbm_binary
    training_epoch: 20        # epochs for the mask-intervention training loop
    lr: 0.001                 # initial learning rate
    regularization_coefficient: 100  # mask sparsity penalty

  prescan:                    # attribution-patching fail-fast gate (#456), pairwise only
    enabled: false            # one-backward gradient x delta-activation scores every cell
    top_k: 10                 # exact interchange then runs only on the top-k survivors
    n_examples: null          # cap on counterfactual pairs scored; null = full test set
```

**Task config** may declare `target_variables: [v1, v2, ...]` (plural) to loop; the legacy singular `target_variable` still works.

### Interchange modes

- **`pairwise`** (default) — patch each counterfactual's activation into its base and score the patched output per example, then average. The reference is set by `task.intervention_metric`:
  - `causal_label` (default) — exact match of the patched output against the causal model's **expected counterfactual label** (interchange-intervention accuracy).
  - `string_match` — lenient (case-insensitive containment) match against the same label.
  - `output_shift` / `output_shift_hellinger` — KL / Hellinger between the patched output distribution and the **base (pre-intervention)** distribution (how much patching this cell moved the output, regardless of correctness).
  - Legacy `kl` / `hellinger` map to the `output_shift` variants.

  > Pairwise is only informative when `task.resample_variable` is a single variable (the one being localized) — see **docs/CODEBASE.md §5**.
- **`centroid`** — patch per-class centroid activations and compare to per-class average distributions (`intervention_metric: kl`/`hellinger`). Meaningful under `resample_variable: "all"`.

### Attribution pre-scan (fail-fast gate)

`prescan.enabled: true` (pairwise mode, single-model only) runs a one-backward **gradient x delta-activation** approximation of every cell's interchange effect before the exact grid: one forward over the counterfactual batch plus one forward+backward over the base batch scores all cells, exact interchange then runs only on the `top_k` survivors. Cells are ranked by **|approx|** — the linearization's sign is unreliable through many downstream non-linearities, but the magnitude separates live cells from dead ones. `results.json` reports both scores wherever both were computed (`prescan.exact_and_approx`) plus `agreement_at_k` (top-set overlap at k capped to half the both-scored cells, so it can miss), `rank_correlation` (signed) and `abs_rank_correlation`, so the approximation quality stays visible; `prescan_heatmap.*` shows the approximate grid. The approximation linearizes around the base run — saturated cells can mis-rank, which is why the exact scan still runs on the survivors.

---

## Outputs

```
{experiment_root}/locate/{method}/
├── metadata.json
├── results.json               # top-level (first-variable best_cell)
└── {variable}/
    ├── heatmap.pdf            # (layer × position) score heatmap
    ├── prescan_heatmap.pdf    # approx grid (prescan.enabled only)
    ├── results.json           # best_cell, scores_per_cell, scores_per_layer
    │                          #   (+ prescan block when the gate ran)
    └── L{layer}/P{pos_id}/    # centroid mode only
        ├── patched_dists.safetensors
        ├── patched_dists.meta.json
        └── ground_truth_*.pdf
```

### Interpretation

- **`heatmap.pdf`** — rows are layers, columns are token positions, cells are the intervention score. Scores are **higher-is-better** (divergence-based metrics are negated), so the strongest cell is the most localized (layer, position) for the target variable.
- **`results.json/best_cell`** — `{"layer": L, "token_position": P}` of the highest-scoring cell.
- **`results.json/scores_per_layer`** — per-layer summary (best — highest — score across positions at that layer). Used by `visualize` and by downstream analyses that need a single layer.
- **`L{layer}/P{pos_id}/ground_truth_*.pdf`** (centroid mode only) — per-class patched output distributions for that cell; visual check that centroid steering actually produces the expected per-class output.
- **`heatmaps/raw_output_mask.pdf`** (`dbm_binary` only) — single (layer × position) binary mask showing which cells DBM selected. Filled cells indicate units the masking objective kept; empty cells were dropped.

### Downstream consumers

- `subspace`, `activation_manifold` auto-resolve from `results.json/best_layer` if `analysis.layers` is null.

---

## Cross-Model Patching

Set `analysis.source_model` to a model name to enable cross-model patching.  When set, activations are collected from the source model and patched into the primary target model (`cfg.model.name`).  The default is `null`, which gives standard single-model patching. Supported in both `pairwise` and `centroid` modes.

```yaml
locate:
  method: interchange
  mode: pairwise
  source_model: meta-llama/Llama-3.2-1B-Instruct   # collect from here
```

**Constraints:**
- Both models must share the same hidden dimension; mismatched architectures will fail at patching time.
- The source and target models must use compatible tokenizers so that token-position indices remain consistent across both pipelines.

**Validation:** Setting `source_model` to the same checkpoint as `cfg.model.name` (i.e., source == target) is a valid way to verify that cross-model patching produces identical results to single-model patching.
