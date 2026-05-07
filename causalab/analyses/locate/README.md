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
  mode: centroid              # interchange only: "centroid" | "pairwise"
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
```

**Task config** may declare `target_variables: [v1, v2, ...]` (plural) to loop; the legacy singular `target_variable` still works.

---

## Outputs

```
{experiment_root}/locate/{method}/
├── metadata.json
├── results.json               # top-level (first-variable best_cell)
└── {variable}/
    ├── heatmap.pdf            # (layer × position) score heatmap
    ├── results.json           # best_cell, scores_per_cell, scores_per_layer
    └── L{layer}/P{pos_id}/    # centroid mode only
        ├── patched_dists.safetensors
        ├── patched_dists.meta.json
        └── ground_truth_*.pdf
```

### Interpretation

- **`heatmap.pdf`** — rows are layers, columns are token positions, cells are the intervention score (lower is better for KL-based centroid mode). The darkest cell is the most localized (layer, position) for the target variable.
- **`results.json/best_cell`** — `{"layer": L, "token_position": P}` of the argmin cell.
- **`results.json/scores_per_layer`** — per-layer summary (min score across positions at that layer). Used by `visualize` and by downstream analyses that need a single layer.
- **`L{layer}/P{pos_id}/ground_truth_*.pdf`** (centroid mode only) — per-class patched output distributions for that cell; visual check that centroid steering actually produces the expected per-class output.
- **`heatmaps/raw_output_mask.pdf`** (`dbm_binary` only) — single (layer × position) binary mask showing which cells DBM selected. Filled cells indicate units the masking objective kept; empty cells were dropped.

### Downstream consumers

- `subspace`, `activation_manifold` auto-resolve from `results.json/best_layer` if `analysis.layers` is null.

---

## Cross-Model Patching

Set `analysis.source_model` to a model name to enable cross-model patching.  When set, activations are collected from the source model and the centroids are patched into the primary target model (`cfg.model.name`).  The default is `null`, which gives standard single-model patching.

```yaml
locate:
  method: interchange
  mode: centroid      # pairwise + source_model raises ValueError
  source_model: meta-llama/Llama-3.2-1B-Instruct   # collect from here
```

**Constraints:**
- `source_model` is only supported with `mode: centroid`.  Using `mode: pairwise` with a non-null `source_model` raises a `ValueError`.
- Both models must share the same hidden dimension; mismatched architectures will fail at patching time.
- The source and target models must use compatible tokenizers so that token-position indices remain consistent across both pipelines.

**Validation:** Setting `source_model` to the same checkpoint as `cfg.model.name` (i.e., source == target) is a valid way to verify that cross-model patching produces identical results to single-model patching.
