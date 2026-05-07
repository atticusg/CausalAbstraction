# Baseline

Baseline answers: *Can the model solve this task unassisted, and are the task's counterfactual generators well-formed?* It runs the model on task samples without any intervention to measure base accuracy and collect per-class output distributions, and (without touching the model) it checks that each registered counterfactual generator on the task actually deconfounds the intervention variable from the other causal variables. This is the generic first step to run on any task — it gates every downstream intervention analysis.

The artifacts produced here are prerequisites for `locate`, `activation_manifold`, and (transitively) `output_manifold`.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — shared params used by this analysis:
- `experiment_root` — output root (default: `artifacts/${task.name}/${model.id}`)
- `batch_size` — default inference batch size (overridden by `analysis.batch_size`)

**Module config** (`causalab/configs/analysis/baseline.yaml`):

```yaml
analysis:
  _name_: baseline
  batch_size: ${batch_size}   # inference batch size
  seed: 42                    # dataset generation seed
  enumerate_all: true         # enumerate all task examples exhaustively
  n_train: ${task.n_train}    # training set size (used for distribution collection)
  n_test: ${task.n_test}      # test set size
  balanced: false             # balance classes in generated datasets
  visualization:
    figure_format: pdf # png or pdf — confusion + ground_truth_* figures
```

---

## Outputs

### Interpretation

- **`accuracy.json`** — Base accuracy (0–1 float). The primary answer to "can the model solve this task?" A very low value means the model hasn't learned the task and downstream interventions will be uninformative — fix the task prompt or switch models before going further.

- **`counterfactual_sanity.json`** — For every zero-arg counterfactual generator on the task, this records whether interchange interventions on the target variable produce a different output than interventions on the other causal variables, across a sample of pairs. A generator with `distinguishes_target_from_others: true` (proportion ≥ 0.99) is safe to use for localization; a low proportion means the generator's counterfactuals confound the target with other variables and downstream DAS/DBM results on it will be contaminated.

- **`train_samples.json`** / **`test_samples.json`** — All rendered (`raw_input`, `raw_output`) pairs from the generated train and test datasets. Use these to eyeball that the task prompt is well-formed. `test_samples.json` is only produced when `n_test > 0`.

- **`confusion_heatmap.{pdf,png}`** — Per-class average output distribution restricted to concept tokens (rows = ground truth classes, columns = output tokens). Rows should be approximately one-hot; off-diagonal mass indicates systematic confusion between specific classes. Extension is set by `analysis.visualization.figure_format` (PNG is convenient for notebooks).

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `accuracy.json` | `{"accuracy": float}` | human reference |
| `counterfactual_sanity.json` | `{generator_name: {distinguishes_target_from_others, proportion, count, n_samples}}` | human reference |
| `train_samples.json` / `test_samples.json` | `{"samples": [{raw_input, raw_output}, …]}` | human reference |
| `confusion_heatmap.pdf` / `.png` | seaborn heatmap | human reference |
| `per_class_output_dists.safetensors` | `[n_classes, vocab_size]` tensor | `locate`, `activation_manifold` |
| `metadata.json` | run config snapshot | provenance |

`per_class_output_dists.safetensors` stores full-vocabulary softmax averages per class (not restricted to concept tokens). Simplex-geometry artifacts (`per_example_output_dists`, `hellinger_pca.pkl`, `hellinger_pca_3d.html`) are produced by `output_manifold`, not here.
