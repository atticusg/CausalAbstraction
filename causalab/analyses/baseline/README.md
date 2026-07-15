# Baseline

Baseline answers: *Can the model solve this task unassisted?* It runs the model on task samples without any intervention to measure base accuracy and collect per-class output distributions. This is the generic first step to run on any task — it gates every downstream intervention analysis.

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

- **`train_samples.json`** / **`test_samples.json`** — All rendered (`raw_input`, `raw_output`) pairs from the generated train and test datasets. Use these to eyeball that the task prompt is well-formed. `test_samples.json` is only produced when `n_test > 0`.

- **`confusion_heatmap.{pdf,png}`** — Per-class average output distribution restricted to concept tokens (rows = ground truth classes, columns = output tokens). Rows should be approximately one-hot; off-diagonal mass indicates systematic confusion between specific classes. Extension is set by `analysis.visualization.figure_format` (PNG is convenient for notebooks).

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `accuracy.json` | `{"accuracy": float}` | human reference |
| `train_samples.json` / `test_samples.json` | `{"samples": [{raw_input, raw_output}, …]}` | human reference |
| `confusion_heatmap.pdf` / `.png` | seaborn heatmap | human reference |
| `per_class_output_dists.safetensors` | `[n_classes, vocab_size]` tensor | `locate`, `activation_manifold` |
| `top_logits.json` | `{"top_k": int, "examples": [{expected_output, prediction, top_tokens, correct}, …]}` | human reference |
| `metadata.json` | run config snapshot | provenance |

In `top_logits.json` each example records `expected_output` (the task's ground-truth label), `prediction` (the model's decoded top-1 token — i.e. `top_tokens[0]`), the full `top_tokens` list, and `correct` (`prediction == expected_output`). The field is named `expected_output`, not `prediction`, to make clear it is the label rather than the model's output.

`per_class_output_dists.safetensors` stores full-vocabulary softmax averages per class (not restricted to concept tokens). Simplex-geometry artifacts (`per_example_output_dists`, `hellinger_pca.pkl`, `hellinger_pca_3d.html`) are produced by `output_manifold`, not here.
