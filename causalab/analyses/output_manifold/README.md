# Output Manifold

Output manifold answers: *What does a continuous manifold through the model's per-class output distributions look like in probability-simplex geometry?* It collects per-example output distributions over the concept tokens, embeds them in Hellinger space (the isometric embedding of the simplex via √p), fits a 3-component PCA, and then fits a Thin-Plate Spline through the per-class centroids in that space. It also emits a 3D interactive visualization of the simplex embedding.

Reads per-class distributions from `baseline`. Downstream consumers: `pullback` (uses centroids + natural distributions for geodesic construction and belief-space path visualization), `evaluate` (conformal metric, belief-space path visualization).

---

## Configuration

**Root config** (`causalab/configs/config.yaml`):
- `experiment_root` — output root
- `batch_size` — default inference batch size

**Module config** (`causalab/configs/analysis/output_manifold.yaml`):

```yaml
analysis:
  _name_: output_manifold
  _subdir: ${analysis.method}_s${analysis.smoothness}
  _output_dir: ${experiment_root}/output_manifold/${analysis._subdir}
  method: spline                 # TPS fitting method
  smoothness: 0.0                # TPS smoothness parameter
  intrinsic_dim: 1               # intrinsic dimensionality of the output manifold
  intrinsic_mode: pca            # "pca" (project centroids via Hellinger PCA) or "parameter" (use causal-model parameter coords)
  seed: 42                       # dataset generation seed (for the forward-pass pre-pass)
  colormap: ${task.colormap}     # color scheme for 3D visualizations
```

---

## Outputs

### Interpretation

- **`hellinger_pca_3d.html`** (at `{experiment_root}/output_manifold/`) — Interactive 3D scatter of all training examples in Hellinger space, colored by class. Well-separated clusters mean the model's output distributions reliably distinguish classes; overlapping clusters foreshadow a TPS fit that can't be pulled apart on that axis.

- **`{method}_s{smoothness}/output_manifold_3d.html`** — Interactive 3D plot of the fit TPS manifold with per-example points overlaid. A good fit passes through the per-class centroids and interpolates smoothly in the intrinsic coordinate implied by the task structure.

- **`{method}_s{smoothness}/metadata.json`** — Records the intrinsic_mode, fitted ambient/intrinsic dims, and any detected periodicity. Check `ambient_dim` = W+1 for sanity.

### Saved artifacts

Top-level (`{experiment_root}/output_manifold/`) — shared across TPS configs:

| File | Shape / Format | Used by |
|---|---|---|
| `per_example_output_dists.safetensors` | `[n_train, n_classes + 1]` tensor | `pullback` (kNN geodesic init), `evaluate` (conformal metric, belief-space paths), `output_manifold` TPS fit |
| `hellinger_pca.pkl` | sklearn `PCA(n_components=3)` | `output_manifold` TPS fit (pca mode), `evaluate`/`pullback` belief-space visualizations |
| `hellinger_pca_3d.html` | plotly html | human reference |

Per-fit (`{experiment_root}/output_manifold/{method}_s{smoothness}/`):

| File | Shape / Format | Used by |
|---|---|---|
| `manifold.pt` | TPS checkpoint | `pullback`, `evaluate` (distance_from_manifold, mesh rendering) |
| `output_manifold_3d.html` | plotly html | human reference |
| `metadata.json` | run config + fit summary | provenance |

The `+1` last dimension in `per_example_output_dists` is the residual "other" mass: `1 − Σ concept_probs`, so each row sums to 1 over the (W+1)-dimensional simplex.
