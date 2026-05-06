# Activation Manifold

Activation manifold answers: *What is the low-dimensional geometry of the model's internal activations as the causal variable varies?* It takes a pre-computed k-dimensional feature subspace (from `subspace` -- PCA, DAS, or DBM) and fits a smooth manifold (Thin-Plate Spline or normalizing flow) through the per-class activation centroids in that space. The manifold learns a mapping from intrinsic coordinates (the causal parameter values) to the ambient activation space, enabling smooth interpolation between classes.

Reads features and rotation matrices from `subspace`. Downstream consumers: `evaluate` (loads the composed featurizer for intervention scoring), `pullback` (uses manifold centroids and encode/decode for path optimization).

> **Scope:** this analysis only fits geometry. It does not evaluate whether traversing the manifold causally steers the model — that is the **`path_steering`** analysis, which is the natural follow-up stage. By default, activation_manifold skips the optional decoding reconstruction test (step 3b) and runs without loading model weights, working purely on cached features and saving the composed featurizer for downstream consumers. Set `skip_decoding_eval: false` if you want the centroid-only KL sanity check (this requires loading model weights and a `baseline` or cached `ref_dists`).
>
> **Implementation note — model loading:** `LMPipeline` defaults to `load_weights=True` everywhere else in the codebase, since most analyses need forward passes. This analysis is the exception: when `skip_decoding_eval=true` (the default), `main.py` calls `load_lite_pipeline()` instead of `load_pipeline()`, which constructs an `LMPipeline` with `load_weights=False`. That returns a tokenizer + `AutoConfig` only — enough to build `InterchangeTarget`s (which need `pipeline.tokenizer` for token-position resolution and `pipeline.model.config.hidden_size` for unit shape) and save the composed featurizer for downstream consumers, with zero GPU memory used for the base model. Setting `skip_decoding_eval=false` automatically falls back to the full `load_pipeline()` path. Don't pass `load_weights` from a config — the analysis decides based on whether forward passes are actually needed.
>
> **In-sample quirk in the reconstruction test:** when `skip_decoding_eval=false`, the test uses the full **train** dataset both as the carrier-prompt distribution for the steered forward passes and (transitively) as the source of the per-class `ref_dists` it compares against. This is deliberate — both sides must share the carrier distribution, otherwise the comparison is confounded by train/test prompt-distribution shift rather than manifold reconstruction quality. But it does mean Step 3b is an in-sample sanity check, not a generalization claim. For a held-out evaluation that uses test prompts and measures whether traversing the manifold causally steers the model, run **`path_steering`**.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`):
- `experiment_root` -- output root
- `batch_size` -- default inference batch size
- `k_features` -- number of subspace features (shared with `subspace`)
- `manifold_intrinsic_coords` -- intrinsic coordinate mode: `parameter` (use causal-model variable embeddings) or `pca` (use raw PCA coordinates)

**Module config** (`causalab/configs/analysis/activation_manifold.yaml`):

```yaml
analysis:
  _name_: activation_manifold
  _subdir: ${.method}_s${.smoothness}
  _output_dir: ${experiment_root}/activation_manifold/${._subdir}

  method: spline                        # "spline" (TPS) or "flow" (normalizing flow)
  smoothness: 0.0                       # TPS smoothness; 0 = exact interpolation through centroids
  intrinsic_dim: null                   # null = auto-detect from causal model (count of causal parameters)
  n_grid: 21                            # points per intrinsic dimension (used for visualization grid)
  batch_size: ${batch_size}
  seed: 42
  subspace: null                        # subspace subdir to use; null = auto-discover all under subspace/
  n_train: ${task.n_train}
  n_test: ${task.n_test}
  manifold_intrinsic_coords: ${manifold_intrinsic_coords}  # "parameter" or "pca"
  skip_decoding_eval: true              # default: skip the reconstruction test (step 3b). Keeps model weights unloaded.
  colormap: ${task.colormap}            # color scheme for 3D visualization
  embedding_shuffle_seed: null          # seed for permuting embedding coordinate assignments (ablation)
```

---

## Outputs

Directory structure: `{experiment_root}/activation_manifold/{subspace_sub}/{method}_s{smoothness}/`

### Interpretation

- **`visualization/manifold_3d.html`** -- Interactive 3D scatter of training activations in the feature subspace, with the fitted manifold surface overlaid. Points are colored by the intervention variable. A good fit shows the manifold passing through per-class centroids with smooth interpolation between them. If the manifold misses centroids or folds, the feature space may be too low-dimensional or the intrinsic parameterization may be wrong.

- **`metadata.json`** -- Records method, smoothness, subspace source, layer, k_features, and task/model provenance. When the decoding reconstruction test ran, `reconstruction_kl` (mean KL divergence across control points) is the summary metric: lower is better. Values below 0.5 nats typically indicate successful reconstruction; values above 2.0 suggest the subspace or manifold is a poor fit. For richer steering-faithfulness evaluation across the full manifold, run **`path_steering`**.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `manifold_spline/ckpt_final.pt` (or `manifold_flow/`) | TPS/flow checkpoint | `evaluate`, `pullback` (via composed featurizer) |
| `manifold_spline/metadata.json` | fit summary (ambient_dim, intrinsic_dim, periodic_dims) | provenance |
| `models/{layer}__{pos_id}/` | composed featurizer (subspace >> standardize >> manifold) | `evaluate`, `path_steering` (load featurizer chain for intervention scoring) |
| `visualization/manifold_3d.html` | plotly HTML | human reference |
| `metadata.json` | run config (+ `reconstruction_kl` if step 3b ran) | provenance |

The composed featurizer saved under `models/` chains three transforms: (1) the subspace projection (e.g. PCA rotation), (2) standardization (zero-mean, unit-variance), and (3) the manifold encode/decode. Downstream analyses load this as a single unit via `InterchangeTarget.load()`.
