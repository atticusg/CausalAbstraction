# Subspace

Subspace answers: *Which low-dimensional subspace of a layer's residual stream encodes the causal variable?* It establishes a `(d_model, k)` rotation onto that subspace at a chosen `(layer, token_position)`, projects the variable's activations through it, and writes the bundle that the manifold analyses fit on. It runs either as a single cell (one layer/position) or as a grid scan over `layers × {token_positions | heads}`.

The subspace is established by one of five **methods**:

- **`pca`** — top-`k` principal components of the activations (no training; fast).
- **`das`** — supervised rotation trained to maximize causal alignment (DAS).
- **`dbm`** — differentiable binary mask over feature dimensions.
- **`boundless`** — Boundless DAS (learned boundaries, no fixed `k`).
- **`fixed`** — thread a **given/precomputed** rotation (e.g. SAE decoder directions, or any imported basis) through the pipeline **without fitting**. Use this to test a hypothesized subspace rather than discovering one. The producer of the rotation artifact is `characterize_subspace/subspace_builder.py` (SAE clusters → `.safetensors`); `method: fixed` consumes it. **Do not scaffold a session-local `fixed_subspace` analysis** — this is the shipped path.

The artifacts produced here are prerequisites for `activation_manifold`, `output_manifold`, and `path_steering`, all of which auto-discover the subspace directory.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`):
- `experiment_root` — output root.
- `seed` — dataset-generation seed.

**Module config** (`causalab/configs/analysis/subspace.yaml`):
- `method` — `pca | das | dbm | boundless | fixed`.
- `k_features` — subspace dimensionality. For `method: fixed`, must equal the given rotation's column count (validated; mismatch fails fast).
- `layers` — `list[int]`, or `null` to auto-resolve the best cell from `locate/`. A single entry is single-cell mode; multiple entries is grid mode. **`method: fixed` requires a single explicit layer** (a given rotation is tied to one `(layer, token_position)`; grid and locate-auto are rejected).
- `token_positions` — list of position names (residual stream / MLP).
- `component_type` — `residual_stream | attention_head | mlp`.
- `modes` — optional sweep: a list of per-mode overrides, each written under `subspace/{mode._subdir}/{mode.name or mode.component_type}/`. `_subdir` is re-resolved per mode, so entries that differ by a `_subdir`-feeding field (`k_features`, `method`) land in distinct dirs instead of colliding.
- Per-method blocks: `das`, `dbm`, `boundless`, and **`fixed`** (provide exactly one input):
  ```yaml
  fixed:
    artifact: null      # path to rotation .safetensors (tensor key: rotation_matrix), shape (d_model, k)
    source: null        # SAE-cluster spec: {sae_checkpoint, clusters_path, cluster_id, orthonormalize}
    feature_ids: null   # explicit SAE feature ids (list[int]); requires fixed.sae_checkpoint
    sae_checkpoint: null
    orthonormalize: true
  ```

---

## Outputs

Single-cell artifacts land under `subspace/{method}_k{k}/{target_variable}/` (e.g. `subspace/pca_k8/result/`, `subspace/fixed_k8/result/`); grid runs add a `layer_x_pos/L{layer}_{pos}/` level plus `heatmaps/` and `grid_results.json`.

| File | Shape / Format | Used by |
|---|---|---|
| `rotation.safetensors` | key `rotation_matrix` `(d_model, k)` (+ `explained_variance_ratio` for PCA) | `activation_manifold`, `path_steering` (loaded onto the featurizer) |
| `features/training_features.safetensors` | key `features` `(N, k)` — projected | `activation_manifold` (manifold fit) |
| `features/raw_features.safetensors` | key `features` `(N, d_model)` — un-projected | `path_steering` **linear** path mode (absent → linear mode is silently skipped) |
| `train_dataset.json` | counterfactual examples, row-aligned to features | downstream dataset alignment |
| `metadata.json` | flat `{analysis, mode, method, k_features, layer, token_position, …}` | discovery (`load_subspace_metadata`) + provenance |
| `visualization/features_{2d,3d}.{pdf,png,html}` | scatter plots | human reference |
| `das/` (DAS) | trained featurizer checkpoint | `activation_manifold` |

For `method: fixed`, `metadata.json` records `method: "pca"` (the on-disk artifact **is** a rotation matrix, so `load_subspace_onto_target` loads it via the PCA branch — no fixed-specific consumer code), and preserves the fixed origin under `discovery: "fixed"` + `fixed_source`.
