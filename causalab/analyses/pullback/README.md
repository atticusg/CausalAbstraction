# Pullback

Pullback answers: *given a desired path through belief space, what trajectory through the model's activation space realizes it?* It optimizes a trajectory `h(t)` in a k-dimensional PCA subspace such that injecting `h(t)` into the model's residual stream produces a sequence of output distributions matching a target belief path between two class centroids. Two phases run in sequence: a **belief path** is traced through the trained output manifold between the centroids, then an **embedding path** is fit so the model's outputs follow that belief path.

## Prerequisites

Pullback **requires** prior runs of `subspace`, `activation_manifold`, and `output_manifold`. The output manifold checkpoint is loaded unconditionally — there is no fallback. If `belief_path.output_manifold_ckpt` is `null` (the default), pullback auto-discovers it under `{experiment_root}/output_manifold/`; otherwise set it explicitly.

No downstream analysis consumes pullback's outputs — it is a terminal diagnostic.

---

## Overview

```
                     PHASE 1 — BELIEF PATH
                     =====================
  output_manifold              per-class centroids
  (trained spline)             (from per_example_output_dists)
         |                              |
         └─────────── manifold_trace ───┘
                       (linear interp in
                        u-space, decode
                        through spline)
                            v
                    p(t) ∈ Δ^W   for t ∈ [0, 1]

                     PHASE 2 — EMBEDDING PATH
                     ========================
  PCA subspace          activation manifold (spline)
         |                       |
         v                       v
  ┌─────────────────────────────────────────┐
  │  For each (ci, cj) pair:                │
  │    init h(t)  →  optimize via           │
  │    L-BFGS / Adam / NEB / basin-hopping  │
  │    with FeatureInterpolateIntervention  │
  │    loss = d(model(h(t_i)), p(t_i))²     │
  └─────────────────────────────────────────┘
                    v
              h(t) ∈ R^k   for t ∈ [0, 1]
```

The Phase 2 forward pass uses pyvene's `IntervenableModel` with `FeatureInterpolateIntervention`: at each timestep the intervention replaces only the k-dim PCA projection of the residual stream and keeps the orthogonal complement intact, so gradients flow back to `h(t)` through a real model forward.

Two reference paths are evaluated alongside the optimized one: **`geometric`** (geodesic on the trained activation manifold, decoded with the full featurizer) and **`linear`** (straight line in PCA space). Geometric distributions are collected with the full PCA→standardize→manifold featurizer; linear distributions and the optimization itself use the PCA-only featurizer.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — shared params used by this analysis:
- `experiment_root` — output root (default: `artifacts/${task.name}/${model.id}`)
- `seed` — used for pair subsampling and counterfactual generation
- `task.n_train`, `task.n_test`, `task.balanced`, `task.enumerate_all`, `task.resample_variable` — read to regenerate the `output_manifold` training dataset for true-class assignment of natural distributions

**Module config** (`causalab/configs/analysis/pullback.yaml`):

```yaml
analysis:
  _name_: pullback

  subspace: null                  # null = auto-discover under {experiment_root}/subspace/
  activation_manifold: null       # null = auto-discover under {experiment_root}/activation_manifold/<ss>/

  belief_path_only: false         # true = skip Phase 2 entirely
  skip_optimization: false        # run Phase 2 setup but no optimizer (still collects geo/linear)

  selected_pairs: null            # null = all (W choose 2) pairs; or list of [start, end] strings
  max_pairs: 25                   # subsample if total exceeds; deterministic via seed
  n_prompts: 16                   # samples per (base_class, cf_class) pair
  batch_size: 32

  # ── Phase 1: belief path between class centroids ────────────────────────────
  belief_path:
    n_steps: 20                   # waypoints along the path (incl. endpoints)
    output_manifold_ckpt: null    # null = auto-discover under {experiment_root}/output_manifold/

  # ── Phase 2: embedding path in k-dim PCA subspace ───────────────────────────
  embedding_optim:
    n_steps: 20                   # may differ from belief_path.n_steps; targets are resampled
    init: knn_graph               # "linear" or "knn_graph"
    k_init: 20                    # k for embedding-space graph init
    fix_start: false
    fix_end: false
    path_length_weight: 0         # ||h_{i+1} - h_i|| penalty (0 = off)
    convergence_window: 2
    convergence_tol: 0.001
    fwd_chunk_size: 4             # batch this many timesteps per forward pass (0 = sequential)
    max_path_snapshots: 1         # optimization snapshots saved for visualization (0 = all)
    log_spaced_snapshots: true
    trajectory_param: free_points # "free_points" or "tps"
    name: lbfgs                   # "lbfgs", "adam", "basin_hopping", or "neb"
    spring_k: 0.1
    tps:
      n_control_points: 10
      smoothness: 0.0
    lbfgs:
      steps: 50
      lr: 1.0
      max_iter: 5
      line_search_fn: strong_wolfe
    adam:
      steps: 250
      lr: 0.5
    basin_hopping:
      n_hops: 50
      step_size: 0.5
      temperature: 1.0
      n_candidates: 5
      lbfgs_steps: 5
      lbfgs_lr: 1.0
      lbfgs_max_iter: 20
      lbfgs_line_search_fn: strong_wolfe

  visualization:
    figure_format: pdf            # "pdf" or "png" — static figure format
    belief_trajectories: true     # per-pair belief trajectory plots
    embedding_diagnostics: true   # per-pair embedding conformal diagnostics
    embedding_3d: true            # 3D activation-space trajectories
    hellinger_pca_3d: true        # Hellinger-PCA belief-space overlay
    plot_belief_target: true      # overlay the fitted belief-target (red) on the Hellinger PCA plot

  path_modes:                     # comparison paths to evaluate alongside the optimized one
    - geometric
    - linear
```

The belief path is built by linearly interpolating the two class centroids' coordinates on the trained `output_manifold` (with periodic shortest-arc wrap for periodic dims) and decoding back to the simplex — no optimization. This requires the `output_manifold` analysis to have been run first.

---

## Outputs

Output root is `{experiment_root}/pullback/{subspace}/{activation_manifold}/[{target_variable}/]`.

### Interpretation

- **`visualization/activation_paths/activation_{ci}_{cj}.html`** — Interactive 3D plot per pair. Optimized trajectory (teal), `geometric` (blue), and `linear` (dark blue) overlaid on the activation manifold mesh and training features colored by class. A healthy optimized path stays close to the manifold and clears the same class basins in order; a path that shoots off into off-manifold space indicates the optimizer found a degenerate solution that satisfies the belief target via residual leakage rather than on-manifold travel.

- **`visualization/belief_space/belief_space_{ci}_{cj}.html`** — Same trajectories projected into Hellinger-PCA belief space, overlaid on the trained output manifold. Use this to confirm the optimized path actually realizes the target belief sequence — its markers should fall on top of the target's. Divergence here means the optimizer failed to hit the belief target even though Phase 1 chose a reachable path.

- **`visualization/belief_paths/{target_path,optimized,geometric,linear}/…`** — Per-pair line plots of per-class probability over `t ∈ [0, 1]` for each path family. The `optimized` panel should track `target_path` closely (this is what Phase 2 minimizes); `geometric` and `linear` reveal how much the manifold or a straight line already get you for free.

- **`metrics/path_recapitulation_{label}.json`** — Three parameterization-invariant scalars per pair, comparing `optimized` and `linear` paths to the activation-manifold geodesic (`v_geo`) as a curve in PCA k-space. Each metric uses orthogonal projection onto the v_geo polyline (closest point on segments, not 1-NN to vertices), so it does not depend on how the paths are parameterized.
  - **`r_squared`** — `1 − RSS/TSS`, clamped to `[0, 1]`. RSS = sum of squared orthogonal residuals from each path point to v_geo's polyline; TSS = total variance of the path around its own centroid. The "fraction of the path's structure explained by v_geo's curve". Primary headline metric.
  - **`mean_dist_from_geometric`** — Mean closest-point distance from each path point to v_geo. In PCA k-space units; useful for absolute magnitude.
  - **`arc_length_ratio`** — `L(path) / L(v_geo)`. ~1 means comparable arc length; <1 indicates a shorter path or partial coverage; >1 indicates a detour or stuttering.
  - A paired t-test on `r_squared` (optimized vs linear) is reported under `paired_t_test_r_squared_optimized_vs_linear` — the "did optimization beat the trivial chord at recapitulating v_geo?" question.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `geodesic_paths.pt` | dict `(ci, cj) → [n_steps, W+1] tensor` | Phase 2 input; visualization |
| `optimization_results.pt` | dict `(ci, cj) → result_dict` | visualization; recapitulation metrics |
| `metrics/path_recapitulation_{label}.json` | `{n_pairs, optimized: {metric: {mean, se}}, linear: {…}, per_pair: {…}, paired_t_test_r_squared_optimized_vs_linear}` | quantitative comparison |
| `visualization/activation_paths/*.html` | Plotly HTML, one per pair | human reference |
| `visualization/belief_space/*.html` | Plotly HTML, one per pair | human reference |
| `visualization/belief_paths/{target_path,optimized,geometric,linear}/*` | static figures, format from `visualization.figure_format` | human reference |
| `metadata.json` | run config snapshot | provenance |

`result_dict` (per-pair entries inside `optimization_results.pt`) contains:
- `v_optimized_k`, `v_geometric_k`, `v_linear_k` — `[n_steps, k]` activation-space trajectories
- `opt_probs_raw`, `geo_probs_raw`, `lin_probs_raw` — `[n_steps, n_samples, W]` per-sample concept probabilities (for error bars)
- `opt_probs_AW1`, `geo_probs_AW1`, `lin_probs_AW1` — `[n_steps, W+1]` mean concept-plus-other distributions
- `p_target_AW1`, `p_start`, `p_end` — `[n_steps, W+1]` belief target and `[W+1]` centroid endpoints
- `loss_history`, `per_step_loss`, `path_history_k` — optimization diagnostics
- `recapitulation_optimized`, `recapitulation_linear` — `dict[str, float]` with keys `r_squared`, `mean_dist_from_geometric`, `arc_length_ratio`
- `base_class`, `cf_class`, `n_samples`, `f_start_k`, `f_end_k` — pair metadata
