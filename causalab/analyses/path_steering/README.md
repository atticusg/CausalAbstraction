# Path Steering

Path steering answers: *does traversing the learned activation manifold between two
class centroids actually steer the model's output distribution the way the
manifold's geometry predicts — and does the geometric (geodesic) path do it better
than a straight line?* For each pair of intervention-value centroids it builds a
steering path (a **geodesic** in the manifold's intrinsic coordinates, or a
**linear** straight line in raw/PCA space), injects each point along the path into
the residual stream, collects the output distribution at every step over several
base prompts, and scores the resulting path family under geometric-quality
criteria (`coherence`, `distance_from_behavior_manifold`, `isometry`). Path modes
are compared with paired t-tests and rendered as interactive activation↔belief
manifold viewers.

Path steering **requires** prior runs of `subspace` and `activation_manifold` (it
loads their composed featurizer) and, for the belief-relative criteria and
visualizations, `output_manifold`. It is a terminal analysis — no downstream
analysis consumes its outputs. (Other analyses' docs refer to this stage by its
former name, **`evaluate`**.)

---

## Overview

```
   subspace (PCA rotation)      activation_manifold (spline)     output_manifold (optional)
          │                            │                                │
          └────── composed featurizer ─┘                                │  belief manifold
                        │                                               │  (Hellinger PCA + spline)
         per-class centroids  (intrinsic u-space / raw / PCA)           │
                        │                                               │
   ┌────────────────────────────────────────────────┐                  │
   │ For each class pair (ci, cj) and each path mode:  │                  │
   │   geometric → geodesic in intrinsic u-space       │                  │
   │   linear    → straight line in raw activation     │                  │
   │   build_path → grid_points  [num_steps, d]        │                  │
   └────────────────────────────────────────────────┘                  │
                        │                                               │
     inject each grid point into the residual stream                    │
     (collect_grid_distributions over n_prompts base prompts)           │
                        ▼                                               │
   pair_distributions  [n_pairs, num_steps, n_prompts, W]               │
                        │                                               │
        ┌───────────────┼────────────────────────────────┐            │
        ▼               ▼                                  ▼            │
   coherence    distance_from_behavior_manifold      isometry ◄─────────┘
   P(concept)   cumulative Bhattacharyya to the      corr(activation-geometry,
   along path   belief manifold                       belief-geometry)
        └───────────────┴────────────────────────────────┘
                        │
          paired t-tests (geometric vs linear)
                        │
   viz: 3D paths · belief-space · dual_manifold · (receptive_field, opt-in)
```

Each path point is an interchange intervention: it replaces only the
featurizer's projection of the residual stream (the manifold coordinate) and
keeps the orthogonal complement intact, so the collected distribution reflects a
real model forward at that coordinate. The `geometric` mode walks the geodesic in
the manifold's intrinsic space (with periodic shortest-arc handling per dim);
`linear` walks a straight chord in raw activation space; `linear_subspace` walks a
straight line in the PCA subspace. `coherence` and
`distance_from_behavior_manifold` are path-based (scored on the collected
distributions); `isometry` is a pure property of the two learned manifolds and
needs no model forward.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — `experiment_root`, `seed`,
`figure_format`. The task config must set `colormap` and `colormap2` (declared in
`REQUIRED_TASK_KEYS`); the belief-space and dual-manifold viewers additionally
read `task.color_by_dim` and the task's dataset knobs (`n_train`, `n_test`,
`enumerate_all`, `resample_variable`) to regenerate row-aligned class labels.

**Module config** (`causalab/configs/analysis/path_steering.yaml`):

```yaml
# @package path_steering
_name_: path_steering
_output_dir: ${experiment_root}/path_steering

eval_criteria:                   # which quality criteria to compute
  - isometry
  - coherence
  - distance_from_behavior_manifold

path_modes:                      # which steering paths to build + compare
  - geometric                    # geodesic in intrinsic manifold coords
  - linear                       # straight line in raw activation space
                                 # (also available: linear_subspace — straight line in PCA space)

# --- Pair selection ---
selected_pairs: null   # null = unbiased sample (writes aggregate stats); a list of
                       # [start, end] string pairs computes ONLY those (aggregate
                       # criteria writes are then skipped — a subset can't be unbiased)
max_pairs: 50          # consulted only when selected_pairs is null; subsample if total exceeds (deterministic via seed)
n_extra_pairs: 0       # extra cross-path pairs (support union >= 3) for path-based criteria
n_prompts: 16          # base prompts per centroid pair
n_eval_samples: 100    # counterfactual samples generated before taking the first n_prompts
batch_size: 32
subspace: null                 # null = auto-discover under {experiment_root}/subspace/
activation_manifold: null      # null = auto-discover under {experiment_root}/activation_manifold/<ss>/
encode_mode: nearest_centroid  # or "nearest_point" (Gauss-Newton projection)

# --- Run modes ---
replot_only: false        # true = render plots from cached pair_distributions only (requires selected_pairs)
recompute_isometry: false # true = recompute only isometry from cached manifolds (no forward passes)

# Shared defaults for path-based metrics
num_steps_along_path: 50  # waypoints per path (endpoints included)

isometry:
  n_arc_steps: 150            # arc-length integration resolution
  distance_function: null     # override task default; null = task config
  n_interior_per_pair: 0      # K equispaced interior points per centroid-pair geodesic (0 = centroids only)

oversteer:
  frac: 0.0        # 0 = disabled; how far past the target to continue (fraction of path length)
  num_steps: 10    # steps in the overshoot region

# --- Visualization ---
visualization:
  figure_format: ${figure_format}   # png or pdf — invariant 6

visualizations:                # which viz to render (dispatched by name in main.py)
  - path_visualization
  - isometry_visualization
  - dual_manifold
  # `receptive_field` is OPT-IN: it runs NEW forward passes (grid_res^2 * n_prompts)

path_visualization:
  colormaps: [${task.colormap}, ${task.colormap2}]
  path_colors: {geometric: black, linear: darkgray}
  full_vocab_softmax: true
  pca_components: null          # PCA indices for the 3D path plots (length 3); null = [0, 1, 2]
  colored_concepts_in_legend: null  # highlight only these concept values; null = color all
  figsize: null                # [width, height] inches for per-pair plots; null = defaults
  font_scale: 1.0              # multiplier on per-pair plot font sizes

isometry_visualization:
  figure_format: ${..visualization.figure_format}
  n_mds_components: 3
  hover_label_format: grid_coords

# `dual_manifold` takes no per-viz config — colormap / color_by_dim come from task_cfg.

receptive_field:               # opt-in decision map over the top-2 subspace PCs
  grid_res: 11                 # cells per axis; cost = grid_res^len(pca_components) * n_prompts forwards
  pca_components: [0, 1]       # 2 dims -> 2D field; [0, 1, 2] -> 3D field
  n_prompts: 8                 # base prompts averaged per cell
  full_vocab_softmax: true     # confidence = max class prob
  encode_confidence: true      # confidence -> per-cell opacity
  show_scatter: true           # training point-cloud overlay
  show_centroids: true         # class-centroid diamonds
  show_paths: true             # geo + linear path overlay (dropdown per pair)
  range_pad: 0.05              # fractional padding on each PC axis range
```

**Criteria (`eval_criteria`)** — `coherence` is the fraction of output mass on the
concept tokens along the path (higher is better); `distance_from_behavior_manifold`
is the cumulative Bhattacharyya distance from each step to the belief manifold
(lower is better; needs `output_manifold`); `isometry` correlates
activation-manifold geometry with belief-manifold geometry (higher is better;
needs `output_manifold`).

**Path modes (`path_modes`)** — listing two or more triggers the paired t-tests
that compare them. `linear_subspace` collapses to `linear` for the isometry metric.

**Pair selection** — a `null` `selected_pairs` runs the unbiased sample and writes
all aggregate stats; a cherry-picked list computes only those pairs and skips
aggregate writes (a subset cannot yield unbiased statistics). Per-pair caches are
still merged so a later unbiased run reuses the work.

**Run modes** — `replot_only` re-renders from cached distributions without a model
load; `recompute_isometry` recomputes only the isometry score from cached
manifolds. They are mutually exclusive.

Each `(subspace, activation_manifold)` combination discovered under
`experiment_root` is evaluated into its own output subtree (see below).

---

## Outputs

Output root is
`{experiment_root}/path_steering/{subspace}/{activation_manifold}/[{target_variable}/]`.

### Interpretation

- **`results_summary.csv`** — The headline answer, one flat `criterion,value` row
  per metric/reduction. Read `coherence/{mode}/mean` (how much output mass stays
  on-target along the path — higher is a cleaner steer),
  `distance_from_behavior_manifold/{mode}/mean` (how far the path drifts off the
  behavior manifold — lower is more natural), and `isometry/{mode}/pearson_r` (does
  the activation manifold's geometry match the belief manifold's — closer to 1 is
  better). Skipped under cherry-pick (`selected_pairs`).

- **`criteria/{coherence,distance_from_behavior_manifold}/paired_ttest_geometric_vs_linear*.json`**
  — The key comparison: *did the geodesic beat the straight line?* Reports
  `mean_diff`, `t_statistic`, `p_value`, and the per-pair samples. For coherence a
  positive `geometric − linear` `mean_diff` means the geodesic keeps more on-target
  mass; for distance a negative `mean_diff` means the geodesic stays closer to the
  behavior manifold. A significant result here is the primary evidence that
  manifold geometry, not just endpoint direction, matters for steering.

- **`vis/dual_manifold.html`** (and `_bars` / `_gridview` variants) — Interactive
  side-by-side activation↔belief viewer: pick a class pair, drag the slider, and
  watch the point move along the path in both spaces at once. A good result: moving
  along the geodesic sweeps smoothly and monotonically from the start class region
  to the end class region in belief space. A bad result: the belief-space marker
  sits near the center / doesn't track the class transition (a degenerate or
  off-manifold path).

- **`vis/paths/3d_paths/{a}_{b}.html`** — 3D activation-space plot of the geodesic
  (black) and linear (gray) paths on the manifold mesh, colored by class. The
  geodesic should hug the manifold; a linear chord that leaves it shows what the
  geometry buys you.

- **`vis/paths/{mode}/pair_{a}_{b}.{png,pdf}`** — Per-pair P(class) vs. path
  fraction `alpha`; the on-target class should rise and the off-target class fall
  as you move along the path.

- **`vis/belief_space/*`** and **`vis/isometry/{mode}/*`** — Paths projected into
  Hellinger belief space, and the isometry scatter of activation-distance vs.
  belief-distance (points should fall on a line if the manifolds are isometric).

- **`vis/receptive_field.html`** (opt-in) — Decision map over the top-2 subspace
  PCs: each grid cell colored by the argmax output class, with the point cloud,
  centroids, and both path families overlaid.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `results_summary.csv` | `criterion,value` rows | human reference |
| `criteria/coherence/{mode}/metrics.json` | `{mean, se, worst_mean, worst_se}` | human reference |
| `criteria/coherence/{mode}/per_pair_scores{,_worst}.safetensors` (+`.meta.json`) | `[n_pairs]` tensor | paired t-tests |
| `criteria/distance_from_behavior_manifold/{mode}/metrics.json` | `{mean, se, geodesic_mean, geodesic_se}` | human reference |
| `criteria/distance_from_behavior_manifold/{mode}/per_pair_scores{,_geodesic}.safetensors` | `[n_pairs]` tensor | paired t-tests |
| `criteria/isometry/{mode}/tensors.safetensors` + `metrics.json` + `metadata.json` | `D_X`, `D_Y`, grid; `{pearson_r, n_pairs}` | `isometry_visualization` |
| `criteria/{metric}/paired_ttest_{a}_vs_{b}{suffix}.json` | `{mean_diff, t_statistic, p_value, samples, differences}` | human reference |
| `paths/{mode}/pair_distributions.safetensors` | `[n_pairs, num_steps, n_prompts, W]` + `pair_grid_points` | cache; `replot_only`; viz |
| `paths/{mode}/pairs.json` | `{pairs, values, n_normal_steps}` | cache index |
| `paths/{mode}/pair_distributions_extra.safetensors` + `extras_meta.json` | `[n_extra, num_steps, n_prompts, W]` | cache (`n_extra_pairs > 0`) |
| `receptive_field/receptive_field.safetensors` + `.json` | grid + scatter/centroid/path overlays | `receptive_field` viz |
| `vis/**.html`, `vis/paths/**.{png,pdf}` | interactive + static plots | human reference |
| `metadata.json` | run config snapshot + full `criteria` results | provenance |

`W` is the number of concept (intervention) values; `num_steps` is
`num_steps_along_path` plus any oversteer steps. `pair_distributions` is collected
with `full_vocab_softmax` over the concept slice, so each step's `W` entries sum to
≤ 1 and the deficit is off-target ("other") mass. `{mode}` is one of `geometric`,
`linear`, `linear_subspace`; the linear baseline (`activation_manifold == linear`)
skips the manifold-only visualizations.
