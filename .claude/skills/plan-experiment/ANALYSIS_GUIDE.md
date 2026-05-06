# Analysis Guide

Reference for the 8 analyses: what research question each answers, the dependency chain, how to write runner YAMLs, and which parameters require decisions.

---

## Analysis Dependency DAG

```
baseline ──────────────────────────────────────────────────────────────────►
    │
    ├──► locate ──────────────────────────────────────────────────────────►
    │        │
    │        └──► subspace ──────────────────────────────────────────────►
    │                 │
    │                 └──► activation_manifold ────────────────────────►
    │                              │
    │                              ├──► path_steering
    │                              └──► pullback ◄────────────────────┐
    │                                                                  │
    └──► output_manifold ──────────────────────────────────────────────┘

attention_pattern   (independent — no upstream dependencies)
```

**Always run `baseline` first.** Each analysis auto-discovers upstream outputs when config values are set to `null` (see Auto-Discovery section below).

---

## Research Questions → Analyses

| Analysis | Research question | Depends on | Key outputs |
|---|---|---|---|
| `baseline` | Can the model solve the task? Are counterfactual generators well-formed? | — | `accuracy.json`, `per_class_output_dists.safetensors`, `counterfactual_sanity.json` |
| `locate` | Which (layer, token_position) cell encodes each causal variable? | `baseline` (reuses output distributions) | `{variable}/heatmap.pdf`, `{variable}/results.json` with `best_cell` |
| `subspace` | What k-dimensional subspace captures the variable's representation? | `locate` (auto-resolves `best_cell`) | rotation/transform `.pt`, `training_features.safetensors`, 3D scatter `.html` |
| `activation_manifold` | What is the geometric structure of activations as the variable varies? | `subspace`, `baseline` | `manifold_spline/ckpt_final.pt`, `manifold_3d.html`, `reconstruction_kl` metric |
| `output_manifold` | What is the geometry of output distributions on the probability simplex? | `baseline` | `manifold.pt`, `output_manifold_3d.html` |
| `path_steering` | Does the subspace/manifold faithfully preserve causal structure? (isometry, coherence, conformal) | `subspace`, `activation_manifold` | per-metric scores and plots, `path_visualization/` |
| `pullback` | What activation trajectories realize prescribed belief-space paths? | `activation_manifold`, `output_manifold` | `belief_paths/`, `embedding_paths/`, `results.json` |
| `attention_pattern` | Which attention heads attend to which token types? | — | per-head heatmaps, `results.json` with entropy/max-attention stats |

---

## Runner YAML Patterns

Runner configs are primary Hydra configs that live in `causalab/configs/` and are selected with `--config-name <name>`. Each inherits `base` for shared globals, declares task and model directly (no `override`), and lists analysis steps in the `defaults:` block.

### Minimal single analysis (baseline only)
```yaml
defaults:
  - base
  - task: weekdays
  - model: llama31_8b
  - analysis/baseline
  - _self_

task:
  n_train: 200
  n_test: 100
```

### Two-step pipeline (baseline → locate)
```yaml
defaults:
  - base
  - task: weekdays
  - model: llama31_8b
  - analysis/baseline
  - analysis/locate
  - _self_

task:
  n_train: 200
  n_test: 100
  target_variables: [day, offset]

locate:
  method: interchange
  layers: [0, 4, 8, 12, 16, 20, 24, 28]
  mode: centroid
```

### Full pipeline (baseline → locate → subspace → activation_manifold)
```yaml
defaults:
  - base
  - task: weekdays
  - model: llama31_8b
  - analysis/baseline
  - analysis/locate
  - analysis/subspace
  - analysis/activation_manifold
  - _self_

task:
  n_train: 200
  n_test: 100
  target_variables: [day]

locate:
  method: interchange
  layers: [0, 4, 8, 12, 16, 20, 24, 28]
subspace:
  method: pca
  k_features: 8
  # layers: null → auto-resolves best_cell from locate
activation_manifold:
  method: spline
  smoothness: 0.0
  # subspace: null → auto-discovers from subspace step

post:
  - type: variable_localization_heatmap
    source_step: locate
    source_method: interchange
```

### Custom step names

Each analysis lives at the package declared in its `# @package <name>` directive — typically the analysis name itself. Mounting the same analysis under two different keys in one runner is no longer supported; if you need to sweep an analysis's hyperparameters, run it twice as separate experiments (or via a Hydra multi-run sweep) rather than chaining two copies in one config.

### Verify the resolved config before running
```bash
uv run python -m causalab.runner.run_exp --config-name {config_name} --cfg job
```

This prints every resolved parameter. Review it before executing the run.

---

## Running

The wrapper script `scripts/run_exp.sh` is the single entry point for both inline and slurm runs. It passes `--config-name <name>` to Hydra and dispatches to `causalab.runner.run_exp`.

### Inline (laptop or interactive cluster session)

```bash
./scripts/run_exp.sh he_locate
```

### CLI overrides

Hydra overrides have the highest precedence and can override anything set by the runner config. Use them for one-off variations without editing the YAML:

```bash
./scripts/run_exp.sh he_locate locate.layers=[15,20]
./scripts/run_exp.sh weekdays_pipeline subspace.k_features=16
```

### Debug pattern (small dataset, few layers)

Always run a small debug pass before the full run:

```bash
./scripts/run_exp.sh he_locate task.n_train=16 task.n_test=16 locate.layers=[0,8,16]
```

Once it produces sane artifacts, drop the overrides and run the full preset.

### Slurm dispatch

Add `--slurm`. Resources resolve from the model config's `slurm.gpus` and the runner's `slurm.time` (default in `causalab/configs/base.yaml`). CLI flags `--gpus`, `--time`, `--qos` override the resolved values.

```bash
./scripts/run_exp.sh --slurm age_8b_k64
./scripts/run_exp.sh --slurm --qos=opportunistic --time=08:00:00 alphabet_70b_k128
```

### Where artifacts land

Per ARCHITECTURE.md §3 invariant 7, all outputs go to `${experiment_root}/{analysis}/...`. In dev mode `experiment_root` defaults to `artifacts/{task}/{model}`; in research mode pass `--experiment-root ${SESSION_DIR}/artifacts/{task}/{model}` so outputs land inside the active session. Re-running the same runner config rewrites that directory — there is no automatic timestamping. To preserve an old run, copy the artifact directory aside or branch the runner config and rename it before re-running.

---

## Common Pitfalls

### `task.resample_variable` × `locate.mode` interaction

`locate.mode: pairwise` is only informative when `task.resample_variable` is set to a single variable name (the variable being localized). Using `pairwise` with `resample_variable: "all"` produces near-zero scores everywhere except the last token. See **ARCHITECTURE.md §5** for the full explanation. Centroid mode (`locate.mode: centroid`) remains meaningful under `resample_variable: "all"` — use it for variable-agnostic localization.

### `balanced` precedence

`task.balanced: true` overrides `resample_variable` when both are set. If you need single-variable counterfactuals, leave `balanced: false`.

---

## Auto-Discovery

When a config param is set to `null` (or omitted, taking the analysis default of `null`), the runner searches for prior outputs automatically:

| Analysis | Config param | What gets auto-discovered |
|---|---|---|
| `subspace` | `layers: null` | `best_cell` from `{experiment_root}/locate/interchange/{variable}/results.json` |
| `subspace` | `token_positions: null` | all defined task positions |
| `activation_manifold` | `subspace: null` | most recent subspace dir under `{experiment_root}/subspace/` |
| `activation_manifold` | `layers: null` | `best_cell` from subspace metadata |
| `path_steering` | `subspace: null` | most recent subspace dir |
| `path_steering` | `activation_manifold: null` | most recent activation_manifold dir |
| `pullback` | `activation_manifold: null` | most recent activation_manifold dir |
| `pullback` | `belief_path.output_manifold_ckpt: null` | most recent output_manifold dir (the output_manifold analysis is required — no fallback) |

**Prefer `null` over hardcoded paths.** It makes runner configs portable and avoids breaking when artifact directories change.

---

## Key Parameter Decisions

Only the parameters that require a decision are listed here. Everything else defaults sensibly from `configs/analysis/{name}.yaml`.

### `baseline`
- **`n_train` / `n_test`** (in `task:`): dataset sizes. Use `enumerate_all: true` to exhaust the full combinatorial space.
- **`batch_size`**: inference batch size (default 32; increase if GPU memory allows).

### `locate`
- **`method`**: `interchange` (fast, no training, recommended first pass) vs `dbm_binary` (trained masks, finds minimal component set).
- **`mode`**: `centroid` (default — distributions are compared class-by-class; works with any counterfactual type) vs `pairwise` (compares base to CF output exactly; only informative when `task.resample_variable` is set to the specific variable being localized — see ARCHITECTURE.md §5).
- **`layers`**: Start with a coarse scan (every 4th layer). Narrow to the relevant region for a fine-grained follow-up run.
- **`token_positions`**: `null` scans all task-defined positions. Specify a subset to save time once you know where to look.

### `subspace`
- **`method`**: `pca` (no training, fast); `das` (supervised linear subspace, learns rotation that maximizes causal alignment); `dbm` (binary mask over feature dimensions).
- **`k_features`**: subspace dimensionality. Rule of thumb: 2–3× the number of distinct variable values for PCA; can be lower for DAS/DBM. Start with 8–16 and compare.

### `activation_manifold`
- **`smoothness`**: TPS regularization. `0.0` = exact interpolation through all centroids. Increase (e.g., `1.0`, `10.0`) if the manifold is noisy or classes are close together.
- **`skip_decoding_eval: true`**: skips the reconstruction steering test (saves significant time during exploration). Enable it for final results.
- **`subspace`**: explicit subspace dir name (e.g., `"pca_k8"`) if you have multiple subspace runs and want to select one; otherwise `null` auto-discovers.

### `path_steering`
- **`eval_criteria`**: which distortion metrics to compute. Start with `["isometry"]`. Add `["coherence", "conformal"]` for a full characterization.
- **`path_modes`**: comparison paths to evaluate (default `[geometric, linear]`).

### `output_manifold`
- **`intrinsic_mode`**: `pca` (default; uses Hellinger PCA coordinates) vs `parameter` (uses causal model parameter coordinates — better when parameters are meaningful ordinal values like months or weekdays).
- **`smoothness`**: same as activation_manifold.

### `pullback`
- **`belief_path.n_steps`**: target-path resolution (number of waypoints, including endpoints).
- **`selected_pairs`** / **`max_pairs`**: which `(start, end)` class pairs to compute. `selected_pairs: null` samples up to `max_pairs` deterministically.
- **`embedding_optim.*`**: Phase 2 trajectory optimizer knobs (optimizer name, n_steps, k_opt, etc.). See `causalab/configs/analysis/pullback.yaml`.
- The `output_manifold` analysis must have run first — pullback loads its checkpoint unconditionally.

### `attention_pattern`
- **`layers`** / **`heads`**: `null` = all. Specify subsets to target a known region.
- **`source_token_types`** / **`target_token_types`**: task position names (e.g., `["io_name", "s_name", "verb"]`) to compute token-type attention statistics.
