# Architecture

## 1. Package Structure

| Module | Named for |
|---|---|
| `causal/` | causal model primitives |
| `tasks/` | task definitions |
| `neural/` | neural network interface |
| `methods/` | interpretability methods |
| `io/` | disk I/O + shared plotting primitives |
| `analyses/` | research analyses |
| `runner/` | run orchestration |

**Dependency flow:** `tasks/` and `causal/` are independent. `neural/` depends on neither. `io/` depends on `neural/`, `tasks/`, and `causal/` only — it is the lowest application layer above third-party libs. `methods/` depends on `neural/`, `causal/`, and `io/`. `analyses/` depends on all four. `runner/` is a thin shell over `analyses/`.


## 2. Runner Config System

The runner uses [Hydra](https://hydra.cc/) for configuration. Configs live in `causalab/configs/`.

### Config groups — module defaults

Each config group holds defaults for one concern. Switching a group swaps the entire module config:

```
configs/
├── config.yaml        # root: sets default group selections + experiment_root
├── task/              # one file per task (weekdays, months, age, ...)
├── model/             # one file per model (llama31_8b, gpt2, ...)
├── analysis/          # one file per analysis type — default values only (see below)
└── runners/            # named full-run presets (see below)
```

Shared global fields (`seed`, `experiment_root`, Hydra output settings) live in `base.yaml` and are inherited by both `config.yaml` and every runner config.

### Analysis configs — self-contained defaults

Each file in `analysis/` (e.g. `subspace.yaml`, `locate.yaml`) defines the complete set of defaults for one analysis type. Each file's **first line is `# @package <name>`** — Hydra mounts the file at `cfg.<name>` automatically. Every parameter has a **concrete default value** — no root-level interpolation for shared params:

```yaml
# analysis/subspace.yaml
# @package subspace
_name_: subspace
_subdir: ${.method}_k${.k_features}
_output_dir: ${experiment_root}/subspace/${._subdir}
method: pca
k_features: 32      # concrete default, not ${k_features}
batch_size: 32       # concrete default, not ${batch_size}
```

Analysis configs contain only analysis-specific defaults. Dataset construction parameters (`n_train`, `n_test`, `enumerate_all`, `balanced`) live in task configs and are read via `cfg.task.*`. The global `seed` lives in `config.yaml` and is read via `cfg.seed`. See invariant 12 in §3.

Analysis configs use *relative* OmegaConf interpolations (`${.method}`, `${._subdir}`) for self-references so they resolve correctly inside their own package. References to task/model config groups (`${experiment_root}`, `${task.colormap}`) are absolute and resolve from the root.

### Runner configs — full-run presets

Runner configs are **primary Hydra configs** that live directly in `causalab/configs/` and are selected with `--config-name <name>`. Each runner inherits `base.yaml` for shared globals, declares its own task and model, and pulls analyses by listing them in the defaults list:

The defaults line `- analysis/locate` tells Hydra: *load `analysis/locate.yaml` and merge it at the package declared by its `# @package` directive* (here, `cfg.locate`). The body's `locate.layers` override then merges on top. Multi-step pipelines have one defaults entry per step:

```yaml
# configs/runners/weekdays/weekdays_8b_pipeline.yaml — multi-step pipeline
defaults:
  - base
  - task: natural_domains_arithmetic_weekdays
  - model: llama31_8b
  - analysis/locate
  - analysis/subspace
  - _self_

locate:
  layers: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
subspace:
  method: pca
  k_features: 8

post:
  - type: variable_localization_heatmap
    source_step: locate
    source_method: interchange
```

Order in the defaults list = order of execution. The runner discovers analyses at runtime by scanning top-level keys of `cfg` for entries that carry the `_name_` sentinel — OmegaConf preserves insertion order, so the chain runs in the order you wrote.

### Running experiments

```bash
# Run a named preset
scripts/run_exp.sh he_locate

# Introspect full resolved config before running
scripts/run_exp.sh he_locate --cfg job

# Dispatch as a slurm batch job (resources resolved from the runner config:
scripts/run_exp.sh --slurm he_pipeline
```

Internally `run_exp.sh` passes `--config-name <name>` to Hydra, making the runner config the primary config for that run. CLI overrides have the highest precedence and can override anything set by the preset.

### Execution model

`run_exp.py` iterates the analysis chain by walking top-level keys of `cfg` for any whose value is a `DictConfig` containing `_name_`. The order of those keys matches the order of `- analysis/<name>` entries in the runner's defaults list. Each step's config is passed to its `main(cfg, analysis_cfg)` entry point as the second argument — the runner does not mutate `cfg` to alias it.

**Post-steps.** The optional `post:` key holds cross-analysis visualization steps that run after all analysis steps complete. Each entry has a `type:` that maps to a handler in `causalab/runner/post_steps.py`.

## 3. Layering rule (neural/ vs methods/ vs analyses/)

`neural/` mirrors the pyvene API surface — pyvene intervention modes (`collect`, `replace`, `interchange`, `interpolate`, `mask`, etc.), hook plumbing, the `IntervenableModel` wrapper, and the base `Featurizer` interface. Nothing else.

`methods/` holds mechanistic interpretability **tools** — reusable primitives and compositions of `neural/` primitives:
- featurizer subclasses (rotation for DAS, SAE encoders, PCA, UMAP, spline/flow manifolds, noise, standardize)
- training loops over learned interventions (batching at scale is fine)
- paired-input (base/counterfactual) logic that pairs a mode with a causal hypothesis
- scoring functions and metric definitions

`analyses/` is where research questions live — dataset loading from paths, artifact-directory layout, metadata tagged with `experiment_type`, heatmap/plot generation, and any end-to-end workflow that answers "where does variable X live?" or "does steering Y work?". Each analysis is a Hydra entry point (`main(cfg)`).

### Invariants

1. **`neural/` must not import from `methods/`.** If a module in `neural/` composes a mode with a hypothesis, subclasses a featurizer, or owns a training loop, it belongs in `methods/`.
2. **`methods/` must not import from `runner/` or `analyses/`.** Methods are library code; they receive configuration as plain arguments (or a resolved `DictConfig`) from their caller. They must not reach into `configs/` for defaults. Save/load primitives shared across layers live in `causalab/io/`.
3. **`io/` must not import from `methods/`, `analyses/`, or `runner/`.** It depends only on `neural/`, `tasks/`, `causal/`, and third-party libs. This is what allows `methods/` and `analyses/` to both consume `io/` without circular edges.
4. **`methods/` must not own research-question orchestration.** No dataset loading from a path, no creation of a named artifact directory layout (`heatmaps/`, `train_eval/`, `metadata.json`, …), no metadata dicts tagged with an `experiment_type`. Those are analysis-layer concerns. Methods return in-memory result dicts; `analyses/` decides where results land. (Composite `save_*` helpers that bundle disk layout + plot for a specific method may live beside that method, but they must consume `io/` primitives only — they never re-implement artifact I/O.)
5. **Hydra is the single source of truth for configuration.** `methods/` functions must not embed hyperparameter defaults (epochs, learning rates, batch sizes, regularization coefficients, subspace dimensions, etc.). Defaults live exactly once, in `configs/`. A method either (a) takes explicit keyword arguments with no implicit fallback, or (b) accepts a pre-resolved config object. `config=None` branches that inject hardcoded defaults inside `methods/` are a violation.

6. **Static figure outputs use a standard `figure_format`.** Any function in `causalab/io/plots/` or downstream that saves a matplotlib figure to disk (PNG or PDF) accepts an optional `figure_format: Literal["png", "pdf"]` (default `"pdf"`), or resolves it via `analysis.visualization.figure_format` in Hydra configs. Callers build paths with `causalab.io.plots.figure_format.path_with_figure_format` (or strip/replace a basename extension) so format is never implicit in a hardcoded `.pdf` suffix alone. Interactive-only outputs (Plotly/HTML) are exempt.

7. **`experiment_root` is the single source of truth for output paths.** It is declared once in `causalab/configs/base.yaml` and consumed by every runner, analysis, and method. Two contexts produce two defaults:
   - **Dev mode and demo notebooks** — the Hydra default `experiment_root: artifacts/${task.name}/${model.id}`. Artifacts land in the gitignored `artifacts/` tree at the repo root.
   - **Research-session workflows** — the active session overrides `experiment_root` to `agent_logs/<session>/artifacts/${task.name}/${model.id}` so the session is a self-contained bundle of generated research. See `.claude/skills/research-session/CONVENTIONS.md`.

    Notebooks and scripts must not override `experiment_root` with `tempfile.mkdtemp()` or other ephemeral paths — artifacts written to `/tmp/` are invisible to downstream analyses and lost on reboot. Always pass a path under `artifacts/` (dev) or under the active session (research).

8. **Demo notebooks must not configure logging.** No `logging.basicConfig(...)` or `logging.getLogger().setLevel(...)` in notebooks under `demos/analyses/*/demo.ipynb`. Library code already configures its own loggers; notebook-level overrides pollute the root logger and produce noisy output that obscures the demo's purpose. Users who need verbose output can set log levels in their own session.

9. **Demo notebooks must inline interactive HTML, not use IFrame.** To display interactive Plotly HTML files, read the file content and use `display(HTML(content))` — never `display(IFrame(path))`. Cursor's notebook renderer cannot resolve local filesystem paths in `<iframe src="...">`, so IFrame-based display silently fails (shows only the object repr). The pattern is:
   ```python
   from IPython.display import HTML
   with open(html_path, 'r') as f:
       display(HTML(f.read()))
   ```

10. **Demo notebooks follow the five-section structure.** Each `demos/analyses/*/demo.ipynb` is a self-contained walkthrough: configure, run, and inspect results — all from one place. See `demos/analyses/baseline/demo.ipynb` as the reference implementation. The five sections are:

   **Section 1 — Research question and functionality.** A single markdown cell with the analysis name as a heading, followed by bolded metadata entries:
   - **Research question** — the question this analysis answers (in italics), matching the README opening paragraph.
   - **When to run this analysis** — practical guidance on when this analysis is appropriate.
   - **Dependent on artifacts from analyses** — which upstream analyses must run first (or "None").
   - **Producing artifacts for analyses** — which downstream analyses consume this analysis's outputs.
   - **Produced artifacts** — a bullet list of every output, with the filename in backticks/bold and a one-line description of what it shows.

   **Section 2 — Configuration.** Three cells:
   1. A **markdown cell** with the `## Configuration` header, a description of where the config is written, and a **table** of the key configurable fields (`| Field | What it controls |`).
   2. A **code cell containing only the YAML string** (`config_yaml = """..."""`). This is the single thing the user edits. The YAML follows the runner config structure: `defaults:` with `/base`, `/task`, `/model`, and `- analysis/<name>` entries for each analysis; plus top-level `<name>:` blocks for any overrides.
   3. A **code cell that saves the YAML** to `causalab/configs/<name>_demo.yaml` and **loads the full Hydra config**, printing the resolved result so the user can verify defaults.

   **Section 3 — Run.** A markdown cell with a one-line explanation, followed by a code cell that runs the analysis via `subprocess.run` (not Jupyter `!` magic):
   ```python
   returncode = subprocess.run(
       ["bash", "scripts/run_exp.sh", "<name>_demo"],
       cwd=project_root,
       capture_output=False,
   )
   if returncode.returncode != 0:
       raise Exception(f"Experiment failed with return code {returncode.returncode}")
   ```
   Using `subprocess.run` gives explicit control over the return code, making failures visible as Python exceptions rather than silently passing in the notebook.

   **Section 4 — Artifact inspection.** Split into two subsections with separate headers:
   - **Artifacts: main results** — the outputs that directly answer the research question. A setup code cell derives `artifact_dir` from the config variables defined in section 2 (not hardcoded), then lists the directory contents. Then one **markdown + code cell pair** per main artifact.
   - **Artifacts: logs and sanity checks** — secondary outputs like task samples, counterfactual sanity checks, reference distributions, and metadata. Same markdown + code cell pair structure.

   In both subsections the markdown cell names the artifact (as a `###` heading) and explains what to look for; the code cell loads and displays it (JSON pretty-printed, images shown inline, tensors summarized by shape).

   **Section 5 — Takeaway.** A final markdown cell summarizing what to check before proceeding to downstream analyses.

11. **Each analysis module has a three-section README.** Every package under `analyses/` must include a `README.md` following the template below. See `baseline/README.md` (simple) and `pullback/README.md` (complex) as reference examples.

    **Section 1 — Opening paragraph (no header).** State the research question the analysis answers in one sentence, in italics. Follow with what the analysis does mechanically (one or two sentences) and where it sits in the pipeline — what it reads from and what depends on it. Pattern:
    ```
    # Module Name

    <Name> answers: *<research question in italics>* It <what it does mechanically>.
    The artifacts produced here are prerequisites for `<downstream>`.
    ```
    If the analysis has non-trivial algorithms (optimization loops, learned components), add a `## Overview` section after the opening paragraph with an ASCII diagram of the data flow. Skip this for straightforward forward-pass analyses.

    **Section 2 — Configuration (`## Configuration`).** Cover two scopes:
    - **Root config** (`causalab/configs/config.yaml`) — list only the shared params that this analysis actually reads (e.g. `experiment_root`, `batch_size`, `layer`). One bullet per param with its default.
    - **Module config** (`causalab/configs/analysis/<name>.yaml`) — paste the full YAML block with an inline `# comment` on every field explaining what it controls:
      ```yaml
      analysis:
        _name_: <name>
        param_one: value   # what it does
        param_two: value   # what it does
      ```
      If the module config has sub-groups (e.g. `belief_optim`, `embedding_optim`), document each sub-group under its own bold heading.

    **Section 3 — Outputs (`## Outputs`).** Split into two subsections:
    - `### Interpretation` — one bullet per human-readable output (JSON result, HTML visualization, PDF plot): `**\`filename\`** — What the number or visual shows. What a good result looks like. What a bad result suggests.` Lead with the most important output (the direct answer to the research question). Focus on *what to look for*, not just what the file contains.
    - `### Saved artifacts` — a table of every file saved to disk for downstream consumption:

      | File | Shape / Format | Used by |
      |---|---|---|
      | `file.pt` | `[dim1, dim2]` tensor | `next_analysis` |
      | `metadata.json` | run config snapshot | provenance |

      Follow the table with any notes needed to interpret non-obvious dimensions.

    **What to omit from analysis READMEs:**
    - **How to run** — that belongs in `runner_readme.md`, not here.
    - **Implementation details** — method internals belong in inline code comments, not the README, unless the algorithm is a core research contribution of the analysis.
    - **Redundant config docs** — don't re-document params that are self-evident from their name and default value.

12. **Dataset construction parameters live in task config; seed lives at root.** The four dataset-construction knobs — `n_train`, `n_test`, `enumerate_all`, `balanced` — are defined in task configs (`configs/task/*.yaml`) and read by analyses via `cfg.task.*`. `seed` is defined once in `config.yaml` and read via `cfg.seed`. Analysis configs must not duplicate these fields. Runner configs that need to override dataset sizes for a specific run do so in the `task:` override block, not per-analysis. This ensures every step in a multi-step pipeline uses the same dataset parameters unless the task itself dictates otherwise.

13. **Agent-generated runner configs live under the active session, not in `causalab/configs/`.** During research-session work, finalized runner YAMLs (the ones invoked by `scripts/run_exp.sh <name>`) are written to `${SESSION_DIR}/code/configs/runners/<group>/<name>.yaml`. The shipped `causalab/configs/` tree — including `causalab/configs/runners/` — is read-only from the agent's perspective: no skill-driven workflow promotes drafts into it, and the session bundle stays self-contained. The runner discovers session-local YAMLs automatically when `--experiment-root` lives under `agent_logs/` (see `scripts/run_exp.sh` and `.claude/skills/research-session/CONVENTIONS.md`). Dev-mode demo notebooks that write to `causalab/configs/<name>_demo.yaml` via `save_runner_config` are exempt — they are part of the shipped demo surface (invariant 10), not agent output.

Practical consequence: `neural/` stays small and boring; `methods/` stays a library of interchangeable tools; `analyses/` is the only layer that knows about disk layout, Hydra, and research intent. This makes the dependency flow declared in §1 enforceable rather than aspirational.


## 4. Artifact serialization policy

**Artifact serialization policy (migration in progress).** The target end state is that all on-disk artifacts produced by `causalab` are split into (1) a tensor payload persisted as `.safetensors` and (2) optional non-tensor metadata persisted as a sibling `.meta.json` under the same stem (e.g. `manifold.safetensors` + `manifold.meta.json`), and that both are written and read exclusively via the helpers in `causalab.io.artifacts`. Migration is staged by artifact kind:

- **Kind-(a) — tensors + scalar meta.** Today, kind-(a) artifacts are serialized via `save_tensors_with_meta` / `save_module` (and their load counterparts) from `causalab.io.artifacts`. This is the steady-state shape and applies now.
- **Kind-(b/c/d) — pickle blobs (sklearn instances), mixed payloads, optimizer state.** These sites still use `torch.save` / `torch.load(weights_only=False)` / `pickle`. Migration of each is tracked in `SAFETENSORS_MIGRATION_PLAN.md` (root of repo). The CI grep rule that bans `torch.save`, `torch.load`, and `pickle` outside `causalab/io/artifacts.py` activates **after** these migrations complete. HuggingFace-managed weights (`.bin`, `model.safetensors` from transformers cache) are out of scope.

**Currently-deferred call sites** (explicit list of the surface that the CI grep rule will cover once it activates):

- `causalab/io/artifacts.py` — `load_pickle` / `save_pickle` helpers (to be deleted after kind-(b/c/d) migration)
- `causalab/analyses/output_manifold/main.py:142-144, 195, 324` — `hellinger_pca` pickle (PR follow-up)
- `causalab/analyses/path_steering/main.py:1211, 1221, 1515, 1533` — `hellinger_pca` readers
- `causalab/methods/spline/belief_fit.py:61` — `torch.load(weights_only=False)` (PR follow-up)
- `causalab/analyses/pullback/main.py:471, 492, 580, 833` — `torch.load(weights_only=False)`
- `causalab/neural/units.py:269` — `torch.load(weights_only=False)`


## 5. Config notes

Config knobs do not compose freely. The combinations below are known to interact, and setting one without the matching other produces misleading results rather than an error.

### `task.resample_variable` × `locate.mode`

`task.resample_variable` (in every `configs/task/*.yaml`) controls how counterfactuals are generated by `generate_datasets` (`runner/helpers.py`):

- `"all"` — the counterfactual is a fresh independent sample; every input variable may differ from the original. This is what every task's hand-written `counterfactuals.generate_dataset` does by default, and it's the right setting for centroid-mode analyses and any analysis that compares *distributions* rather than exact tokens.
- A variable name (e.g. `"entity"`, `"number"`) — the counterfactual is a copy of the original with only that one input variable resampled to a different value. The task's generator is bypassed.

`locate.mode: pairwise` scores each `(layer, token_position)` cell by whether patching the residual stream at that cell causes the model to emit the *counterfactual's full target answer* (under `intervention_metric: string_match`). With `resample_variable: "all"`, multiple input variables differ between original and counterfactual — so patching only *one* variable's position produces an output that reflects a mixture (e.g. `original_number + cf_entity`), which does not match the CF's target. Entity/number positions score near zero even when the variable is clearly encoded there; only the last-token position scores high, because by then the full answer has been composed.

**Rule:** `locate.mode: pairwise` is only informative when `task.resample_variable` is set to a single variable name — the variable whose representation you are trying to localize. Target that variable with both knobs together:

```yaml
task:
  resample_variable: entity   # CFs differ from originals only in `entity`

locate:
  mode: pairwise              # scores how cleanly patching each cell
                              # flips the model to the CF's answer
```

Centroid mode (`locate.mode: centroid`) scores distributional KL to per-class references and remains meaningful under `resample_variable: "all"` — use it when you want a variable-agnostic localization pass.

`balanced: true` takes precedence over `resample_variable` when both are set; if you need single-variable counterfactuals, leave `balanced: false`.
