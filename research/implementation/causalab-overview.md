# Causalab — codebase & workflow overview

This document explains the workflow layout, important directories, experiment
execution, and available pipelines. It is a concise companion to
[`../../docs/CODEBASE.md`](../../docs/CODEBASE.md), which is authoritative for
architecture, layering, and invariants. Follow the code and `docs/CODEBASE.md`
when this older overview disagrees with them.

## The workflow layout

Silico exposes this research guidance through its `causalab-pipeline` routing
skill. The substantive documents live in this directory:

- **The research protocol** — `../answer-research-question/answer-research-question.md`, the current and non-stale entry point. Six steps (behavioral analysis → exploratory experimentation → hypothesis generation → hypothesis testing → generalize → save), a roadmap that plans them, and explicit routing back into exploration or hypothesis generation when a test comes back negative. Its steps are documents in their own right, not sibling workflows: `.../exploratory-experimentation/exploratory-experimentation.md` and `.../hypothesis-generation/hypothesis-generation.md` are entered *through* it.
- **The older entry paths** — `../explore-subspace/explore-subspace.md` (a subspace → autointerp) and the end-to-end `../subspace-causal-analysis-pipeline/subspace-causal-analysis-pipeline.md` (is this given subspace causal?). Both predate the protocol refactor and are stale.
- **[`implementation.md`](implementation.md)** — orientation for causalab plus the codebase how-to (this doc lives here). It covers the model-support boundary and getting started (below), the references for building new tasks / primitives / analyses (the `setup-task`, `setup-methods`, and `setup-analyses` guides), running a configured experiment end-to-end (`running-experiments.md`), the analysis catalog (`ANALYSIS_GUIDE.md`), common problems (`COMMON_PROBLEMS.md`), and the report-format templates under [`references/`](references/).

These workflows are the common *shapes*, not a fixed menu — the analyses compose freely, so when a question doesn't fit one of them, build the path for it rather than forcing the question into a box.

Silico's own planner/worker/critic harness owns session bootstrap and stage sequencing. The causalab `standalone-orchestrator` and `development-session` skills stayed in the causalab repo and are **not** part of silico.

## Model support — check this first

Causalab's harness loads **transformer language models** via HuggingFace. For a model it can't load — a vision model (e.g. DINO), an audio model, a biology / life-sciences foundation model, anything that is not a transformer LM — the causalab **code won't run**, but the **experimental-design knowledge fully carries over**:

- Use the research protocol's methods (correlation vs. causation, designing causal models and counterfactual datasets, choosing an intervention, quantifying effects, and reporting), adapted to the model by hand.
- The hypothesis-generation step is **CPU-only and fully model-agnostic** — a high-level causal model plus the counterfactual datasets that distinguish hypotheses — so `../answer-research-question/hypothesis-generation/hypothesis-generation.md` applies directly regardless of the target model.
- Write results up against the report-format templates in [`references/`](references/) and [`../answer-research-question/hypothesis-generation/hypothesis-report-format.md`](../answer-research-question/hypothesis-generation/hypothesis-report-format.md) — they hold no matter which model or tool produced the numbers.

When the target *is* a supported transformer LM, the causalab library runs the interventions end-to-end.

## New to causalab

Start hands-on with the onboarding tutorial notebooks under `~/.silico/libraries/causalab-internal/demos/onboarding_tutorial/` (`01_define_MCQA_task` → `02_trace_residual_stream` → `03_localize_with_patching`; `04`–`10` cover DAS/DBM, PCA, Boundless DAS, cross-model patching, attention, steering). For an end-to-end pipeline, run the weekdays demo: `cd ~/.silico/libraries/causalab-internal && ./scripts/run_exp.sh weekdays_8b_pipeline` (add `--slurm` to dispatch; needs one GPU with ≥24 GB).

## Key directories

Task-specific definitions live under `causalab/tasks/`; the task-agnostic experiment engine is the trio `causalab/analyses/` + `causalab/configs/` + `causalab/runner/`.

A **task** is the bridge between a behavioral hypothesis and the engine. The engine is completely task-agnostic — it knows nothing about IOI, arithmetic, or multiple choice. All task-specific knowledge lives in a package under `causalab/tasks/<name>/` (`causal_models.py`, `counterfactuals.py`, `token_positions.py`, plus supporting files), consumed through a standard interface. For the file-by-file breakdown and the conventions every task follows, see the task-package-layout reference (`setup-task/instructions/task_package_layout.md`); the setup-task guide (`setup-task/setup-task.md`) creates these files from a markdown spec.

```
causalab/
├── analyses/                       # One package per analysis type (each ships a README)
│   ├── baseline/
│   ├── locate/
│   ├── subspace/
│   ├── activation_manifold/
│   ├── output_manifold/
│   ├── path_steering/
│   ├── pullback/
│   ├── attention_pattern/
│   └── ...                         # ablation, causal_sufficiency, logit_lens, path_patching, ...
├── configs/                        # Hydra configs (composed at runtime)
│   ├── analysis/                   # Per-analysis defaults (one .yaml per analysis)
│   ├── model/                      # Per-model configs (gpt2, llama31_8b, llama31_70b, ...)
│   ├── task/                       # Per-task configs
│   ├── runners/                    # Composed run configs, grouped by task
│   │   ├── demos/                  # Onboarding/demo runner configs
│   │   ├── IOI/, mcqa/, age/, weekdays/, alphabet/, hours/, integer/,
│   │   │   months/, graph_walk/, hierarchical_equality/
│   ├── base.yaml
│   └── config.yaml
├── runner/                         # Hydra dispatcher (run_exp.py)
├── causal/                         # Causal model primitives
├── methods/                        # Shared method code (DAS, DBM, PCA, ...)
├── neural/                         # Model loading + activation hooks
├── io/                             # Save/load utilities
└── tasks/                          # (described above)
```

## Running an experiment

The unit of experimentation is an **analysis** (a module under `causalab/analyses/<name>/`) plus a **runner config** (a YAML under `causalab/configs/runners/<group>/<name>.yaml`). The runner config selects a task, a model, and one or more analyses, then sets analysis-specific knobs. The dispatcher in `causalab/runner/run_exp.py` composes the Hydra config and invokes each analysis in dependency order.

A minimal runner config (adapted from `configs/runners/demos/locate_demo.yaml`):

```yaml
# @package _global_
defaults:
  - /base
  - /task: natural_domains_arithmetic_weekdays
  - /model: llama31_8b
  - /analysis/locate
  - _self_
task:
  target_variable: result
  resample_variable: entity
locate:
  method: interchange
  layers: [0, 8, 16, 24]
```

To **chain analyses** in a single run, add more `- /analysis/<name>` entries to the `defaults:` list — the dispatcher runs them in dependency order (e.g. `baseline` → `locate` → `subspace`). Filtering of correct-only examples is built into the pipeline.

Invoke the dispatcher via the wrapper script from the library checkout:

```
cd ~/.silico/libraries/causalab-internal
./scripts/run_exp.sh {runner_config_name}                # run inline
./scripts/run_exp.sh --slurm {runner_config_name}        # dispatch as an sbatch job
```

For example, `./scripts/run_exp.sh locate_demo`. The leading `./` enables the registered tab-completion (`scripts/completion.bash`), which auto-discovers configs by basename across the `causalab/configs/runners/<group>/` subdirectories. Under the hood this runs `python -m causalab.runner.run_exp` with the appropriate Hydra overrides. When `--slurm` is set, the wrapper resolves `--gres=gpu:N` from the model config's `slurm.gpus` and `--time` from the runner's `slurm.time` (default in `causalab/configs/base.yaml`); manual `--gpus`, `--time`, and `--qos` flags override the resolved values.

### Where results go

Outputs land in a path that encodes the run's task, model, and analysis. Results are not timestamped — re-running the same runner config rewrites that directory:

```
artifacts/{task}/{model}/{analysis}/
├── accuracy.json               # Or analysis-specific JSON results
├── metadata.json               # Snapshot of the resolved Hydra config
├── *.safetensors / *.pt        # Tensors (e.g. full_output_dists)
├── *.png / *.pdf               # Heatmaps, confusion matrices, plots
└── ...                         # Analysis-specific outputs
```

The path encodes the config, making it trivial to compare results across models or analyses for the same task. The runner config under `causalab/configs/runners/<group>/<name>.yaml` is checked into git and is the source of truth for a run; `metadata.json` snapshots the resolved Hydra config that produced the artifacts. To keep an old run, copy its artifact directory aside before re-running.

## Analyses

Each analysis answers one research question and may depend on outputs from earlier analyses. The canonical chain:

| Analysis | Research question | Depends on |
|---|---|---|
| **baseline** | Can the model solve the task? Are the counterfactual generators well-formed? | — |
| **locate** | Which `(layer, token_position)` encodes each causal variable? | baseline |
| **subspace** | What k-dimensional subspace captures the variable's representation? | locate |
| **activation_manifold** | What is the geometric structure of activations as the variable varies? | subspace |
| **output_manifold** | What is the geometry of output distributions on the probability simplex? | baseline |
| **path_steering** | Does the subspace/manifold faithfully preserve causal structure? | subspace, activation_manifold |
| **pullback** | What activation trajectories realize prescribed belief-space paths? | activation_manifold, output_manifold |
| **attention_pattern** | Which attention heads attend to which token types? | — |

Beyond the chain, `ablation`, `causal_sufficiency` (causal tracing, below), `logit_lens`, and `path_patching` ship as additional analyses, and several analyses back the research workflows (e.g. `develop_hypothesis`, `exploration`, `characterize_subspace`, `manifold_bundle_ingest`). The per-analysis `README.md` and `causalab/configs/analysis/<name>.yaml` are authoritative — consult them rather than trusting this list to stay exhaustive. Method-level techniques — **DAS**, **DBM**, **PCA**, **Boundless DAS** — live in `causalab/methods/` and are selected as options inside the analyses (e.g. `subspace.method: das`, `locate.method: interchange`).

### Manifold analyses

The manifold-steering method described in the handbook is implemented by the `activation_manifold` and `path_steering` analyses (`causalab/analyses/activation_manifold/`, `causalab/analyses/path_steering/`); `output_manifold` and `pullback` extend it into output (belief) space. When running inside the Silico environment, the `silico-capabilities:steering-pipeline` skill wraps the operational steering-vector pipeline — SAE-feature extraction, difference-of-means/contrastive (DOM), or a geometry-aware manifold follow-up, plus the standard validation battery.

### Causal tracing on causalab

The restoration sweep described under "Causal tracing" in the handbook ships as the `causal_sufficiency` analysis (`causalab/analyses/causal_sufficiency/`, config `causalab/configs/analysis/causal_sufficiency.yaml`), a ROME-style trace. It corrupts the entry site — the residual stream at layer `-1` (embeddings) over the configured span — with **zero**, **mean**, or **seeded noise**, then **restores one clean site at a time** across a chosen grid (attention head / attention-output / MLP / residual, keyed `(layer, head)` or `(layer, position)`), optionally a centered `window` of consecutive layers per cell (ROME's severed traces; `window=10`). Corruption and restore happen together in a **single** forward pass — the restored site's clean value propagates through an already-corrupted network, no two-pass collect/inject:

- `zero`/`mean` corruption are broadcast `replace` interventions (any span); the restore is a per-example `replace` of each example's own clean activation (collected once, so restore sites are single-token).
- `noise` corruption uses the dynamic seeded **noise** intervention (PR #331), adding independent Gaussian per token at scale `noise_scale × σ` (σ = the subject-embedding std, so the default `3.0` is ROME's `3σ`), and so spans the whole multi-token subject; it is mixed with the `replace` restore in one pass via `run_steering_interventions`' `type_by_unit` map.

Recovery per cell is `restored_metric − corrupted_floor` (optionally normalized to the `clean_ceiling − floor` band), scored by softmax `P(answer)` (`prob`, ROME), the correct-vs-distractor logit difference (`logit_diff`), or a single-token logit (`logit`), laid out as the familiar layer × site heatmap.

## Turnkey pipelines

When you do choose to run on causalab, pick your entry point by what you start from:

- **`../subspace-causal-analysis-pipeline/subspace-causal-analysis-pipeline.md`** — start from a **given** subspace bundle ("is this subspace causal?"). Drives characterize → develop-causal-model → setup-task → plan → run → interpret end to end, running interchange-IIA / mediation on the fixed rotation.
- **`../answer-research-question/answer-research-question.md`** — start from a **task/behavior** ("which subspace mediates this latent?"). Not a turnkey pipeline and not stale: it is the current six-step protocol, and it expects you to plan it with a roadmap and to make the routing decisions yourself rather than running gate-free. Prefer it.

## Report-format references

Report-format templates fix the tone, macrostructure, and figure conventions for writing up a method's results. (Behavior-exploration reporting is owned by the `../answer-research-question/exploratory-experimentation/exploratory-experimentation.md` workflow itself — its report/publish steps fix the tone, macrostructure, and the interactive logit-lens / interchange / PCA figure contracts — so it has no separate format doc.)

- [`references/interchange-das-localization-report-format.md`](references/interchange-das-localization-report-format.md) (beside this doc) — the fixed template for reporting **neural localization** (full-vector interchange / activation patching and DAS) against a certified causal model: the positive-control discipline, the variable-by-variable macrostructure (inputs → output → intermediates), per-variable contents, and plot-type rules.
- [`../answer-research-question/hypothesis-generation/hypothesis-report-format.md`](../answer-research-question/hypothesis-generation/hypothesis-report-format.md) — the fixed tone, macrostructure, per-dataset distinguishability-heatmap rules, and figure-caption convention for a hypothesis-generation report (competing causal models + counterfactual datasets and their CPU distinguishability).
