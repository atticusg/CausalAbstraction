---
name: getting-started
description: Onboarding guide for new users. Explains the causalab codebase, how to set up tasks, what experiments are available, and how to run them with Claude. Use when the user asks "how do I use this?", "what can I do?", or needs a walkthrough.
---

# Getting Started with Causalab

This skill walks new users through the causalab codebase: what it does, how it's organized, how to set up tasks, and how to run experiments — all with Claude's help.

**When invoked:** Start by presenting the topics covered in this guide and letting the user choose how they want to proceed. Do NOT immediately dump all content.

## Step 0: Orientation (ALWAYS start here)

Present the following to the user:

> **Hi! I’m Hume, your AI assistant. Welcome to CausaLab!** I can help with the following topics:
>
> 1. **What is Causalab?** — Core concepts: causal models, counterfactuals, interchange interventions
> 2. **The key directories** — `causalab/tasks/` (task definitions), `causalab/analyses/` (analysis modules), `causalab/configs/` (Hydra configs and runner configs)
> 3. **How to set up a new task** — Defining a causal model for a behavior you want to study
> 4. **What experiments you can run** — From simple activation patching to DAS and DBM
> 5. **How to run experiments with Claude** — The end-to-end workflow using skills
> 6. **Quick reference** — All available skills at a glance

>
> **Recommended starting point:** If you're new, start with `demos/causal_model_demo.ipynb` for the causal-model primer, and `demos/weekdays_geometry.ipynb` for the end-to-end pipeline.
>
> Notebooks `04–10` cover advanced topics (DAS/DBM training, PCA, Boundless DAS, cross-model patching, attention patterns, steering interventions) once you have the basics down.
>
> These notebooks give you hands-on experience before diving into the full framework.
>
> **Want to see an end-to-end research pipeline instead?** Run the weekdays demo, which chains baseline → subspace → activation_manifold → output_manifold → path_steering → pullback on Llama-3.1-8B:
> - **Notebook:** `demos/weekdays_geometry.ipynb`
> - **Script:** `./scripts/run_exp.sh weekdays_8b_pipeline` (add `--slurm` to dispatch as a job)
>
> Minimum hardware: **1 GPU with ≥24 GB VRAM** (e.g., a single H100 or A100). The runner config uses `model: llama31_8b` (`slurm.gpus: 1`).

Then ask the user:

> "Would you like a **high-level overview** of everything, or do you want to **dive into a specific topic** (1–6)?"

- If they pick a specific topic, jump directly to that section.
- If they want a high-level overview, walk through each section briefly, pausing between sections to ask if they want more detail.

Most workflows start with a session skill: `/research-session` for experiments and analysis, `/development-session` for codebase work. CLAUDE.md's mode detection usually invokes the right one for you automatically, but you can invoke either explicitly.

---

## Section 1: What is Causalab?

Explain the following to the user:

> **Causalab** is a framework for **mechanistic interpretability** — reverse-engineering what algorithms neural networks (especially language models) use internally.
>
> It uses **causal abstraction**: you write a high-level causal model that describes *how you think* the LM solves a task, then you run experiments to test whether the LM's internal components actually implement that algorithm.

### Core Concepts

Explain these concepts clearly:

1. **Causal Model** — A hypothesis about how the LM solves a task. It defines:
   - **Variables** (e.g., "subject name", "indirect object", "answer")
   - **Values** each variable can take (e.g., list of names)
   - **Mechanisms** — functions that compute each variable from its parents
   - Must always include `raw_input` (the text prompt) and `raw_output` (the expected answer)

2. **Counterfactuals** — Pairs of inputs where specific variables are swapped. For example: same sentence but with the subject name changed. These let you test whether a specific internal component carries information about that variable.

3. **Interchange Interventions** — The key technique. You:
   - Run the model on input A
   - Run the model on input B (the counterfactual)
   - *Replace* the activations at a specific component (layer, position, head) from B into A
   - Check if the output changes to match what the causal model predicts
   - If it does, that component carries the swapped variable's information

4. **Token Positions** — Where in the input sequence to intervene. Each variable is associated with specific token positions (e.g., the subject name tokens, the final token before the answer).

### Ask the user:

> "Would you like to see a concrete example of how this works, or shall we move on to how the codebase is organized?"

If they want an example, point them to:
- `demos/causal_model_demo.ipynb` — Defining a causal model
- `demos/weekdays_geometry.ipynb` — End-to-end manifold steering pipeline

---

## Section 2: The Key Directories

The directories the user needs to understand are `causalab/tasks/` (task-specific definitions) and the trio `causalab/analyses/` + `causalab/configs/` + `causalab/runner/` (the task-agnostic experiment engine). Present them in depth.

### `causalab/tasks/` — Task definitions

A **task** is the bridge between a behavioral hypothesis and the experiment engine. The experiment code is completely task-agnostic — it doesn't know anything about IOI, arithmetic, or multiple choice. All task-specific knowledge lives in a Python package under `causalab/tasks/<name>/`, which the experiments consume through a standard interface (`causal_models.py`, `counterfactuals.py`, `token_positions.py`, plus a few supporting files).

For the file-by-file breakdown of what each module is responsible for and the conventions every task must follow, see [`.claude/skills/setup-task/instructions/task_package_layout.md`](../setup-task/instructions/task_package_layout.md). The `/setup-task` skill creates these files from a markdown spec.

### `causalab/analyses/`, `causalab/configs/`, `causalab/runner/` — The experiment engine

The task-agnostic machinery that runs causal abstraction experiments is split across three sibling directories. An **analysis** module defines what gets measured, a **runner config** (Hydra YAML) defines what to run on, and the **runner** dispatcher composes them.

```
causalab/
├── analyses/                       # One package per analysis type
│   ├── baseline/
│   ├── locate/
│   ├── subspace/
│   ├── activation_manifold/
│   ├── output_manifold/
│   ├── path_steering/
│   ├── pullback/
│   └── attention_pattern/
├── configs/                        # Hydra configs (composed at runtime)
│   ├── analysis/                   # Per-analysis defaults (one .yaml per analysis)
│   ├── model/                      # Per-model configs (gpt2, llama31_8b, llama31_70b, ...)
│   ├── task/                       # Per-task configs
│   ├── runners/                    # Composed run configs, grouped by task
│   │   ├── demos/                  # Demo runner configs
│   │   ├── age/, alphabet/, graph_walk/, months/, weekdays/
│   ├── base.yaml
│   └── config.yaml
├── runner/                         # Hydra dispatcher (run_exp.py)
├── causal/                         # Causal model primitives
├── methods/                        # Shared method code (DAS, DBM, PCA, ...)
├── neural/                         # Model loading + activation hooks
├── io/                             # Save/load utilities
└── tasks/                          # (described above)
```

Explain the key insight:

> The unit of experimentation is an **analysis** (a module under `causalab/analyses/<name>/`) plus a **runner config** (a YAML under `causalab/configs/runners/<group>/<name>.yaml`). The runner config selects a task, a model, and one or more analyses, then sets analysis-specific knobs. The dispatcher in `causalab/runner/run_exp.py` composes the Hydra config and invokes each analysis in dependency order.
>
> A minimal runner config (adapted from `configs/runners/demos/locate_demo.yaml`):
>
> ```yaml
> # @package _global_
> defaults:
>   - /base
>   - /task: natural_domains_arithmetic_weekdays
>   - /model: llama31_8b
>   - /analysis/locate
>   - _self_
> task:
>   target_variable: result
>   resample_variable: entity
> locate:
>   method: interchange
>   layers: [0, 8, 16, 24]
> ```
>
> To **chain analyses** in a single run, add more `- /analysis/<name>` entries to the `defaults:` list — the dispatcher will run them in dependency order (e.g. `baseline` → `locate` → `subspace`). Filtering of correct-only examples is built into the pipeline; you don't need to invoke it separately.

### `artifacts/` — Where results are saved

Outputs land in a path that encodes the run's task, model, and analysis. Results are not timestamped — re-running the same runner config rewrites that directory:

```
artifacts/{task}/{model}/{analysis}/
├── accuracy.json               # Or analysis-specific JSON results
├── metadata.json               # Snapshot of the resolved Hydra config
├── *.safetensors / *.pt        # Tensors (e.g. full_output_dists)
├── *.png / *.pdf               # Heatmaps, confusion matrices, plots
└── ...                         # Analysis-specific outputs
```

Explain the key points:

> - **Path encodes config** — `artifacts/{task}/{model}/{analysis}/` makes it trivial to compare results across models or across analyses for the same task.
> - **Reproducibility** — The runner config under `causalab/configs/runners/<group>/<name>.yaml` is checked into git and is the source of truth for a run. `metadata.json` snapshots the resolved Hydra config that produced the artifacts.
> - **No automatic timestamping** — Re-running overwrites. If you want to keep an old run, copy its artifact directory aside, or branch and edit the runner config name before re-running.

---

## Section 3: How to Set Up a New Task

Explain the workflow:

> Setting up a task means defining a causal model of how you think the LM solves a specific behavior, and writing the code that generates inputs, counterfactuals, and checks outputs.

### What you need before starting

1. **A behavioral task** — Something the LM can do (e.g., "predict the indirect object in a sentence", "answer a multiple-choice question", "add two numbers")
2. **A causal hypothesis** — What variables matter and how they relate (e.g., "the answer depends on the subject and the verb")
3. **A model to test on** 

### Using Claude to set up a task

Tell the user:

> You can ask Claude to set up a task for you. There are two ways:
>
> **Option A: From a specification file**
> If you have a markdown file describing the task (a `set_up_task.md`), just say:
> ```
> /setup-task path/to/my_task_spec.md
> ```
>
> **Option B: Interactively**
> If you don't have a spec yet, just say:
> ```
> /setup-task
> ```
> Claude will walk you through creating the specification step by step — defining variables, templates, counterfactuals, and token positions.
>
> **Option C: From a research paper**
> If you want to replicate a task from a paper, provide the PDF:
> ```
> /setup-task path/to/paper.pdf
> ```

### What the skill creates

The `setup-task` skill generates all required files:

| File | Purpose |
|------|---------|
| `config.py` | Constants: variable value lists, max tokens, task name |
| `causal_models.py` | Causal model definition with variables and mechanisms |
| `templates.py` | Input text templates with placeholders |
| `counterfactuals.py` | Functions that generate counterfactual pairs |
| `token_positions.py` | Maps variable names to token positions in the input |
| `checker.py` | Validates model output against expected answer |
| `metrics.py` | Scoring functions for intervention results |
| `__init__.py` | Package exports |
| `demo.ipynb` | Overview notebook showing examples |
| `README.md` | Task-specific notes and usage |

It also validates the task end-to-end: loads the model, checks accuracy, verifies token alignment.

---

## Section 4: What Experiments Can You Run?

The current engine offers 8 named **analyses**. Each one answers a specific research question and may depend on outputs from earlier analyses. Chain them inside a single runner config by adding multiple `- /analysis/<name>` entries to the `defaults:` list.

| Analysis | Research question | Depends on |
|---|---|---|
| **baseline** | Can the model solve the task? Are counterfactual generators well-formed? | — |
| **locate** | Which (layer, token_position) encodes each causal variable? | baseline |
| **subspace** | What k-dimensional subspace captures the variable's representation? | locate |
| **activation_manifold** | What is the geometric structure of activations as the variable varies? | subspace |
| **output_manifold** | What is the geometry of output distributions on the probability simplex? | baseline |
| **path_steering** | Does the subspace/manifold faithfully preserve causal structure? | subspace, activation_manifold |
| **pullback** | What activation trajectories realize prescribed belief-space paths? | activation_manifold, output_manifold |
| **attention_pattern** | Which attention heads attend to which token types? | — |

Each analysis is configured by a Hydra YAML at `causalab/configs/analysis/<name>.yaml` and invoked through a runner config under `causalab/configs/runners/<group>/<name>.yaml`. Method-level techniques like **DAS** (Distributed Alignment Search), **DBM** (Desiderata-Based Masking), **PCA**, and **Boundless DAS** live in `causalab/methods/` and are selected as options inside the analyses (e.g. `subspace.method: das` or `locate.method: interchange`).

For a hands-on walkthrough of the legacy DAS / DBM / PCA / Boundless DAS workflows, see notebooks `04_train_DAS_and_DBM.ipynb`, `05_PCA_DBM_DAS.ipynb`, and `06_boundless_DAS.ipynb` in `demos/onboarding_tutorial/`.

---

## Section 5: How to Run Experiments with Claude

### The simple way

Tell the user:

> Just tell Claude what you want to investigate. For example:
>
> ```
> /run-experiment
> ```
>
> Claude will walk you through:
> 1. Selecting a task
> 2. Choosing which analyses to run (and how to chain them)
> 3. Configuring hyperparameters (layers, target/resample variables, methods, etc.)
> 4. Composing the runner config under `causalab/configs/runners/<group>/<name>.yaml`
> 5. Running the experiment via the dispatcher
> 6. Analyzing the results

### The manual way

If you'd rather invoke the dispatcher yourself, use the wrapper script:

```
./scripts/run_exp.sh {runner_config_name}                # run inline
./scripts/run_exp.sh --slurm {runner_config_name}        # dispatch as sbatch job
```

For example: `./scripts/run_exp.sh locate_demo`. The leading `./` enables the registered tab-completion (see `scripts/completion.bash`), which auto-discovers configs by basename across all `causalab/configs/runners/<group>/` subdirectories. Under the hood this invokes `python -m causalab.runner.run_exp` with the appropriate Hydra overrides.

When `--slurm` is set, the wrapper resolves `--gres=gpu:N` from the model config's `slurm.gpus` and `--time` from the runner's `slurm.time` (default in `causalab/configs/base.yaml`). Manual `--gpus`, `--time`, and `--qos` flags override the resolved values.

### Where results go

Results are saved under `artifacts/`, keyed by task / model / analysis:

```
artifacts/{task}/{model}/{analysis}/
├── accuracy.json              # Or analysis-specific JSON results
├── metadata.json              # Resolved Hydra config snapshot
├── *.safetensors / *.pt       # Tensors
├── *.png / *.pdf              # Heatmaps, confusion matrices, plots
└── ...                        # Analysis-specific outputs
```

---

## Section 6: Quick Reference — All Available Commands

Summarize the skills:

| Command | What it does |
|---------|-------------|
| `/research-session` | Bootstrap a session directory at the start of a research workflow |
| `/development-session` | Load development-mode context at the start of codebase work |
| `/getting-started` | This guide — learn how to use causalab with Claude |
| `/setup-task` | Create a new task or explore an existing one |
| `/plan-experiment` | Crystallize a research objective into a detailed plan (`RESEARCH_OBJECTIVE.md` + `PLAN.md`) — runs before `/run-experiment` |
| `/run-experiment` | Materialize runner config(s) from the plan and execute |
| `/interpret-experiment` | Auto-invoked after `/run-experiment` — writes a single `result/REPORT.md` grounded in the plan |
| `/replicate-paper` | Replicate results from a research paper |
| `/document-issues` | Document problems encountered during work |

---

## Closing

Ask the user:

> "Now that you have an overview, what would you like to do?"
>
> Options:
> 1. **Set up a new task** — Define a causal model for a behavior I want to study
> 2. **Run experiments on an existing task** — Pick from weekdays, months, age, alphabet, or graph_walk
> 3. **Explore the demo notebooks** — `demos/weekdays_geometry.ipynb` or `demos/causal_model_demo.ipynb`
> 4. **Something else** — Ask me anything about the codebase

Route to the appropriate skill based on the user's choice.
