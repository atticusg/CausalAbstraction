# Causal Abstraction for Mechanistic Interpretability

![Tests](https://github.com/goodfire-ai/causalab/actions/workflows/test.yml/badge.svg)

A framework for **mechanistic interpretability** — reverse-engineering the algorithms language models use internally using **causal abstraction**.

You write a high-level causal model describing *how you think* an LM solves a task, then run experiments to test whether the LM's internal components actually implement that algorithm.

## Quick Start

Causalab is built to be driven by a **coding agent** (e.g. Claude Code). The fast path:

1. Clone and install:
   ```bash
   git clone https://github.com/goodfire-ai/causalab.git
   cd causalab
   uv sync
   ```
2. Open the directory in your coding agent of choice.
3. Describe what you want to do — e.g. *"walk me through the codebase"*, *"set up a new task from this spec"*, *"run the weekdays pipeline"*. The agent routes the request through the matching skill (`/getting-started`, `/setup-task`, `/run-experiment`, …).

Prefer to look around first? Run the end-to-end weekdays pipeline (Llama-3.1-8B, ≥24 GB VRAM):

```bash
./scripts/run_exp.sh weekdays_8b_pipeline          # inline
./scripts/run_exp.sh --slurm weekdays_8b_pipeline  # sbatch
```

Or open [`demos/weekdays_geometry.ipynb`](demos/weekdays_geometry.ipynb) for the same pipeline rendered as a notebook.

## Working with a coding agent

The full workflow is skill-driven. Each skill is a focused entry point — invoke it by name (`/<skill>`), or describe your goal and let the agent route to it.

### Investigating a task — `/setup-task`

Use `/setup-task` to explore, understand, or create a task. Whether you want to inspect an existing task's causal model, browse its counterfactuals, or build a new task from scratch, this is the entry point.

```
/setup-task                     # interactive
/setup-task path/to/spec.md     # from a spec file
/setup-task path/to/paper.pdf   # from a paper PDF
```

### Running experiments — `/plan-experiment` then `/run-experiment`

`/plan-experiment` crystallizes a research objective into `RESEARCH_OBJECTIVE.md` + `PLAN.md` (analysis DAG, sweep strategy, expected artifacts). `/run-experiment` then materializes the runner config(s) and executes the pipeline. `/interpret-experiment` is auto-invoked at the end and writes a single `result/REPORT.md` grounded in the plan.

```
/plan-experiment
/run-experiment
```

### All available skills

| Command | What it does |
|---------|-------------|
| `/research-session` | Bootstrap a session directory at the start of a research workflow |
| `/development-session` | Load engineering context at the start of codebase work |
| `/getting-started` | Onboarding walkthrough |
| `/setup-task` | Create, explore, or investigate a task |
| `/plan-experiment` | Crystallize a research objective into `RESEARCH_OBJECTIVE.md` + `PLAN.md` |
| `/run-experiment` | Materialize runner configs from the plan and execute |
| `/interpret-experiment` | Auto-invoked after `/run-experiment` — writes `result/REPORT.md` |
| `/replicate-paper` | Reproduce results from a research paper |
| `/document-issues` | Document failures, confusions, and workarounds |

## Core Concepts

### Causal Models

A causal model is your hypothesis about how the LM solves a task. It consists of:

- **Variables**: concepts that might be represented in the network (e.g., "subject name", "indirect object")
- **Values**: possible assignments to each variable
- **Parent–Child Relationships**: directed dependencies
- **Mechanisms**: functions that compute a variable's value given its parents'

### Causal Abstraction

Mechanistic interpretability aims to reverse-engineer the algorithm a network implements. Causal abstraction grounds this: an algorithm is a causal model, a network is a causal model, and "implementation" is the abstraction relation between two models. The algorithm is a **high-level causal model**, the network is a **low-level causal model**, and when the high-level mechanisms are accurate simplifications of the low-level mechanisms, the algorithm is a **causal abstraction** of the network.

### Interchange Interventions

Interchange interventions test whether a high-level variable aligns with specific features in the LM. The intervention replaces activations from one input with activations from a counterfactual input, isolating one causal pathway at a time.

Method-level techniques for *constructing* the feature space being intervened on — DAS, DBM, PCA, Boundless DAS, SAE — live in [`causalab/methods/`](causalab/methods/) and are selected as options inside analyses (e.g. `subspace.method: das`, `locate.method: interchange`).

## Available Analyses

The runner is built around eight named **analyses**. Each answers a specific research question and may consume artifacts from earlier analyses. Chain them in a single run by listing multiple `- /analysis/<name>` entries in a runner config's `defaults:` block.

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

Each analysis is configured by a Hydra YAML at `causalab/configs/analysis/<name>.yaml` and invoked through a runner config under `causalab/configs/runners/<group>/<name>.yaml`.

## Repository Structure

The codebase follows a strict layering. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full breakdown, layering invariants, and config conventions.

```
causalab/
├── causal/        # Causal model primitives
├── tasks/         # Task definitions (causal_models.py, counterfactuals.py, …)
├── neural/        # Pyvene API surface — pipeline.py, units.py, LM_units.py,
│                  #   featurizer.py, activations/
├── methods/       # Reusable interpretability tools — DAS, DBM, PCA, SAE,
│                  #   manifold builders, scoring metrics
├── io/            # Single source of truth for disk I/O + shared plot primitives
├── analyses/      # Research-question wrappers (baseline/, locate/, subspace/, …)
├── runner/        # Hydra dispatcher — run_exp.py
└── configs/       # Hydra configs — analysis/, model/, task/, runners/
demos/             # Onboarding notebooks + the weekdays_geometry pipeline notebook
artifacts/         # Run outputs, keyed by task / model / analysis (gitignored)
```

**Dependency flow:** `tasks/` and `causal/` are independent. `neural/` depends on neither. `io/` depends only on `neural/`, `tasks/`, `causal/`. `methods/` depends on `neural/`, `causal/`, `io/`. `analyses/` depends on all four. `runner/` is a thin shell over `analyses/`.

### Task packages (`causalab/tasks/<name>/`)

Each task is a self-contained Python package consumed by the analyses through a fixed interface:

| File | Purpose |
|------|---------|
| `causal_models.py` | Causal model: variables, values, mechanisms |
| `counterfactuals.py` | Generates counterfactual pairs for each variable |
| `token_positions.py` | Maps variable names to token positions in the input |
| `config.py` | Constants: variable value lists, max tokens, task name |
| `templates.py` | Input text templates with placeholders |

Tasks and analyses are fully separated — define a new task and every analysis works automatically.

### Where results land

```
artifacts/{task}/{model}/{analysis}/
├── *.json / metadata.json     # Results + resolved Hydra config snapshot
├── *.safetensors / *.pt       # Tensors (activations, weights, distributions)
├── *.png / *.pdf              # Plots
└── *.html                     # Interactive visualizations
```

The path encodes the run, so cross-model and cross-analysis comparisons are direct file-system operations. Re-running the same runner config rewrites the directory; copy artifacts aside if you want to keep an old run.

## Getting Started

### Installation

```bash
git clone https://github.com/goodfire-ai/causalab.git
cd causalab
uv sync
```

For development:

```bash
uv run pre-commit install  # set up git hooks
```

### Recommended starting points

- **End-to-end pipeline (recommended):** [`demos/weekdays_geometry.ipynb`](demos/weekdays_geometry.ipynb) chains baseline → subspace → activation_manifold → output_manifold → path_steering → pullback on Llama-3.1-8B. The same pipeline runs from the CLI as `./scripts/run_exp.sh weekdays_8b_pipeline`. Minimum hardware: 1 GPU with ≥24 GB VRAM.
- **Causal model primer:** [`demos/causal_model_demo.ipynb`](demos/causal_model_demo.ipynb) walks through defining a causal model and counterfactual dataset.

### Tab completion for `run_exp.sh`

Tab-complete runner config names when invoking `./scripts/run_exp.sh`:

```bash
# bash
source scripts/completion.bash
# zsh
source scripts/completion.zsh
```

To enable permanently from the repo root:

```bash
# bash
echo "source $(pwd)/scripts/completion.bash" >> ~/.bashrc
# zsh
echo "source $(pwd)/scripts/completion.zsh" >> ~/.zshrc
```

### Submitting a slurm job

`run_exp.sh` is the single entry point for both inline and slurm runs. Pass `--slurm` to dispatch as `sbatch`; `--gres=gpu:N` is resolved from the model config's `slurm.gpus` and `--time` from the runner's `slurm.time` (default in `causalab/configs/base.yaml`). CLI flags `--gpus`, `--time`, `--qos` override.

```bash
./scripts/run_exp.sh --slurm weekdays_8b_pipeline
./scripts/run_exp.sh --slurm --qos=opportunistic --time=08:00:00 weekdays_8b_pipeline
```

### Running tests

```bash
uv run pytest -m "not slow and not gpu"  # quick
uv run pytest                            # full
```
