# Task Definitions for Causal Abstraction Experiments

Each task is a self-contained package under `causalab/tasks/<name>/` that
defines a causal model, counterfactual generation, and tokenization helpers.
Tasks are loaded at runtime via `load_task()` in `causalab.tasks.loader`, and a
runner selects one with `- /task: <name>` (resolved from a Hydra config in
`causalab/configs/task/<name>.yaml`).

Standing up a runnable task has three parts: (1) the task **package** (`causalab/tasks/<name>/`), (2) the Hydra **task config** (`causalab/configs/task/<name>.yaml`), and (3) **validation** against a model. All three are described below.

## 1. The task package (`causalab/tasks/<name>/`)

Create a directory `causalab/tasks/<name>/` with these modules:

| File | Required | Purpose |
|------|----------|---------|
| `causal_models.py` | yes | Causal model + the exports `load_task()` reads |
| `counterfactuals.py` | yes | Counterfactual-pair generation |
| `token_positions.py` | for interventions | Maps variable names → token positions |
| `config.py` | yes | Constants: task name, value lists, token budgets |
| `templates.py` | yes | Input text templates + fill function |
| `checker.py` | optional | Custom output matcher (see below) |
| `metrics.py` | optional | Task-specific metric helpers |
| `__init__.py` | yes | Package exports |
| `summary.ipynb` | optional | CPU-only task overview notebook (no model load) |

### causal_models.py (required)

Defines the causal model and exports that `load_task()` reads by convention.

**Singleton tasks** (fixed structure, e.g. weekdays, months, years):

| Export | Type | Required |
|--------|------|----------|
| `CAUSAL_MODEL` | `CausalModel` | yes |
| `VARIABLE_VALUES` | `dict[str, list[str]]` — var name → values | yes |
| `CYCLIC_VARIABLES` | `set[str]` — which variables wrap cyclically | yes (may be empty) |
| `EMBEDDINGS` | `dict[str, Callable]` — var name → embedding fn | yes |
| `PERIODIC_INFO` | `dict[str, int]` — var name → period length | no |
| `TEMPLATE` | `str` — prompt template | no |
| `TARGET_VARIABLE` | `str` — variable being steered | no |
| `RANDOM_CAUSAL_MODEL` | `CausalModel` — random baseline model | no |
| `RANDOM_VARIABLE_VALUES` | `dict[str, list[str]]` — values for random baseline | no |

The per-value output forms used for scoring are **not** a task export — they are
declared on the `CausalModel` itself via `output_tokens` (`{variable: {value:
[surface form, ...]}}`; build the mechanical `[" v", v]` map with
`build_output_tokens`). The probability-path score tokens and the string
`checker` are both derived from it, so a task that declares `output_tokens` needs
no `checker.py`. Optional `match_modes={variable: "prefix"}` accepts output that
continues past the answer.

**Factory tasks** (parameterized, e.g. graph_walk, natural_domains_arithmetic):

| Export | Type | Required |
|--------|------|----------|
| `CREATE_CAUSAL_MODEL` | `Callable[[config], CausalModel]` | yes |
| `GET_VARIABLE_VALUES` | `Callable[[CausalModel], dict[str, list[str]]]` | yes |
| `CYCLIC_VARIABLES` | `set[str]` | yes (may be empty) |
| `EMBEDDINGS` | `dict[str, Callable]` | yes |
| `GET_CYCLIC_VARIABLES` | `Callable[[CausalModel], set[str]]` | no (overrides `CYCLIC_VARIABLES`) |
| `GET_EMBEDDINGS` | `Callable[[CausalModel], dict[str, Callable]]` | no (overrides `EMBEDDINGS`) |
| `GET_PERIODIC_INFO` | `Callable[[CausalModel], dict[str, int] \| None]` | no |
| `CREATE_RANDOM_CAUSAL_MODEL` | `Callable[[config], CausalModel]` | no |
| `TARGET_VARIABLE` | `str` | no |

### counterfactuals.py (required)

```python
generate_dataset(causal_model, n_examples, seed) → list[dict]
```

Each dict has `"input"` (a causal trace) and `"counterfactual_inputs"`
(list of counterfactual traces).

### token_positions.py (required for intervention experiments)

```python
create_token_positions(pipeline, ...) → dict[str, TokenPosition]
```

Maps position names to `TokenPosition` objects that locate where in
the token sequence to intervene.

### config.py (required)

Task constants: `TASK_NAME` (must equal the package directory name — the loader
key), the input variable value lists, and the token budgets `MAX_TASK_TOKENS`
(max input length) and `MAX_NEW_TOKENS` (tokens the model generates; `1` for
single-token-answer tasks).

### templates.py (required)

The input text templates and a `fill_template(...)` function. Every placeholder
in a template must correspond to at most one causal-model variable — don't
pre-concatenate variables into intermediate strings; the template's `.format()`
is the formatting step.

### checker.py (optional)

```python
checker(neural_output: dict, causal_output: str) -> bool
```

Decides whether the model's output (`neural_output["string"]`) matches the
expected answer. `checker.py` is **optional**: when a task declares
`output_tokens` on its `CausalModel`, the string checker (and the
probability-path score tokens) are derived from it automatically. Ship a
`checker.py` only when you need a genuinely-custom matcher — it takes precedence
over the derived one. A task that offers **neither** `output_tokens` **nor**
`checker.py` for its target variable cannot grade its output and fails to load.
Pick the semantics your task needs: exact stripped equality for single-token
answers, `actual.startswith(expected)` for `max_new_tokens > 1` tasks whose model
continues past the answer, etc.

### metrics.py / __init__.py / summary.ipynb

`metrics.py` holds optional task-specific metric helpers. `__init__.py` re-exports
the package's public surface (at minimum `CAUSAL_MODEL` / the factory). An optional
`summary.ipynb` demonstrates the *task* (causal model, samples, token positions,
counterfactuals) on CPU — it must not load a language model.

## 2. Registering the task with Hydra (`causalab/configs/task/<name>.yaml`)

A task package is not runnable until it has a Hydra config in the `task` config
group. This file is what makes a runner's `- /task: <name>` resolve; it mounts at
`cfg.task` by group-default packaging, so it carries **no** `# @package` directive
(matching the other files in `causalab/configs/task/`).

```yaml
name: <name>                  # must equal the task package dir name (the loader key)

# Intervention target. `target_variables` (plural) is the canonical key that
# `locate` reads first; several analyses (baseline, activation_manifold,
# output_manifold, path_steering, pullback) read only the singular
# `target_variable`, and task resolution raises if it is null. Emit BOTH,
# pointing at the same variable, and keep them in sync.
target_variable: <var>
target_variables: [<var>]

# Generation / decoding
max_new_tokens: 1             # = MAX_NEW_TOKENS in config.py (1 for single-token tasks)

# Dataset sizing (root-level knobs — see docs/CODEBASE.md invariant 12)
n_train: 1000
n_test: 50
enumerate_all: false          # true only when the input space is small enough to enumerate exhaustively
balanced: false               # balance the generated dataset over target_variable
resample_variable: all        # "all" = CF resamples every input var; a var name = CF differs in only that one

# Scoring
intervention_metric: string_match   # "string_match" for single-token answers; "kl" for distributional targets

# Visualization / geometry
colormap: viridis             # resolution-critical (see note below)
colormap2: null               # resolution-critical for path_steering
distance_function: hellinger  # defensive; only used by a task's own isometry: block
```

**Resolution-critical keys.** `colormap` and `colormap2` are read via
`${task.colormap}` / `${task.colormap2}` by the shipped viz/manifold analysis
configs (`subspace`, `activation_manifold`, `output_manifold`, `path_steering`), so
a runner that mounts one of those analyses fails to resolve `- /task: <name>` if
they are absent — keep them present even for non-manifold tasks. See
`docs/CODEBASE.md` §5 for the full required-keys contract.

Sanity-check the config parses:

```bash
uv run python -c "from omegaconf import OmegaConf; print(OmegaConf.to_yaml(OmegaConf.load('causalab/configs/task/<name>.yaml')))"
```

## 3. Validating a new task

Before running the full pipeline, confirm the task tokenizes cleanly and the model
can actually solve it.

**Pre-flight tokenizer check (model-free, blocking).** Catches tokenization
mismatches (e.g. an orphaned trailing space in the answer) without loading the
model:

```bash
uv run python -m causalab.tasks.preflight --task <name> --model <model_name>
```

Exit 0 = clean; exit 1 = tokenization error to fix before proceeding; exit 2 = the
check couldn't run (e.g. a factory task that needs its run config to sample) — fall
through to the accuracy check.

**Accuracy gate.** Run `baseline` (or a short ad-hoc `generate` loop) on ~64
examples. If the model solves **< 20%**, the task is behaviorally inert for that
model — downstream interventions produce degenerate geometry (near-100% "other"
probability mass). Fix the prompt/templates or switch models before continuing.

**Single-token / spacing.** For clean token-level interventions, filter variable
values to single-token-in-context values, and confirm which spacing variant the
model actually emits (leading space vs. none) so intervention token alignment stays
correct; update `templates.py` / `config.py` to match.

## Active tasks

| Task | Description | Dimensionality |
|------|-------------|----------------|
| `natural_domains_arithmetic` | Unified weekdays/months/hours/age/integer/alphabet | factory, 1D (cyclic or linear) |
| `graph_walk` | Next-node prediction on graphs | factory, 1D or 2D |
| `entity_binding` | Positional entity retrieval | — |
| `hierarchical_equality` | Hierarchical variable equality | — |
| `identity_naming` | Entity → canonical name lookup (factory) | — |
| `MCQA` | Multiple-choice question answering | — |
| `IOI` | Indirect object identification (coverage-oriented runner) | — |
| `hex_color` | Hex-code → color-name mapping | — |
| `subject_object_relations` | Subject→object relation recall (LRE-style) | factory |
