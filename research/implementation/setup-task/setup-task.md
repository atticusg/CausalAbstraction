---
name: setup-task
description: Create, explore, investigate, or add a new task from a markdown specification. Use when the user wants to set up a new task, add a task, explore an existing task's structure, investigate a task or dataset, look at task data, understand a dataset, inspect variables or templates, examine counterfactuals, or do anything related to understanding or building a task.
---

# Setup Task Skill

Narrate like a colleague walking the user through building a task — say what you extracted from the spec and what you're about to create in plain prose; don't echo internal variable names back at them.

Creates all task files for a new task from a markdown specification. The task is scaffolded **out of tree**, under the run's working directory at `$WORKDIR/code/tasks/<task_name>/` (mirroring the setup-methods and setup-analyses guides); the runner imports it via the session-local `tasks.<name>` fallback when it runs with `CAUSALAB_SESSION_CODE=$WORKDIR` set. Shipped `causalab.tasks.<name>` always takes precedence and is read-only reference — never author task code inside the lab-managed library checkout at `~/.silico/libraries/causalab-internal/` (it is re-synced / image-resident and would be clobbered or is unwritable). Shipping a stabilized task upstream is a separate causalab-repo PR, not part of this run.

## Required Reading

Before running this skill, read:
- `instructions/task_package_layout.md` — the file-by-file structure of a task package and the conventions every task must follow.
- `instructions/task_quality_objectives.md` — quality objectives a task must satisfy to support causal abstraction (granularity, grading totality, input determinism, single-token decoding, value coverage). Use as a checklist when drafting or reviewing a specification.
- `CONVENTIONS.md` — the task-package invariants (required variables, naming, counterfactuals, token positions, metrics, checker) and the reference helper snippets the steps below point to.

Below, `$WORKDIR` stands for a working directory under your experiment's output — where the session-local `code/` tree, drafts, and scratch files land. Drafts that go through iteration with the user — task specifications drafted from a PDF or through interactive creation — land in `$WORKDIR/set_up_task_draft.md` first. Copy to the task package's `set_up_task.md` (under `$WORKDIR/code/tasks/{task_name}/`) only on user approval.

## Files Created (All)

Task package (under `$WORKDIR/code/tasks/<task_name>/`):
1. config.py
2. templates.py
3. causal_models.py
4. counterfactuals.py
5. token_positions.py
6. checker.py - only if the task needs a custom matcher; otherwise derived from `output_tokens` (see 4.6)
7. metrics.py
8. __init__.py
9. summary.ipynb - Task overview notebook

Plus one Hydra config-group file (a *different* directory):
10. `$WORKDIR/code/configs/task/<task_name>.yaml` — the task config that lets a runner's `- /task: <task_name>` resolve. Mirrors how the setup-analyses guide emits `configs/analysis/<name>.yaml`.

## Workflow

### Step 1: Get Specification

The specification path comes from the argument the skill was invoked with, if any; otherwise look for a `PATH_TO_SEED` mentioned in the conversation context. If a specification path is already available, use it and skip to Step 2. Otherwise, ask the user:

> "Do you have an existing specification for this task?"
>
> Options:
> 1. **Yes, I have an MD file** — provide the path
> 2. **Yes, I have a PDF document (e.g., a paper)** — provide the path
> 3. **No, let's create one together**

**Existing markdown spec:** Ask for the path → store it → go to Step 2.

**PDF document:** Ask for the PDF path → read instructions from `instructions/create_specification_from_pdf.md` → follow those instructions to extract information from the PDF and create the specification section by section, getting user approval for each section. While iterating, write the draft to `$WORKDIR/set_up_task_draft.md`. After the user approves the full draft:
- Create the task folder: `$WORKDIR/code/tasks/{task_name}/`
- Copy the draft: copy `$WORKDIR/set_up_task_draft.md` to `$WORKDIR/code/tasks/{task_name}/set_up_task.md`.
- Store that path as the specification file → go to Step 2.

**Interactive creation:** Read instructions from `instructions/create_specification.md` → follow those instructions to guide the user through creating a task specification file section by section. While iterating, write the draft to `$WORKDIR/set_up_task_draft.md`. After the user approves the full draft:
- Create the task folder: `$WORKDIR/code/tasks/{task_name}/`
- Copy the draft: copy `$WORKDIR/set_up_task_draft.md` to `$WORKDIR/code/tasks/{task_name}/set_up_task.md`.
- Store that path as the specification file → go to Step 2.

### Step 2: Read Specification and Extract Information

Read the MD file and extract:
- Task name (from YAML frontmatter `name:` field or filename)
- Input/output/intermediate variables
- Templates
- Counterfactuals
- Token positions
- **Model** (from the `models:` YAML block in the specification)
- **Output token mode** (from the `output_token_mode:` field, defaults to `"full"` if not specified). One of:
  - `"full"` — evaluate the complete generated output against `raw_output`
  - `"single_constrained"` — `raw_output` must always be a single token; filter values during validation to ensure this
  - `"first_token_only"` — `raw_output` can be multi-token, but the checker only compares the first generated token against the first token of `raw_output`

Print a brief summary of what was extracted. Flag if something is not clear or missing.

Confirm with the user before creating the task files:
> "I've extracted the following from the specification:
> - Task name: the task name
> - Variables: the list of variables
> - Templates: how many templates
> - Counterfactuals: the counterfactual types
> - Model: the model
> - Output token mode: the output token mode
>
> Does this look correct? Should I proceed with creating the task files?"

### Step 3: Determine Output Directory

Scaffold out of tree, under the run's working directory, mirroring the setup-methods and setup-analyses guides:

```bash
TASK_DIR="$WORKDIR/code/tasks/{task_name}"
TASK_CONFIG="$WORKDIR/code/configs/task/{task_name}.yaml"   # Hydra config-group file (Step 4.10)
```

The runner imports a session-local task through the `tasks.<name>` fallback in `load_task` when it runs with `CAUSALAB_SESSION_CODE=$WORKDIR` set — `scripts/run_exp.sh` then puts `$WORKDIR/code/` on `PYTHONPATH` and appends `$WORKDIR/code/configs/` to Hydra's search path, so the task config in `$TASK_CONFIG` is what makes a runner's `- /task: {task_name}` resolve. Shipped `causalab.tasks.<name>` always takes precedence (the loader resolves shipped first). Everything this skill creates is session-local under `$WORKDIR/code/`; never write into the lab-managed library checkout.

**Collision guard:** refuse to scaffold if the name is already taken — either a shipped `causalab.tasks.{task_name}` or an existing `$WORKDIR/code/tasks/{task_name}/`. A session-local name must not shadow a shipped task (the loader resolves shipped first, so a colliding session-local task would never load). If the name collides, stop and ask the user for a different name.

Confirm with the user before creating the files:
> "I will create the following files in the task folder:
> - config.py
> - causal_models.py
> - counterfactuals.py
> - token_positions.py
> - checker.py (only if a custom matcher is needed — otherwise derived from output_tokens, see 4.6)
> - metrics.py
> - __init__.py
> - summary.ipynb
>
> plus the Hydra task config in the `configs/task/` group.
>
> Should I proceed with creating these files?"

Create `$TASK_DIR` if it doesn't exist.

**Session-local import namespace.** Because the task lives under `$WORKDIR/code/tasks/<name>/`, Python imports use the bare `tasks.<name>` namespace (not `causalab.tasks.<name>`), which requires `$WORKDIR/code` on `PYTHONPATH`/`sys.path`. The runner sets this automatically when `CAUSALAB_SESSION_CODE=$WORKDIR` is set (`scripts/run_exp.sh`); for ad-hoc `python`/notebook use, prepend it first (`export PYTHONPATH="$WORKDIR/code:$PYTHONPATH"`, or a `sys.path.insert` bootstrap cell). The example snippets below use `tasks.<name>`.

### Step 4: Create All Files

**Create files in order:**

#### 4.1 config.py
- Read template: `setup-task/templates/config.py`
- Create the file with constants from the specification

#### 4.2 templates.py
- Read template: `setup-task/templates/templates.py`
- Always create this file, even for single-template tasks

#### 4.3 causal_models.py
- Read template: `setup-task/templates/causal_models.py`
- Create with variables and mechanisms from specification

#### 4.4 counterfactuals.py
- Read template: `setup-task/templates/counterfactuals.py`
- Create counterfactual generators from specification

#### 4.5 token_positions.py
- Read template: `setup-task/templates/token_positions.py`
- Create token position definitions from specification

#### 4.6 checker.py
- The single match authority for base accuracy and intervention scoring. A task whose causal model declares `output_tokens` needs no `checker.py` — `derive_checker` derives the checker from it; write one only for a genuinely-custom matcher (it takes precedence over the derived checker — see `templates/causal_models.py`).
- Read template: `setup-task/templates/checker.py`
- If writing one, create output validation logic based on `output_token_mode`:
  - `"full"`: compare full stripped strings (`actual == expected`)
  - `"single_constrained"`: same as `"full"` (raw_output is guaranteed single-token by filtering)
  - `"first_token_only"`: compare only the first token of each. The checker must accept a `tokenizer` parameter (passed during setup) and compare `tokenizer.encode(actual)[0] == tokenizer.encode(expected)[0]`

#### 4.7 metrics.py
- Read template: `setup-task/templates/metrics.py`
- Create metrics functions

#### 4.8 __init__.py
- Read template: `setup-task/templates/__init__.py`
- Create with proper exports

#### 4.9 summary.ipynb
Create a CPU-only task overview notebook at `$TASK_DIR/summary.ipynb` that demonstrates the *task* (causal model, samples, token positions, counterfactuals), not the language model.

- **Never load a model here.** Do not add a cell that loads an `LMPipeline` or runs an LM forward pass — no `pipeline.generate(...)`, no `m.run_forward(...)`-style line. Show outputs with `CAUSAL_MODEL.sample_input()` only. Model validation belongs in Step 5, not the notebook.
- **Build the notebook with `nbformat`** so every cell gets a unique `"id"` (hand-written JSON raises `MissingIDFieldWarning`). The `nbformat` builder snippet and the required cell order are in `CONVENTIONS.md` § "Notebook builder".

#### 4.10 configs/task/{task_name}.yaml (Hydra task config)

This is the file that lets a runner's `- /task: {task_name}` resolve. Earlier revisions had to hand-write it because the skill skipped this step (issue #263). It lives in a **different directory** from the package — the `configs/task/` group, not `$TASK_DIR`.

- Read template: `setup-task/templates/task.yaml`
- Write it to `$TASK_CONFIG` (`$WORKDIR/code/configs/task/{task_name}.yaml`), creating the `configs/task/` parents if needed.
- Substitutions:
  - `{{TASK_NAME}}` → the task name (must equal the package dir name `{task_name}`, the loader key).
  - `{{TARGET_VARIABLE}}` → the intervention target: the intermediate (or output) variable under test, drawn from the spec. It fills **both** `target_variable` (singular) and `target_variables: [...]` (plural) — the template emits both. Plural is the canonical shipped key (`locate` reads it first); singular is still needed because several analyses (`baseline`, `activation_manifold`, `output_manifold`, `path_steering`, `pullback`) read only it and `resolve_task()` raises if it is null. Keep the two in sync.
  - `{{MAX_NEW_TOKENS}}` → the `MAX_NEW_TOKENS` value from config.py (1 for single-token tasks).
- Leave the remaining dataset / metric / viz keys at their template defaults, adjusting only where the spec dictates (e.g. `intervention_metric: kl` for a distributional target, `balanced: true` to balance over the target, `enumerate_all: true` for a small enumerable input space).
- Keep `colormap` / `colormap2` / `distance_function` present even if the task isn't a manifold task. `colormap` and `colormap2` are **resolution-critical**: shipped viz/manifold analysis configs interpolate `${task.colormap}` (subspace, activation_manifold, output_manifold, path_steering) and `${task.colormap2}` (path_steering), so a runner that mounts one of those analyses fails to resolve `- /task: {task_name}` if they're absent. `distance_function` is **defensive** — no analysis config interpolates it (it's only self-referenced inside a task's own `isometry:` block, which this template omits) and `path_steering` reads it with a default — kept for forward-compat / a manifold task that later adds an `isometry:` block.

Sanity-check the emitted config:
```bash
uv run python -c "from omegaconf import OmegaConf; print(OmegaConf.to_yaml(OmegaConf.load('$TASK_CONFIG')))"
```

The config carries no `# @package` directive: it mounts at `cfg.task` via group-default packaging when selected with `- /task: {task_name}`, matching the shipped `causalab/configs/task/*.yaml`. If Hydra rejects the file, check the YAML indentation and re-emit.

### Step 5: Model Validation

Load the model specified in the specification and validate the task works end-to-end.

#### 5.0 Pre-flight tokenizer check (model-free, blocking gate)

Run the model-free tokenizer pre-flight **before loading the model**, on the freshly-scaffolded session-local task with the model from the spec's `models:` block:

```bash
# Session-local task needs $WORKDIR/code on PYTHONPATH (a shipped task does not).
cd ~/.silico/libraries/causalab-internal
PYTHONPATH="$WORKDIR/code${PYTHONPATH:+:$PYTHONPATH}" \
  uv run python -m causalab.tasks.preflight --task {task_name} --model {model_name}
```

- **Exit 0** — clean, proceed to 5.1.
- **Exit 1** — tokenization error(s) found. **Block.** Apply the fix it names, then re-run; do not load the model until it passes.
- **Exit 2** — the check could not run (e.g. a factory task that needs its run config to sample). Not a tokenization finding; note it and fall through to 5.1, relying on the token-alignment test (5.4) instead.

What this check catches, the #169 orphan-trailing-space trap, and the example blocking error are documented in `../running-experiments.md` § "Pre-flight tokenizer check (model-free, blocking)".

#### 5.1 Load the model

```python
import torch
from causalab.neural.pipeline import LMPipeline
from tasks.[task_name].config import MAX_TASK_TOKENS, MAX_NEW_TOKENS
from tasks.[task_name].causal_models import CAUSAL_MODEL

try:  # custom checker.py, when the task ships one
    from tasks.[task_name].checker import checker
except ImportError:  # none — derive it from output_tokens, as the loader does
    from causalab.causal.causal_model import derive_checker
    from tasks.[task_name].causal_models import TARGET_VARIABLE
    checker = derive_checker(
        CAUSAL_MODEL.output_tokens[TARGET_VARIABLE],
        (CAUSAL_MODEL.match_modes or {}).get(TARGET_VARIABLE, "exact"),
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = LMPipeline(
    "[model_name]",
    max_new_tokens=MAX_NEW_TOKENS,
    device=DEVICE,
    max_length=MAX_TASK_TOKENS,
)
```

**⚠️ GPU memory rule for every validation `generate()` call below.** Never pass a
whole sample set (or all few-shot prompts) to a single `pipeline.generate(...)` call.
Batched attention plus full-vocab logits scale with batch × sequence length; with
four-shot prompts a 64-prompt batch has been observed to OOM at ~71 GB. Always chunk
to a small `EVAL_BATCH_SIZE` and pass `output_scores=False` unless you actually need
the logits. The accuracy test in 5.3 shows the pattern.

#### 5.2 Single-token variable filtering

**Behavior depends on `output_token_mode`:**

- **`"single_constrained"`**: Single-token filtering is **required**. Filter both input variable values and output values so that `raw_output` always tokenizes to a single token — filtering is mandatory, no confirmation needed.
- **`"first_token_only"`**: Single-token filtering of output values is **not needed** (the checker handles multi-token outputs). Still optionally filter input variable values for cleaner interventions.
- **`"full"`**: Single-token filtering is **optional**, as before.

For `"full"` or `"first_token_only"` mode, ask the user:
> "Do you want to filter input variable values to be single-token only?
> This is recommended when token-level interventions need consistent alignment."

If the user says yes (or mode is `"single_constrained"`), filter each variable's value list to single-token-in-context values — the `filter_single_token` helper and the leading-space-in-context rule are in `CONVENTIONS.md` § "Single-token filtering". After filtering, update `config.py` with the filtered lists. If a variable list drops below 5 values, warn the user:
> "⚠️ This variable only has a few single-token values remaining. Consider expanding the value pool or relaxing the single-token constraint."

#### 5.3 Accuracy test (64 examples)

Test whether the model can actually solve the task. Sample 64 examples up front and score them in chunks with the checker — the chunked accuracy-test loop (which mode picks the first-token vs. standard checker, and the GPU-memory chunking pattern) is in `CONVENTIONS.md` § "Accuracy-test loop". Report `correct/total` and the accuracy.

**If accuracy < 20%** the task is behaviorally inert for this model — it violates
Objective 3 (input determinism), and running the downstream pipeline on it produces
degenerate geometry (near-100% "other" probability mass). Treat sub-threshold accuracy
as a rejection, not a soft warning: the task needs *repair* (explicit framing /
in-context demonstrations to fix the determinism) or *exclusion* before it can be
re-run. Do not finalize the task or run any downstream analysis on it. Surface the
rejection to the user — the task name, model, accuracy (correct over total), and
gold-token probability mass (the prob_accuracy from the accuracy test, now meaningful
with space-/case-agnostic scoring):
  > "⚠️ The model only solves a small fraction of the examples, below the 20% minimum
  > threshold. The task may not be suitable for this model, or there may be a template or
  > spacing issue. Investigating spacing variants next..."

**If accuracy >= 20%:** Report success and proceed to spacing check.

#### 5.4 Spacing check

Even if accuracy is acceptable, verify token alignment, then test both spacing variants and pick the one the model actually emits — the alignment-inspection and two-variant loops are in `CONVENTIONS.md` § "Spacing check".

**MANDATORY hard automated step: test both spacing variants, pick the one the
model actually emits, and record it.** Auto-detecting and recording the working spacing
runs unconditionally. With the space-/case-agnostic scoring fix the probability instrumentation
is robust to either spacing, but the spec must still record the variant the model emits so
intervention token-alignment stays clean.

After choosing the working variant, update templates.py and causal_models.py accordingly, then re-run the accuracy test on 16 examples to confirm.

Report the validation results to the user:
> "Model validation results for this model:
> - Accuracy: correct over total
> - Token alignment: matched or mismatched
> - Any fixes applied
>
> If accuracy is below 20%: 'The model struggles with this task. Should we continue, try a different model, or adjust the templates?'
> If accuracy is at or above 20%: 'The model can solve this task. Should I proceed with final verification?'"

### Step 6: Verify All Files Created

Run verification checks:

```bash
# Session-local imports need $WORKDIR/code on PYTHONPATH (a shipped task would
# instead import as `causalab.tasks.[task_name]`). Run from the library checkout
# so `causalab` (the shipped package + preflight) also resolves.
cd ~/.silico/libraries/causalab-internal
export PYTHONPATH="$WORKDIR/code${PYTHONPATH:+:$PYTHONPATH}"

# Check all files exist
ls -la "$TASK_DIR/"

# Test imports
uv run python -c "from tasks.[task_name] import CAUSAL_MODEL; print('Import successful')"

# Test sampling
uv run python -c "from tasks.[task_name] import CAUSAL_MODEL; s = CAUSAL_MODEL.sample_input(); print('Sample:', s['raw_input'][:50], '...')"

# Test counterfactuals
uv run python -c "from tasks.[task_name].counterfactuals import COUNTERFACTUAL_GENERATORS; print('Counterfactuals:', list(COUNTERFACTUAL_GENERATORS.keys()))"

# Verify the Hydra task config parses and carries the required keys
uv run python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$WORKDIR/code/configs/task/[task_name].yaml')
required = ['name', 'n_train', 'n_test', 'enumerate_all', 'balanced', 'resample_variable', 'max_new_tokens', 'intervention_metric']
missing = [k for k in required if k not in c]
assert not missing, f'task config missing required keys: {missing}'
# loader contract: either singular or plural satisfies the intervention target
assert ('target_variable' in c) or ('target_variables' in c), 'need target_variable or target_variables'
assert c.name == '[task_name]', f'task config name {c.name!r} != [task_name]'
print('Task config OK:', c.name)
"
```

**If verification fails:** Fix the error and re-run verification.

### Step 7: Report Results

Print summary:
```
Task Setup Complete!
====================

Created files (under $WORKDIR/code/tasks/[task_name]/):
  - config.py
  - causal_models.py
  - counterfactuals.py
  - token_positions.py
  - checker.py (only if a custom matcher was needed; otherwise derived from output_tokens)
  - metrics.py
  - __init__.py
  - summary.ipynb

Hydra task config (configs/task/ group):
  - the task config, which lets the runner's task selection resolve

Model validation:
  - Accuracy: correct over total
  - Token alignment: matched, or fixed (trailing space / leading space)

Verification:
  - Import successful
  - Sampling works
  - Counterfactuals work

Usage (session-local — the runner puts $WORKDIR/code on PYTHONPATH):
  from tasks.[task_name] import CAUSAL_MODEL
  sample = CAUSAL_MODEL.sample_input()

Shipping upstream (out of scope for this run): promoting a stabilized task into the
shipped causalab/tasks/ tree is a separate causalab-repo PR.

Summary notebook:
  Open the task's summary.ipynb to explore the task
```

After printing the summary, ask the user:

> "Now you have all the files you need to start running experiments. Would you like me to help you run an experiment?"

If the user says yes, hand off to running the experiment. Run mechanics live at `../running-experiments.md`.
