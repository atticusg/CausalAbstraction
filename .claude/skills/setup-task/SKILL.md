---
name: setup-task
description: Create, explore, investigate, or add a new task from a markdown specification. Use when the user wants to set up a new task, add a task, explore an existing task's structure, investigate a task or dataset, look at task data, understand a dataset, inspect variables or templates, examine counterfactuals, or do anything related to understanding or building a task.
args: <path_to_specification>
---

# Setup Task Skill

Creates all task files in `causalab/tasks/<task_name>/` from a markdown specification.

## Required Reading

Before running this skill, read:
- `.claude/skills/research-session/CONVENTIONS.md` — research-session layout and active-session protocol.
- `.claude/skills/setup-task/instructions/task_package_layout.md` — the file-by-file structure of a `causalab/tasks/<name>/` package and the conventions every task must follow.

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop.

## Autonomous vs Interactive Mode

**Check the `CAUSALAB_AUTONOMOUS` environment variable:**
```bash
echo $CAUSALAB_AUTONOMOUS
```

- **If set to "1"**: Run autonomously without asking for approval
- **If not set or empty**: Run interactively - ask for user approval before each major step

## Arguments

When invoked, the skill receives a single argument: the path to the specification file.
- If argument provided: use it as the specification path
- If no argument: look for PATH_TO_SEED in the conversation context

## Issue Tracking

**Use the `document-issues` skill throughout this workflow.** Issues are written to `${SESSION_DIR}/issues.md` (top level — see `.claude/skills/research-session/CONVENTIONS.md`).

## Step 0a: Resolve the Active Session

```bash
if [ ! -f agent_logs/.current ]; then
    echo "No active research session. Run /research-session first." >&2
    exit 1
fi
SESSION_NAME="$(cat agent_logs/.current)"
SESSION_DIR="agent_logs/${SESSION_NAME}"
if [ ! -d "${SESSION_DIR}" ]; then
    echo "Active session marker points at missing directory ${SESSION_DIR}. Run /research-session." >&2
    exit 1
fi
```

In autonomous mode (`CAUSALAB_AUTONOMOUS=1`), the harness pre-creates the session and writes `.current` before invoking the skill. The skill assumes `${SESSION_DIR}` is valid and proceeds.

Drafts that go through iteration with the user — task specifications drafted via Option 2 (PDF) or Option 3 (interactive) — land in `${SESSION_DIR}/plan/set_up_task_draft.md` first. Promote to `causalab/tasks/{task_name}/set_up_task.md` only on user approval.

## Files Created (All)
1. config.py
2. templates.py
3. causal_models.py
4. counterfactuals.py
5. token_positions.py
6. checker.py
7. metrics.py
8. __init__.py
9. summary.ipynb - Task overview notebook

## Workflow

### Step 0: Check Mode and Get Specification

First, check if running in autonomous mode:
```bash
echo $CAUSALAB_AUTONOMOUS
```

Store the result - if "1", skip all interactive prompts. If empty/unset, run interactively.

#### Autonomous mode (CAUSALAB_AUTONOMOUS=1)

Use the specification path from:
1. The argument passed to the skill (if any)
2. PATH_TO_SEED mentioned in the conversation

Skip directly to Step 1.

#### Interactive mode (default)

If a specification path was provided as an argument, use it and skip to Step 1.

Otherwise, ask the user:

> "Do you have an existing specification for this task?"
>
> Options:
> 1. **Yes, I have an MD file** — provide the path
> 2. **Yes, I have a PDF document (e.g., a paper)** — provide the path
> 3. **No, let's create one together**

**Option 1 (MD file):** Ask for the path → store it → go to Step 1.

**Option 2 (PDF document):** Ask for the PDF path → read instructions from `.claude/skills/setup-task/instructions/create_specification_from_pdf.md` → follow those instructions to extract information from the PDF and create the specification section by section, getting user approval for each section. While iterating, write the draft to `${SESSION_DIR}/plan/set_up_task_draft.md`. After the user approves the full draft:
- Create the task folder: `causalab/tasks/{task_name}/`
- Promote the draft: copy `${SESSION_DIR}/plan/set_up_task_draft.md` to `causalab/tasks/{task_name}/set_up_task.md`.
- Store the promoted path as the specification file → go to Step 1.

**Option 3 (Interactive creation):** Read instructions from `.claude/skills/setup-task/instructions/create_specification.md` → follow those instructions to guide the user through creating a task specification file section by section. While iterating, write the draft to `${SESSION_DIR}/plan/set_up_task_draft.md`. After the user approves the full draft:
- Create the task folder: `causalab/tasks/{task_name}/`
- Promote the draft: copy `${SESSION_DIR}/plan/set_up_task_draft.md` to `causalab/tasks/{task_name}/set_up_task.md`.
- Store the promoted path as the specification file → go to Step 1.

### Step 1: Read Specification and Extract Information

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

**🔒 APPROVAL CHECKPOINT (if not autonomous):**
> "I've extracted the following from the specification:
> - Task name: {task_name}
> - Variables: {list of variables}
> - Templates: {number of templates}
> - Counterfactuals: {list of counterfactual types}
> - Model: {model_name}
> - Output token mode: {output_token_mode}
>
> Does this look correct? Should I proceed with creating the task files?"

If autonomous mode: proceed immediately.

### Step 2: Determine Output Directory

Use the default: `causalab/tasks/<task_name>/`

**🔒 APPROVAL CHECKPOINT (if not autonomous):**
> "I will create the following files in `causalab/tasks/{task_name}/`:
> - config.py
> - causal_models.py
> - counterfactuals.py
> - token_positions.py
> - checker.py
> - metrics.py
> - __init__.py
> - summary.ipynb
>
> Should I proceed with creating these files?"

If autonomous mode: proceed immediately.

Create the directory if it doesn't exist.

### Step 3: Create All Files

**Create files in order:**

#### 3.1 config.py
- Read template: `.claude/skills/setup-task/templates/config.py`
- Create the file with constants from the specification

#### 3.2 templates.py
- Read template: `.claude/skills/setup-task/templates/templates.py`
- Always create this file, even for single-template tasks

#### 3.3 causal_models.py
- Read template: `.claude/skills/setup-task/templates/causal_models.py`
- Create with variables and mechanisms from specification

#### 3.4 counterfactuals.py
- Read template: `.claude/skills/setup-task/templates/counterfactuals.py`
- Create counterfactual generators from specification

#### 3.5 token_positions.py
- Read template: `.claude/skills/setup-task/templates/token_positions.py`
- Create token position definitions from specification

#### 3.6 checker.py
- Read template: `.claude/skills/setup-task/templates/checker.py`
- Create output validation logic based on `output_token_mode`:
  - `"full"`: compare full stripped strings (`actual == expected`)
  - `"single_constrained"`: same as `"full"` (raw_output is guaranteed single-token by filtering)
  - `"first_token_only"`: compare only the first token of each. The checker must accept a `tokenizer` parameter (passed during setup) and compare `tokenizer.encode(actual)[0] == tokenizer.encode(expected)[0]`

#### 3.7 metrics.py
- Read template: `.claude/skills/setup-task/templates/metrics.py`
- Create metrics functions

#### 3.8 __init__.py
- Read template: `.claude/skills/setup-task/templates/__init__.py`
- Create with proper exports

#### 3.9 summary.ipynb
Create a task overview notebook at `causalab/tasks/[task_name]/summary.ipynb` that demonstrates the task.

**Notebook structure:**
1. **Title & Description** - Task name and brief description from specification
2. **Setup** - Import the task modules
3. **Causal Model Overview** - Show the causal model variables and their relationships
4. **Sample Generation** - Generate and display sample inputs
5. **Token Positions** - Visualize token positions on a sample input
6. **Counterfactual Generation** - Show example counterfactuals

**Write the notebook as JSON using the Write tool:**
```json
{
  "cells": [...],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

**Required cells:**
1. Markdown: Task title and description
2. Code: Imports (`from causalab.tasks.[task_name] import causal_model, COUNTERFACTUAL_GENERATORS`)
3. Markdown: "Causal Model Variables"
4. Code: `print(causal_model.values)` or similar to show variables
5. Markdown: "Sample Generation"
6. Code: Generate and print several samples
7. Markdown: "Token Positions"
8. Code: Show token position definitions and visualize on a sample (tokenize input, highlight positions)
9. Markdown: "Counterfactual Generation"
10. Code: Generate and display a counterfactual example

### Step 4: Model Validation

Load the model specified in the specification and validate the task works end-to-end.

#### 4.1 Load the model

```python
import torch
from causalab.neural.pipeline import LMPipeline
from causalab.tasks.[task_name].config import MAX_TASK_TOKENS, MAX_NEW_TOKENS
from causalab.tasks.[task_name].causal_models import causal_model
from causalab.tasks.[task_name].checker import checker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = LMPipeline(
    "[model_name]",
    max_new_tokens=MAX_NEW_TOKENS,
    device=DEVICE,
    max_length=MAX_TASK_TOKENS,
)
```

#### 4.2 Single-token variable filtering

**Behavior depends on `output_token_mode`:**

- **`"single_constrained"`**: Single-token filtering is **required**. Filter both input variable values and output values so that `raw_output` always tokenizes to a single token. Skip the approval checkpoint — filtering is mandatory.
- **`"first_token_only"`**: Single-token filtering of output values is **not needed** (the checker handles multi-token outputs). Still optionally filter input variable values for cleaner interventions.
- **`"full"`**: Single-token filtering is **optional**, as before.

**🔒 APPROVAL CHECKPOINT (if not autonomous, and mode is `"full"` or `"first_token_only"`):**
> "Do you want to filter input variable values to be single-token only?
> This is recommended when token-level interventions need consistent alignment."

If the user says yes (or mode is `"single_constrained"`), filter each variable's value list using the model's tokenizer.

**Critical: check tokenization in context, not in isolation.** Tokenizers prepend a leading space to tokens that appear mid-sentence. For example, in `"The capital city of France is"`, the tokenizer sees `" France"` (with leading space), not `"France"`. So you must check the token count of the value *as it would appear in the template*.

**How to determine if a value gets a leading space:**

1. Look at the template string (e.g., `"The capital city of {country} is"`)
2. Find where `{country}` appears — is it preceded by a space or at the start of the string?
3. If preceded by a space (the common case): the tokenizer will encode the value with a leading space. Check `" France"`, not `"France"`.
4. If at the very start of the template: the tokenizer will NOT add a leading space. Check `"France"` as-is.

**Filtering logic:**

```python
from causalab.tasks.[task_name].config import TEMPLATE  # or TEMPLATES[0]

# For each variable with a value list (e.g., NAMES, OBJECTS, PLACES):
def filter_single_token(values: list[str], var_placeholder: str, template: str) -> list[str]:
    """Filter values to those that tokenize as a single token in context."""
    # Determine if variable gets a leading space in the template
    # Find the placeholder position and check what precedes it
    placeholder = "{" + var_placeholder + "}"
    idx = template.find(placeholder)
    has_leading_space = idx > 0 and template[idx - 1] == " "

    kept = []
    for value in values:
        # Tokenize the value as it appears in context
        token_input = (" " + value) if has_leading_space else value
        token_ids = pipeline.tokenizer.encode(token_input, add_special_tokens=False)
        if len(token_ids) == 1:
            kept.append(value)

    return kept
```

**Apply to each variable list and report results:**

```python
# Example for a task with NAMES and OBJECTS
original_names = NAMES  # from config.py
filtered_names = filter_single_token(NAMES, "name", template)
print(f"NAMES: {len(original_names)} -> {len(filtered_names)} single-token values")
print(f"  Removed: {set(original_names) - set(filtered_names)}")

# Repeat for each variable list...
```

**After filtering:** Update `config.py` with the filtered lists. If a variable list drops below 5 values, warn the user:
> "⚠️ {variable} only has {n} single-token values remaining. Consider expanding the value pool or relaxing the single-token constraint."

#### 4.3 Accuracy test (64 examples)

Test whether the model can actually solve the task. The checker behavior depends on `output_token_mode`:

- **`"full"` or `"single_constrained"`**: Use the standard checker (exact match on full/single-token output).
- **`"first_token_only"`**: Use the first-token checker, which compares only the first generated token against the first token of the expected output using the tokenizer.

```python
correct = 0
total = 64
for _ in range(total):
    full_setting = causal_model.sample_input()
    lm_output = pipeline.generate([full_setting], output_scores=False)
    if checker(lm_output, full_setting["raw_output"]):
        correct += 1

accuracy = correct / total
print(f"Model accuracy: {correct}/{total} = {accuracy:.1%}")
```

**If accuracy < 20%:** Flag this to the user:
> "⚠️ The model only solves {accuracy:.0%} of examples ({correct}/{total}).
> This is below the 20% minimum threshold. The task may not be suitable for this model,
> or there may be a template/spacing issue. Investigating spacing variants next..."

**If accuracy >= 20%:** Report success and proceed to spacing check.

#### 4.4 Spacing check

Even if accuracy is acceptable, verify token alignment. Test **all four spacing variants** and report which one the model actually produces:

```python
# Get 16 examples
examples = [causal_model.sample_input() for _ in range(16)]

# For each example, check what the model actually generates
for full_setting in examples[:3]:  # Show details for first 3
    lm_output = pipeline.generate([full_setting], output_scores=False)
    generated = lm_output["string"]
    expected = full_setting["raw_output"]

    # Check token-level alignment
    expected_ids = pipeline.tokenizer.encode(expected, add_special_tokens=False)
    actual_ids = lm_output["sequences"][0].tolist()
    tokens_match = actual_ids == expected_ids

    print(f"Input:    {full_setting['raw_input']!r}")
    print(f"Expected: {expected!r} -> tokens {expected_ids}")
    print(f"Actual:   {generated!r} -> tokens {actual_ids}")
    print(f"Token match: {tokens_match}")
```

**MANDATORY: Test both spacing variants and use the one that works.**

Some models (e.g., GPT-2) use BPE tokenization where word tokens include a leading space (Ġ prefix). In these cases, the template should NOT end with a trailing space, and `raw_output` should include a leading space (e.g., `" France"`). Other models work better with a trailing space on the template and no leading space on `raw_output`.

**Test both variants on 16 examples and pick the one with higher accuracy:**

```python
# Variant A: trailing space on template, no leading space on raw_output
# Variant B: no trailing space on template, leading space on raw_output
# Test both, pick the one that gives higher accuracy and better token alignment
```

After choosing the working variant, update templates.py and causal_models.py accordingly, then re-run the accuracy test on 16 examples to confirm.

**🔒 APPROVAL CHECKPOINT (if not autonomous):**
> "Model validation results for {model_name}:
> - Accuracy: {correct}/{total} ({accuracy:.0%})
> - Token alignment: {match/mismatch}
> - {Any fixes applied}
>
> {If accuracy < 20%: 'The model struggles with this task. Should we continue, try a different model, or adjust the templates?'}
> {If accuracy >= 20%: 'The model can solve this task. Should I proceed with final verification?'}"

### Step 5: Verify All Files Created

Run verification checks:

```bash
# Check all files exist
ls -la causalab/tasks/[task_name]/

# Test imports
uv run python -c "from causalab.tasks.[task_name] import causal_model; print('Import successful')"

# Test sampling
uv run python -c "from causalab.tasks.[task_name] import causal_model; s = causal_model.sample_input(); print('Sample:', s['raw_input'][:50], '...')"

# Test counterfactuals
uv run python -c "from causalab.tasks.[task_name].counterfactuals import COUNTERFACTUAL_GENERATORS; print('Counterfactuals:', list(COUNTERFACTUAL_GENERATORS.keys()))"
```

**If verification fails:** Fix the error and re-run verification. Use the `document-issues` skill to log any problems encountered.

### Step 6: Report Results

Print summary:
```
Task Setup Complete!
====================

Created files:
  - causalab/tasks/[task_name]/config.py
  - causalab/tasks/[task_name]/causal_models.py
  - causalab/tasks/[task_name]/counterfactuals.py
  - causalab/tasks/[task_name]/token_positions.py
  - causalab/tasks/[task_name]/checker.py
  - causalab/tasks/[task_name]/metrics.py
  - causalab/tasks/[task_name]/__init__.py
  - causalab/tasks/[task_name]/summary.ipynb

Model validation ({model_name}):
  - Accuracy: {correct}/{total} ({accuracy:.0%})
  - Token alignment: ✓ matched / ✗ fixed (trailing space / leading space)

Verification:
  - Import successful
  - Sampling works
  - Counterfactuals work

Usage:
  from causalab.tasks.[task_name] import causal_model
  sample = causal_model.sample_input()

Summary notebook:
  Open causalab/tasks/[task_name]/summary.ipynb to explore the task
```

After printing the summary, ask the user:

> "Now you have all the files you need to start running experiments. Would you like me to help you run an experiment?"

If the user says yes, invoke the `run-experiment` skill.

## Key Conventions

**Required:** Every causal model MUST have `raw_input` and `raw_output` variables.


**Variable naming:** Use the EXACT variable names from the specification. Do not rename, re-case, or abbreviate variables. Use `snake_case` with underscores for multi-word names (e.g., `var_1` not `var1`, `s_name` not `S_NAME`). Token position dictionary keys should be lowercase (e.g., `capital`, `country`, `end`).

**Counterfactuals:** Implement ALL counterfactual types listed in the specification. Do not skip any. Each counterfactual type should have its own generator function and be included in `COUNTERFACTUAL_GENERATORS`. Use `.copy()` and `.intervene()` on traces to create counterfactuals.

**Template variable:** For tasks with a single template, do NOT include `template` as a causal variable. Instead, use `TEMPLATE = TEMPLATES[0]` as a module-level constant in causal_models.py. Only include `template` as an input variable when the task has multiple templates that the model must handle.

**Token positions:** The `create_token_positions` function MUST return the result of `build_token_position_factories()` directly — a `Dict[str, Callable]` of factory functions. Do NOT call the factories yourself (e.g., `factory(pipeline)`) — the experiment framework calls them later. The declarative `build_token_position_factories` approach works well for simple tasks with clear variable positions. However, use custom Python functions (the fallback approach) when:
- The task uses in-context learning (ICL) with many repeated examples — the declarative system finds the LAST occurrence, which may not be correct
- A variable appears multiple times in the template and you need a specific occurrence (not the last one)
- Positions require regex or complex string parsing to locate

**Metrics:** The `metric` function signature is ALWAYS `metric(neural_output: dict, causal_output: str) -> bool`. It receives the model's output dict (with a `"string"` key) and the expected causal output string. It does NOT have access to logits, the pipeline, or the tokenizer. Do not implement logit-based or probability-based metrics in `metrics.py` — those are computed separately during experiments.

**Checker:** Default to use `startswith`.

## Restrictions
- ONLY read templates from `.claude/skills/setup-task/templates/`
