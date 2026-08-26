# Setup-task conventions

Reference material for `setup-task.md` — the invariants every task package must satisfy, the restriction on template sources, and the long helper snippets the recipe points to (single-token filtering, the accuracy-test loop, the spacing check, the notebook builder). The recipe owns the step flow; this file owns the contracts and copy-in code.

## Key Conventions

**Required:** Every causal model MUST have `raw_input` and `raw_output` variables.

**Variable naming:** Use the EXACT variable names from the specification. Do not rename, re-case, or abbreviate variables. Use `snake_case` with underscores for multi-word names (e.g., `var_1` not `var1`, `s_name` not `S_NAME`). Token position dictionary keys should be lowercase (e.g., `capital`, `country`, `end`).

**Counterfactuals:** Implement ALL counterfactual types listed in the specification. Do not skip any. Each counterfactual type should have its own generator function and be included in `COUNTERFACTUAL_GENERATORS`. Use `.copy()` and `.intervene()` on traces to create counterfactuals.

**Template variable:** For tasks with a single template, do NOT include `template` as a causal variable. Instead, use `TEMPLATE = TEMPLATES[0]` as a module-level constant in causal_models.py. Only include `template` as an input variable when the task has multiple templates that the model must handle.

**Token positions:** The `create_token_positions` function MUST return `Dict[str, TokenPosition]` — materialized `TokenPosition` **instances**, not factory functions. Use the `build_token_positions` helper, which builds the declarative specs and materializes them against the pipeline in one call:

```python
from causalab.neural.token_positions import build_token_positions

return build_token_positions(token_position_specs, template, pipeline)
```

Consumers (e.g. the runner's `build_targets_for_grid`) read `.id` off the returned values; returning the bare factories from `build_token_position_factories()` crashes the `locate` step with `AttributeError: 'function' object has no attribute 'id'`. The declarative spec approach works well for simple tasks with clear variable positions. However, use custom Python functions (the fallback approach) when:
- The task uses in-context learning (ICL) with many repeated examples — the declarative system finds the LAST occurrence, which may not be correct
- A variable appears multiple times in the template and you need a specific occurrence (not the last one)
- Positions require regex or complex string parsing to locate

**Metrics:** The `metric` function signature is ALWAYS `metric(neural_output: dict, causal_output: str) -> bool`. It receives the model's output dict (with a `"string"` key) and the expected causal output string. It does NOT have access to logits, the pipeline, or the tokenizer. Do not implement logit-based or probability-based metrics in `metrics.py` — those are computed separately during experiments.

**Checker:** Default to exact match on the stripped strings (`actual == expected`, as in `templates/checker.py`); use something less strict like `startswith` only when the task needs it.

## Restrictions

- ONLY read templates from `setup-task/templates/`.

## Notebook builder (summary.ipynb)

Build `summary.ipynb` with `nbformat` so every cell carries a unique `"id"` — nbformat ≥ 4.5 requires it, and hand-written JSON raises `MissingIDFieldWarning` and gets silently rewritten by tools. `new_*_cell` assigns the ids and `nbformat.write` writes a schema-valid file. Adapt the cell bodies to the task; keep the section order.

```python
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells = [
    new_markdown_cell("# [Task Name]\n\n[one-line task description]"),
    # sys.path bootstrap so the session-local `tasks.<name>` namespace resolves
    new_code_cell(
        "import sys\n"
        'sys.path.insert(0, "$WORKDIR/code")'
    ),
    new_code_cell(
        "from tasks.[task_name] import CAUSAL_MODEL, COUNTERFACTUAL_GENERATORS"
    ),
    new_markdown_cell("## Causal Model Variables"),
    new_code_cell("print(CAUSAL_MODEL.variables)  # or CAUSAL_MODEL.values for value lists"),
    new_markdown_cell("## Sample Generation"),
    new_code_cell(
        "for _ in range(3):\n"
        "    s = CAUSAL_MODEL.sample_input()\n"
        '    print(repr(s["raw_input"]), "->", repr(s["raw_output"]))'
    ),
    new_markdown_cell("## Token Positions"),
    new_code_cell(
        "from tasks.[task_name].token_positions import create_token_positions\n"
        "# show the token-position specs and highlight them on one sample"
    ),
    new_markdown_cell("## Counterfactual Generation"),
    new_code_cell(
        "gen = next(iter(COUNTERFACTUAL_GENERATORS.values()))\n"
        "# generate and display one counterfactual example"
    ),
]
nbformat.write(nb, "${TASK_DIR}/summary.ipynb")
```

Required cells, in order:
1. Markdown: Task title and description
2. Code: `sys.path` bootstrap so the session-local namespace resolves
3. Code: Imports (`from tasks.[task_name] import CAUSAL_MODEL, COUNTERFACTUAL_GENERATORS`)
4. Markdown: "Causal Model Variables"
5. Code: show the causal model variables (`CAUSAL_MODEL.variables` / `.values`)
6. Markdown: "Sample Generation"
7. Code: Generate and print several samples (via `CAUSAL_MODEL.sample_input()`)
8. Markdown: "Token Positions"
9. Code: Show token position definitions and visualize on a sample (tokenize input, highlight positions)
10. Markdown: "Counterfactual Generation"
11. Code: Generate and display a counterfactual example

## Single-token filtering (`filter_single_token`)

Tokenizers prepend a leading space to tokens that appear mid-sentence: in `"The capital city of France is"` the tokenizer sees `" France"` (with leading space), not `"France"`. So check the token count of the value *as it would appear in the template* — find where the placeholder appears, and if it is preceded by a space (the common case) check the leading-space form; if it is at the very start of the template, check the bare form.

```python
from tasks.[task_name].config import TEMPLATE  # or TEMPLATES[0]

# For each variable with a value list (e.g., NAMES, OBJECTS, PLACES):
def filter_single_token(values: list[str], var_placeholder: str, template: str) -> list[str]:
    """Filter values to those that tokenize as a single token in context."""
    # Determine if variable gets a leading space in the template
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

Apply to each variable list and report what was dropped:

```python
# Example for a task with NAMES and OBJECTS
original_names = NAMES  # from config.py
filtered_names = filter_single_token(NAMES, "name", template)
print(f"NAMES: {len(original_names)} -> {len(filtered_names)} single-token values")
print(f"  Removed: {set(original_names) - set(filtered_names)}")

# Repeat for each variable list...
```

After filtering, update `config.py` with the filtered lists.

## Accuracy-test loop (64 examples)

Sample all examples up front, then validate in small chunks. Never pass a whole sample set (or all few-shot prompts) to a single `pipeline.generate(...)` call — batched attention plus full-vocab logits scale with batch × sequence length, and a 64-prompt four-shot batch has been observed to OOM at ~71 GB. Chunk to `EVAL_BATCH_SIZE` and pass `output_scores=False` (accuracy matching needs no logits).

```python
EVAL_BATCH_SIZE = 8  # lower this if you still OOM on long / many-shot prompts

settings = [CAUSAL_MODEL.sample_input() for _ in range(64)]
total = len(settings)
correct = 0
for start in range(0, total, EVAL_BATCH_SIZE):
    chunk = settings[start : start + EVAL_BATCH_SIZE]
    out = pipeline.generate(chunk, output_scores=False)
    # generate() returns "string" as a list for a multi-example batch; slice
    # per example so the per-example checker (reads neural_output["string"]) works.
    strings = out["string"] if isinstance(out["string"], list) else [out["string"]]
    for full_setting, gen in zip(chunk, strings):
        if checker({"string": gen}, full_setting["raw_output"]):
            correct += 1

accuracy = correct / total
print(f"Model accuracy: {correct}/{total} = {accuracy:.1%}")
```

For `output_token_mode: "first_token_only"`, use the first-token checker (compares only the first generated token against the first token of the expected output via the tokenizer) instead of the standard exact-match checker.

## Spacing check (token alignment)

Even when accuracy is acceptable, verify token alignment, then test both spacing variants and record the one the model actually emits:

```python
# Get 16 examples
examples = [CAUSAL_MODEL.sample_input() for _ in range(16)]

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

Some models (e.g., GPT-2) use BPE tokenization where word tokens include a leading space (Ġ prefix). In these cases the template should NOT end with a trailing space and `raw_output` should include a leading space (e.g., `" France"`); other models work better with a trailing space on the template and no leading space on `raw_output`.

```python
# Template-side space: trailing space on template, no leading space on raw_output
# Output-side space: no trailing space on template, leading space on raw_output
# Test both on 16 examples, pick the one with higher accuracy and better token alignment
```

After choosing the working variant, update templates.py and causal_models.py accordingly, then re-run the accuracy test on 16 examples to confirm.
