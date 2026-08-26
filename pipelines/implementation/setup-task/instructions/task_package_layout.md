# Task Package Layout

A **task** is the bridge between a behavioral hypothesis and the experiment engine. Experiment code is task-agnostic — it doesn't know IOI, arithmetic, or MCQA. All task-specific knowledge lives in the task package, consumed through a standard interface.

Every task is a Python package with the same set of files. The setup-task guide scaffolds it **out of tree** at `$WORKDIR/code/tasks/<name>/` (imported as the session-local `tasks.<name>` when `CAUSALAB_SESSION_CODE=$WORKDIR` is set); a shipped task lives at `causalab/tasks/<name>/` in the library checkout (imported as `causalab.tasks.<name>`, and taking precedence). The file layout is identical in both:

```
<tasks-root>/<name>/        # $WORKDIR/code/tasks/<name>/  or  causalab/tasks/<name>/
├── __init__.py            # Exports everything experiments need
├── causal_models.py       # The causal model: variables, values, mechanisms
├── counterfactuals.py     # Generates counterfactual pairs for each variable
├── token_positions.py     # Maps variable names → token positions in the input
├── config.py              # Constants: variable value lists, max tokens, task name
├── templates.py           # Input text templates with placeholders
├── checker.py             # (optional) custom output matcher; derived from output_tokens when absent
├── metrics.py             # Custom scoring functions
├── summary.ipynb          # Hands-on overview notebook (sample inputs, CFs)
├── set_up_task.md         # The original markdown spec this skill consumed
└── README.md              # (optional) task-specific notes
```

## Why this structure

Each file answers a specific question the experiment engine needs answered:

| File | Question it answers |
|---|---|
| `causal_models.py` | What is your hypothesis? — variables, values, mechanisms (the causal graph that gets tested) |
| `counterfactuals.py` | How do I generate test cases? — paired inputs where specific variables are swapped |
| `token_positions.py` | Where in the input should I intervene? — variable name → token positions in the prompt |
| `config.py` / `templates.py` | What are the concrete values and sentence structures? — raw materials drawn by `counterfactuals.py` |
| `checker.py` / `metrics.py` | How do I validate the model's output? — exact match, first-token match, custom comparison |

This separation means you can define a completely new task without touching any experiment code — implement these files and every analysis (interchange scoring, DAS, DBM, PCA, manifold fitting, …) works automatically.

## Key conventions

- **Required:** Every causal model MUST have `raw_input` and `raw_output` variables.
- **Variable naming:** Use the EXACT variable names from the specification. `snake_case`, never re-cased or abbreviated. Token-position dict keys are lowercase (e.g. `capital`, `country`, `end`).
- **Counterfactuals:** Implement ALL counterfactual types listed in the specification — each has its own generator function and is included in `COUNTERFACTUAL_GENERATORS`.
- **Single-template tasks:** Do NOT include `template` as a causal variable. Use `TEMPLATE = TEMPLATES[0]` as a module-level constant in `causal_models.py`. Only include `template` as an input variable for tasks with multiple templates the model must handle.
- **Token positions:** `create_token_positions` MUST return `Dict[str, TokenPosition]` — materialized instances, not factories. Use the helper: `return build_token_positions(specs, template, pipeline)` (from `causalab.neural.token_positions`), which builds the declarative specs and materializes them in one call. Consumers read `.id` off the values; returning the bare factories from `build_token_position_factories()` crashes `locate` with `AttributeError: 'function' object has no attribute 'id'`. Use custom Python (not the declarative builder) when (a) the task uses ICL with repeated examples, (b) a variable appears multiple times and you need a specific occurrence, or (c) positions need regex/complex parsing.
- **Metrics:** The `metric` function signature is ALWAYS `metric(neural_output: dict, causal_output: str) -> bool`. It receives the model's output dict (with `"string"` key) and the expected causal output string. It does NOT have access to logits, the pipeline, or the tokenizer.
- **Checker:** Defaults to exact match (`actual == expected`) on the stripped output; relax to `startswith` only when the task needs it.
