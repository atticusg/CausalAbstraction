# Task Package Layout

A **task** is the bridge between a behavioral hypothesis and the experiment engine. Experiment code is task-agnostic — it doesn't know weekdays, graph_walk, or arithmetic. All task-specific knowledge lives in the task package, consumed through a standard interface.

Every task is a Python package under `causalab/tasks/<name>/` with the same set of files:

```
causalab/tasks/<name>/
├── __init__.py            # Exports everything experiments need
├── causal_models.py       # The causal model: variables, values, mechanisms
├── counterfactuals.py     # Generates counterfactual pairs for each variable
├── token_positions.py     # Maps variable names → token positions in the input
├── config.py              # Constants: variable value lists, max tokens, task name
├── templates.py           # Input text templates with placeholders
├── checker.py             # Validates model output against expected answer
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
- **Token positions:** `create_token_positions` MUST return the result of `build_token_position_factories()` directly — a `Dict[str, Callable]`. Do NOT call the factories yourself; the experiment framework calls them later. Use custom Python (not the declarative builder) when (a) the task uses ICL with repeated examples, (b) a variable appears multiple times and you need a specific occurrence, or (c) positions need regex/complex parsing.
- **Metrics:** The `metric` function signature is ALWAYS `metric(neural_output: dict, causal_output: str) -> bool`. It receives the model's output dict (with `"string"` key) and the expected causal output string. It does NOT have access to logits, the pipeline, or the tokenizer.
- **Checker:** Defaults to `startswith` on the stripped output.
