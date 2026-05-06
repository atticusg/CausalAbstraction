# Characterize the Task → `PLAN.md` §B

Goal of this step: fill `PLAN.md` §B (causal model & dataset) with enough detail that, if the task package does not yet exist, `/setup-task` can be invoked autonomously from `/run-experiment` later — no further human Q&A.

## Two paths

### Path A: Task package exists at `causalab/tasks/{task_name}/`

Read these files (whichever are present) and summarize their contents into §B:

- `causal_models.py` — variables, value spaces, mechanisms.
- `counterfactuals.py` — `COUNTERFACTUAL_GENERATORS` dict, sampling rules.
- `token_positions.py` — declared positions for each variable.
- `config.py` — `MAX_TASK_TOKENS`, `MAX_NEW_TOKENS`, value-list constants.
- `templates.py` — input templates with placeholders.
- `set_up_task.md` (if present) — the original spec, often the cleanest source.

Fill §B from what you find. Do not invent fields the existing task does not declare. Note in the plan: **Status: exists**.

### Path B: Task package does not exist

§B becomes a `/setup-task`-compatible spec. It must cover **all eight** fields surfaced by the codebase survey of `/setup-task` requirements:

1. **Task identity** — `name:` (snake_case slug) and a one-paragraph description of what the model solves (input/output format). This is the YAML frontmatter for `set_up_task.md` later.
2. **Variables & value spaces** — input variable names, types (categorical / ordinal / numeric), value lists or ranges. Output variable name and how it relates to inputs. Any intermediate variables (parents, compute logic).
3. **Templates & rendering** — raw input template(s) with placeholders (e.g. `"If today is {day}, {offset} days from now is"`). For multiple templates, the rendering rule.
4. **Causal mechanisms** — for each derived variable, the parent set and the compute function (lambda or descriptive logic).
5. **Counterfactual generation rules** — per-input-variable CFs (change one input, output must change), the fully random CF (change all inputs), and intermediate-variable CFs if any exist. Note any custom sampling distributions.
6. **Token positions** — named position definitions per variable. For simple tasks, just variable names. For ICL or repeated patterns, custom Python logic.
7. **Validation & checking** — checker rule (default `startswith`); metric signature `metric(neural_output: dict, causal_output: str) -> bool`; whether to filter input values to single-token-only (depends on `output_token_mode`).
8. **Configuration constants** — `MAX_TASK_TOKENS` (max context), `MAX_NEW_TOKENS` (typically 1 for single-token tasks).

Plus the **`output_token_mode`** decision (one of `full` / `single_constrained` / `first_token_only`). This is a non-autonomous decision; surface it explicitly.

Note in the plan: **Status: to be created**, then list the eight fields above as a `set_up_task.md`-shaped block that `/setup-task` can ingest verbatim.

## Five non-autonomous decisions

`/setup-task` cannot resolve these autonomously. The plan must surface them so the user (or autonomous harness, with logged defaults) approves them now, before runs start:

1. **`output_token_mode` choice** — `full` vs `single_constrained` vs `first_token_only`. Affects validation and filtering strategy.
2. **Single-token filtering threshold** — if `single_constrained`, the tool will filter input values to single-token only; some values will be removed.
3. **Low-accuracy fallback** — what to do if model accuracy < 20% in `/setup-task` Step 4 (continue, swap model, adjust template).
4. **Space-variant tokenization** — if the tokenizer adds leading/trailing spaces, which spacing variant the model actually produces (template trailing space + raw_output bare, or vice versa).
5. **Custom CF sampling distributions** — defaults are auto-proposed, but deviations (e.g. "sample only values far from base") need user confirmation.

In the plan, list each with the chosen default and a one-line rationale. In autonomous mode, log the defaults to `${SESSION_DIR}/plan/approval.log`.

## Counterfactual generator vs `task.resample_variable`

The plan must declare which `task.resample_variable` setting each downstream `locate` analysis will use. Per `ARCHITECTURE.md` §5:

- `"all"` — counterfactual is a fresh independent sample. Use with `locate.mode: centroid` only.
- A single variable name (e.g. `"day"`) — counterfactual differs from the original only in that variable. Required for `locate.mode: pairwise` on that variable.

If the plan localizes more than one variable in pairwise mode, **each variable's pairwise locate must run with its own `resample_variable`** — that is one separate runner config per variable. Sharing `experiment_root` across these is a hazard (see `plan_sweep.md`) because `locate/{method}/results.json` would be clobbered.

## Dataset sizing

Fill the `task:` block:

```yaml
task:
  n_train: 200      # rationale
  n_test: 100       # rationale
  enumerate_all: false  # true exhausts the combinatorial space; rationale if true
  balanced: false       # true overrides resample_variable; rationale if true
```

Per ARCHITECTURE.md §3 invariant 12, these live in the task config and are read by every analysis via `cfg.task.*`. The plan declares them once; runner configs inherit.
