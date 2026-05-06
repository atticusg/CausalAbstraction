---
name: setup-methods
description: Scaffold one or more session-local interpretability methods (reusable primitives — featurizers, scorers, training loops, distances, …) in a single invocation. Use when an experiment needs methods that don't yet exist in causalab/methods/. Generates code under ${SESSION_DIR}/code/methods/<name>/ — never inside causalab/. Pair with /setup-analyses to wrap methods in Hydra entry points.
args: <one or more spec paths, space-separated; or empty for interactive single-spec elicitation>
---

# Setup Methods Skill

Scaffolds one or more *methods* (reusable interpretability primitives) under the active research session. Methods are the middle layer in `ARCHITECTURE.md` §3: pure library code that depends on `neural/`, `io/`, `causal/`, `tasks/` but never on `analyses/` or `runner/`. They take inputs as plain arguments (or a resolved `DictConfig`), return in-memory results, and never touch disk.

The skill writes to `${SESSION_DIR}/code/methods/<name>/` — never to `causalab/`. Promotion to `causalab/methods/` is a manual `git mv` step taken only after the prototype stabilizes (see `.claude/skills/research-session/CONVENTIONS.md`).

## Batch invocation

This skill is **loaded once** per `/run-experiment` Step 4 invocation. The caller passes the full list of method specs (one per `custom` §D node in `PLAN.md`) at once, and the skill loops Steps 1–5 over each spec sequentially. Step 0 and Step 6 run once for the whole batch. This keeps the scaffolding canonical (single source of truth in this skill) without re-loading SKILL.md per primitive.

When called with no arguments, the skill falls back to the legacy single-spec interactive flow (Step 1 elicits one spec; Steps 3–5 run once).

## Required Reading

Before running this skill, read:

1. `.claude/skills/research-session/CONVENTIONS.md` — research-session layout (especially "What goes in `code/`") and active-session protocol.
2. `ARCHITECTURE.md` §3 invariants 1, 2, 4, 5 — the layering rules every session-local method must respect:
   - **Inv 1**: `neural/` cannot import from `methods/`. Reverse holds — methods can read neural primitives.
   - **Inv 2**: `methods/` must not import from `runner/` or `analyses/`. Configuration is passed as plain kwargs or a resolved `DictConfig`.
   - **Inv 4**: methods do not own research-question orchestration — no dataset loading from a path, no artifact-directory layouts, no metadata dicts tagged with `experiment_type`. Return an in-memory dict; let the analysis decide where it lands.
   - **Inv 5**: methods must not embed hyperparameter defaults. Either take explicit kwargs with no implicit fallback, or accept a resolved config object.

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop.

## Autonomous vs Interactive Mode

```bash
echo $CAUSALAB_AUTONOMOUS
```

- **`"1"`**: Run autonomously. The skill requires at least one spec path (the harness — typically `/run-experiment` Step 4 — is responsible for producing them). Skip approval checkpoints; log decisions to `${SESSION_DIR}/issues.md` if anything ambiguous comes up.
- **Empty / unset**: Interactive. Walk through the steps with explicit user approval at the checkpoints below. If no spec paths are provided, fall back to single-spec interactive elicitation. With one or more spec paths, scaffold each spec sequentially and surface a single batch-level approval (Step 2).

## Issue Tracking

Use `/document-issues` whenever you hit a blocker, a layering violation you cannot resolve, or a missing primitive that arguably belongs in `causalab/methods/`. Issues land in `${SESSION_DIR}/issues.md`.

---

## Step 0: Resolve the Active Session

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

In autonomous mode the harness pre-creates the session and writes `.current`; trust the marker.

---

## Step 1: Read or Elicit the Specifications

The skill consumes one or more markdown specs — each `set_up_method.md` laid out per `SET_UP_METHOD_TEMPLATE.md`. Input shapes:

1. **Args are paths** (one or more space-separated paths to existing markdown files) → read each and use directly. Order is preserved; specs are processed in argv order.
2. **No args, interactive mode** → run `instructions/create_specification.md` and elicit a single spec section by section, writing the draft to `${SESSION_DIR}/code/methods/<name>/set_up_method.md` as it grows. Get user approval at each section. (Interactive elicitation is single-spec only; batches must come in via paths.)
3. **No args, autonomous mode** → fail with a clear message: this skill requires at least one spec path in autonomous mode.

After this step every `${SESSION_DIR}/code/methods/<name>/set_up_method.md` referenced in argv exists and is approved.

### Refuse name collisions

For **each** spec, before proceeding, check that its `<name>` does not already exist under `causalab/methods/<name>/` or `causalab/methods/<name>.py`. If a collision is found, refuse the **whole batch** with:

> "A method named `<name>` already ships under `causalab/methods/`. Pick a different name; session-local code must not shadow shipped methods. (Batch aborted before any scaffolding ran.)"

Surface all collisions first, then abort, so the caller can fix names in one pass.

---

## Step 2: 🔒 Batch Approval Checkpoint (interactive only)

Print a single block listing **every** method in the batch:

> **Batch (N methods):**
>
> 1. **`<name_1>`** — signature `<…>`, deps `<…>`, hyperparameters `<…>`
> 2. **`<name_2>`** — signature `<…>`, deps `<…>`, hyperparameters `<…>`
> …
>
> **Files about to be created** (one bundle per method, under `${SESSION_DIR}/code/methods/<name>/`):
> - `__init__.py`, `<name>.py`, `set_up_method.md` (already saved), `tests/test_<name>.py`
>
> Approve all, edit one, or cancel batch?

Proceed only on approval. "Edit one" returns to Step 1 for that single spec, then re-enters Step 2 with the revised batch. In autonomous mode, log "batch scaffold proceeding, no approval gate (autonomous)" in `${SESSION_DIR}/issues.md` once for the whole batch.

---

## Step 3: Scaffold from Templates

**Loop Steps 3, 4, and 5 per spec, in argv order.** All other steps run once per batch.

For the current spec, create the directory and files:

```
${SESSION_DIR}/code/methods/<name>/
├── __init__.py            from templates/__init__.py
├── <name>.py              from templates/method.py
├── set_up_method.md       (already saved in Step 1)
└── tests/
    └── test_<name>.py     from templates/test_method.py
```

Substitute the spec values into the templates:

- `<name>` → method name (snake_case).
- Function/class signature, type annotations, and the docstring purpose paragraph — straight from §1–§3 of the spec.
- Imports listed in §3 of the spec — emit them at the top of `<name>.py`. Validate that none come from `causalab/runner/` or `causalab/analyses/` (refuse and revise if they do — invariants 1, 2).
- Hyperparameters from §4 — emit as keyword arguments with **no defaults**. The function body remains `raise NotImplementedError(...)`.
- Test scaffold — one test that calls the method on randomly-generated tensors of the input shape and asserts the output shape and dtype. The body of the test asserts `pytest.raises(NotImplementedError)` initially so the file passes immediately; the agent flips that assertion to a real shape check during Step 4.

Verify the imports parse:

```bash
uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', '${SESSION_DIR}/code/methods/<name>/<name>.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('parsed ok')"
```

---

## Step 4: Implement the Body

Now fill in the function body. The flow is test-first:

1. Replace the `pytest.raises(NotImplementedError)` shape check in `tests/test_<name>.py` with a real shape/dtype assertion.
2. Run the test — it now fails because the body still raises.
3. Implement the body. Iterate `uv run pytest ${SESSION_DIR}/code/methods/<name>/tests/ -v` until it passes.

Implementation rules (from `ARCHITECTURE.md` §3):

- No hyperparameter defaults inside the function (`def f(x, *, k_features)` — no `= 8`).
- No disk I/O. The method returns an in-memory dict; the caller (an analysis) decides what to persist.
- Imports stay restricted to `causalab/{neural,methods,io,causal,tasks}/` and standard third-party libs. No `causalab.runner.*`, no `causalab.analyses.*`.

In interactive mode, ask the user once before each non-trivial design decision (e.g. choice of optimizer, batching strategy). In autonomous mode, take the simplest viable path and document it in the docstring.

---

## Step 5: Layering Audit

Before declaring the method done, run a quick audit:

```bash
grep -rE "from causalab\.(runner|analyses)" "${SESSION_DIR}/code/methods/<name>/" || echo "no forbidden imports"
grep -rE "torch\.save|safetensors|json\.dump|open\(" "${SESSION_DIR}/code/methods/<name>/<name>.py" || echo "no disk I/O"
```

If either grep matches, treat it as a layering violation. Either fix the code (refactor disk I/O up to the analysis layer) or, if the violation is intentional and unavoidable, log an entry to `${SESSION_DIR}/issues.md` via `/document-issues` so the issue is visible at promotion time.

If the method depends on a primitive that arguably belongs in `causalab/methods/` (e.g. it had to re-implement a small distance function because the existing one is private), file an issue.

---

## Step 6: Hand-off

After every spec in the batch has cleared Steps 3–5, print one summary:

```
Batch scaffolded (N methods):
  - <name_1>   ${SESSION_DIR}/code/methods/<name_1>/
  - <name_2>   ${SESSION_DIR}/code/methods/<name_2>/
  …

Use them from session-local analyses:
    from methods.<name> import <main_callable>

Run all tests:
    uv run pytest ${SESSION_DIR}/code/methods/

Next: wrap each in a Hydra entry point with /setup-analyses if the experiment plan
expects analysis-level nodes (the corresponding analysis specs go to that skill in a
single batch), or call them directly from a session notebook.
```

---

## Important Notes

### What this skill does NOT do

- **Does not edit anything under `causalab/`.** The shipped package is read-only here.
- **Does not register methods with the runner.** Methods are library code; runner-side wiring happens in `/setup-analyses`.
- **Does not add Hydra defaults.** Hyperparameters live in the analysis's `analysis.yaml`, never in method code.
- **Does not promote.** Promotion to `causalab/methods/` is manual.

### Restrictions

- Only edit files under `${SESSION_DIR}/code/methods/<name>/` and `${SESSION_DIR}/issues.md`.
- Read templates only from `.claude/skills/setup-methods/templates/`.
- Refuse names that collide with any directory or file under `causalab/methods/`.
