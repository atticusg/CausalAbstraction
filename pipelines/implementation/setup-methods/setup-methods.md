---
name: setup-methods
description: Scaffold one or more interpretability methods (reusable primitives — featurizers, scorers, training loops, distances, …) in a single invocation. Use when an experiment needs methods that don't yet exist in causalab/methods/. Generates code out of tree under $WORKDIR/code/methods/<name>/. Pair with the setup-analyses guide to wrap methods in Hydra entry points.
---

# Setup Methods Skill

Talk like a colleague: tell the user which primitives you're scaffolding and why, and ask before each non-trivial design call.

Scaffolds one or more *methods* (reusable interpretability primitives). The input is one or more spec paths — markdown files laid out per `SET_UP_METHOD_TEMPLATE.md`, one method each — passed space-separated; with no path given, the skill elicits a single spec interactively. Methods are the middle layer in `docs/CODEBASE.md` §3: pure library code that depends on `neural/`, `io/`, `causal/`, `tasks/` but never on `analyses/` or `runner/`. They take inputs as plain arguments (or a resolved `DictConfig`), return in-memory results, and never touch disk.

The skill writes **out of tree** to `$WORKDIR/code/methods/<name>/`, where `$WORKDIR` is the run's working directory. The runner picks it up via the session-local `methods.<name>` fallback when it runs with `CAUSALAB_SESSION_CODE=$WORKDIR` set. Never author method code inside the lab-managed library checkout at `~/.silico/libraries/causalab-internal/` (it is re-synced / image-resident and would be clobbered or is unwritable); shipping a stabilized method upstream is a separate causalab-repo PR, not part of this run.

## Batch invocation

This skill is **loaded once** per run-phase setup pass. The caller passes the full list of method specs (one per §D node in `PLAN.md` whose Coverage status is `method-gap` or `both-gap`) at once, and the skill loops the per-spec steps (Scaffold → Implement → Audit) over each spec sequentially. The remaining steps run once for the whole batch. This keeps the scaffolding canonical (single source of truth in this skill) without re-loading this document per primitive.

When called with no spec path, the skill falls back to the single-spec interactive flow (Step 1 elicits one spec; the per-spec steps run once).

## Required Reading

Before running this skill, read `docs/CODEBASE.md` §3 — the layering rules every method must respect. The relevant invariants (§3 invariants 1, 2, 4, 5 — read the section for the authoritative wording):
   - `neural/` cannot import from `methods/`. Reverse holds — methods can read neural primitives.
   - `methods/` must not import from `runner/` or `analyses/`. Configuration is passed as plain kwargs or a resolved `DictConfig`.
   - methods do not own research-question orchestration — no dataset loading from a path, no artifact-directory layouts, no metadata dicts tagged with `experiment_type`. Return an in-memory dict; let the analysis decide where it lands.
   - methods must not embed hyperparameter defaults. Either take explicit kwargs with no implicit fallback, or accept a resolved config object.

The skill requires at least one spec path. When one or more spec paths are provided, scaffold each spec sequentially and confirm the batch with the user once (Step 2); when none is provided, fall back to single-spec interactive elicitation. If anything is ambiguous or blocked — a layering violation you cannot resolve, or a missing primitive that arguably belongs in `causalab/methods/` — surface it to the user rather than guessing.

---

## Step 1: Read or Elicit the Specifications

The skill consumes one or more markdown specs — each `set_up_method.md` laid out per `SET_UP_METHOD_TEMPLATE.md`. Input shapes:

1. **Spec paths given** (one or more space-separated paths to existing markdown files) → read each and use directly. Order is preserved; specs are processed in the order received.
2. **No paths** → run `instructions/create_specification.md` and elicit a single spec section by section, writing the draft to `$WORKDIR/code/methods/<name>/set_up_method.md` as it grows. Get user approval at each section. (Interactive elicitation is single-spec only; batches must come in via paths.)

After this step every `$WORKDIR/code/methods/<name>/set_up_method.md` referenced exists and is approved.

### Refuse name collisions

For **each** spec, before proceeding, check that its `<name>` does not already exist as a shipped `causalab/methods/<name>/` or `causalab/methods/<name>.py`, nor as an existing `$WORKDIR/code/methods/<name>/`. If a collision is found, refuse the **whole batch** with:

> "A method named `<name>` already ships under `causalab/methods/`. Pick a different name; a session-local method must not shadow a shipped one (the loader resolves shipped first). (Batch aborted before any scaffolding ran.)"

Surface all collisions first, then abort, so the caller can fix names in one pass.

---

## Step 2: Batch Approval Checkpoint

Print one plain-prose block that walks the user through every method in the batch — name each one, what it takes and returns, what it depends on, and which hyperparameters it exposes — then name the files each will create (a per-method bundle under `$WORKDIR/code/methods/<name>/`: the package init, the method module, the spec already saved, and the test). Close by asking whether to approve all, revise one, or cancel the batch.

Proceed only on approval. Revising one returns to Step 1 for that single spec, then re-enters Step 2 with the revised batch.

---

## Step 3: Scaffold from Templates

**Loop Steps 3, 4, and 5 per spec, in argv order.** All other steps run once per batch.

For the current spec, create the directory and files:

```
$WORKDIR/code/methods/<name>/
├── __init__.py            from templates/__init__.py
├── <name>.py              from templates/method.py
├── set_up_method.md       (already saved in Step 1)
└── tests/
    └── test_<name>.py     from templates/test_method.py
```

Substitute the spec values into the templates:

- `<name>` → method name (snake_case).
- Function/class signature, type annotations, and the docstring purpose paragraph — straight from §1–§3 of the spec.
- Imports listed in §3 of the spec — emit them at the top of `<name>.py`. Validate that none come from `causalab/runner/` or `causalab/analyses/` (refuse and revise if they do — see the layering invariant in `docs/CODEBASE.md` §3, invariant 2).
- Hyperparameters from §4 — emit as keyword arguments with **no defaults**. The function body remains `raise NotImplementedError(...)`.
- Test scaffold — one test that calls the method on randomly-generated tensors of the input shape and asserts the output shape and dtype. The body of the test asserts `pytest.raises(NotImplementedError)` initially so the file passes immediately; the agent flips that assertion to a real shape check during Step 4.

Verify the imports parse:

```bash
cd ~/.silico/libraries/causalab-internal
PYTHONPATH="$WORKDIR/code${PYTHONPATH:+:$PYTHONPATH}" \
  uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', '$WORKDIR/code/methods/<name>/<name>.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('parsed ok')"
```

---

## Step 4: Implement the Body

Now fill in the function body. The flow is test-first:

1. Replace the `pytest.raises(NotImplementedError)` shape check in `tests/test_<name>.py` with a real shape/dtype assertion.
2. Run the test — it now fails because the body still raises.
3. Implement the body. Iterate `PYTHONPATH="$WORKDIR/code:$PYTHONPATH" uv run pytest $WORKDIR/code/methods/<name>/tests/ -v` (from the library checkout, so `causalab` also resolves) until it passes.

Implementation rules (from `docs/CODEBASE.md` §3):

- No hyperparameter defaults inside the function (`def f(x, *, k_features)` — no `= 8`).
- No disk I/O. The method returns an in-memory dict; the caller (an analysis) decides what to persist.
- Imports stay restricted to `causalab/{neural,methods,io,causal,tasks}/` and standard third-party libs. No `causalab.runner.*`, no `causalab.analyses.*`.

Ask the user once before each non-trivial design decision (e.g. choice of optimizer, batching strategy); when running without a user in the loop, take the simplest viable path and document it in the docstring.

---

## Step 5: Layering Audit

Before declaring the method done, run a quick audit:

```bash
grep -rE "from causalab\.(runner|analyses)" "$WORKDIR/code/methods/<name>/" || echo "no forbidden imports"
grep -rE "torch\.save|safetensors|json\.dump|open\(" "$WORKDIR/code/methods/<name>/<name>.py" || echo "no disk I/O"
```

If either grep matches, treat it as a layering violation. Either fix the code (refactor disk I/O up to the analysis layer) or, if the violation is intentional and unavoidable, surface it to the user so the exception stays visible.

If the method depends on a primitive that arguably belongs in `causalab/methods/` (e.g. it had to re-implement a small distance function because the existing one is private), flag it to the user.

---

## Step 6: Hand-off

After every spec in the batch has cleared the per-spec steps (Scaffold → Implement → Audit), print one summary:

```
Batch scaffolded (N methods):
  - <name_1>   $WORKDIR/code/methods/<name_1>/
  - <name_2>   $WORKDIR/code/methods/<name_2>/
  …

Use them from session-local analyses:
    from methods.<name> import <main_callable>

Run all tests (from ~/.silico/libraries/causalab-internal):
    PYTHONPATH="$WORKDIR/code:$PYTHONPATH" uv run pytest $WORKDIR/code/methods/

Next: wrap each in a Hydra entry point via the setup-analyses guide if the experiment plan
expects analysis-level nodes (the corresponding analysis specs go to that guide in a
single batch), or call them directly from a notebook.
```

---

## Important Notes

### What this skill does NOT do

- **Does not touch the runner or analyses.** Methods are library code; runner-side wiring happens via the setup-analyses guide.
- **Does not add Hydra defaults.** Hyperparameters live in the analysis's `analysis.yaml`, never in method code.

### Restrictions

- Only create/edit files under `$WORKDIR/code/methods/<name>/`.
- Read templates only from `setup-methods/templates/`.
- Refuse names that collide with any directory or file under `causalab/methods/`.
