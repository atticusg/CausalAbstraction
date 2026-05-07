---
name: setup-analyses
description: Scaffold one or more session-local analyses (research-question wrappers, Hydra entry points) under the active research session in a single invocation. Use when an experiment plan needs analyses that don't yet exist in causalab/analyses/. Generates code under ${SESSION_DIR}/code/analyses/<name>/ and Hydra configs under ${SESSION_DIR}/code/configs/analysis/<name>.yaml — never inside causalab/. Pair with /setup-methods for missing primitives.
args: <one or more spec paths, space-separated; or empty for interactive single-spec elicitation>
---

# Setup Analyses Skill

Scaffolds one or more *analyses* (research-question wrappers) under the active research session. Analyses are the top library layer in `ARCHITECTURE.md` §3: they orchestrate I/O and Hydra config, chain methods, own artifact-directory layouts, and expose a `main(cfg)` Hydra entry point. The runner picks them up via session-code path injection (see "Execution wiring" in this file).

## Batch invocation

This skill is **loaded once** per `/run-experiment` Step 4 invocation. The caller passes the full list of analysis specs (one per `custom` §D analysis node in `PLAN.md`) at once, and the skill loops Steps 1–5 over each spec sequentially. Step 0 and Step 6 run once for the whole batch. This keeps the scaffolding canonical (single source of truth in this skill) without re-loading SKILL.md per analysis.

When called with no arguments, the skill falls back to the legacy single-spec interactive flow (Step 1 elicits one spec; Steps 3–5 run once).

The skill writes to:
- `${SESSION_DIR}/code/analyses/<name>/` — Python module (`main.py`, `__init__.py`, `README.md`, `set_up_analysis.md`).
- `${SESSION_DIR}/code/configs/analysis/<name>.yaml` — Hydra config that the runner mounts at `cfg.<name>`.

It never edits anything under `causalab/`. Promotion to `causalab/analyses/` is a manual `git mv` step taken only after the prototype stabilizes (see `.claude/skills/research-session/CONVENTIONS.md`).

## Required Reading

Before running this skill, read:

1. `.claude/skills/research-session/CONVENTIONS.md` — research-session layout (especially "What goes in `code/`") and active-session protocol.
2. `ARCHITECTURE.md` §3 invariants 2, 3, 4, 5, 7, 11, 12 — analysis-layer rules:
   - **Inv 2**: methods don't import from `runner/` or `analyses/`. The reverse is fine — analyses orchestrate methods.
   - **Inv 3**: `io/` doesn't import from `methods/`, `analyses/`, or `runner/`. Analyses save through `causalab.io.*` primitives only.
   - **Inv 4**: analyses own dataset loading, artifact-directory layout, and metadata. Methods don't.
   - **Inv 5**: defaults live in Hydra (`analysis.yaml`), not in code.
   - **Inv 7**: `experiment_root` is the single source of truth for output paths. Analyses consume `cfg.experiment_root`; never override it inline.
   - **Inv 11**: every analysis package has a 3-section README (opening / Configuration / Outputs).
   - **Inv 12**: dataset-construction params (`n_train`, `n_test`, `enumerate_all`, `balanced`) live in `cfg.task.*`; `seed` lives in `cfg.seed`. Analysis configs don't duplicate them.
3. `.claude/skills/plan-experiment/ANALYSIS_GUIDE.md` — the analysis dependency DAG, auto-discovery rules, and runner-YAML composition patterns.
4. `ARCHITECTURE.md` §2 — the runner config system (`# @package`, `_name_`, `_subdir`, `_output_dir`).

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop.

## Autonomous vs Interactive Mode

```bash
echo $CAUSALAB_AUTONOMOUS
```

- **`"1"`**: Run autonomously. At least one spec path is required; the harness — typically `/run-experiment` Step 4 — is responsible for producing them. Skip approval checkpoints.
- **Empty / unset**: Interactive. Walk through the steps with explicit user approval at the checkpoints. If no spec paths are provided, fall back to single-spec interactive elicitation per `instructions/create_specification.md`. With one or more spec paths, scaffold each spec sequentially and surface a single batch-level approval (Step 2).

## Issue Tracking

Use `/document-issues` whenever a layering rule is hard to satisfy, an upstream artifact is missing, or the analysis depends on a method that arguably belongs in `causalab/methods/`. Issues land in `${SESSION_DIR}/issues.md`.

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

The skill consumes one or more markdown specs — each `set_up_analysis.md` laid out per `SET_UP_ANALYSIS_TEMPLATE.md`. Input shapes:

1. **Args are paths** (one or more space-separated paths to existing markdown files) → read each and use directly. Order is preserved; specs are processed in argv order.
2. **No args, interactive mode** → run `instructions/create_specification.md` and elicit a single spec section by section, writing the draft to `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md` as it grows. Get user approval at each section. (Interactive elicitation is single-spec only; batches must come in via paths.)
3. **No args, autonomous mode** → fail with a clear message: this skill requires at least one spec path in autonomous mode.

After this step every `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md` referenced in argv exists and is approved.

### Refuse name collisions

For **each** spec, before proceeding, check that its `<name>` does not already exist under `causalab/analyses/<name>/` or `causalab/configs/analysis/<name>.yaml`. If a collision is found, refuse the **whole batch** with:

> "An analysis named `<name>` already ships in `causalab/`. Pick a different name; session-local code must not shadow shipped analyses (the dispatcher prefers shipped over session). (Batch aborted before any scaffolding ran.)"

Surface all collisions first, then abort, so the caller can fix names in one pass.

---

## Step 2: 🔒 Batch Approval Checkpoint (interactive only)

Print a single block listing **every** analysis in the batch:

> **Batch (N analyses):**
>
> 1. **`<name_1>`** — *<research question>*; reads `<…>`; methods `<…>`; config defaults `<…>`
> 2. **`<name_2>`** — *<research question>*; reads `<…>`; methods `<…>`; config defaults `<…>`
> …
>
> **Files about to be created** (one bundle per analysis):
> - `${SESSION_DIR}/code/analyses/<name>/{main.py, __init__.py, README.md, set_up_analysis.md}`
> - `${SESSION_DIR}/code/configs/analysis/<name>.yaml`
>
> Approve all, edit one, or cancel batch?

Proceed only on approval. "Edit one" returns to Step 1 for that single spec, then re-enters Step 2 with the revised batch.

---

## Step 3: Scaffold from Templates

**Loop Steps 3, 4, and 5 per spec, in argv order.** All other steps run once per batch.

For the current spec, create the directories and files. Substitutions follow the spec:

```
${SESSION_DIR}/code/analyses/<name>/
├── __init__.py            from templates/__init__.py
├── main.py                from templates/main.py
├── README.md              from templates/README.md
└── set_up_analysis.md     (already saved in Step 1)

${SESSION_DIR}/code/configs/analysis/<name>.yaml   from templates/analysis.yaml
```

**`main.py` substitutions:**
- `<name>` → analysis name (lowercase, snake_case).
- `ANALYSIS_NAME = "<name>"` constant — must match `_name_` in the YAML (the runner cross-checks this at line 146 of `causalab/runner/run_exp.py`).
- Imports section: emit the methods listed in spec §3 plus standard helpers from `causalab.runner.helpers` (`resolve_task`, `generate_datasets`) and `causalab.io.artifacts` (`save_json_results`, `save_tensor_results`, `save_experiment_metadata`).
- Body: a stub that loads the task, generates datasets, calls each method, then saves outputs. Each step is a `# TODO: ` comment listing what the spec says to do; the function ends with `raise NotImplementedError(...)` so the agent has to fill it in during Step 4.

**`analysis.yaml` substitutions** (per ARCHITECTURE §2):
- First line `# @package <name>` — Hydra mounts the file at `cfg.<name>`.
- `_name_: <name>` — sentinel the dispatcher uses to detect analysis steps.
- `_subdir: ${.method}_<key>${.<key>}` — the `_subdir` pattern decides where this analysis writes inside `${experiment_root}/<name>/`. Keep at minimum `_subdir: default` if the analysis has no method/parameter axis to encode.
- `_output_dir: ${experiment_root}/<name>/${._subdir}` — every method that saves uses this as its base.
- Every config knob from spec §4 with its proposed default. **Do not** include `n_train`, `n_test`, `enumerate_all`, `balanced`, or `seed` — those live in `task:` / root (invariant 12).
- A `visualization:` block with `figure_format: pdf` if the analysis emits matplotlib figures (invariant 6).

**`README.md` substitutions** (per ARCHITECTURE §3 invariant 11):
- Opening paragraph: name, italicized research question, what it does mechanically, where it sits in the pipeline.
- `## Configuration` section: root-config params it reads (e.g. `experiment_root`, `seed`); module-config block with inline `#` comments on every field.
- `## Outputs` section split into `### Interpretation` (per-output bullet, what to look for) and `### Saved artifacts` (table).

**`__init__.py`:** `from .main import main` so the dispatcher can `import analyses.<name>.main`.

After scaffolding, sanity-check the YAML:

```bash
uv run python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('${SESSION_DIR}/code/configs/analysis/<name>.yaml'); print(OmegaConf.to_yaml(cfg))"
```

The output should show a top-level config with `_name_`, `_subdir`, `_output_dir`, plus the spec's knobs. If Hydra rejects the file, inspect the `# @package` directive and re-emit.

---

## Step 4: Implement the Body

Now fill in `main.py`. The shape mirrors `causalab/analyses/baseline/main.py`:

```python
def main(cfg: DictConfig) -> dict[str, Any]:
    analysis = cfg[ANALYSIS_NAME]
    out_dir = os.path.join(cfg.experiment_root, ANALYSIS_NAME, analysis._subdir)
    os.makedirs(out_dir, exist_ok=True)

    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )

    train_dataset, test_dataset = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )

    pipeline = load_pipeline(model_name=cfg.model.name, ...)

    # Call methods listed in set_up_analysis.md §3
    results = <method>(...)

    save_json_results(results, out_dir, "results.json")
    save_experiment_metadata(out_dir, cfg)
    return results
```

Implementation rules:
- Save through `causalab.io.*` primitives only (invariant 3).
- `experiment_root` consumed via `cfg.experiment_root`. Never override.
- Auto-discover upstream artifacts using `causalab.io.pipelines` scanners when applicable (`ANALYSIS_GUIDE.md` "Auto-Discovery").
- No hyperparameter defaults inline — read from `analysis.<knob>`.

Test by composing a small runner config and dispatching the `--cfg job` introspection (per `/run-experiment` Step 4). The dispatcher imports `analyses.<name>.main` once `${SESSION_DIR}/code/` is on `PYTHONPATH` (set automatically by `scripts/run_exp.sh` when `--experiment-root` lives under `agent_logs/`).

---

## Step 5: Layering Audit

```bash
grep -rE "torch\.save|safetensors|json\.dump\(" "${SESSION_DIR}/code/analyses/<name>/main.py" \
  | grep -vE "save_(json|tensor)_results|save_experiment_metadata"  # OK
```

Anything left after the `grep -v` is a hand-rolled disk write that should go through `causalab.io.artifacts` instead.

```bash
grep -E "^\s*(epochs|batch_size|lr|learning_rate|smoothness|reg_coef|k_features)\s*=" \
    "${SESSION_DIR}/code/analyses/<name>/main.py"
```

Any literal hyperparameter assignment in code is an invariant-5 violation — move it to `analysis.yaml`.

If the analysis depends on a method that arguably belongs in `causalab/methods/`, file an issue via `/document-issues` so promotion-time decisions stay visible.

---

## Step 6: Hand-off

After every spec in the batch has cleared Steps 3–5, print one summary:

```
Batch scaffolded (N analyses):
  - <name_1>   ${SESSION_DIR}/code/analyses/<name_1>/  +  configs/analysis/<name_1>.yaml
  - <name_2>   ${SESSION_DIR}/code/analyses/<name_2>/  +  configs/analysis/<name_2>.yaml
  …

Reference each from a runner config (place under ${SESSION_DIR}/code/configs/runners/<group>/):

    defaults:
      - base
      - task: <task>
      - model: <model>
      - analysis/<name>            # resolves to session-local YAML via search-path injection
      - _self_

The runner picks up session-local analyses automatically when --experiment-root lives
under agent_logs/. See /run-experiment Step 5 for the env requirement.

Next: hand off to /run-experiment. If any analyses' §3 dependencies still need
session-local methods scaffolded, invoke /setup-methods (batch the remaining specs)
before continuing.
```

---

## Important Notes

### What this skill does NOT do

- **Does not edit anything under `causalab/`.** The shipped package is read-only here.
- **Does not write a runner config.** Runner-config drafting belongs to `/run-experiment` Step 3.
- **Does not run the analysis.** The Hydra dispatch happens in `/run-experiment`.
- **Does not promote.** Promotion to `causalab/analyses/` is manual.

### Restrictions

- Only edit files under `${SESSION_DIR}/code/analyses/<name>/`, `${SESSION_DIR}/code/configs/analysis/<name>.yaml`, and `${SESSION_DIR}/issues.md`.
- Read templates only from `.claude/skills/setup-analyses/templates/`.
- Refuse names that collide with any directory or file under `causalab/analyses/` or `causalab/configs/analysis/`.
