---
name: setup-analyses
description: Scaffold one or more analyses (research-question wrappers, Hydra entry points) in a single invocation. Use when an experiment plan needs analyses that don't yet exist in causalab/analyses/. Generates code out of tree under $WORKDIR/code/analyses/<name>/ and Hydra configs under $WORKDIR/code/configs/analysis/<name>.yaml. Pair with the setup-methods guide for missing primitives.
---

# Setup Analyses Skill

Tell the user, in plain prose, which analyses you're scaffolding and why; surface the batch for approval, then report what landed.

Scaffolds one or more *analyses* (research-question wrappers) under the active research session. Analyses are the top library layer in `docs/CODEBASE.md` §3: they orchestrate I/O and Hydra config, chain methods, own artifact-directory layouts, and expose a `main(cfg)` Hydra entry point. The runner picks them up via session-code path injection (see Step 5 hand-off).

> **Before scaffolding, check it isn't already shipped.** Some capabilities that look like a missing analysis already exist in `causalab/analyses/`:
> - **Threading a given/precomputed subspace** (SAE decoder directions, an imported rotation) through the manifold pipeline → use the shipped **`subspace` analysis with `method: fixed`** (`subspace.fixed.{artifact|source|feature_ids}`, single explicit `layers: [L]`). It writes the bundle `activation_manifold`/`path_steering` auto-discover, including `raw_features` for the linear path mode. **Do NOT scaffold a new `fixed_subspace` analysis** (it was re-invented 9+ times and re-hit the same traps — issue #218).

## Batch invocation

This skill is **loaded once** per run-phase setup invocation. The caller passes the full list of analysis specs (one per §D node in `PLAN.md` whose Coverage status is `analysis-gap` or `both-gap`) at once, and the skill loops Steps 3–5 over each spec sequentially. The session resolves once, and the Step 6 hand-off runs once for the whole batch. This keeps the scaffolding canonical (single source of truth in this skill) without re-loading this document per analysis.

When called with no spec paths, the skill falls back to the legacy single-spec interactive flow (Step 1 elicits one spec; Steps 3–5 run once).

The skill writes **out of tree**, under the run's working directory `$WORKDIR`:
- `$WORKDIR/code/analyses/<name>/` — Python module (`main.py`, `__init__.py`, `README.md`, `set_up_analysis.md`).
- `$WORKDIR/code/configs/analysis/<name>.yaml` — Hydra config that the runner mounts at `cfg.<name>`.

The runner picks these up via the session-local `analyses.<name>` fallback when it runs with `CAUSALAB_SESSION_CODE=$WORKDIR` set. Never author analysis code inside the lab-managed library checkout at `~/.silico/libraries/causalab-internal/` (it is re-synced / image-resident and would be clobbered or is unwritable); shipping a stabilized analysis upstream is a separate causalab-repo PR, not part of this run.

## Required Reading

Before running this skill, read:

1. `docs/CODEBASE.md` §3 — the analysis-layer invariants (§3 invariants 2–5, 7, 11, 12): analyses orchestrate methods while `methods/` and `io/` stay independent, defaults live in Hydra not code, `experiment_root` is the single source of truth for output paths, every package ships a 3-section README, and dataset-construction params live under `cfg.task.*` while `seed` lives at `cfg.seed`. Read the section for the authoritative numbered list — don't rely on numbers transcribed here.
2. `../ANALYSIS_GUIDE.md` — the analysis dependency DAG, auto-discovery rules, and runner-YAML composition patterns.
3. `docs/CODEBASE.md` §2 — the runner config system (`# @package`, `_name_`, `_subdir`, `_output_dir`).

When one or more spec paths are provided, scaffold each spec sequentially and confirm the batch with the user once (Step 2); when none is provided, fall back to single-spec interactive elicitation per `instructions/create_specification.md`. Surface anything blocked or ambiguous to the user — a layering rule that is hard to satisfy, a missing upstream artifact, or an analysis that depends on a method that arguably belongs in `causalab/methods/`.

---

## Step 1: Read or Elicit the Specifications

The skill consumes one or more markdown specs — each `set_up_analysis.md` laid out per `SET_UP_ANALYSIS_TEMPLATE.md`. The input is one or more spec paths; describe them to the user in prose ("the analysis specs you handed me"), never as a typed argument. Input shapes:

1. **Spec paths given** (one or more, space-separated, to existing markdown files) → read each and use directly. Order is preserved; specs are processed in the order received.
2. **No paths** → run `instructions/create_specification.md` and elicit a single spec section by section, writing the draft to `$WORKDIR/code/analyses/<name>/set_up_analysis.md` as it grows. Get user approval at each section. (Interactive elicitation is single-spec only; batches must come in via paths.)

After this step every `$WORKDIR/code/analyses/<name>/set_up_analysis.md` referenced in the input exists and is approved.

### Refuse name collisions

For **each** spec, before proceeding, check that its `<name>` does not already exist as a shipped `causalab/analyses/<name>/` or `causalab/configs/analysis/<name>.yaml`, nor as an existing `$WORKDIR/code/analyses/<name>/`. If a collision is found, refuse the **whole batch** with:

> "An analysis named `<name>` already ships in `causalab/`. Pick a different name; a session-local analysis must not shadow a shipped one (the dispatcher prefers shipped over session-local). (Batch aborted before any scaffolding ran.)"

Surface all collisions first, then abort, so the caller can fix names in one pass.

---

## Step 2: Batch Approval Checkpoint

Lay out the whole batch for the user in plain prose, one analysis per line: its name, the research question it answers, the upstream artifacts it reads, the methods it chains, and the config defaults you propose. Then say which files each analysis will create — the Python bundle under `$WORKDIR/code/analyses/<name>/` and the matching Hydra config under `$WORKDIR/code/configs/analysis/`. Close by asking whether to approve all, edit one, or cancel the batch.

Proceed only on approval. "Edit one" returns to Step 1 for that single spec, then re-enters Step 2 with the revised batch.

---

## Step 3: Scaffold from Templates

**Loop Steps 3, 4, and 5 per spec, in argv order.** All other steps run once per batch.

For the current spec, create the directories and files. Substitutions follow the spec:

```
$WORKDIR/code/analyses/<name>/
├── __init__.py            from templates/__init__.py
├── main.py                from templates/main.py
├── README.md              from templates/README.md
└── set_up_analysis.md     (already saved in Step 1)

$WORKDIR/code/configs/analysis/<name>.yaml   from templates/analysis.yaml
```

**`main.py` substitutions:**
- `<name>` → analysis name (lowercase, snake_case).
- `ANALYSIS_NAME = "<name>"` constant — must match `_name_` in the YAML (the runner cross-checks the module's `ANALYSIS_NAME` against the slice's `_name_` in `causalab/runner/run_exp.py::_run_steps`; a mismatch is a hard error).
- Imports section: emit the methods listed in spec §3 plus standard helpers from `causalab.runner.helpers` (`resolve_task`, `generate_datasets`) and `causalab.io.artifacts` (`save_json_results`, `save_tensor_results`, `save_experiment_metadata`).
- Body: a stub that loads the task, generates datasets, calls each method, then saves outputs. Each step is a `# TODO: ` comment listing what the spec says to do; the function ends with `raise NotImplementedError(...)` so the agent has to fill it in during Step 4.

**`analysis.yaml` substitutions** (per docs/CODEBASE.md §2):
- First line `# @package <name>` — Hydra mounts the file at `cfg.<name>`.
- `_name_: <name>` — sentinel the dispatcher uses to detect analysis steps.
- `_subdir: ${.method}_<key>${.<key>}` — the `_subdir` pattern decides where this analysis writes inside `${experiment_root}/<name>/`. **Never a fixed literal like `_subdir: default`**, even when the analysis has no method/parameter axis to encode — concurrent or follow-up dispatches into the same literal dir overwrite each other (#171 C4); pattern it on a field that distinguishes the runs you might launch (`${.target_variable}`, `L${.layers[0]}`, … — see `SET_UP_ANALYSIS_TEMPLATE.md`).
- `_output_dir: ${experiment_root}/<name>/${._subdir}` — every method that saves uses this as its base.
- Every config knob from spec §4 with its proposed default. **Do not** include `n_train`, `n_test`, `enumerate_all`, `balanced`, or `seed` — those live in `task:` / root (dataset-construction params, per the docs/CODEBASE.md §3 invariant 12).
- A `visualization:` block with `figure_format: ${figure_format}` if the analysis emits matplotlib figures — inherits the global PNG default; override per-analysis only for PDF (the figure-format invariant, docs/CODEBASE.md §3 invariant 6).

**Make it fan-out-friendly (cheap, optional).** If the analysis has a scan dimension (layers, token positions, an index, …), expose it as a config knob whose body **restricts its work to a single value when given one** — that lets `causalab/runner/fanout.py` shard one GPU job per value (see `docs/CODEBASE.md` "Fan-out"). For `--collect` to recombine the shards into one artifact rather than leaving per-shard files, either encode the shard axis in `_subdir` so each shard writes a disjoint path, or emit the `locate`-style `results.json` with a top-level `scores_per_cell` keyed `"<layer>|<pos_id>"`. Writing only under `_output_dir` (the `experiment_root`-is-the-single-source-of-truth invariant, docs/CODEBASE.md §3 invariant 7) is what makes per-shard isolation automatic — never write to a fixed/global path.

**`README.md` substitutions** (the 3-section-README invariant, docs/CODEBASE.md §3 invariant 11):
- Opening paragraph: name, italicized research question, what it does mechanically, where it sits in the pipeline.
- `## Configuration` section: root-config params it reads (e.g. `experiment_root`, `seed`); module-config block with inline `#` comments on every field.
- `## Outputs` section split into `### Interpretation` (per-output bullet, what to look for) and `### Saved artifacts` (table).

**`__init__.py`:** `from .main import main` so the dispatcher can `import analyses.<name>.main`.

After scaffolding, sanity-check the YAML (run from the library checkout so `omegaconf` resolves):

```bash
cd ~/.silico/libraries/causalab-internal
uv run python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('$WORKDIR/code/configs/analysis/<name>.yaml'); print(OmegaConf.to_yaml(cfg))"
```

The output should show a top-level config with `_name_`, `_subdir`, `_output_dir`, plus the spec's knobs. If Hydra rejects the file, inspect the `# @package` directive and re-emit.

---

## Step 4: Implement the Body

Now fill in `main.py`. The scaffold is `templates/main.py` — the single source of truth for the module shape (the `_locate_analysis_cfg` helper, the load-task / build-datasets / load-pipeline preamble, and the commented save-and-return tail). Don't re-derive that shape from memory: open the template, then replace its `# TODO` / `NotImplementedError` body with the method calls from spec §3, reaching into the analysis config slice for every knob.

Implementation rules:
- Save through `causalab.io.*` primitives only — the payload comes first, the output dir second.
- `experiment_root` consumed via `cfg.experiment_root`. Never override.
- Auto-discover upstream artifacts using `causalab.io.pipelines` scanners when applicable (`../ANALYSIS_GUIDE.md` "Auto-Discovery").
- No hyperparameter defaults inline — read every knob from the analysis config slice.
- **Locate the analysis slice via `_locate_analysis_cfg(cfg)`, not `cfg[ANALYSIS_NAME]`.** The runner passes the whole cfg and dispatches by the slice's `_name_`, so the top-level key only equals `ANALYSIS_NAME` under the default `# @package {{ANALYSIS_NAME}}`. If you mount the config under a different package (`# @package <other>`, the CONFIG_KEY ≠ ANALYSIS_NAME case), the helper still finds the slice by its `_name_`. The shipped template already does this — keep it.

Test by composing a small runner config and dispatching the `--cfg job` introspection (run mechanics live in `../running-experiments.md`). The dispatcher imports `analyses.<name>.main` once `$WORKDIR/code/` is on `PYTHONPATH` (set automatically by `scripts/run_exp.sh` when it runs with `CAUSALAB_SESSION_CODE=$WORKDIR`).

---

## Step 5: Layering Audit

```bash
grep -rE "torch\.save|safetensors|json\.dump\(" "$WORKDIR/code/analyses/<name>/main.py" \
  | grep -vE "save_(json|tensor)_results|save_experiment_metadata"  # OK
```

Anything left after the `grep -v` is a hand-rolled disk write that should go through `causalab.io.artifacts` instead.

```bash
grep -E "^\s*(epochs|batch_size|lr|learning_rate|smoothness|reg_coef|k_features)\s*=" \
    "$WORKDIR/code/analyses/<name>/main.py"
```

Any literal hyperparameter assignment in code violates the defaults-live-in-Hydra invariant (docs/CODEBASE.md §3 invariant 5) — move it to `analysis.yaml`.

If the analysis depends on a method that arguably belongs in `causalab/methods/`, flag it to the user so the decision stays visible.

---

## Step 6: Hand-off

After every spec in the batch has cleared Steps 3–5, print one summary:

```
Batch scaffolded (N analyses):
  - <name_1>   $WORKDIR/code/analyses/<name_1>/  +  $WORKDIR/code/configs/analysis/<name_1>.yaml
  - <name_2>   $WORKDIR/code/analyses/<name_2>/  +  $WORKDIR/code/configs/analysis/<name_2>.yaml
  …

Reference each from a runner config (under $WORKDIR/code/configs/runners/<group>/):

    defaults:
      - base
      - task: <task>
      - model: <model>
      - analysis/<name>            # resolves to session-local YAML via search-path injection
      - _self_

The runner picks up session-local analyses automatically when it runs with
CAUSALAB_SESSION_CODE=$WORKDIR set. See running-experiments.md for the env requirement.

Next: hand back to running the experiment (run mechanics in running-experiments.md).
If any analyses' §3 dependencies still need methods scaffolded, invoke the
setup-methods guide (batch the remaining specs) before continuing.
```

---

## Important Notes

### What this skill does NOT do

- **Does not write a runner config.** Runner-config drafting belongs to running the experiment (`../running-experiments.md`).
- **Does not run the analysis.** The Hydra dispatch happens when the experiment runs.

### Restrictions

- Only create/edit files under `$WORKDIR/code/analyses/<name>/` and `$WORKDIR/code/configs/analysis/<name>.yaml`.
- Read templates only from `setup-analyses/templates/`.
- Refuse names that collide with any directory or file under `causalab/analyses/` or `causalab/configs/analysis/`.
