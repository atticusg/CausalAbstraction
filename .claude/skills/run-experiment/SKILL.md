---
name: run-experiment
description: Run causal abstraction experiments on language models. Guides through task selection, model configuration, counterfactual generation, and intervention experiments. Use when the user wants to run experiments, run an experiment, execute experiments, launch experiments, start experiments, test interventions, do causal analysis, analyze model internals, probe a model, or do anything related to running or executing experiments on models.
---

# Run Experiment Skill

Autonomous executor for causal abstraction experiments. Consumes the plan produced by `/plan-experiment` and runs it end-to-end without prompting the user. The decision-making — task choice, analysis chain, knobs, sweep strategy — happens upstream in `/plan-experiment`. This skill materializes the runner config(s), executes them, and verifies artifacts.

The execution model is **runner-config–driven** — you do NOT write per-run Python scripts. You compose a YAML at `${SESSION_DIR}/code/configs/runners/<group>/<name>.yaml`, run `./scripts/run_exp.sh <name>` with the session-scoped `--experiment-root` override, and outputs land under `${SESSION_DIR}/artifacts/{task}/{model}/{analysis}/`. Stable presets live at `causalab/configs/runners/<group>/` and serve as starting points to copy from — but the runner you write for this session stays in the session's `code/` tree (per the session-dir invariant in `CONVENTIONS.md`). See `ARCHITECTURE.md` §2 for the config system and §3 invariant 7 for the `experiment_root` routing rule.

## Required Reading

Before running any experiments, read:

1. **`.claude/skills/research-session/CONVENTIONS.md`** — research-session layout, active-session detection protocol, `${SESSION_DIR}` resolution, per-skill write rules.
2. **`.claude/skills/plan-experiment/ANALYSIS_GUIDE.md`** — analysis dependency DAG, runner-YAML composition patterns, auto-discovery rules, per-analysis parameter decisions, CLI overrides, slurm dispatch, common pitfalls.

`ARCHITECTURE.md` and `COMMON_PROBLEMS.md` are referenced inline at the points they bite — read them then, not up front.

## Issue Tracking

**Use the `document-issues` skill throughout this workflow.** Issues go to `${SESSION_DIR}/issues.md` (top level — see `.claude/skills/research-session/CONVENTIONS.md`).

---

## Step 0a: Resolve or Bootstrap the Active Session

```bash
if [ -f agent_logs/.current ]; then
    SESSION_NAME="$(cat agent_logs/.current)"
    SESSION_DIR="agent_logs/${SESSION_NAME}"
fi
```

- If `agent_logs/.current` and `${SESSION_DIR}` both resolve → continue to Step 0b.
- Otherwise hand off to `/research-session` rather than failing:
  1. Determine the research objective from the user's most recent message and conversation context.
  2. If the objective is unclear or absent, ask the researcher one focused question to elicit it. Do not invoke `/research-session` blind — that produces a session with a generic auto-slug.
  3. Invoke `/research-session` via the Skill tool, passing the objective.
  4. After it returns, re-read `agent_logs/.current` and proceed.

---

## Step 0b: Locate PLAN.md

```bash
PLAN="${SESSION_DIR}/plan/PLAN.md"
[ -f "${PLAN}" ] || { echo "missing"; }
```

- Present → continue to Step 1.
- Missing → tell the user: *"No PLAN.md in this session — please run `/plan-experiment` to produce one. `/run-experiment` consumes the plan; it does not redo planning."* Stop. Do **not** auto-invoke `/plan-experiment` — that skill is interactive by design and needs the user driving objective elicitation.

---

## Step 1: Read the Plan

Read both files produced by `/plan-experiment`:

- `${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md` — objective, scope, success criteria, hypotheses.
- `${SESSION_DIR}/plan/PLAN.md` — task spec (§B), neural surface (§C), analysis-chain DAG with per-node knobs and pre-flight checks (§D), risk register (§E), runner-config names + sweep strategy + expected artifact tree (§F).

Extract two lists used by later steps:

- **Setup queue** (ordered: task first, then methods, then analyses) — items PLAN.md marks as needing scaffolding, each with the `Spec path:` field PLAN.md provides.
  - §B `Status: to be created` → one task entry.
  - §D nodes whose first line reads `custom — out of scope, planned but not implemented` → one entry per missing method/analysis.
- **Runner configs** — the §F names + the §D per-node knobs that hydrate them in Step 5. If §F declares a sweep with a `sweep_id`, every runner config in this run inherits `experiment_root = ${SESSION_DIR}/artifacts/{task}/{model}/{sweep_id}`.

The plan is the single source of truth. This skill executes it; it does not redo planning decisions.

---

## Step 2: Optional Task Setup

If §B `Status` reads `to be created (autonomous /setup-task input below)`:

1. Confirm `causalab/tasks/{task_name}/` does **not** already exist. If it does, PLAN.md is stale — log via `/document-issues` and skip to Step 3.
2. Resolve the companion spec at `${SESSION_DIR}/plan/setup_task_{task_name}.md`:
   - If §B has a `Spec path:` field and the file exists → use it.
   - Otherwise → **synthesize the spec yourself**. PLAN.md §B is structured (per PLAN_TEMPLATE.md line 13) to be lifted verbatim into a task spec, so copy §B's content into the new file: causal variables table, mechanism summary, expected behavior, edge / stress cases, counterfactual generator config, dataset sizing. Match the structure expected by `/setup-task` — see `.claude/skills/setup-task/TASK_TEMPLATE.md`.
   - For any field §B leaves unspecified, pick a defensible default and **log the gap** via `/document-issues` (one issue per gap, with the field name and the default used). Do not stop on missing detail — continue autonomously.
3. Invoke `/setup-task` via the Skill tool, passing the spec path.
4. Wait for completion. Re-verify `causalab/tasks/{task_name}/` now contains `causal_models.py` + `counterfactuals.py`.

If §B `Status` reads `exists`, skip this step.

Read the implementation files in `causalab/tasks/{task_name}/` (and the spec at `causalab/tasks/{task_name}/set_up_task.md` if present) before proceeding.

---

## Step 3: Validate Task Setup

**CRITICAL**: Before composing a runner config, validate the task setup so that subsequent runs don't silently produce noise.

If the task has a `tests/` directory, run:
```bash
uv run pytest causalab/tasks/{task_name}/tests/ -v
```

Otherwise, run the validation script `causalab/tasks/{task_name}/tests/test_with_model.py` (or the template at `.claude/skills/setup-task/templates/tests/test_with_model.py`). It runs:

1. **Forward pass** — sample an input, run `pipeline.generate`, check the checker.
2. **Token positions** — verify each declared position lands on the intended token.
3. **Token alignment (Test 3)** — verify `raw_output` tokenizes to the same tokens the model actually generates. A mismatch here silently breaks every token-level intervention. See `COMMON_PROBLEMS.md` for fixes.

Only proceed once all three pass.

---

## Step 4: Optional Method / Analysis Setup

Walk PLAN.md §D nodes once and **partition** every card whose first line reads `custom — out of scope, planned but not implemented` into two lists:

- **Method specs** — node names a primitive that would live under `causalab/methods/<name>/` (a featurizer, scorer, distance, training loop, …). Companion spec at `${SESSION_DIR}/plan/setup_method_{name}.md`.
- **Analysis specs** — node names a Hydra entry point that would live under `causalab/analyses/<name>/` (a research-question wrapper). Companion spec at `${SESSION_DIR}/plan/setup_analysis_{name}.md`.

For each node in either list, resolve the companion spec:
- If the node has a `Spec path:` field and the file exists → use it.
- Otherwise → **synthesize the spec yourself** from the §D card (research question, method choice, non-default knobs, upstream/downstream artifacts, pre-flight check) plus relevant §C/§E context. Follow the structure expected by `/setup-methods` (see `.claude/skills/setup-methods/SET_UP_METHOD_TEMPLATE.md`) or `/setup-analyses` (see `.claude/skills/setup-analyses/SET_UP_ANALYSIS_TEMPLATE.md`). Save the synthesized spec under `${SESSION_DIR}/plan/setup_method_{name}.md` or `${SESSION_DIR}/plan/setup_analysis_{name}.md`.
- For anything the plan doesn't pin down (function signature, internal hyperparameters, IO contract details), pick a defensible default and **log the gap** via `/document-issues` (one issue per uncertainty, naming the field and the default used). Do not stop on missing detail — continue autonomously.

Once both lists are fully resolved, invoke the setup skills **in batch — one call each, passing every spec path at once**:

1. **Methods first.** Invoke `/setup-methods` once, passing the full space-separated list of method spec paths. The skill loads SKILL.md once, scaffolds each spec sequentially, and returns when the batch is done.
2. **Then analyses.** Invoke `/setup-analyses` once, passing the full list of analysis spec paths. Methods must be scaffolded first because an analysis may import a session-local method.
3. **Fill in stub bodies.** After each batch returns, walk the scaffolded files and replace each `raise NotImplementedError(...)` placeholder with the real implementation derived from the spec, so the runner config in Step 5 has something runnable to import. This is the only place `/run-experiment` writes session-local code.

If exactly one custom node is needed, you may still pass a single-element list to the batch skill — there is no separate single-spec path. (The skills' single-spec interactive elicitation flow is reserved for direct user invocation outside `/run-experiment`.)

If no §D node is `custom`, skip this step.

---

## Step 5: Compose the Runner Config

Pick a similar existing preset under `causalab/configs/runners/<group>/` as a starting point — there are 65 presets organized by task family, and `runners/demos/` has 30 minimal single-analysis configs to model from.

Write the runner config to `${SESSION_DIR}/code/configs/runners/<group>/<name>.yaml`, where `<group>` is the task family and `<name>` matches the names declared in PLAN.md §F (e.g. `weekdays_8b_locate.yaml`). Iterate in-place. The wrapper auto-discovers runners under both `causalab/configs/runners/` and `${SESSION_DIR}/code/configs/runners/` (see `scripts/run_exp.sh:99-141`). Promotion to `causalab/configs/runners/` is a manual `git mv` once the preset stabilizes — there is no auto-promotion step.

The YAML must follow the structure in `ANALYSIS_GUIDE.md` "Runner YAML Patterns":

```yaml
defaults:
  - base
  - task: <task_name>
  - model: <model_name>
  - analysis/<step1>
  - analysis/<step2>          # add one entry per analysis to chain
  - _self_

task:
  n_train: 200                # dataset construction lives in task block (ARCHITECTURE invariant 12)
  n_test: 100
  target_variables: [day]
  resample_variable: day      # only when using locate.mode: pairwise (see ARCHITECTURE §5)

<step1>:
  # only knobs that diverge from configs/analysis/<step1>.yaml defaults
locate:
  method: interchange
  layers: [0, 4, 8, 12, 16, 20, 24, 28]
  mode: centroid

post:
  - type: variable_localization_heatmap
    source_step: locate
    source_method: interchange
```

Rules (from `ARCHITECTURE.md` §2 and §3):

- **Order of `- analysis/<name>` entries = order of execution.** Always run `baseline` first.
- **Prefer `null` (auto-discovery) over hardcoded paths** for `subspace.layers`, `activation_manifold.subspace`, `pullback.belief_path.output_manifold_ckpt`, etc. Auto-discovery rules are listed in `ANALYSIS_GUIDE.md`.
- **Dataset construction (`n_train`, `n_test`, `enumerate_all`, `balanced`) lives in the `task:` block**, not in any analysis block (invariant 12).
- **Do not embed `experiment_root` in the YAML.** Pass it on the CLI via `--experiment-root` (Step 7) so the same runner preset runs unchanged in dev mode (`artifacts/...`) and research mode (`${SESSION_DIR}/artifacts/...`). Invariant 7 spells out the two contexts.
- **Do not write per-run scripts.** No file under `experiments/{task}/{timestamp}/`. Each analysis already has a `main(cfg)` Hydra entry point at `causalab/analyses/<name>/main.py`; `causalab/runner/run_exp.py` dispatches to them in order.

---

## Step 6: Inspect the Resolved Config

Print the fully resolved Hydra config and snapshot it. Pass the same session-scoped `experiment_root` you'll use in Step 7 so the snapshot reflects the actual run:

```bash
EXP_ROOT="${SESSION_DIR}/artifacts/{task}/{model}"
uv run python -m causalab.runner.run_exp --config-name {config_name} \
    experiment_root="${EXP_ROOT}" --cfg job \
    | tee "${SESSION_DIR}/run/{config_name}_resolved.yaml"
```

This shows every interpolation resolved (paths, defaults inherited, overrides applied). Pay particular attention to:

- `experiment_root` — should point at `${SESSION_DIR}/artifacts/{task}/{model}/` (with `/{variant}` appended automatically when `task.variant` is set).
- `task.target_variables` and `task.resample_variable` — must be consistent with the analyses you chained (see ARCHITECTURE §5 for `resample_variable` × `locate.mode`).
- Each analysis's `_output_dir` — confirms where its artifacts will land (under the session-scoped root).

The snapshot lands in `run/`. Proceed to Step 7 without prompting.

---

## Step 7: Run

Always pass `--experiment-root` so artifacts land inside the session, not in the global `artifacts/` tree:

```bash
EXP_ROOT="${SESSION_DIR}/artifacts/{task}/{model}"

./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" {config_name}                                      # inline
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" {config_name} task.n_train=16 locate.layers=[0,8,16]   # debug pass
./scripts/run_exp.sh --slurm --experiment-root "${EXP_ROOT}" {config_name}                              # cluster
```

**Session-local analyses (from `/setup-analyses`)** chain into runner configs via `- analysis/<name>` defaults entries that resolve against `${SESSION_DIR}/code/configs/`. The wrapper detects `--experiment-root` under `agent_logs/` automatically and:
- prepends `${SESSION_DIR}/code/` to `PYTHONPATH` so `import analyses.<name>` and `import methods.<name>` resolve;
- appends `${SESSION_DIR}/code/configs/` to Hydra's search path so the YAML resolves.

If your runner chains a session-local analysis but you're invoking `causalab.runner.run_exp` directly (not via the wrapper), set the env vars manually:

```bash
PYTHONPATH="${SESSION_DIR}/code:${PYTHONPATH:-}" \
  uv run python -m causalab.runner.run_exp --config-name {config_name} \
    experiment_root="${EXP_ROOT}" \
    "++hydra.searchpath=[file://${SESSION_DIR}/code/configs]" --cfg job
```

**Always do a debug pass first** (small dataset, coarse layer scan). Once it produces sane artifacts, drop the per-run overrides and run the full preset.

The run is **blocking**. Redirect verbose stdout to a log file inside the session's `run/` dir to avoid filling the tool-result buffer:

```bash
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" {config_name} > "${SESSION_DIR}/run/run.log" 2>&1
```

Slurm resources resolve from `model.slurm.gpus` and `slurm.time`; override with `--gpus`, `--time`, `--qos` if needed. The `--experiment-root` flag is forwarded into the sbatch step automatically.

---

## Step 8: Verify Artifacts

Per ARCHITECTURE.md §3 invariant 7, research-mode artifacts land under `${SESSION_DIR}/artifacts/{task}/{model}/{analysis}/...`. List them:

```bash
ls -R "${SESSION_DIR}/artifacts/{task}/{model}/"
```

**Diff against the `Expected artifact tree` ASCII block in PLAN.md §F.** Anything in the plan that didn't materialize is a failure to investigate.

For each chained analysis, confirm the expected outputs exist (per `ANALYSIS_GUIDE.md` "Research Questions → Analyses" table). Common per-analysis files:

- `baseline/`: `accuracy.json`, `per_class_output_dists.safetensors`, `counterfactual_sanity.json`
- `locate/{method}/{variable}/`: `results.json` (with `best_cell`), `heatmap.pdf`
- `subspace/{method}_k{k}/`: `metadata.json`, rotation/transform `.pt`, `visualization/features_3d.html`
- `activation_manifold/{method}_s{s}/`: `manifold_spline/ckpt_final.pt`, `metadata.json`, `visualization/manifold_3d.html`
- `output_manifold/`, `path_steering/`, `pullback/`, `attention_pattern/` — see ANALYSIS_GUIDE.

If anything is missing, check `${SESSION_DIR}/run/run.log` and use `document-issues` to log the failure under `${SESSION_DIR}/issues.md`.

---

## Step 9: Hand Off to `/interpret-experiment`

Invoke `/interpret-experiment` (no arguments — the skill discovers the active session via `agent_logs/.current` and reads inputs directly from `${SESSION_DIR}/plan/`, `${SESSION_DIR}/run/`, and `${SESSION_DIR}/artifacts/`):

```
/interpret-experiment
```

`interpret-experiment` produces a single consolidated `${SESSION_DIR}/result/REPORT.md` (with embedded figures under `${SESSION_DIR}/result/figures/`) grounded in the session's planned objective, success criteria, and hypotheses. It runs autonomously — no approval prompts. This skill does not write its own results summary — interpretation lives entirely in `/interpret-experiment`.

---

## Iteration

To iterate (try different layers, methods, k_features, etc.) you have two options:

1. **One-off variation via Hydra override:** keep the runner config, override on the CLI (re-pass `--experiment-root`):
   ```bash
   ./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" weekdays_8b_locate locate.layers=[15,20]
   ```
2. **Persistent variation:** copy the runner config to a new name and edit it. Re-running a runner config rewrites its artifact subdirectory under `${EXP_ROOT}` — branch the config name to preserve old runs.

Persistent variation that changes the analysis chain or knobs in non-trivial ways belongs back in `/plan-experiment`, not here.

---

## Important Notes

### Workflow principles

1. **The plan is the source of truth.** If the plan is wrong, fix it via `/plan-experiment` and re-run. Don't patch around a bad plan from inside this skill.
2. **Plan first, run small, then run full.** Always do a debug pass with `task.n_train=16` and a coarse `locate.layers` before the full preset.
3. **Read `ANALYSIS_GUIDE.md` before each new analysis type** — the per-analysis "Key Parameter Decisions" tells you what actually needs a decision; everything else has a sensible default.
4. **Inspect with `--cfg job` before every run** — catches misconfigured paths and `task.resample_variable` × `locate.mode` mismatches early. Use the same `--experiment-root` you'll pass to the run.
5. **Always pass `--experiment-root ${SESSION_DIR}/artifacts/{task}/{model}`** — this is what makes the session self-contained. Don't override `experiment_root` to ephemeral paths like `/tmp/`; those artifacts are invisible to downstream analyses.

### Error handling

If a run fails:
1. Document the error in `${SESSION_DIR}/issues.md` via `document-issues`.
2. Check `${SESSION_DIR}/run/run.log` for the actual exception.
3. Re-run with the failing analysis isolated (run only that step) and `--cfg job` to inspect its config.
4. Common fixes are in `COMMON_PROBLEMS.md`.

### Restrictions

- **Do not prompt the user for approval.** This skill is autonomous. The only legitimate user interaction is the one objective-elicitation question in Step 0a when no session exists. Steps 5–9 must never prompt.
- **Do not write Python scripts that orchestrate runs.** Each analysis already has `main(cfg)`. The dispatcher is `causalab/runner/run_exp.py`.
- **Do not import from `causalab/methods/`, `causalab/runner/`, or `causalab/analyses/` in user-facing code at this skill's level** — those are library code consumed by analyses.
- **Do not modify core library files.** This skill composes runner configs and runs them. The only writes it makes outside `${SESSION_DIR}/code/configs/runners/` and `${SESSION_DIR}/run/` are: companion specs under `${SESSION_DIR}/plan/setup_*.md` (synthesized in Steps 2/4 when the plan didn't pre-write them), and the body of session-local methods/analyses scaffolded in Step 4.
- **Do not skip Step 3 (task validation).** A token-alignment mismatch silently destroys downstream results.
