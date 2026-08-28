---
name: running-experiments
description: How to run a causal-abstraction experiment on causalab — compose a runner config, pre-flight the tokenizer, inspect the resolved config, dispatch a stage (locally or on slurm, optionally fanned out), and verify the artifacts. Workspace-agnostic — it takes an experiment to run, however specified; the caller supplies the output location and decides the sequencing.
---

# Running experiments on causalab

The portable how-to for *running* a configured experiment. It takes **an experiment to run, however specified** — a single analysis, or a chain — and explains the mechanics: how to compose the runner config, validate the task, dispatch the run, and confirm the outputs landed. It does **not** decide *what* to run or *when* — that sequencing (consume a plan, gate between stages, iterate, hand off to interpretation) belongs to whatever drives the run; in silico the planner/worker/critic harness owns it.

The execution model is **runner-config–driven** — you do not write per-run Python scripts. You compose a YAML **out of tree** at `$WORKDIR/code/configs/runners/<group>/<name>.yaml` (under the run's working directory), run `./scripts/run_exp.sh <name>` with `CAUSALAB_SESSION_CODE=$WORKDIR` set and an `--experiment-root` override, and outputs land under `<experiment-root>/{analysis}/` (the `{task}/{model}` segment belongs in the root you pass — see "Where outputs go"). Stable presets ship under `causalab/configs/runners/<group>/` in the library checkout — they are read-only starting points to copy from, but the runner you author lives in `$WORKDIR/code/`, never inside the lab-managed checkout. See `docs/CODEBASE.md` §2 for the config system and §3 invariant 7 for the `experiment_root` routing rule. Each analysis already has a `main(cfg)` Hydra entry point at `causalab/analyses/<name>/main.py`; `causalab/runner/run_exp.py` dispatches to them in order — there are no per-run scripts.

## Where outputs go

`--experiment-root` is the one knob that routes every artifact. The caller supplies it. Below, `$WORKDIR` stands for a working directory under your experiment's output — where you collect this run's artifacts, logs, and resolved-config snapshots.

| Context | `--experiment-root` |
|---|---|
| Driven run (silico) | `$WORKDIR/artifacts/{task}/{model}` (keeps the run's outputs self-contained) |
| Dev / ad-hoc | omit it — the `artifacts/{task}/{model}` default in `causalab/configs/base.yaml` applies |

Pass it on the CLI, never embed it in the YAML — the same preset then runs unchanged in every context (invariant 7). When `task.variant` is set, `apply_experiment_root_variant` appends `/{variant}` automatically.

## Required reading

Before running, read **`ANALYSIS_GUIDE.md`** — the analysis dependency DAG, runner-YAML composition patterns, auto-discovery rules, per-analysis parameter decisions, CLI overrides, slurm dispatch, and common pitfalls. `docs/CODEBASE.md` and the sibling `COMMON_PROBLEMS.md` are referenced inline at the points they bite — read them then, not up front.

## Validate the task before you run it

Validate the task setup first, so a tokenization or alignment bug doesn't silently turn every downstream run into noise. `{task_dir}` is the resolved task dir — `$WORKDIR/code/tasks/{task_name}/` for a session-local task, `causalab/tasks/{task_name}/` for a shipped one.

### Pre-flight tokenizer check (model-free, blocking)

Run this first — it needs only the tokenizer (no weights, no GPU, sub-second) and catches a class of tokenization gotchas that otherwise stay invisible until the baseline returns ~0% accuracy. The classic trap: a `prompt_suffix` ending in a space (e.g. `"\nAnswer: "`). BPE/sentencepiece tokenizers encode that trailing space as its own orphan whitespace token, so the model's next-token target becomes the bare answer form (`"Kate"`) instead of the leading-space merged form (`" Kate"`) the task's declared answer tokens expect — every comparison then fails. The check reads the accepted forms from the task's `CausalModel.output_tokens` declaration and is shipped as `causalab/tasks/preflight.py`:

```bash
# Session-local task needs $WORKDIR/code on PYTHONPATH (a shipped task does not).
cd ~/.silico/libraries/causalab-internal
PYTHONPATH="$WORKDIR/code${PYTHONPATH:+:$PYTHONPATH}" \
  uv run python -m causalab.tasks.preflight --task {task_name} --model {model_name}
```

- **Exit 0** — clean; proceed to the model-dependent tests.
- **Exit 1** — tokenization error(s). Block the run, fix the task per the error (typically: drop the trailing space so the prompt ends on a non-whitespace character — `"\nAnswer:"` not `"\nAnswer: "`), then re-run.
- **Exit 2** — the check could not run (e.g. a factory task that needs its run config to sample). Not a finding; fall through to the model-dependent alignment test below.

### Model-dependent validation

If the task has a `tests/` directory, run `PYTHONPATH="$WORKDIR/code:$PYTHONPATH" uv run pytest {task_dir}/tests/ -v` from the library checkout (the `PYTHONPATH` prefix is only needed for a session-local task). Otherwise run the validation script `{task_dir}/tests/test_with_model.py` (template at `setup-task/templates/tests/test_with_model.py`). It checks a forward pass against the checker, that each declared token position lands on the intended token, and that `raw_output` tokenizes to the tokens the model actually generates — a mismatch there silently breaks every token-level intervention (fixes in `COMMON_PROBLEMS.md`). Proceed only once all pass.

## Compose the runner config

Copy a similar preset under `causalab/configs/runners/<group>/` (the shipped tree) as a starting point (`runners/demos/` holds minimal single-analysis configs). Write yours to `$WORKDIR/code/configs/runners/<group>/<name>.yaml`, where `<group>` is the task family. The wrapper auto-discovers runners by basename under both `causalab/configs/runners/` and `$WORKDIR/code/configs/runners/` (the latter when `CAUSALAB_SESSION_CODE=$WORKDIR` is set).

```yaml
# @package _global_           # runners live in a subdir → must declare _global_
defaults:
  - /base                     # absolute (leading-/) paths so they resolve from the config root
  - /task: <task_name>
  - /model: <model_name>
  - /analysis/<step1>
  - /analysis/<step2>         # one entry per chained analysis; order = order of execution
  - _self_

task:
  n_train: 200                # dataset construction lives in the task block (invariant 12)
  n_test: 100
  target_variables: [day]
  resample_variable: day      # only with locate.mode: pairwise (docs/CODEBASE.md §5)

locate:
  method: interchange
  layers: [0, 4, 8, 12, 16, 20, 24, 28]
  mode: centroid

post:
  - type: variable_localization_heatmap
    source_step: locate
    source_method: interchange
```

Rules (from `docs/CODEBASE.md` §2/§3):

- **Runners need `# @package _global_` + absolute (leading-`/`) defaults paths.** A bare `- analysis/<name>` resolves relative to the runner's own subdir and fails to load — copy a shipped preset (e.g. `causalab/configs/runners/demos/baseline_demo.yaml`) to inherit the form.
- **Order of `- analysis/<name>` entries is the order of execution.** Always run `baseline` first.
- **Prefer `null` (auto-discovery) over hardcoded paths** for `subspace.layers`, `activation_manifold.subspace`, `pullback.belief_path.output_manifold_ckpt`, etc. — rules in `ANALYSIS_GUIDE.md`.
- **Dataset construction (`n_train`, `n_test`, `enumerate_all`, `balanced`) lives in the `task:` block**, not in any analysis block (invariant 12).
- **Do not embed `experiment_root` in the YAML** — pass it on the CLI (above).
- **Do not write per-run scripts** under `experiments/{task}/{timestamp}/`.

## Inspect the resolved config

Print the fully resolved Hydra config before running — pass the same `--experiment-root` you will run with, so what you inspect matches what runs:

```bash
cd ~/.silico/libraries/causalab-internal
export CAUSALAB_SESSION_CODE="$WORKDIR"     # so session-local code + configs resolve
EXP_ROOT="$WORKDIR/artifacts/{task}/{model}"
PYTHONPATH="$WORKDIR/code:$PYTHONPATH" uv run python -m causalab.runner.run_exp --config-name {config_name} \
    experiment_root="${EXP_ROOT}" \
    "++hydra.searchpath=[file://$WORKDIR/code/configs]" --cfg job
```

Check `experiment_root` (points under your output dir, `/{variant}` appended when `task.variant` is set), `task.target_variables` × `task.resample_variable` consistency with the chained analyses (docs/CODEBASE.md §5), and each analysis's `_output_dir`. You need not `tee` this — `run_exp.sh` snapshots the resolved config to `$WORKDIR/run/{config_name}_resolved.yaml` at submit, so the canonical record always exists pre-run.

## Dispatch a run

Always pass `--experiment-root` so artifacts land where the caller collects them, not in the global `artifacts/` tree:

```bash
cd ~/.silico/libraries/causalab-internal
export CAUSALAB_SESSION_CODE="$WORKDIR"     # wrapper adds $WORKDIR/code to PYTHONPATH + Hydra search path
EXP_ROOT="$WORKDIR/artifacts/{task}/{model}"

./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" {config_name}                                          # inline
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" {config_name} task.n_train=16 locate.layers=[0,8,16]   # debug pass
./scripts/run_exp.sh --slurm --experiment-root "${EXP_ROOT}" {config_name}                                  # cluster
```

**Sbatch scripts you author must be self-contained**: reference only your
experiment worktree and artifact directories — never `~/.silico/libraries`,
which is a Lab path, not a job path. The job image already provides causalab;
invoke the runner directly, with the same session-local wiring as the
direct-invocation form above — pointed at the *delivered* copy of your
`$WORKDIR` inside the worktree, since pod paths do not exist in the job:

```bash
workdir="{delivered_workdir}"   # your $WORKDIR as delivered in the worktree, e.g. "$PWD/$SILICO_EXPERIMENT_RELATIVE_DIR/<session-dir>"
CAUSALAB_SESSION_CODE="$workdir" PYTHONPATH="$workdir/code${PYTHONPATH:+:$PYTHONPATH}" \
uv run python -m causalab.runner.run_exp --config-name {config_name} \
    experiment_root={cluster_artifacts_dir} \
    "++hydra.searchpath=[file://$workdir/code/configs]"
```

(A shipped preset config needs none of the session-local wiring — just the
`uv run python -m causalab.runner.run_exp` line.)

**Always do a debug pass first** (small dataset, coarse layer scan); once it produces sane artifacts, drop the per-run overrides and run the full preset. The run is **blocking** — redirect verbose stdout to a log inside your working dir's `run/` folder so it doesn't fill the tool-result buffer:

```bash
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" {config_name} > "$WORKDIR/run/run.log" 2>&1
```

**Session-local analyses and methods** (authored via the setup-analyses / setup-methods guides) live under `$WORKDIR/code/analyses/<name>/` and `$WORKDIR/code/methods/<name>/`, with Hydra configs under `$WORKDIR/code/configs/`. They chain in via `- /analysis/<name>` defaults entries and import as the bare `analyses.<name>` / `methods.<name>`. The wrapper wires them up when `CAUSALAB_SESSION_CODE=$WORKDIR` is set: it prepends `$WORKDIR/code/` to `PYTHONPATH` (so `import analyses.<name>` / `import methods.<name>` / `import tasks.<name>` resolve) and appends `$WORKDIR/code/configs/` to Hydra's search path. Shipped `causalab.*` always takes precedence. If you invoke `causalab.runner.run_exp` directly (not via the wrapper), set those manually:

```bash
CAUSALAB_SESSION_CODE="$WORKDIR" PYTHONPATH="$WORKDIR/code:${PYTHONPATH:-}" \
  uv run python -m causalab.runner.run_exp --config-name {config_name} \
    experiment_root="${EXP_ROOT}" \
    "++hydra.searchpath=[file://$WORKDIR/code/configs]" --cfg job
```

Slurm resources resolve from `model.slurm.gpus` and `slurm.time`; override with `--gpus`, `--time`, `--qos` if needed. Add `--wait` to block until a cluster job finishes (it exits with the job's status) instead of hand-rolling a `squeue` poll; the wrapper prints the job's log path at submit (`slurm_logs/<job-name>_<jobid>.{out,err}`, relative to the repo root).

**Size the job to the work, then fan it out — don't run a few big jobs.** Before a large run, smoke-measure the per-item rate on a handful of examples, project the full wall-clock, and **fan to available width** (one SLURM array, many single-GPU shards) rather than running a few large jobs. A grid over layers / tokens is embarrassingly parallel; sharding it is the default, not an optimization, and a few large unsharded jobs leave the cluster idle. And if you are tempted to shrink a *planned* result (fewer facts, fewer seeds, a narrower sweep) to fit a time budget — that is a material scope change: surface it as a decision, never silently down-scope.

**Fan a wide scan or sweep out over GPUs.** To run a whole sweep at once, or shard one analysis's scan (one job per layer, or per layer × token) for a fast result, dispatch through the fan-out orchestrator instead of a bare `run_exp.sh`:

```bash
cd ~/.silico/libraries/causalab-internal
export CAUSALAB_SESSION_CODE="$WORKDIR"
uv run python -m causalab.runner.fanout {config_name} --base "${EXP_ROOT}" \
    --axis-each 'locate.layers=range(0,32)' --axis 'locate.token_positions=[subject],[verb]' \
    --gpus 1 --wait --collect
```

It builds the shard manifest, submits one SLURM array (or runs locally across visible GPUs), waits, and `--collect` recombines outputs. Each shard runs under `--experiment-root ${EXP_ROOT}/shards/<id>`, so session-local code and config resolve as above. Limits to respect: only `locate`-shaped `scores_per_cell` results auto-merge (others land as per-shard files), and a dependency-chained pipeline must be fanned **per stage** (collect between stages), never as one unit. Full reference: `docs/CODEBASE.md` "Fan-out".

## The gate concept

A run is only worth interpreting if its prerequisite signal is real. The discipline — borrowed into every plan as a per-stage **pre-flight gate** — is to name, for each analysis, a falsifiable **sign-of-life floor**: a *should-pass* case the analysis must clear and an informative *should-fail* case it must reject (not trivial-by-construction). A `baseline` whose accuracy is at chance, or a `locate` whose should-fail control scores as high as the real variable, means the downstream chain cannot mean anything yet — fix the upstream stage before spending compute below it. (How to *enforce* that floor between stages — the capped iterate-until-signal loop, the halt-the-chain rule, the per-stage interim report — is sequencing, owned by whatever drives the run; in silico the planner/worker/critic harness owns it.) See `ANALYSIS_GUIDE.md` "Designing sign-of-life sweeps" for how to choose the two cases per analysis.

## Verify the artifacts

Research-mode artifacts land under `<experiment-root>/{analysis}/...` (invariant 7 — the `{task}/{model}` segment is already part of the `--experiment-root` you passed). List them and confirm each chained analysis produced its expected outputs (per the `ANALYSIS_GUIDE.md` "Research Questions → Analyses" table). Common per-analysis files:

- `baseline/`: `accuracy.json`, `per_class_output_dists.safetensors`
- `locate/{method}/{variable}/`: `results.json` (with `best_cell`), `heatmap.png`
- `subspace/{method}_k{k}/`: `metadata.json`, rotation/transform `.pt`, `visualization/features_3d.html`
- `activation_manifold/{method}_s{s}/`: `manifold_spline/ckpt_final.pt`, `metadata.json`, `visualization/manifold_3d.html`
- `output_manifold/`, `path_steering/`, `pullback/`, `attention_pattern/` — see `ANALYSIS_GUIDE.md`.

If something is missing, check `$WORKDIR/run/run.log` for the exception. Common fixes are in `COMMON_PROBLEMS.md`.

## Iterate

To try different layers, methods, `k_features`, etc., either override on the CLI (re-passing `--experiment-root`):

```bash
cd ~/.silico/libraries/causalab-internal
export CAUSALAB_SESSION_CODE="$WORKDIR"
./scripts/run_exp.sh --experiment-root "${EXP_ROOT}" weekdays_8b_locate locate.layers=[15,20]
```

or copy the runner config to a new name and edit it. Re-running a runner config rewrites its artifact subdirectory under the experiment root — branch the config name to preserve old runs. A change to the analysis chain or knobs that is non-trivial belongs back in planning, not in a one-off override.

## Principles

1. **Plan first, run small, then run full.** Always do a debug pass (`task.n_train=16`, a coarse `locate.layers`) before the full preset.
2. **Read `ANALYSIS_GUIDE.md` before each new analysis type** — its "Key Parameter Decisions" tells you what actually needs a decision; everything else has a sensible default.
3. **Inspect with `--cfg job` before every run** — catches misconfigured paths and `resample_variable` × `locate.mode` mismatches early. Use the same `--experiment-root` you will run with.
4. **Keep outputs inside the caller's experiment root.** Don't override `experiment_root` to ephemeral paths like `/tmp/`; those artifacts are invisible to downstream analyses.
5. **Don't write Python that orchestrates runs.** Each analysis already has `main(cfg)`; the dispatcher is `causalab/runner/run_exp.py`.
