# Runner

`causalab/runner/` is the orchestration layer: a thin shell over `analyses/`
that turns a Hydra config into a run. A **runner config** names one task, one
model, and an ordered list of analyses; the runner composes them, dispatches each
analysis in order, and routes every output under a single `experiment_root`.

This doc is the practical "how to run" reference. For the config *system* itself
(config groups, the `# @package` mechanism, how a runner composes its defaults
list), see [`docs/CODEBASE.md` §2](../../docs/CODEBASE.md#2-runner-config-system).

## Running an experiment

The wrapper `scripts/run_exp.sh` is the normal entry point. It discovers a runner
config by basename, then shells out to the Hydra entry point:

```bash
# Basename is enough — the wrapper finds it under causalab/configs/runners/<group>/
scripts/run_exp.sh baseline_demo

# Print the fully resolved config without running
scripts/run_exp.sh baseline_demo --cfg job
```

The equivalent direct invocation (no discovery — pass the config path relative to
`causalab/configs/`):

```bash
uv run python -m causalab.runner.run_exp --config-name runners/demos/baseline_demo
```

**Overriding config values.** Everything after the runner name is a Hydra
override (`key=value`), and CLI overrides have the highest precedence:

```bash
scripts/run_exp.sh he_pipeline subspace.k_features=8 locate.layers=[10,11,12]
```

`--experiment-root DIR` is a wrapper convenience for `experiment_root=DIR`. Hydra
introspection flags (`--cfg job|hydra|all`, `--resolve`, `--package`) may be
passed anywhere and are forwarded to Hydra.

## How a runner config is composed

A runner config is a **primary Hydra config** (`# @package _global_`) living under
`causalab/configs/runners/<group>/`. It inherits `/base`, selects a `/task:` and a
`/model:`, and pulls analyses by listing them in its defaults list. **Order in the
defaults list is order of execution** — the runner walks the composed config for
top-level keys carrying a `_name_` sentinel and dispatches each in insertion
order.

```yaml
# @package _global_
defaults:
  - /base
  - /task: hierarchical_equality
  - /model: llama31_8b
  - /analysis/locate      # runs first
  - /analysis/subspace    # runs second
  - _self_
```

See [`docs/CODEBASE.md` §2](../../docs/CODEBASE.md#2-runner-config-system) for the
full composition rules (absolute `- /analysis/<name>` paths, `_subdir`/`_output_dir`
directives, `post:` visualization steps).

## Where artifacts land (`--experiment-root`)

`experiment_root` is the single source of truth for output paths. Its default is
`artifacts/${task.name}/${model.id}` — the gitignored `artifacts/` tree at the
repo root. Each analysis writes under `${experiment_root}/<analysis_name>/…`.
Override per run with `--experiment-root <path>` (or `experiment_root=<path>`).

A step that crashes has its partial output dir removed (unless it pre-existed the
run) so a half-written result is never mistaken for a complete one.

## SLURM dispatch (`--slurm`)

`--slurm` submits the run as an sbatch job instead of running inline. GPU count,
walltime, and job name are resolved **from the runner config** — `model.slurm.gpus`
and the `slurm.time` block (default in `base.yaml`) — so the config stays the
single source of truth. `causalab/runner/slurm_args.py` reads them; the wrapper
forwards them to `sbatch`.

```bash
scripts/run_exp.sh --slurm age_8b_k64
scripts/run_exp.sh --slurm --wait age_8b_k64                 # block until it finishes
scripts/run_exp.sh --slurm --qos=opportunistic --time=08:00:00 --gpus=2 alphabet_70b_k128
```

- `--gpus N` / `--time HH:MM:SS` override the config-resolved resources.
- `--qos` is one of `normal | opportunistic | scavenge`.
- `--wait` blocks until the job completes and exits with its status.
- Logs land in `slurm_logs/<job-name>_<jobid>.{out,err}`, relative to the repo
  root; the resolved path is printed at submit.

### Fan-out across many GPUs

To split one experiment into many independent shards (e.g. one job per `layer`, or
per `layer × token`), use `causalab/runner/fanout.py`
(`uv run python -m causalab.runner.fanout <runner> --axis …`). It is
analysis-agnostic and reuses this wrapper per shard. See
[`docs/CODEBASE.md` §2 "Fan-out"](../../docs/CODEBASE.md#2-runner-config-system).

## Session-local code injection

*(Optional power-user feature — most runs never touch this.)*

By default a run only sees task, model, method, and analysis code from the
**installed `causalab` package**. Session-local code injection lets a run instead
pick up prototypes that live *beside its own experiment outputs*, so you can
iterate on a new task/method/analysis (or a one-off runner YAML) without editing
the installed package.

It activates automatically when `--experiment-root` lives under an
`agent_logs/<session>/` directory. The parent `<session>/` is then treated as a
**session directory** (referred to below as `${SESSION_DIR}`), and the runner:

1. **Prepends `${SESSION_DIR}/code/` to `PYTHONPATH`**, so top-level imports like
   `import analyses.<name>`, `import methods.<name>`, and `import tasks.<name>`
   resolve to prototypes under `code/` — used as a *fallback* only, after the
   shipped `causalab.*` namespace, so a session-local module never shadows a
   shipped one.
2. **Adds `${SESSION_DIR}/code/configs/` to Hydra's search path**, so a runner's
   `- /analysis/<name>` defaults entries — and runner YAMLs under
   `code/configs/runners/` — resolve from beside the outputs.

`CAUSALAB_SESSION_CODE` is the environment variable that carries the session
directory across an sbatch re-exec: sbatch does not reliably propagate the
inherited environment into the job step, so `--slurm`/fan-out dispatch forwards it
explicitly (`--export=ALL,CAUSALAB_SESSION_CODE=<session>`), and the in-job
re-exec re-detects the session from it.

The wrapper (`scripts/run_exp.sh`) performs this detection and injection; the
Python side (`run_exp.py`, `causalab/tasks/loader.py`, `fanout.py`) only consults
`CAUSALAB_SESSION_CODE` to decide whether the session-local fallback is in play.
