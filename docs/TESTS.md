# Causalab Testing Conventions

Causalab is a codebase for trusted mechanistic interpretability experiments. It contains tested primitives of interpretability methods and analyses, which can be chained together to answer research questions. The tests must be held to a high standard to create confidence in the results. Trustworthiness and reputation is the most valuable asset of us researchers, we need to back it up with consistently correct experiments.


## Quick Start

Run all tests via `uv run pytest`. Select tiers with pytest markers — for example `uv run pytest -m "not golden"` runs every CPU tier and skips the GPU-only `golden` tier, and `uv run pytest -m golden` runs only the golden tier (which needs a GPU).

## Overview of test types

Every test belongs to exactly one tier, declared via a pytest marker. The five markers fall into two families: **unit tests** — CPU-only, fast, narrow functionality (`numerical_unit`, `property`, `unit`) — and **end-to-end tests** — user-facing experiment behavior (`smoke`, `golden`).

| Tier | Marker | What it asserts | Required for (`causalab/` subdir) | Wall budget |
| --- | --- | --- | --- | --- |
| `numerical_unit` | `@pytest.mark.numerical_unit` | Expected input–output pairs pinned on fixed seeds — losses, metric values, sample outputs at known inputs (CPU); catches sign flips. This tier covers the CPU task-numerical-pin and module-level value tests; runner-level value pinning is the `golden` tier, not this one. | `methods`, `tasks` | <2 min total |
| `property` | `@pytest.mark.property` | Object properties: shape & dtype contracts at module boundaries; invariances / equivariances / determinism (same seed → byte-identical artifact). For `tasks/`, also hypothesis-driven, LM-free causal-model invariants — counterfactual roundtrip, off-path no-ops, mechanism determinism. | `causal`, `methods`, `neural`, `tasks` | <1 min total |
| `unit` | `@pytest.mark.unit` | Pure-function tests, parsers, small utilities (the "else" bucket). The default tier — marker still required (enforcement does not infer it). | `causal`, `io`, `neural`, `runner` | <5 min total |
| `smoke` | `@pytest.mark.smoke` | Tiny-config end-to-end runs of every smoke runner survive on `tiny-random` (CPU): fixed seed, smallest model, ≤8 examples. Asserts only that the declared `expected_artifacts` are produced — output content is garbage, no numerical checks (use `assert_runner_completed`). | `analyses`, `methods`, `tasks` | <30s per runner, <5 min total |
| `golden` | `@pytest.mark.golden` | Value-pinned full-pipeline runner goldens on the real, open-weight coherent model (`chat-coherent`, GPU) — the **sole GPU tier**; results must replicate within tolerance. Auto-discovered by `tests/end_to_end/test_goldens.py` from `tests/end_to_end/goldens/*.json`; new goldens need only a config + bootstrapped JSON, no new test code. | `analyses`, `methods` | <10 min total |

All five markers are registered in `pyproject.toml`. `tests/conftest.py` installs a `pytest_collection_modifyitems` hook that **fails the run** with a `pytest.UsageError` listing offending nodeids if any test lacks a tier marker.

The **Required for** column names the `causalab/` subdirs whose modules must reach that tier. `unit` / `numerical_unit` / `property` are required **direct** (a test written for the module); `smoke` / `golden` are required **transitive** (a passing end-to-end runner touches the module — neither has a per-module form). Two subdirs are unlisted on purpose: `configs/` is exempt (it has no importable modules of its own — its behavior is exercised by the end-to-end config tests) and `__init__.py` files are excluded.


## Conventions

### Mocking policy

1. **Tiny real over mocks.** Use the smallest real implementations: `tiny_random_model()` instead of mocking a neural network, tiny datasets, ...
2. **Mock only at system boundaries.** Reserve mocks for transactional dependencies CI shouldn't hit: SLURM, wandb, HTTP APIs, paid services, time/random, forced error paths.

### Unit tests

1. **Required tiers by subdir.** [Overview of test types](#overview-of-test-types) describes tiers a script must carry.
2. **Test file location.** A test for `causalab/<subdir>/<stem>.py` belongs at `tests/<subdir>/test_<stem>.py`. Tier is declared via marker (`pytestmark = pytest.mark.<tier>`, at module, class, or function scope).
3. **One class per tier.** A single file may hold multiple tiers; the typical pattern is one class each:

   ```python
   class TestThingUnit:
       pytestmark = pytest.mark.unit
       def test_returns_expected_shape(self): ...

   class TestThingNumericalUnit:
       pytestmark = pytest.mark.numerical_unit
       def test_loss_at_fixed_seed(self): ...
   ```

4. **Cross-cutting end-to-end tests.** The four of them live under `tests/end_to_end/` because they parametrize over every YAML in `tests/end_to_end/configs/`, not a single source module. When adding coverage, add per-module test files under `tests/` mirroring the `causalab/` layout, covering the tiers required for that subdir.

5. **Task numerical pin.** Guard each task with an LM-free symbolic pin on a task's `CausalModel` and counterfactual generator. It lives beside the task's tests as a sidecar `tests/tasks/<task>/pinned_samples.json`:

   ```json
   {
     "task": "MCQA",
     "seeds": [0, 1, 2, 3, 4],
     "samples": [
       {"seed": 0,
        "input": {"raw_input": "...", "raw_output": "...", "...": "..."},
        "counterfactual_inputs": [{"raw_input": "...", "...": "..."}]}
     ]
   }
   ```
   Update task numerical pins via `scripts/update_task_pins.py --task=<name>`, review the diff, then if correct rerun with `--i-have-reviewed-the-diff`.
6. **Factory-task gap.** `graph_walk`, `identity_naming`, and `natural_domains_arithmetic` have no task numerical pin yet — `walk_task_samples` raises `NotImplementedError` for them until a serialisable `task_cfg` shim lands, so they carry no `numerical_unit` direct coverage until then.

### End-to-end tests

Config driven end to end tests checking pipeline behavior. Each config is scored against up to four gates (cheapest first):

| Gate | Dir | Model | Asserts | Breadth | CI job |
|---|---|---|---|---|---|
| **compose** | `configs/{smoke,golden}/*` | — | Hydra composes the full config tree without error (`tests/end_to_end/test_compose.py`) | every e2e config; <10 s CPU | `test` (`not golden`) |
| **dispatch** | `configs/{smoke,golden}/*` | — | `_iter_analysis_steps` names an importable `causalab.analyses.<name>.main` (same file) | every e2e config | `test` (`not golden`) |
| **smoke** | `configs/smoke/*` | `tiny-random` (CPU) | each config's declared `expected_artifacts` exist | broad: every task baseline + every tiny-random-able analysis chain | `test` (`not golden`) |
| **golden** | `configs/golden/*` | `chat-coherent` (GPU) | value pins (accuracy and/or extracted analysis scalars) within tolerance | narrow: tasks/chains a real model handles | GPU (run manually) |

#### Smoke

1. **Model and resources.** Every `configs/smoke/*` config executes end-to-end at tiny scale on the random-init Llama stub (`tests/end_to_end/test_smoke.py::test_runner_smoke`) on CPU. Each config pins `/model: tiny-random`, bakes its own task scale, and declares its `expected_artifacts:`; the test only fixes `experiment_root`.
2. **Coverage.** One smoke config per shipped task baseline, plus the analysis chains `tiny-random` can express.
3. **Needs coherent English or real activation structure → golden.** Any test needing a model that generates coherent English (or real activation structure) should be a golden, not a smoke: e.g. `output_manifold` and `pullback` are mathematically undefined on random-init weights (`causalab/methods/spline/cubic.py:106-110`), so they're golden-only.

#### Golden
A runner-scope, full-pipeline, real-model end-to-end pin under `tests/end_to_end/goldens/<runner>.json` — the output of a small model that reliably solves the task at a fixed seed. The only test class that uses a GPU.

1. **Model and resources.** The sole coherent fixture is `chat-coherent` = `Qwen/Qwen3-4B-Instruct-2507`: ungated on HuggingFace, a standard decoder-only architecture, and (chat template + a terse answer directive) clears the `0.9` gate on every shipped golden task.
2. **Numeric gate** (`tests/end_to_end/test_goldens.py`) — pinned values must re-run within tolerance; the backstop against a refactor silently shifting a loss or tensor layout. For baseline: `accuracy ≥ 0.9` on `n_samples ≥ 30` at the pinned seed, `prob_accuracy` defined (not `null`), and the per-key reductions (`mean/std/first/last`) reproduce within tolerance on rerun.
3. **Sample size.** `n_samples ∈ {30, 50, 100}` (upper bound = the `numerical_unit` wall budget), set via `cfg.task.n_train` — the baseline analysis measures accuracy on the *train* split (`causalab/analyses/baseline/main.py:209`), not `n_test`.
4. **Golden test format.** Each golden runner has a JSON in `tests/end_to_end/goldens/<runner>.json`:

   ```json
   {
     "runner": "golden/mcqa",
     "seed": 0,
     "model": "chat-coherent",
     "deterministic": false,
     "tolerance": {"default": 1e-5, "<metric_name>": 1e-3},
     "values": {"accuracy.accuracy": 1.0, "metric_a": 0.0}
   }
   ```

5. **Comparison.** `extract_values` walks the whole output tree — baseline `accuracy.json` metrics plus every `*.safetensors` as a `.sha256` and per-tensor `mean/std/first/last/shape` (non-finite reductions dropped, only `shape` anchors them). Each value must reproduce within its `tolerance`: a flat `{key: tolerance}` map, missing keys → `default` (`1e-5`); `accuracy.accuracy` → `0` (exact ratio), `accuracy.prob_accuracy` → `1e-4` (BLAS/CUDA drift), reductions → `1e-5`. Absolute, not relative (`tests/end_to_end/test_goldens.py:72`).
6. **Determinism is not a gate on GPU.** All goldens run on GPU (non-deterministic), so `.sha256` is logged for completeness but not asserted in practice.
7. **Updating goldens.** `tests/end_to_end/update_goldens.py` prints a rich diff and refuses to write without `--i-have-reviewed-the-diff`:

   ```bash
   # inspect current vs new (no writes):
   uv run python tests/end_to_end/update_goldens.py --baseline golden/age

   # accept the diff (writes the JSON):
   uv run python tests/end_to_end/update_goldens.py --baseline golden/age \
       --i-have-reviewed-the-diff
   ```


## GitHub CI

`.github/workflows/test.yml` runs on every pull request and on push to `main`, on GitHub-hosted CPU runners. Two jobs:

1. **`lint`** — `uv run pre-commit run --all-files` (ruff + the stack); fails the job on any finding.
2. **`test`** — `uv run pytest` over the CPU tiers (`unit` / `numerical_unit` / `property` / `smoke`), publishing a JUnit report.

The `golden` tier is the sole GPU tier and is not part of the CPU pull-request pipeline; run it on a GPU node with `uv run pytest -m golden` (see [Golden](#golden)).