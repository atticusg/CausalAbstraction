# Causalab Testing Conventions

Causalab is a codebase for trusted mechanistic interpretability experiments. It contains tested primitives of interpretability methods, expressed as serializable intervention protocols (docs/intervention_protocol.md) executed by engines. The tests must be held to a high standard to create confidence in the results. Trustworthiness and reputation is the most valuable asset of us researchers, we need to back it up with consistently correct experiments.


## Quick Start

Run all CPU tests via `uv run pytest -m "not golden"` (what CI runs). `uv run pytest -m golden` runs only the golden tier, which needs an accelerator and real model weights (see [Golden](#golden)).

## Overview of test types

Every test belongs to exactly one tier, declared via a pytest marker. The five markers fall into two families: **unit tests** — CPU-only, fast, narrow functionality (`numerical_unit`, `property`, `unit`) — and **end-to-end tests** — user-facing behavior through the real CLI (`smoke`, `golden`).

| Tier | Marker | What it asserts | Wall budget |
| --- | --- | --- | --- |
| `numerical_unit` | `@pytest.mark.numerical_unit` | Expected input–output pairs pinned on fixed seeds (CPU): the frozen parity goldens (`tests/neural/parity/goldens/*.json` replayed through protocol documents on tiny-random), task numerical pins (`tests/tasks/<task>/pinned_samples.json`), metric/train-loop values, and the task-driven end-to-end IIA pin (`tests/neural/engines/pytorch_hooks/test_end_to_end_iia.py` — a serialized task table driven through the CLI). Catches sign flips. | <2 min total |
| `property` | `@pytest.mark.property` | Object properties: shape & dtype contracts, invariances / equivariances / determinism (load twice → same digests and canonical form), causal-model invariants. | <1 min total |
| `unit` | `@pytest.mark.unit` | Pure-function tests, parsers, validation rules, small utilities (the "else" bucket). The default tier — marker still required (enforcement does not infer it). Includes the golden tier's CPU structural guard (`tests/golden/test_structural.py`). | <5 min total |
| `smoke` | `@pytest.mark.smoke` | Corpus documents execute end-to-end on `tiny-random` through the real CLI (`tests/neural/engines/pytorch_hooks/test_run_corpus.py`, `test_workflow_run.py`): artifact existence, shapes, indicator ranges — no numerical pins. | <5 min total |
| `golden` | `@pytest.mark.golden` | Real-model runs on an accelerator — the **sole GPU tier**, two sub-tiers under `tests/golden/` with opposite provenance (see [Golden](#golden)). | model-bound; run per document set |

All five markers are registered in `pyproject.toml`. `tests/conftest.py` installs a `pytest_collection_modifyitems` hook that **fails the run** with a `pytest.UsageError` listing offending nodeids if any test lacks a tier marker.

## Conventions

### Mocking policy

1. **Tiny real over mocks.** Use the smallest real implementations: `hf-internal-testing/tiny-random-*` checkpoints instead of mocking a neural network, tiny committed fixture datasets, ...
2. **Mock only at system boundaries.** Reserve mocks for transactional dependencies CI shouldn't hit: HTTP APIs, paid services, time/random, forced error paths. (The CLI's lazily-imported execution engine is stubbed via `sys.modules` in `tests/protocol/test_cli.py` — the same idea.)

### Unit tests

1. **Test file location.** A test for `causalab/<subdir>/<stem>.py` belongs at `tests/<subdir>/test_<stem>.py`. Tier is declared via marker (`pytestmark = pytest.mark.<tier>`, at module, class, or function scope).
2. **One class per tier.** A single file may hold multiple tiers; the typical pattern is one class each.
3. **Task numerical pin.** Guard each task with an LM-free symbolic pin on a task's `CausalModel` and counterfactual generator, as a sidecar `tests/tasks/<task>/pinned_samples.json`. Update via `scripts/update_task_pins.py --task=<name>`, review the diff, then if correct rerun with `--i-have-reviewed-the-diff`.

### Pinned-artifact discipline

Every pinned file follows the same rule: **regenerate via its script, review the diff, never hand-edit.**

| Pin | Script | What a diff means |
|---|---|---|
| `tests/protocol/corpus_digests.json` | `tests/protocol/update_corpus_digests.py` | the canonical form changed — spec §7 treats it as a loader migration |
| `tests/protocol/workflow_digests.json` | `tests/protocol/update_workflow_digests.py` | a shipped workflow's canonical form changed (workflow spec §7). It is also what makes "adding a step type changed no existing document" a check rather than a claim |
| `tests/golden/golden_digests.json` | `tests/golden/update_golden_digests.py` | a golden document or its fixture dataset changed (dataset content digests are part of the canonical form) |
| `tests/golden/drift/drift_goldens.json` | `tests/golden/drift/update_drift_goldens.py` (GPU) | the stack's real-model numerics moved |
| `tests/tasks/<task>/pinned_samples.json` | `scripts/update_task_pins.py` | task prompt/causal-model semantics changed |
| `tests/neural/parity/goldens/*.json` | none — **deliberately frozen** pre-migration captures | do not regenerate; recapturing from the new stack would defeat the anchor |
| `tests/protocol/fixtures/data/weekdays/task_n4_s0.json` | `scripts/build_task_dataset.py` (parameters in its `.manifest.json` sidecar; `--check` verifies without writing) | the task's generator or causal model changed — a committed table is a build product, so the manifest is the recipe |

## End-to-end tests

### Smoke

Corpus documents (`tests/protocols/*_im.json`) run through the real CLI on `hf-internal-testing/tiny-random-LlamaForCausalLM` with `--set` overrides retargeting layers/model at tiny scale. Assertions are existence/shape/dtype only — tiny-random output content is garbage by design. The workflow capstone (`tests/neural/engines/pytorch_hooks/test_workflow_run.py`) runs the whole weekdays pipeline shape the same way, and `test_bundle_entries_run.py` covers the two tensor handoffs between steps: a *swept* fit applied at one selected coordinate (the capstone deliberately collapses that sweep, which is how the gap went unseen) and a mean-ablation harvest reduced at save time. `test_script_step_run.py` is the `script`-step capstone, covering both of its directions in one pipeline: protocol → script → script, and protocol → script → protocol (a fitted basis re-entering a model-touching run, with its ArtifactIdentity checked). The capstone also pins `--resume`: an edit to a step's *script* busts the reuse, which is why the script's content hash is in the digest.

Step scripts themselves are unit-tested under `tests/steps/`: each shipped script against a hand-computed oracle plus a determinism assertion (`numerical_unit`), the `select`/`plot` reductions and their refusals (`unit`), and the isolation path proved by comparing pids. That a real `causalab validate` of a workflow whose script imports torch never itself imports torch is checked in a subprocess (`tests/protocol/test_load_is_torch_free.py`), since `tests/conftest.py` imports torch at session scope.

### Golden

The sole accelerator tier, under `tests/golden/`, in two sub-tiers with **opposite provenance rules**:

1. **Paper goldens** (`tests/golden/test_paper_goldens.py`): protocol documents on real open models (gpt2, gpt2-xl, Llama-3.1-8B, gemma-2-2b-it), asserted against values from **published papers or the VeriFires task packages** (`tests/golden/paper_goldens.json`, one provenance entry per value — paper quote, VeriFires task + leaf id, sidedness, band/floor). No value in that file may come from running this stack. Fixture datasets are seeded, committed JSON from `tests/golden/fixtures/generators/`.
2. **Drift pins** (`tests/golden/drift/`): the successor of the retired chat-coherent tier — Qwen3-4B documents whose values **are** pinned from a reviewed run of this stack on the canonical cuda box (`update_drift_goldens.py --i-have-reviewed-the-diff`), replayed within tolerance for run-to-run drift detection. Baseline gate: accuracy ≥ 0.9. Until the first capture lands the replay skips. Fidelity gap vs the retired tier, recorded here: protocol v1 has no chat-template path, so the drift documents use raw completions (the old tier's chat template + answer directive is not reproduced).

Practicalities:

- Run with `uv run pytest tests/golden -m golden` on a box with an accelerator. Gated models (`meta-llama/Llama-3.1-8B`, `google/gemma-2-2b-it`) need a licensed `HF_TOKEN` — without one the load 401s with nothing naming the cause.
- The reference engine runs each (model, input) forward group as **one batch**; the largest paper-golden document (hours, 1,152 rows on Llama-8B) wants ~35GB free in bf16.
- A CPU structural guard (`tests/golden/test_structural.py`, tier `unit`, runs in CI) keeps the tier honest without loading models: documents load and digest to their pins, every non-pending goldens entry is claimed by exactly one test, provenance fields are present.
- VRAM reclamation: `tests/conftest.py::pytest_runtest_teardown` runs `gc.collect()` + `torch.cuda.empty_cache()` after every `golden`-marked test.

## GitHub CI

`.github/workflows/test.yml` runs on every pull request and on push to `main`, on GitHub-hosted CPU runners. Two jobs:

1. **`lint`** — `uv run pre-commit run --all-files` (ruff + the stack); fails the job on any finding.
2. **`test`** — `uv run pytest -m "not golden"` over the CPU tiers (`unit` / `numerical_unit` / `property` / `smoke`), publishing a JUnit report.

The `golden` tier is the sole accelerator tier and is not part of the CPU pull-request pipeline; run it on a GPU node with `uv run pytest -m golden` (see [Golden](#golden)).
